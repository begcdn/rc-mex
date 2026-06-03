from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "whose", "with", "tell", "me", "give", "show", "list", "does", "did",
}


@dataclass
class SmokeExample:
    question_id: str
    question: str
    program_index: int
    start_entity_ids: set[str]
    start_entity_name: str
    gold_answer_ids: set[str]
    gold_answer_labels: list[str]


@dataclass
class SearchState:
    frontier_ids: set[str]
    score: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    relation_sequences: list[list[str]] = field(default_factory=list)
    entity_sequences: list[list[str]] = field(default_factory=list)
    soft_signals: dict[str, float] = field(default_factory=dict)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)

    log_stage(1, 4, f"Loading KQA Pro KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    log_line(f"Entities: {len(graph.entities)}")
    log_line(f"Relation+direction keys: {len(graph.relations())}")

    log_stage(2, 4, f"Selecting simple 2-hop questions from {args.questions}")
    questions = load_json(args.questions)
    examples, selection_stats = select_examples(graph, questions, args.max_examples, args.max_questions)
    log_line(f"Selected questions: {len(examples)}")
    log_line(f"Skipped unsupported: {selection_stats.get('unsupported_program', 0)}")
    log_line(f"Skipped empty gold execution: {selection_stats.get('empty_gold_execution', 0)}")

    log_stage(3, 4, "Running baseline, soft proof-state, and future-aware proof-state beams")
    rows = []
    for index, example in enumerate(examples, start=1):
        baseline = run_baseline_path_beam(
            graph=graph,
            example=example,
            top_k=args.top_k,
            beam_width=args.beam_width,
            relation_cap=args.relation_cap,
            sample_entities=args.sample_entities,
            max_branch_entities=args.max_branch_entities,
            debug_trace=args.debug_trace,
        )
        proof_state = run_soft_proof_state_beam(
            graph=graph,
            example=example,
            top_k=args.top_k,
            beam_width=args.beam_width,
            relation_cap=args.relation_cap,
            sample_entities=args.sample_entities,
            max_branch_entities=args.max_branch_entities,
            noisy_branch_threshold=args.noisy_branch_threshold,
            debug_trace=args.debug_trace,
        )
        future_aware = run_future_aware_proof_state_beam(
            graph=graph,
            example=example,
            top_k=args.top_k,
            beam_width=args.beam_width,
            relation_cap=args.relation_cap,
            sample_entities=args.sample_entities,
            max_branch_entities=args.max_branch_entities,
            noisy_branch_threshold=args.noisy_branch_threshold,
            debug_trace=args.debug_trace,
        )
        row = build_prediction_row(graph, example, baseline, proof_state, future_aware)
        print_runtime_log(row, index, len(examples))
        rows.append(row)

    log_stage(4, 4, "Writing smoke-test outputs")
    metrics = compute_metrics(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "selection_stats": selection_stats,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    (output_dir / "report.md").write_text(write_report(metrics, rows), encoding="utf-8")
    error_overlap = build_error_overlap(rows)
    write_json(output_dir / "error_overlap.json", error_overlap)
    (output_dir / "error_overlap.md").write_text(write_error_overlap_markdown(error_overlap), encoding="utf-8")
    log_line("predictions.jsonl")
    log_line("metrics.json")
    log_line("report.md")
    log_line("error_overlap.json")
    log_line("error_overlap.md")
    if args.debug_trace:
        trace_rows = select_trace_rows(rows, args.debug_limit)
        write_jsonl(output_dir / "debug_trace.jsonl", [debug_trace_json_row(row) for row in trace_rows])
        (output_dir / "debug_trace.md").write_text(write_debug_trace_markdown(trace_rows), encoding="utf-8")
        log_line("debug_trace.jsonl")
        log_line("debug_trace.md")
    print(f"Wrote proof-state search smoke outputs to {output_dir}")


def select_examples(
    graph: KnowledgeGraph,
    samples: list[dict[str, Any]],
    max_examples: int,
    max_questions: int | None,
) -> tuple[list[SmokeExample], dict[str, int]]:
    examples: list[SmokeExample] = []
    stats: dict[str, int] = defaultdict(int)
    for program_index, sample in enumerate(samples):
        if max_questions is not None and stats["questions_seen"] >= max_questions:
            break
        stats["questions_seen"] += 1
        try:
            example = parse_simple_two_hop(graph, sample, program_index)
        except ValueError as exc:
            stats[str(exc)] += 1
            continue
        examples.append(example)
        stats["examples_selected"] += 1
        if len(examples) >= max_examples:
            break
    return examples, dict(stats)


def parse_simple_two_hop(graph: KnowledgeGraph, sample: dict[str, Any], program_index: int) -> SmokeExample:
    program = sample.get("program", []) or []
    try:
        return parse_legacy_exact_two_relate(graph, sample, program_index)
    except ValueError as legacy_error:
        try:
            return parse_linear_two_relate_entity_answer(graph, sample, program_index)
        except ValueError:
            raise legacy_error


def parse_legacy_exact_two_relate(graph: KnowledgeGraph, sample: dict[str, Any], program_index: int) -> SmokeExample:
    program = sample.get("program", []) or []
    if len(program) != 3:
        raise ValueError("unsupported_program")
    find_step, first_relate, second_relate = program
    if find_step.get("function") != "Find":
        raise ValueError("unsupported_program")
    if first_relate.get("function") != "Relate" or second_relate.get("function") != "Relate":
        raise ValueError("unsupported_program")
    if first_relate.get("dependencies") != [0] or second_relate.get("dependencies") != [1]:
        raise ValueError("unsupported_program")
    find_inputs = find_step.get("inputs", []) or []
    if not find_inputs:
        raise ValueError("unsupported_program")
    start_ids = graph.find_entities(str(find_inputs[0]))
    if not start_ids:
        raise ValueError("empty_start_entity")
    gold_state = set(start_ids)
    for step in [first_relate, second_relate]:
        inputs = step.get("inputs", []) or []
        if len(inputs) < 2:
            raise ValueError("unsupported_program")
        gold_state, _ = graph.follow(gold_state, str(inputs[0]), str(inputs[1]), max_proofs=10000)
        if not gold_state:
            raise ValueError("empty_gold_execution")
    return SmokeExample(
        question_id=str(sample.get("id") or sample.get("qid") or sample.get("ID") or f"val:{program_index}"),
        question=str(sample.get("question", "")),
        program_index=program_index,
        start_entity_ids=start_ids,
        start_entity_name=str(find_inputs[0]),
        gold_answer_ids=gold_state,
        gold_answer_labels=entity_labels(graph, gold_state),
    )


def parse_linear_two_relate_entity_answer(graph: KnowledgeGraph, sample: dict[str, Any], program_index: int) -> SmokeExample:
    """Accept real KQA Pro simple chains with type filters.

    KQA Pro validation has no literal three-step Find->Relate->Relate programs.
    The simple entity-answer chains usually look like:

      Find -> Relate -> FilterConcept -> Relate -> FilterConcept -> What

    This parser uses the gold program only to select that controlled subset, find
    the topic entity, compute the final gold answer, and infer hop budget=2.
    Search still never sees the gold relation IDs or intermediate prefixes.
    """
    program = sample.get("program", []) or []
    if not program or program[-1].get("function") != "What":
        raise ValueError("unsupported_program")
    chain_indices = dependency_chain_to_root(program, len(program) - 1)
    chain = [program[index] for index in chain_indices]
    functions = [step.get("function") for step in chain]
    if not functions or functions[0] != "Find":
        raise ValueError("unsupported_program")
    allowed = {"Find", "Relate", "FilterConcept", "What"}
    if any(function not in allowed for function in functions):
        raise ValueError("unsupported_program")
    if functions.count("Relate") != 2:
        raise ValueError("unsupported_program")
    if functions[-1] != "What":
        raise ValueError("unsupported_program")

    find_inputs = chain[0].get("inputs", []) or []
    if not find_inputs:
        raise ValueError("unsupported_program")
    start_ids = graph.find_entities(str(find_inputs[0]))
    if not start_ids:
        raise ValueError("empty_start_entity")

    gold_state = execute_linear_gold_chain(graph, start_ids, chain[1:])
    if not gold_state:
        raise ValueError("empty_gold_execution")
    return SmokeExample(
        question_id=str(sample.get("id") or sample.get("qid") or sample.get("ID") or f"val:{program_index}"),
        question=str(sample.get("question", "")),
        program_index=program_index,
        start_entity_ids=start_ids,
        start_entity_name=str(find_inputs[0]),
        gold_answer_ids=gold_state,
        gold_answer_labels=entity_labels(graph, gold_state),
    )


def dependency_chain_to_root(program: list[dict[str, Any]], final_index: int) -> list[int]:
    out = []
    seen = set()
    index = final_index
    while True:
        if index in seen or index < 0 or index >= len(program):
            raise ValueError("unsupported_program")
        seen.add(index)
        out.append(index)
        dependencies = program[index].get("dependencies", []) or []
        if not dependencies:
            break
        if len(dependencies) != 1:
            raise ValueError("unsupported_program")
        index = int(dependencies[0])
    return list(reversed(out))


def execute_linear_gold_chain(
    graph: KnowledgeGraph,
    start_ids: set[str],
    chain: list[dict[str, Any]],
) -> set[str]:
    state = set(start_ids)
    for step in chain:
        function = step.get("function")
        inputs = step.get("inputs", []) or []
        if function == "Relate":
            if len(inputs) < 2:
                raise ValueError("unsupported_program")
            state, _ = graph.follow(state, str(inputs[0]), str(inputs[1]), max_proofs=10000)
        elif function == "FilterConcept":
            if not inputs:
                raise ValueError("unsupported_program")
            concept_ids = graph.find_concepts(str(inputs[0]))
            if not concept_ids:
                raise ValueError("unsupported_program")
            state = {entity_id for entity_id in state if graph.is_instance_of_any(entity_id, concept_ids)}
        elif function == "What":
            continue
        else:
            raise ValueError("unsupported_program")
        if not state:
            return set()
    return state


def run_baseline_path_beam(
    graph: KnowledgeGraph,
    example: SmokeExample,
    top_k: int,
    beam_width: int,
    relation_cap: int,
    sample_entities: int,
    max_branch_entities: int,
    debug_trace: bool = False,
) -> dict[str, Any]:
    states = [
        SearchState(
            frontier_ids={entity_id},
            score=0.0,
            entity_sequences=[[graph.entity_name(entity_id)]],
            relation_sequences=[[]],
        )
        for entity_id in sorted(example.start_entity_ids)
    ]
    expansion_count = 0
    trace: list[dict[str, Any]] = []
    for hop in [1, 2]:
        next_states: list[SearchState] = []
        for state in states:
            source_id = next(iter(state.frontier_ids))
            frontier = graph.candidate_relations([source_id], cap=relation_cap, sample_entities=sample_entities)
            ranked = rank_relations(example.question, frontier)
            for candidate in ranked[:top_k]:
                targets = relation_targets(graph, source_id, candidate, max_branch_entities)
                expansion_count += 1
                for target_id in targets:
                    step = evidence_step(graph, source_id, candidate, target_id, hop, len(targets))
                    next_states.append(
                        SearchState(
                            frontier_ids={target_id},
                            score=state.score + candidate["label_score"],
                            evidence=state.evidence + [step],
                            relation_sequences=[(state.relation_sequences[0] if state.relation_sequences else []) + [candidate["relation_id"]]],
                            entity_sequences=[(state.entity_sequences[0] if state.entity_sequences else [graph.entity_name(source_id)]) + [graph.entity_name(target_id)]],
                            soft_signals={
                                "lexical_similarity": candidate["lexical_similarity"],
                                "frequency_bonus": candidate["frequency_bonus"],
                                "label_score": candidate["label_score"],
                            },
                        )
                    )
        next_states.sort(key=lambda item: (-item.score, state_sort_key(item)))
        if debug_trace:
            trace.append(
                {
                    "hop": hop,
                    "top_candidate_states": [
                        summarize_baseline_state(state, rank)
                        for rank, state in enumerate(next_states[:5], start=1)
                    ],
                }
            )
        states = next_states[:beam_width]
    return search_result(
        graph,
        states,
        example.gold_answer_ids,
        expansion_count,
        mode="baseline_path_beam",
        debug_trace=trace,
    )


def run_soft_proof_state_beam(
    graph: KnowledgeGraph,
    example: SmokeExample,
    top_k: int,
    beam_width: int,
    relation_cap: int,
    sample_entities: int,
    max_branch_entities: int,
    noisy_branch_threshold: int,
    debug_trace: bool = False,
) -> dict[str, Any]:
    answer_type = guess_answer_type(example.question)
    states = [
        SearchState(
            frontier_ids=set(example.start_entity_ids),
            score=0.0,
            entity_sequences=[[graph.entity_name(entity_id)] for entity_id in sorted(example.start_entity_ids)],
            relation_sequences=[[]],
            soft_signals={"answer_type_known": 1.0 if answer_type else 0.0},
        )
    ]
    expansion_count = 0
    trace: list[dict[str, Any]] = []
    for hop in [1, 2]:
        next_states: list[SearchState] = []
        for state in states:
            expansion_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source_id in sorted(state.frontier_ids):
                frontier = graph.candidate_relations([source_id], cap=relation_cap, sample_entities=sample_entities)
                ranked = rank_relations(example.question, frontier)
                for candidate in ranked[:top_k]:
                    targets = relation_targets(graph, source_id, candidate, max_branch_entities)
                    expansion_count += 1
                    for target_id in targets:
                        expansion_groups[target_id].append(
                            {
                                "source_id": source_id,
                                "candidate": candidate,
                                "branch_size": len(targets),
                            }
                        )
            for target_id, fragments in expansion_groups.items():
                steps = [
                    evidence_step(graph, fragment["source_id"], fragment["candidate"], target_id, hop, fragment["branch_size"])
                    for fragment in fragments
                ]
                signals = soft_fragment_signals(
                    graph=graph,
                    question=example.question,
                    answer_type=answer_type,
                    state=state,
                    target_id=target_id,
                    steps=steps,
                    hop=hop,
                    noisy_branch_threshold=noisy_branch_threshold,
                )
                relation_sequences = extend_relation_sequences(state.relation_sequences, steps)
                entity_sequences = extend_entity_sequences(graph, state.entity_sequences, steps)
                next_states.append(
                    SearchState(
                        frontier_ids={target_id},
                        score=state.score + signals["total_delta"],
                        evidence=state.evidence + steps,
                        relation_sequences=relation_sequences,
                        entity_sequences=entity_sequences,
                        soft_signals=merge_signal_dicts(state.soft_signals, signals),
                    )
                )
        next_states.sort(key=lambda item: (-item.score, state_sort_key(item)))
        if debug_trace:
            trace.append(
                {
                    "hop": hop,
                    "top_candidate_states": [
                        summarize_proof_state(state, rank)
                        for rank, state in enumerate(next_states[:5], start=1)
                    ],
                }
            )
        states = next_states[:beam_width]
    return search_result(
        graph,
        states,
        example.gold_answer_ids,
        expansion_count,
        mode="soft_proof_state_beam",
        debug_trace=trace,
    )


def run_future_aware_proof_state_beam(
    graph: KnowledgeGraph,
    example: SmokeExample,
    top_k: int,
    beam_width: int,
    relation_cap: int,
    sample_entities: int,
    max_branch_entities: int,
    noisy_branch_threshold: int,
    debug_trace: bool = False,
) -> dict[str, Any]:
    answer_type = guess_answer_type(example.question)
    states = [
        SearchState(
            frontier_ids=set(example.start_entity_ids),
            score=0.0,
            entity_sequences=[[graph.entity_name(entity_id)] for entity_id in sorted(example.start_entity_ids)],
            relation_sequences=[[]],
            soft_signals={"answer_type_known": 1.0 if answer_type else 0.0},
        )
    ]
    expansion_count = 0
    trace: list[dict[str, Any]] = []
    for hop in [1, 2]:
        next_states: list[SearchState] = []
        for state in states:
            expansion_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source_id in sorted(state.frontier_ids):
                frontier = graph.candidate_relations([source_id], cap=relation_cap, sample_entities=sample_entities)
                ranked = rank_relations(example.question, frontier)
                for candidate in ranked[:top_k]:
                    targets = relation_targets(graph, source_id, candidate, max_branch_entities)
                    expansion_count += 1
                    for target_id in targets:
                        expansion_groups[target_id].append(
                            {
                                "source_id": source_id,
                                "candidate": candidate,
                                "branch_size": len(targets),
                            }
                        )
            for target_id, fragments in expansion_groups.items():
                steps = [
                    evidence_step(graph, fragment["source_id"], fragment["candidate"], target_id, hop, fragment["branch_size"])
                    for fragment in fragments
                ]
                signals = future_aware_fragment_signals(
                    graph=graph,
                    question=example.question,
                    answer_type=answer_type,
                    state=state,
                    target_id=target_id,
                    steps=steps,
                    hop=hop,
                    relation_cap=relation_cap,
                    sample_entities=sample_entities,
                    noisy_branch_threshold=noisy_branch_threshold,
                )
                relation_sequences = extend_relation_sequences(state.relation_sequences, steps)
                entity_sequences = extend_entity_sequences(graph, state.entity_sequences, steps)
                next_states.append(
                    SearchState(
                        frontier_ids={target_id},
                        score=state.score + signals["total_delta"],
                        evidence=state.evidence + steps,
                        relation_sequences=relation_sequences,
                        entity_sequences=entity_sequences,
                        soft_signals=merge_signal_dicts(state.soft_signals, signals),
                    )
                )
        next_states.sort(key=lambda item: (-item.score, state_sort_key(item)))
        if debug_trace:
            trace.append(
                {
                    "hop": hop,
                    "top_candidate_states": [
                        summarize_future_aware_state(state, rank)
                        for rank, state in enumerate(next_states[:5], start=1)
                    ],
                }
            )
        states = next_states[:beam_width]
    return search_result(
        graph,
        states,
        example.gold_answer_ids,
        expansion_count,
        mode="future_aware_proof_state_beam",
        debug_trace=trace,
    )


def rank_relations(question: str, frontier: list[Any]) -> list[dict[str, Any]]:
    ranked = []
    for relation in frontier:
        label = f"{relation.predicate.replace('_', ' ')} {relation.direction}"
        lexical_similarity = char_ngram_similarity(question, label)
        frequency_bonus = 0.03 * min(1.0, float(relation.frequency) / 5.0)
        ranked.append(
            {
                "relation_id": relation.predicate,
                "direction": relation.direction,
                "frequency": relation.frequency,
                "lexical_similarity": lexical_similarity,
                "frequency_bonus": frequency_bonus,
                "label_score": lexical_similarity + frequency_bonus,
            }
        )
    ranked.sort(key=lambda item: (-item["label_score"], item["relation_id"], item["direction"]))
    return ranked


def relation_targets(graph: KnowledgeGraph, source_id: str, candidate: dict[str, Any], max_branch_entities: int) -> list[str]:
    targets = []
    for relation in graph.iter_relations(source_id):
        if relation.get("predicate") != candidate["relation_id"]:
            continue
        if relation.get("direction") != candidate["direction"]:
            continue
        target_id = str(relation.get("object", ""))
        if target_id:
            targets.append(target_id)
    return sorted(set(targets))[:max_branch_entities]


def evidence_step(
    graph: KnowledgeGraph,
    source_id: str,
    candidate: dict[str, Any],
    target_id: str,
    hop: int,
    branch_size: int,
) -> dict[str, Any]:
    return {
        "hop": hop,
        "from_entity_id": source_id,
        "from_entity": graph.entity_name(source_id),
        "relation_id": candidate["relation_id"],
        "direction": candidate["direction"],
        "to_entity_id": target_id,
        "to_entity": graph.entity_name(target_id),
        "to_types": graph.entity_type_names(target_id),
        "relation_label_score": candidate["label_score"],
        "lexical_similarity": candidate.get("lexical_similarity", candidate["label_score"]),
        "frequency_bonus": candidate.get("frequency_bonus", 0.0),
        "branch_size": branch_size,
        "readable": f"{graph.entity_name(source_id)} --{candidate['relation_id']}[{candidate['direction']}]--> {graph.entity_name(target_id)}",
    }


def soft_fragment_signals(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    state: SearchState,
    target_id: str,
    steps: list[dict[str, Any]],
    hop: int,
    noisy_branch_threshold: int,
) -> dict[str, float]:
    best_label = max((float(step["relation_label_score"]) for step in steps), default=0.0)
    avg_label = sum(float(step["relation_label_score"]) for step in steps) / max(1, len(steps))
    convergence = 0.18 * max(0, len(steps) - 1)
    branch_size = max((int(step["branch_size"]) for step in steps), default=1)
    noisy_penalty = -0.18 if branch_size >= noisy_branch_threshold else 0.0
    redundancy = -0.10 if repeats_relation_pattern(state, steps) else 0.0
    type_score = type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0
    progress = plausible_progress(question, graph, target_id, steps, hop)
    uncertainty_floor = 0.03 if best_label < 0.08 and noisy_penalty == 0.0 and redundancy == 0.0 else 0.0
    total = (
        0.65 * best_label
        + 0.20 * avg_label
        + convergence
        + 0.16 * type_score
        + progress
        + uncertainty_floor
        + noisy_penalty
        + redundancy
    )
    return {
        "best_label_similarity": best_label,
        "avg_label_similarity": avg_label,
        "convergence_bonus": convergence,
        "type_compatibility": type_score,
        "plausible_progress": progress,
        "uncertainty_floor": uncertainty_floor,
        "noisy_branch_penalty": noisy_penalty,
        "redundancy_penalty": redundancy,
        "total_delta": total,
    }


def future_aware_fragment_signals(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    state: SearchState,
    target_id: str,
    steps: list[dict[str, Any]],
    hop: int,
    relation_cap: int,
    sample_entities: int,
    noisy_branch_threshold: int,
) -> dict[str, float]:
    relation_text = " ".join(step["relation_id"].replace("_", " ") for step in state.evidence + steps)
    remaining_terms = remaining_question_terms(question, relation_text)
    current_relevance = max((float(step["relation_label_score"]) for step in steps), default=0.0)
    future = future_satisfiability(
        graph=graph,
        question=question,
        remaining_terms=remaining_terms,
        entity_id=target_id,
        relation_cap=relation_cap,
        sample_entities=sample_entities,
    )
    type_score = type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0
    progress = plausible_progress(question, graph, target_id, steps, hop)
    useful_convergence = useful_convergence_bonus(
        graph=graph,
        target_id=target_id,
        steps=steps,
        future_satisfiability_score=future,
        type_score=type_score,
    )
    surface_penalty = surface_convergence_penalty(
        graph=graph,
        state=state,
        target_id=target_id,
        steps=steps,
        future_satisfiability_score=future,
        answer_type=answer_type,
    )
    redundancy = -0.12 if repeats_relation_pattern(state, steps) else 0.0
    branch_size = max((int(step["branch_size"]) for step in steps), default=1)
    noisy_penalty = -0.20 if branch_size >= noisy_branch_threshold else 0.0
    drift = drift_penalty(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        remaining_terms=remaining_terms,
        future_satisfiability_score=future,
        hop=hop,
    )
    total = (
        0.58 * current_relevance
        + 0.34 * future
        + useful_convergence
        + 0.14 * type_score
        + progress
        + surface_penalty
        + redundancy
        + drift
        + noisy_penalty
    )
    return {
        "current_relevance": current_relevance,
        "future_satisfiability": future,
        "useful_convergence": useful_convergence,
        "type_compatibility": type_score,
        "progress": progress,
        "surface_convergence_penalty": surface_penalty,
        "redundancy_penalty": redundancy,
        "drift_penalty": drift,
        "noisy_branch_penalty": noisy_penalty,
        "remaining_terms_count": float(len(remaining_terms)),
        "total_delta": total,
    }


def future_satisfiability(
    graph: KnowledgeGraph,
    question: str,
    remaining_terms: set[str],
    entity_id: str,
    relation_cap: int,
    sample_entities: int,
) -> float:
    frontier = graph.candidate_relations([entity_id], cap=relation_cap, sample_entities=sample_entities)
    if not frontier:
        return 0.0
    remaining_text = " ".join(sorted(remaining_terms))
    best_remaining = 0.0
    best_question = 0.0
    for relation in frontier:
        label = relation.predicate.replace("_", " ")
        best_question = max(best_question, char_ngram_similarity(question, label))
        if remaining_text:
            best_remaining = max(best_remaining, token_overlap_score(remaining_terms, tokenize_content(label)))
    return min(1.0, 0.70 * best_remaining + 0.30 * best_question)


def useful_convergence_bonus(
    graph: KnowledgeGraph,
    target_id: str,
    steps: list[dict[str, Any]],
    future_satisfiability_score: float,
    type_score: float,
) -> float:
    if len(steps) < 2:
        return 0.0
    relation_labels = {step["relation_id"] for step in steps}
    different_relations = len(relation_labels) > 1
    future_helpful = future_satisfiability_score > 0.12
    type_helpful = type_score > 0.0
    if future_helpful or type_helpful or different_relations:
        return 0.16 + 0.04 * min(2, len(steps) - 2)
    return 0.0


def surface_convergence_penalty(
    graph: KnowledgeGraph,
    state: SearchState,
    target_id: str,
    steps: list[dict[str, Any]],
    future_satisfiability_score: float,
    answer_type: str,
) -> float:
    penalty = 0.0
    if len(steps) >= 2 and future_satisfiability_score < 0.08:
        penalty -= 0.16
    if has_inverse_loop(state, steps):
        penalty -= 0.18
    relation_labels = [step["relation_id"] for step in steps]
    if len(steps) >= 2 and len(set(relation_labels)) == 1:
        penalty -= 0.08
    target_text = " ".join([graph.entity_name(target_id), *graph.entity_type_names(target_id)]).casefold()
    if answer_type in {"person", "work"} and any(word in target_text for word in ["country", "organization", "location"]):
        penalty -= 0.08
    return penalty


def drift_penalty(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    target_id: str,
    remaining_terms: set[str],
    future_satisfiability_score: float,
    hop: int,
) -> float:
    target_text = " ".join([graph.entity_name(target_id), *graph.entity_type_names(target_id)]).casefold()
    generic_terms = ["country", "diplomatic", "time zone", "located in", "administrative", "language"]
    generic = any(term in target_text for term in generic_terms)
    expected = type_compatibility(graph, target_id, answer_type) > 0.0 if answer_type else False
    needed_text = " ".join(sorted(remaining_terms))
    target_match = char_ngram_similarity(needed_text, target_text) if needed_text else 0.0
    if hop == 2 and answer_type and not expected and future_satisfiability_score < 0.06 and target_match < 0.05:
        return -0.14
    if generic and future_satisfiability_score < 0.08 and target_match < 0.08:
        return -0.10
    return 0.0


def has_inverse_loop(state: SearchState, steps: list[dict[str, Any]]) -> bool:
    previous_edges = {
        (step["from_entity_id"], step["relation_id"], step["to_entity_id"])
        for step in state.evidence
    }
    for step in steps:
        reverse = (step["to_entity_id"], step["relation_id"], step["from_entity_id"])
        if reverse in previous_edges:
            return True
    return False


def plausible_progress(question: str, graph: KnowledgeGraph, target_id: str, steps: list[dict[str, Any]], hop: int) -> float:
    target_text = " ".join([graph.entity_name(target_id), *graph.entity_type_names(target_id)])
    entity_match = char_ngram_similarity(question, target_text)
    relation_text = " ".join(step["relation_id"].replace("_", " ") for step in steps)
    relation_match = char_ngram_similarity(question, relation_text)
    if relation_match > 0.18:
        return 0.12
    if hop == 1 and entity_match > 0.10:
        return 0.08
    if entity_match > 0.16:
        return 0.10
    return 0.0


def repeats_relation_pattern(state: SearchState, steps: list[dict[str, Any]]) -> bool:
    previous = [step["relation_id"] for step in state.evidence]
    current = [step["relation_id"] for step in steps]
    if not previous or not current:
        return False
    return all(relation in previous for relation in current)


def type_compatibility(graph: KnowledgeGraph, entity_id: str, answer_type: str) -> float:
    if not answer_type:
        return 0.0
    type_names = " ".join(graph.entity_type_names(entity_id)).casefold()
    name = graph.entity_name(entity_id).casefold()
    checks = {
        "person": ["person", "human", "people"],
        "location": ["location", "city", "country", "state", "province", "place", "territorial"],
        "time": ["time", "date", "year"],
        "number": ["number", "quantity"],
        "film": ["film", "movie", "television"],
        "organization": ["organization", "company", "institution", "team"],
        "work": ["work", "book", "film", "song", "album"],
    }
    for needle in checks.get(answer_type, [answer_type]):
        if needle in type_names or needle in name:
            return 1.0
    return 0.0


def guess_answer_type(question: str) -> str:
    q = question.casefold()
    if q.startswith("who") or "which person" in q:
        return "person"
    if q.startswith("where") or "which country" in q or "which city" in q or "what country" in q:
        return "location"
    if q.startswith("when") or "what year" in q or "date" in q:
        return "time"
    if "how many" in q or "number" in q:
        return "number"
    if "film" in q or "movie" in q:
        return "film"
    if "company" in q or "organization" in q or "team" in q:
        return "organization"
    return ""


def extend_relation_sequences(previous: list[list[str]], steps: list[dict[str, Any]]) -> list[list[str]]:
    base = previous or [[]]
    out = []
    for seq in base:
        for step in steps:
            out.append(seq + [step["relation_id"]])
    return out[:8]


def extend_entity_sequences(graph: KnowledgeGraph, previous: list[list[str]], steps: list[dict[str, Any]]) -> list[list[str]]:
    base = previous or [[]]
    out = []
    for seq in base:
        for step in steps:
            if seq:
                out.append(seq + [step["to_entity"]])
            else:
                out.append([step["from_entity"], step["to_entity"]])
    return out[:8]


def merge_signal_dicts(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    merged = dict(previous)
    for key, value in current.items():
        merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def search_result(
    graph: KnowledgeGraph,
    states: list[SearchState],
    gold_answer_ids: set[str],
    expansion_count: int,
    mode: str,
    debug_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for state in states:
        for entity_id in state.frontier_ids:
            row = grouped.setdefault(
                entity_id,
                {
                    "answer_id": entity_id,
                    "answer_label": graph.entity_name(entity_id),
                    "score": 0.0,
                    "best_path_score": float("-inf"),
                    "paths": [],
                    "is_gold": entity_id in gold_answer_ids,
                },
            )
            row["score"] += state.score
            row["best_path_score"] = max(row["best_path_score"], state.score)
            row["paths"].append(
                {
                    "path_score": state.score,
                    "evidence": state.evidence,
                    "relation_sequences": state.relation_sequences,
                    "entity_sequences": state.entity_sequences,
                    "soft_signals": state.soft_signals,
                    "readable": " | ".join(step["readable"] for step in state.evidence),
                }
            )
    candidates = list(grouped.values())
    for candidate in candidates:
        candidate["num_paths"] = len(candidate["paths"])
        if candidate["best_path_score"] == float("-inf"):
            candidate["best_path_score"] = 0.0
    candidates.sort(key=lambda item: (-item["score"], -item["best_path_score"], item["answer_label"], item["answer_id"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    predicted = {candidates[0]["answer_id"]} if candidates else set()
    p = precision(gold_answer_ids, predicted)
    r = recall(gold_answer_ids, predicted)
    return {
        "mode": mode,
        "candidate_answers": candidates,
        "top_answer": candidates[0] if candidates else None,
        "predicted_answer_ids": sorted(predicted),
        "hits_at_1": bool(predicted & gold_answer_ids),
        "exact_match": predicted == gold_answer_ids,
        "final_answer_precision": p,
        "final_answer_recall": r,
        "final_answer_f1": f1(p, r),
        "gold_generated": any(candidate["is_gold"] for candidate in candidates),
        "final_result_size": len(predicted),
        "candidate_count": len(candidates),
        "expansion_count": expansion_count,
        "debug_trace": debug_trace or [],
    }


def summarize_baseline_state(state: SearchState, rank: int) -> dict[str, Any]:
    last_step = state.evidence[-1] if state.evidence else {}
    return {
        "rank": rank,
        "hop": last_step.get("hop"),
        "candidate_path": " | ".join(step["readable"] for step in state.evidence),
        "relation_chosen": last_step.get("relation_id", ""),
        "direction": last_step.get("direction", ""),
        "lexical_similarity": last_step.get("lexical_similarity", 0.0),
        "frequency_bonus": last_step.get("frequency_bonus", 0.0),
        "final_baseline_score": state.score,
        "target_entity": next(iter(state.frontier_ids), ""),
        "target_label": last_step.get("to_entity", ""),
    }


def summarize_proof_state(state: SearchState, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "hop": state.evidence[-1].get("hop") if state.evidence else None,
        "candidate_evidence_state": " | ".join(step["readable"] for step in state.evidence),
        "fragments": [step["readable"] for step in state.evidence],
        "best_label_similarity": state.soft_signals.get("best_label_similarity", 0.0),
        "avg_label_similarity": state.soft_signals.get("avg_label_similarity", 0.0),
        "convergence_bonus": state.soft_signals.get("convergence_bonus", 0.0),
        "type_compatibility": state.soft_signals.get("type_compatibility", 0.0),
        "plausible_progress": state.soft_signals.get("plausible_progress", 0.0),
        "uncertainty_floor": state.soft_signals.get("uncertainty_floor", 0.0),
        "noisy_branch_penalty": state.soft_signals.get("noisy_branch_penalty", 0.0),
        "redundancy_penalty": state.soft_signals.get("redundancy_penalty", 0.0),
        "final_proof_state_score": state.score,
        "target_entity": next(iter(state.frontier_ids), ""),
    }


def summarize_future_aware_state(state: SearchState, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "hop": state.evidence[-1].get("hop") if state.evidence else None,
        "candidate_evidence_state": " | ".join(step["readable"] for step in state.evidence),
        "fragments": [step["readable"] for step in state.evidence],
        "current_relevance": state.soft_signals.get("current_relevance", 0.0),
        "future_satisfiability": state.soft_signals.get("future_satisfiability", 0.0),
        "useful_convergence": state.soft_signals.get("useful_convergence", 0.0),
        "type_compatibility": state.soft_signals.get("type_compatibility", 0.0),
        "progress": state.soft_signals.get("progress", 0.0),
        "surface_convergence_penalty": state.soft_signals.get("surface_convergence_penalty", 0.0),
        "redundancy_penalty": state.soft_signals.get("redundancy_penalty", 0.0),
        "drift_penalty": state.soft_signals.get("drift_penalty", 0.0),
        "noisy_branch_penalty": state.soft_signals.get("noisy_branch_penalty", 0.0),
        "final_score": state.score,
        "target_entity": next(iter(state.frontier_ids), ""),
    }


def build_prediction_row(
    graph: KnowledgeGraph,
    example: SmokeExample,
    baseline: dict[str, Any],
    proof_state: dict[str, Any],
    future_aware: dict[str, Any],
) -> dict[str, Any]:
    baseline_correct = baseline["hits_at_1"]
    proof_correct = proof_state["hits_at_1"]
    future_correct = future_aware["hits_at_1"]
    if baseline_correct and proof_correct and future_correct:
        failure_type = "both_correct"
    elif future_correct and not proof_correct:
        failure_type = "future_aware_correct"
    elif proof_correct:
        failure_type = "proof_state_correct"
    elif baseline_correct:
        failure_type = "baseline_correct"
    elif not baseline["gold_generated"] and not proof_state["gold_generated"] and not future_aware["gold_generated"]:
        failure_type = "gold_not_generated"
    elif not baseline["candidate_answers"] or not proof_state["candidate_answers"] or not future_aware["candidate_answers"]:
        failure_type = "empty_frontier"
    else:
        failure_type = "both_fail"
    return {
        "question_id": example.question_id,
        "program_index": example.program_index,
        "question": example.question,
        "gold_answer_ids": sorted(example.gold_answer_ids),
        "gold_answers": example.gold_answer_labels,
        "start_entity": {
            "name": example.start_entity_name,
            "entity_ids": sorted(example.start_entity_ids),
            "labels": entity_labels(graph, example.start_entity_ids),
        },
        "baseline_path_beam": baseline,
        "soft_proof_state_beam": proof_state,
        "future_aware_proof_state_beam": future_aware,
        "failure_type": failure_type,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row["baseline_path_beam"] for row in rows]
    proof = [row["soft_proof_state_beam"] for row in rows]
    future = [row["future_aware_proof_state_beam"] for row in rows]
    return {
        "number_of_selected_questions": len(rows),
        "baseline_hits_at_1": avg_bool(item["hits_at_1"] for item in baseline),
        "proof_state_hits_at_1": avg_bool(item["hits_at_1"] for item in proof),
        "future_aware_hits_at_1": avg_bool(item["hits_at_1"] for item in future),
        "baseline_exact_match": avg_bool(item["exact_match"] for item in baseline),
        "proof_state_exact_match": avg_bool(item["exact_match"] for item in proof),
        "future_aware_exact_match": avg_bool(item["exact_match"] for item in future),
        "baseline_final_answer_f1": avg(item["final_answer_f1"] for item in baseline),
        "proof_state_final_answer_f1": avg(item["final_answer_f1"] for item in proof),
        "future_aware_final_answer_f1": avg(item["final_answer_f1"] for item in future),
        "baseline_gold_generated_rate": avg_bool(item["gold_generated"] for item in baseline),
        "proof_state_gold_generated_rate": avg_bool(item["gold_generated"] for item in proof),
        "future_aware_gold_generated_rate": avg_bool(item["gold_generated"] for item in future),
        "average_candidate_count_baseline": avg(item["candidate_count"] for item in baseline),
        "average_candidate_count_proof_state": avg(item["candidate_count"] for item in proof),
        "average_candidate_count_future_aware": avg(item["candidate_count"] for item in future),
        "average_expansion_count_baseline": avg(item["expansion_count"] for item in baseline),
        "average_expansion_count_proof_state": avg(item["expansion_count"] for item in proof),
        "average_expansion_count_future_aware": avg(item["expansion_count"] for item in future),
        "average_final_result_size_baseline": avg(item["final_result_size"] for item in baseline),
        "average_final_result_size_proof_state": avg(item["final_result_size"] for item in proof),
        "average_final_result_size_future_aware": avg(item["final_result_size"] for item in future),
        "proof_state_wins": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ),
        "baseline_wins": sum(
            1 for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "future_aware_wins_over_current_proof_state": sum(
            1 for row in rows
            if row["future_aware_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "current_proof_state_wins_over_future_aware": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["future_aware_proof_state_beam"]["final_answer_f1"]
        ),
        "both_correct": sum(
            1 for row in rows
            if row["baseline_path_beam"]["hits_at_1"]
            and row["soft_proof_state_beam"]["hits_at_1"]
            and row["future_aware_proof_state_beam"]["hits_at_1"]
        ),
        "both_fail": sum(
            1 for row in rows
            if not row["baseline_path_beam"]["hits_at_1"]
            and not row["soft_proof_state_beam"]["hits_at_1"]
            and not row["future_aware_proof_state_beam"]["hits_at_1"]
        ),
        "future_aware_avoids_surface_convergence": sum(
            1 for row in rows
            if future_top_surface_penalty(row) < 0.0
            and row["future_aware_proof_state_beam"]["final_answer_f1"] >= row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "future_aware_hurts_current_proof_state": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["hits_at_1"] and not row["future_aware_proof_state_beam"]["hits_at_1"]
        ),
        "failure_counts": dict(Counter(row["failure_type"] for row in rows)),
    }


def write_report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Proof-State Search Smoke Test",
        "",
        "This tests whether soft proof-state search shows signal over independent path/entity beam search on simple KQA Pro two-hop chains.",
        "",
        "No gold relation IDs, gold prefixes, relation cards, LLM constraint extraction, ToG/Freebase, or quantum-inspired scoring are used during search.",
        "",
        "This run compares three modes: `baseline_path_beam`, current `soft_proof_state_beam`, and `future_aware_proof_state_beam`.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Future-Aware Wins Over Current Proof-State",
        "",
        *debug_section_rows(select_debug_rows(rows, "future"), limit=5),
        "## Current Proof-State Wins Over Future-Aware",
        "",
        *debug_section_rows(select_debug_rows(rows, "current_over_future"), limit=5),
        "## Future-Aware Surface-Convergence Avoidance Cases",
        "",
        *debug_section_rows(select_debug_rows(rows, "surface_avoidance"), limit=5),
        "## Future-Aware Hurts Current Proof-State Cases",
        "",
        *debug_section_rows(select_debug_rows(rows, "future_hurts"), limit=5),
        "",
        "## Proof-State Wins",
        "",
        *debug_section_rows(select_debug_rows(rows, "proof"), limit=5),
        "## Baseline Wins",
        "",
        *debug_section_rows(select_debug_rows(rows, "baseline"), limit=5),
        "## Both Fail",
        "",
        *debug_section_rows(select_debug_rows(rows, "both_fail"), limit=5),
    ]
    return "\n".join(lines)


def select_trace_rows(rows: list[dict[str, Any]], debug_limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind in ["future", "current_over_future", "surface_avoidance", "future_hurts", "proof", "baseline", "both_fail"]:
        for row in select_debug_rows(rows, kind):
            key = str(row["question_id"])
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= debug_limit:
                return selected
    for row in rows:
        key = str(row["question_id"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= debug_limit:
            break
    return selected


def future_top_surface_penalty(row: dict[str, Any]) -> float:
    top = row["future_aware_proof_state_beam"].get("top_answer") or {}
    paths = top.get("paths", []) if top else []
    if not paths:
        return 0.0
    return float(paths[0].get("soft_signals", {}).get("surface_convergence_penalty", 0.0))


def debug_trace_json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "program_index": row["program_index"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "start_entity": row["start_entity"],
        "failure_type": row["failure_type"],
        "baseline_correct": row["baseline_path_beam"]["hits_at_1"],
        "proof_state_correct": row["soft_proof_state_beam"]["hits_at_1"],
        "future_aware_correct": row["future_aware_proof_state_beam"]["hits_at_1"],
        "baseline_top_answer": row["baseline_path_beam"]["top_answer"],
        "proof_state_top_answer": row["soft_proof_state_beam"]["top_answer"],
        "future_aware_top_answer": row["future_aware_proof_state_beam"]["top_answer"],
        "baseline_debug_trace": row["baseline_path_beam"].get("debug_trace", []),
        "proof_state_debug_trace": row["soft_proof_state_beam"].get("debug_trace", []),
        "future_aware_debug_trace": row["future_aware_proof_state_beam"].get("debug_trace", []),
        "explanation": proof_state_choice_explanation(row),
        "future_aware_explanation": future_aware_choice_explanation(row),
    }


def write_debug_trace_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Proof-State Search Debug Trace",
        "",
        "This file explains why the current heuristic baseline and proof-state searches chose their top path/state.",
        "",
        "No search logic, scores, prompts, LLMs, or relation cards are changed by this debug mode.",
        "",
    ]
    for row in rows:
        baseline = row["baseline_path_beam"]
        proof = row["soft_proof_state_beam"]
        future = row["future_aware_proof_state_beam"]
        baseline_top = baseline["top_answer"] or {}
        proof_top = proof["top_answer"] or {}
        future_top = future["top_answer"] or {}
        lines.extend(
            [
                f"## {row['question_id']}",
                "",
                f"Question: {row['question']}",
                "",
                f"Gold answer: `{', '.join(row['gold_answers'])}`",
                "",
                f"Baseline top final path: {top_path_readable(baseline_top)}",
                "",
                f"Proof-state top final state: {top_path_readable(proof_top)}",
                "",
                f"Future-aware top final state: {top_path_readable(future_top)}",
                "",
                f"Baseline correct: `{baseline['hits_at_1']}`",
                f"Proof-state correct: `{proof['hits_at_1']}`",
                f"Future-aware correct: `{future['hits_at_1']}`",
                "",
                "### Baseline Hop Trace",
                "",
                *baseline_trace_lines(baseline.get("debug_trace", [])),
                "### Proof-State Hop Trace",
                "",
                *proof_trace_lines(proof.get("debug_trace", [])),
                "### Future-Aware Proof-State Hop Trace",
                "",
                *future_aware_trace_lines(future.get("debug_trace", [])),
                "### Why Proof-State Chose This Over Baseline",
                "",
                *proof_state_choice_explanation(row),
                "### Why Future-Aware Chose This",
                "",
                *future_aware_choice_explanation(row),
                "",
            ]
        )
    return "\n".join(lines)


def baseline_trace_lines(trace: list[dict[str, Any]]) -> list[str]:
    if not trace:
        return ["_No baseline trace captured._", ""]
    lines: list[str] = []
    for hop in trace:
        lines.append(f"#### Hop {hop['hop']}")
        lines.append("")
        lines.append("| Rank | Candidate path | Relation | Lexical similarity | Frequency bonus | Final baseline score |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | `{relation}` `{direction}` | {lex:.4f} | {freq:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_path", "")),
                    relation=escape_md(state.get("relation_chosen", "")),
                    direction=escape_md(state.get("direction", "")),
                    lex=float(state.get("lexical_similarity", 0.0)),
                    freq=float(state.get("frequency_bonus", 0.0)),
                    score=float(state.get("final_baseline_score", 0.0)),
                )
            )
        lines.append("")
    return lines


def proof_trace_lines(trace: list[dict[str, Any]]) -> list[str]:
    if not trace:
        return ["_No proof-state trace captured._", ""]
    lines: list[str] = []
    for hop in trace:
        lines.append(f"#### Hop {hop['hop']}")
        lines.append("")
        lines.append(
            "| Rank | Candidate evidence state | Best label | Avg label | Convergence | Type | Progress | Uncertain | Noisy | Redundant | Final score |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | {best:.4f} | {avg:.4f} | {conv:.4f} | {typ:.4f} | {prog:.4f} | {unc:.4f} | {noisy:.4f} | {red:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_evidence_state", "")),
                    best=float(state.get("best_label_similarity", 0.0)),
                    avg=float(state.get("avg_label_similarity", 0.0)),
                    conv=float(state.get("convergence_bonus", 0.0)),
                    typ=float(state.get("type_compatibility", 0.0)),
                    prog=float(state.get("plausible_progress", 0.0)),
                    unc=float(state.get("uncertainty_floor", 0.0)),
                    noisy=float(state.get("noisy_branch_penalty", 0.0)),
                    red=float(state.get("redundancy_penalty", 0.0)),
                    score=float(state.get("final_proof_state_score", 0.0)),
                )
            )
        lines.append("")
    return lines


def future_aware_trace_lines(trace: list[dict[str, Any]]) -> list[str]:
    if not trace:
        return ["_No future-aware proof-state trace captured._", ""]
    lines: list[str] = []
    for hop in trace:
        lines.append(f"#### Hop {hop['hop']}")
        lines.append("")
        lines.append(
            "| Rank | Candidate evidence state | Current | Future | Convergence | Type | Progress | Surface | Redundant | Drift | Noisy | Final score |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | {cur:.4f} | {future:.4f} | {conv:.4f} | {typ:.4f} | {prog:.4f} | {surf:.4f} | {red:.4f} | {drift:.4f} | {noisy:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_evidence_state", "")),
                    cur=float(state.get("current_relevance", 0.0)),
                    future=float(state.get("future_satisfiability", 0.0)),
                    conv=float(state.get("useful_convergence", 0.0)),
                    typ=float(state.get("type_compatibility", 0.0)),
                    prog=float(state.get("progress", 0.0)),
                    surf=float(state.get("surface_convergence_penalty", 0.0)),
                    red=float(state.get("redundancy_penalty", 0.0)),
                    drift=float(state.get("drift_penalty", 0.0)),
                    noisy=float(state.get("noisy_branch_penalty", 0.0)),
                    score=float(state.get("final_score", 0.0)),
                )
            )
        lines.append("")
    return lines


def proof_state_choice_explanation(row: dict[str, Any]) -> list[str]:
    proof_top = row["soft_proof_state_beam"].get("top_answer") or {}
    paths = proof_top.get("paths", []) if proof_top else []
    signals = paths[0].get("soft_signals", {}) if paths else {}
    positive_components = {
        key: float(signals.get(key, 0.0))
        for key in [
            "best_label_similarity",
            "avg_label_similarity",
            "convergence_bonus",
            "type_compatibility",
            "plausible_progress",
            "uncertainty_floor",
        ]
    }
    negative_components = {
        key: float(signals.get(key, 0.0))
        for key in ["noisy_branch_penalty", "redundancy_penalty"]
    }
    strongest = max(positive_components.items(), key=lambda item: item[1], default=("", 0.0))
    lines = [
        f"- Strongest positive component: `{strongest[0] or 'none'}` = `{strongest[1]:.4f}`",
        f"- Avoided noisy branch: `{negative_components.get('noisy_branch_penalty', 0.0) >= 0.0}` "
        f"(noisy penalty `{negative_components.get('noisy_branch_penalty', 0.0):.4f}`)",
        f"- Convergence helped: `{positive_components.get('convergence_bonus', 0.0) > 0.0}` "
        f"(bonus `{positive_components.get('convergence_bonus', 0.0):.4f}`)",
        f"- Type compatibility helped: `{positive_components.get('type_compatibility', 0.0) > 0.0}` "
        f"(score `{positive_components.get('type_compatibility', 0.0):.4f}`)",
        f"- Redundancy penalty applied: `{negative_components.get('redundancy_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('redundancy_penalty', 0.0):.4f}`)",
    ]
    if row["soft_proof_state_beam"]["hits_at_1"] and not row["baseline_path_beam"]["hits_at_1"]:
        lines.append("- Outcome: proof-state beat baseline on this question.")
    elif row["baseline_path_beam"]["hits_at_1"] and not row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: baseline beat proof-state on this question.")
    elif row["baseline_path_beam"]["hits_at_1"] and row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: both methods were correct.")
    else:
        lines.append("- Outcome: both methods failed.")
    return lines


def future_aware_choice_explanation(row: dict[str, Any]) -> list[str]:
    future_top = row["future_aware_proof_state_beam"].get("top_answer") or {}
    paths = future_top.get("paths", []) if future_top else []
    signals = paths[0].get("soft_signals", {}) if paths else {}
    positive_components = {
        key: float(signals.get(key, 0.0))
        for key in ["current_relevance", "future_satisfiability", "useful_convergence", "type_compatibility", "progress"]
    }
    negative_components = {
        key: float(signals.get(key, 0.0))
        for key in ["surface_convergence_penalty", "redundancy_penalty", "drift_penalty", "noisy_branch_penalty"]
    }
    strongest = max(positive_components.items(), key=lambda item: item[1], default=("", 0.0))
    lines = [
        f"- Strongest positive component: `{strongest[0] or 'none'}` = `{strongest[1]:.4f}`",
        f"- Future satisfiability helped: `{positive_components.get('future_satisfiability', 0.0) > 0.0}` "
        f"(score `{positive_components.get('future_satisfiability', 0.0):.4f}`)",
        f"- Useful convergence helped: `{positive_components.get('useful_convergence', 0.0) > 0.0}` "
        f"(bonus `{positive_components.get('useful_convergence', 0.0):.4f}`)",
        f"- Surface convergence penalty applied: `{negative_components.get('surface_convergence_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('surface_convergence_penalty', 0.0):.4f}`)",
        f"- Drift penalty applied: `{negative_components.get('drift_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('drift_penalty', 0.0):.4f}`)",
        f"- Noisy branch penalty applied: `{negative_components.get('noisy_branch_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('noisy_branch_penalty', 0.0):.4f}`)",
    ]
    if row["future_aware_proof_state_beam"]["hits_at_1"] and not row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: future-aware beat current proof-state on this question.")
    elif row["soft_proof_state_beam"]["hits_at_1"] and not row["future_aware_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: future-aware hurt a case current proof-state got right.")
    elif row["future_aware_proof_state_beam"]["hits_at_1"] and row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: both proof-state variants were correct.")
    else:
        lines.append("- Outcome: both proof-state variants failed.")
    return lines


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def select_debug_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "future":
        selected = [
            row for row in rows
            if row["future_aware_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "current_over_future":
        selected = [
            row for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["future_aware_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "surface_avoidance":
        selected = [
            row for row in rows
            if future_top_surface_penalty(row) < 0.0
            and row["future_aware_proof_state_beam"]["final_answer_f1"] >= row["soft_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "future_hurts":
        selected = [
            row for row in rows
            if row["soft_proof_state_beam"]["hits_at_1"] and not row["future_aware_proof_state_beam"]["hits_at_1"]
        ]
    elif kind == "proof":
        selected = [
            row for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ]
    elif kind == "baseline":
        selected = [
            row for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ]
    else:
        selected = [
            row for row in rows
            if not row["baseline_path_beam"]["hits_at_1"] and not row["soft_proof_state_beam"]["hits_at_1"]
        ]
    return selected[:5]


def debug_section_rows(rows: list[dict[str, Any]], limit: int) -> list[str]:
    if not rows:
        return ["_None._", ""]
    lines = []
    for row in rows[:limit]:
        baseline_top = row["baseline_path_beam"]["top_answer"] or {}
        proof_top = row["soft_proof_state_beam"]["top_answer"] or {}
        future_top = row["future_aware_proof_state_beam"]["top_answer"] or {}
        baseline_path = top_path_readable(baseline_top)
        proof_path = top_path_readable(proof_top)
        future_path = top_path_readable(future_top)
        likely = likely_reason(row)
        lines.extend(
            [
                f"### {row['question_id']}",
                f"- Question: {row['question']}",
                f"- Gold answer: {row['gold_answers']}",
                f"- Baseline top answer: `{baseline_top.get('answer_label', '')}`",
                f"- Baseline top path: {baseline_path}",
                f"- Proof-state top answer: `{proof_top.get('answer_label', '')}`",
                f"- Proof-state evidence: {proof_path}",
                f"- Future-aware top answer: `{future_top.get('answer_label', '')}`",
                f"- Future-aware evidence: {future_path}",
                f"- Likely reason: {likely}",
                "",
            ]
        )
    return lines


def build_error_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [error_overlap_case(row) for row in rows]
    summary_keys = [
        "same_top_answer_as_baseline",
        "same_top_answer_as_current_proof_state",
        "same_first_hop_relation_as_baseline",
        "same_first_hop_relation_as_current_proof_state",
        "same_second_hop_relation_as_baseline",
        "same_relation_sequence_as_baseline",
        "same_drift_family_as_baseline",
        "future_repeats_baseline_mistake",
        "future_repeats_baseline_first_hop_different_final",
        "future_loses_current_proof_state_win",
        "future_avoids_surface_convergence_successfully",
        "penalties_tiny_compared_with_positive_terms",
    ]
    summary = {key: sum(1 for case in cases if case[key]) for key in summary_keys}
    summary["total_questions"] = len(cases)
    summary["drift_family_counts_baseline"] = dict(Counter(case["baseline_drift_family"] for case in cases))
    summary["drift_family_counts_future_aware"] = dict(Counter(case["future_aware_drift_family"] for case in cases))
    return {
        "summary": summary,
        "cases": cases,
        "examples": {
            "future_repeats_baseline_mistakes": select_overlap_examples(cases, "future_repeats_baseline_mistake"),
            "future_avoids_baseline_mistakes": select_overlap_examples(cases, "future_avoids_baseline_mistake"),
            "future_loses_to_current_proof_state": select_overlap_examples(cases, "future_loses_current_proof_state_win"),
        },
    }


def error_overlap_case(row: dict[str, Any]) -> dict[str, Any]:
    baseline = row["baseline_path_beam"]
    current = row["soft_proof_state_beam"]
    future = row["future_aware_proof_state_beam"]
    baseline_top = baseline.get("top_answer") or {}
    current_top = current.get("top_answer") or {}
    future_top = future.get("top_answer") or {}
    baseline_evidence = top_evidence(baseline_top)
    current_evidence = top_evidence(current_top)
    future_evidence = top_evidence(future_top)
    baseline_first = hop_relation_set(baseline_evidence, 1)
    current_first = hop_relation_set(current_evidence, 1)
    future_first = hop_relation_set(future_evidence, 1)
    baseline_second = hop_relation_set(baseline_evidence, 2)
    future_second = hop_relation_set(future_evidence, 2)
    baseline_sequence = relation_sequence(baseline_evidence)
    future_sequence = relation_sequence(future_evidence)
    baseline_family = drift_family(baseline_top)
    future_family = drift_family(future_top)
    score_breakdown = future_score_breakdown(future_top)
    same_top_as_baseline = same_answer_id(future_top, baseline_top)
    same_top_as_current = same_answer_id(future_top, current_top)
    same_first_as_baseline = bool(future_first & baseline_first)
    same_first_as_current = bool(future_first & current_first)
    same_second_as_baseline = bool(future_second & baseline_second)
    same_sequence_as_baseline = bool(future_sequence) and future_sequence == baseline_sequence
    same_family_as_baseline = future_family == baseline_family and future_family != "unknown"
    future_repeats_baseline_mistake = (
        not baseline.get("hits_at_1", False)
        and not future.get("hits_at_1", False)
        and (same_top_as_baseline or same_sequence_as_baseline or same_family_as_baseline)
    )
    future_repeats_first_hop_different_final = (
        not baseline.get("hits_at_1", False)
        and not future.get("hits_at_1", False)
        and same_first_as_baseline
        and not same_top_as_baseline
    )
    future_loses_current = current.get("hits_at_1", False) and not future.get("hits_at_1", False)
    future_avoids_baseline_mistake = not baseline.get("hits_at_1", False) and future.get("hits_at_1", False)
    future_avoids_surface = (
        future.get("hits_at_1", False)
        and not baseline.get("hits_at_1", False)
        and (
            score_breakdown["surface_convergence_penalty"] < 0.0
            or (same_family_as_baseline is False and baseline_family != "unknown")
        )
    )
    penalties_tiny = (
        score_breakdown["positive_score_sum"] > 0.0
        and score_breakdown["penalty_to_positive_ratio"] < 0.15
        and not future.get("hits_at_1", False)
    )
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "baseline_top_answer": baseline_top.get("answer_label", ""),
        "current_proof_state_top_answer": current_top.get("answer_label", ""),
        "future_aware_top_answer": future_top.get("answer_label", ""),
        "baseline_correct": baseline.get("hits_at_1", False),
        "current_proof_state_correct": current.get("hits_at_1", False),
        "future_aware_correct": future.get("hits_at_1", False),
        "baseline_top_path": top_path_readable_plain(baseline_top),
        "current_proof_state_top_path": top_path_readable_plain(current_top),
        "future_aware_top_path": top_path_readable_plain(future_top),
        "baseline_first_hop_relations": sorted(format_relation_set(baseline_first)),
        "current_proof_state_first_hop_relations": sorted(format_relation_set(current_first)),
        "future_aware_first_hop_relations": sorted(format_relation_set(future_first)),
        "baseline_second_hop_relations": sorted(format_relation_set(baseline_second)),
        "future_aware_second_hop_relations": sorted(format_relation_set(future_second)),
        "baseline_relation_sequence": [f"{relation}/{direction}" for relation, direction in baseline_sequence],
        "future_aware_relation_sequence": [f"{relation}/{direction}" for relation, direction in future_sequence],
        "baseline_drift_family": baseline_family,
        "future_aware_drift_family": future_family,
        "same_top_answer_as_baseline": same_top_as_baseline,
        "same_top_answer_as_current_proof_state": same_top_as_current,
        "same_first_hop_relation_as_baseline": same_first_as_baseline,
        "same_first_hop_relation_as_current_proof_state": same_first_as_current,
        "same_second_hop_relation_as_baseline": same_second_as_baseline,
        "same_relation_sequence_as_baseline": same_sequence_as_baseline,
        "same_drift_family_as_baseline": same_family_as_baseline,
        "future_repeats_baseline_mistake": future_repeats_baseline_mistake,
        "future_repeats_baseline_first_hop_different_final": future_repeats_first_hop_different_final,
        "future_loses_current_proof_state_win": future_loses_current,
        "future_avoids_baseline_mistake": future_avoids_baseline_mistake,
        "future_avoids_surface_convergence_successfully": future_avoids_surface,
        "penalties_tiny_compared_with_positive_terms": penalties_tiny,
        "score_breakdown": score_breakdown,
    }


def select_overlap_examples(cases: list[dict[str, Any]], flag: str, limit: int = 5) -> list[dict[str, Any]]:
    return [case for case in cases if case.get(flag)][:limit]


def write_error_overlap_markdown(error_overlap: dict[str, Any]) -> str:
    summary = error_overlap["summary"]
    lines = [
        "# Future-Aware Error Overlap Diagnostic",
        "",
        "This diagnostic compares `baseline_path_beam`, current `soft_proof_state_beam`, and `future_aware_proof_state_beam` on the same selected questions.",
        "",
        "It does not change search, scoring, prompts, or candidate generation.",
        "",
        "## Summary Counts",
        "",
        "| Diagnostic | Count |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "### Drift Families",
            "",
            "**Baseline**",
            "",
            "```json",
            json.dumps(summary.get("drift_family_counts_baseline", {}), indent=2, sort_keys=True),
            "```",
            "",
            "**Future-aware**",
            "",
            "```json",
            json.dumps(summary.get("drift_family_counts_future_aware", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Future-Aware Repeats Baseline Mistakes",
            "",
            *overlap_example_lines(error_overlap["examples"]["future_repeats_baseline_mistakes"]),
            "## Future-Aware Avoids Baseline Mistakes",
            "",
            *overlap_example_lines(error_overlap["examples"]["future_avoids_baseline_mistakes"]),
            "## Future-Aware Loses To Current Proof-State",
            "",
            *overlap_example_lines(error_overlap["examples"]["future_loses_to_current_proof_state"]),
        ]
    )
    return "\n".join(lines)


def overlap_example_lines(cases: list[dict[str, Any]]) -> list[str]:
    if not cases:
        return ["_None._", ""]
    lines: list[str] = []
    for case in cases[:5]:
        score = case["score_breakdown"]
        lines.extend(
            [
                f"### {case['question_id']}",
                f"- Question: {case['question']}",
                f"- Gold answer: {case['gold_answers']}",
                f"- Baseline top: `{case['baseline_top_answer']}`",
                f"- Baseline path: `{case['baseline_top_path']}`",
                f"- Current proof-state top: `{case['current_proof_state_top_answer']}`",
                f"- Current proof-state path: `{case['current_proof_state_top_path']}`",
                f"- Future-aware top: `{case['future_aware_top_answer']}`",
                f"- Future-aware path: `{case['future_aware_top_path']}`",
                f"- Same top as baseline: `{case['same_top_answer_as_baseline']}`",
                f"- Same first hop as baseline: `{case['same_first_hop_relation_as_baseline']}`",
                f"- Same second hop as baseline: `{case['same_second_hop_relation_as_baseline']}`",
                f"- Baseline drift family: `{case['baseline_drift_family']}`",
                f"- Future-aware drift family: `{case['future_aware_drift_family']}`",
                "- Future-aware score breakdown:",
                f"  - future_bonus_total: `{score['future_bonus_total']:.4f}`",
                f"  - useful_convergence: `{score['useful_convergence']:.4f}`",
                f"  - surface_convergence_penalty: `{score['surface_convergence_penalty']:.4f}`",
                f"  - redundancy_penalty: `{score['redundancy_penalty']:.4f}`",
                f"  - drift_penalty: `{score['drift_penalty']:.4f}`",
                f"  - noisy_branch_penalty: `{score['noisy_branch_penalty']:.4f}`",
                f"  - positive_score_sum: `{score['positive_score_sum']:.4f}`",
                f"  - negative_penalty_sum: `{score['negative_penalty_sum']:.4f}`",
                f"  - penalty_to_positive_ratio: `{score['penalty_to_positive_ratio']:.4f}`",
                "",
            ]
        )
    return lines


def same_answer_id(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return bool(left and right and left.get("answer_id") == right.get("answer_id"))


def top_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    paths = candidate.get("paths", []) if candidate else []
    if not paths:
        return []
    return paths[0].get("evidence", []) or []


def top_path_readable_plain(candidate: dict[str, Any]) -> str:
    paths = candidate.get("paths", []) if candidate else []
    if not paths:
        return ""
    return str(paths[0].get("readable", ""))


def hop_relation_set(evidence: list[dict[str, Any]], hop: int) -> set[tuple[str, str]]:
    return {
        (str(step.get("relation_id", "")), str(step.get("direction", "")))
        for step in evidence
        if step.get("hop") == hop and step.get("relation_id")
    }


def relation_sequence(evidence: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(step.get("relation_id", "")), str(step.get("direction", "")))
        for step in evidence
        if step.get("relation_id")
    )


def format_relation_set(values: set[tuple[str, str]]) -> set[str]:
    return {f"{relation}/{direction}" for relation, direction in values}


def drift_family(candidate: dict[str, Any]) -> str:
    evidence = top_evidence(candidate)
    if not evidence:
        return "unknown"
    fragments = []
    for step in evidence:
        fragments.append(" ".join(str(step.get(key, "")) for key in ["from_entity", "relation_id", "direction", "to_entity"]))
        fragments.append(" ".join(str(type_name) for type_name in (step.get("to_types", []) or [])))
    path_text = " ".join(fragments).casefold().replace("_", " ")
    relation_ids = [str(step.get("relation_id", "")) for step in evidence]
    entity_chain = [str(step.get("from_entity_id", "")) for step in evidence] + [str(step.get("to_entity_id", "")) for step in evidence]
    if has_path_inverse_loop(evidence):
        return "inverse_loop"
    if any(term in path_text for term in ["subsidiary", "parent organization", "owner of", "owned by"]):
        return "organization_parent_subsidiary_loop"
    if any(term in path_text for term in ["award", "famous people", "notable", "winner", "nominated"]):
        return "award_or_fame_drift"
    if any(term in path_text for term in ["cast member", "film", "movie", "director", "producer", "screenwriter", "composer"]):
        return "cast_or_film_drift"
    if any(term in path_text for term in ["country", "capital", "shares border", "diplomatic relation", "administrative territorial"]):
        return "geography_drift"
    if any(term in path_text for term in ["location", "located in", "city", "state", "province"]):
        return "broad_location_or_country_branch"
    if len(relation_ids) != len(set(relation_ids)) or len(entity_chain) != len(set(entity_chain)):
        return "generic_relation_loop"
    return "unknown"


def has_path_inverse_loop(evidence: list[dict[str, Any]]) -> bool:
    seen = set()
    for step in evidence:
        edge = (step.get("from_entity_id"), step.get("relation_id"), step.get("to_entity_id"))
        reverse = (step.get("to_entity_id"), step.get("relation_id"), step.get("from_entity_id"))
        if reverse in seen:
            return True
        seen.add(edge)
    return False


def future_score_breakdown(future_top: dict[str, Any]) -> dict[str, float]:
    paths = future_top.get("paths", []) if future_top else []
    signals = paths[0].get("soft_signals", {}) if paths else {}
    future_bonus_total = float(signals.get("future_satisfiability", 0.0)) + float(signals.get("useful_convergence", 0.0))
    useful_convergence = float(signals.get("useful_convergence", 0.0))
    surface = float(signals.get("surface_convergence_penalty", 0.0))
    redundancy = float(signals.get("redundancy_penalty", 0.0))
    drift = float(signals.get("drift_penalty", 0.0))
    noisy = float(signals.get("noisy_branch_penalty", 0.0))
    positive_sum = sum(
        max(0.0, float(signals.get(key, 0.0)))
        for key in ["current_relevance", "future_satisfiability", "useful_convergence", "type_compatibility", "progress"]
    )
    negative_sum = sum(abs(min(0.0, value)) for value in [surface, redundancy, drift, noisy])
    return {
        "future_bonus_total": future_bonus_total,
        "useful_convergence": useful_convergence,
        "surface_convergence_penalty": surface,
        "redundancy_penalty": redundancy,
        "drift_penalty": drift,
        "noisy_branch_penalty": noisy,
        "positive_score_sum": positive_sum,
        "negative_penalty_sum": negative_sum,
        "penalty_to_positive_ratio": negative_sum / positive_sum if positive_sum else 0.0,
    }


def top_path_readable(candidate: dict[str, Any]) -> str:
    paths = candidate.get("paths", []) if candidate else []
    if not paths:
        return "`<none>`"
    return f"`{paths[0].get('readable', '')}`"


def likely_reason(row: dict[str, Any]) -> str:
    baseline = row["baseline_path_beam"]
    proof = row["soft_proof_state_beam"]
    if proof["hits_at_1"] and not baseline["hits_at_1"]:
        return "proof-state scoring promoted a gold-reaching evidence state over the independent top path"
    if baseline["hits_at_1"] and not proof["hits_at_1"]:
        return "proof-state heuristics likely over-weighted convergence/type/progress signals or under-weighted the direct relation label"
    if not baseline["gold_generated"] and not proof["gold_generated"]:
        return "gold answer was not generated by either search, so scoring cannot fix this case"
    return "both methods failed despite candidate generation; inspect relation choices and noisy branches"


def print_runtime_log(row: dict[str, Any], index: int, total: int) -> None:
    baseline = row["baseline_path_beam"]
    proof = row["soft_proof_state_beam"]
    future = row["future_aware_proof_state_beam"]
    baseline_top = baseline["top_answer"] or {}
    proof_top = proof["top_answer"] or {}
    future_top = future["top_answer"] or {}
    print(f"\nQuestion {index}/{total}", flush=True)
    print(f"Q: {row['question']}", flush=True)
    print(f"Gold answer: {row['gold_answers']}", flush=True)
    print(f"Start entity: {row['start_entity']['name']} {row['start_entity']['labels']}", flush=True)
    print(f"Baseline top answer: {baseline_top.get('answer_label', '<none>')}", flush=True)
    print(f"Proof-state top answer: {proof_top.get('answer_label', '<none>')}", flush=True)
    print(f"Future-aware top answer: {future_top.get('answer_label', '<none>')}", flush=True)
    print(f"Gold generated baseline: {'yes' if baseline['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated proof-state: {'yes' if proof['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated future-aware: {'yes' if future['gold_generated'] else 'no'}", flush=True)
    print(f"Baseline top path: {top_path_readable(baseline_top)}", flush=True)
    print(f"Proof-state evidence: {top_path_readable(proof_top)}", flush=True)
    print(f"Future-aware evidence: {top_path_readable(future_top)}", flush=True)
    print(f"Failure type: {row['failure_type']}", flush=True)


def entity_labels(graph: KnowledgeGraph, entity_ids: set[str], limit: int = 20) -> list[str]:
    return [graph.entity_name(entity_id) for entity_id in sorted(entity_ids, key=lambda eid: graph.entity_name(eid))[:limit]]


def state_sort_key(state: SearchState) -> str:
    entities = ",".join(sorted(state.frontier_ids))
    relations = "|".join("->".join(seq) for seq in state.relation_sequences)
    return f"{entities}:{relations}"


def precision(gold: set[str], predicted: set[str]) -> float:
    if not predicted:
        return 0.0
    return len(gold & predicted) / len(predicted)


def recall(gold: set[str], predicted: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & predicted) / len(gold)


def f1(p: float, r: float) -> float:
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def avg(values: Any) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values) if values else 0.0


def avg_bool(values: Any) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else 0.0


def char_ngram_similarity(left: str, right: str, n: int = 3) -> float:
    left_vec = char_ngram_vector(left, n)
    right_vec = char_ngram_vector(right, n)
    if not left_vec or not right_vec:
        return 0.0
    dot = sum(value * right_vec.get(key, 0) for key, value in left_vec.items())
    left_norm = sum(value * value for value in left_vec.values()) ** 0.5
    right_norm = sum(value * value for value in right_vec.values()) ** 0.5
    return dot / max(1e-12, left_norm * right_norm)


def char_ngram_vector(text: str, n: int) -> dict[str, int]:
    normalized = f"  {normalize_for_similarity(text)}  "
    out: dict[str, int] = {}
    for index in range(max(1, len(normalized) - n + 1)):
        gram = normalized[index : index + n]
        out[gram] = out.get(gram, 0) + 1
    return out


def normalize_for_similarity(text: str) -> str:
    text = str(text).casefold().replace("_", " ")
    return " ".join(token for token in re.split(r"[^a-z0-9]+", text) if token)


def tokenize_content(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(text).casefold().replace("_", " "))
        if len(token) > 2 and token not in STOPWORDS
    }


def remaining_question_terms(question: str, matched_relation_text: str) -> set[str]:
    question_terms = tokenize_content(question)
    matched_terms = tokenize_content(matched_relation_text)
    remaining = question_terms - matched_terms
    return remaining or question_terms


def token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal soft proof-state search smoke test for KQA Pro.")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="runs/proof_state_search_smoke")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--relation-cap", type=int, default=30)
    parser.add_argument("--sample-entities", type=int, default=25)
    parser.add_argument("--max-branch-entities", type=int, default=40)
    parser.add_argument("--noisy-branch-threshold", type=int, default=25)
    parser.add_argument("--debug-trace", action="store_true", help="Write debug_trace.md/jsonl with top hop states and score components.")
    parser.add_argument("--debug-limit", type=int, default=10, help="Maximum number of questions to include in debug trace outputs.")
    return parser.parse_args()


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
