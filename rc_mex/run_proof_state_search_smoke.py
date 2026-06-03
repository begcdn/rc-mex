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

FUTURE_AWARE_V2_CONSTANTS = {
    "future_cap": 0.25,
    "first_hop_relation_family_limit": 1,
    "current_relevance_weight": 0.55,
    "gated_future_weight": 1.0,
    "type_compatibility_weight": 0.16,
    "strong_loop_penalty": -0.45,
    "start_entity_return_penalty": -0.45,
    "surface_convergence_penalty": -0.25,
    "drift_penalty": -0.30,
    "redundancy_penalty": -0.18,
    "noisy_branch_penalty": -0.30,
}

TWO_SCORE_CONSTANTS = {
    "future_cap": 0.25,
    "retention_current_relevance_weight": 0.50,
    "retention_type_weight": 0.10,
    "retention_diversity_bonus": 0.06,
    "retention_soft_loop_penalty": -0.12,
    "retention_soft_drift_penalty": -0.16,
    "retention_noisy_branch_penalty": -0.22,
    "final_role_coverage_weight": 0.55,
    "final_answer_type_weight": 0.20,
    "final_current_relevance_weight": 0.25,
    "final_useful_convergence_bonus": 0.12,
    "unresolved_need_penalty_weight": -0.25,
    "hard_loop_penalty": -0.45,
    "semantic_level_drift_penalty": -0.30,
    "surface_convergence_penalty": -0.20,
    "redundancy_penalty": -0.12,
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

    log_stage(3, 4, "Running baseline, soft proof-state, two-score, and legacy future-aware beams")
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
        two_score = run_two_score_proof_state_beam(
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
        future_aware_v2 = run_future_aware_v2_proof_state_beam(
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
        row = build_prediction_row(graph, example, baseline, proof_state, two_score, future_aware, future_aware_v2)
        print_runtime_log(row, index, len(examples))
        rows.append(row)

    log_stage(4, 4, "Writing smoke-test outputs")
    metrics = compute_metrics(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "scorer_constants": {
            "two_score_proof_state_beam": TWO_SCORE_CONSTANTS,
            "future_aware_v2_proof_state_beam": FUTURE_AWARE_V2_CONSTANTS,
        },
        "selection_stats": selection_stats,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    (output_dir / "report.md").write_text(write_report(metrics, rows), encoding="utf-8")
    error_overlap = build_error_overlap(rows, future_key="two_score_proof_state_beam", diagnostic_label="two_score_proof_state_beam")
    write_json(output_dir / "error_overlap.json", error_overlap)
    (output_dir / "error_overlap.md").write_text(write_error_overlap_markdown(error_overlap), encoding="utf-8")
    gold_survival_audit = build_target_gold_survival_audit(
        graph,
        rows,
        args,
        target_key="two_score_proof_state_beam",
        target_label="Two-Score",
    )
    behavior_audit = build_target_behavior_audit(
        rows,
        gold_survival_audit,
        target_key="two_score_proof_state_beam",
        target_label="Two-Score",
    )
    code_behavior_audit = build_two_score_code_behavior_audit(rows, gold_survival_audit)
    write_json(output_dir / "gold_survival_audit.json", gold_survival_audit)
    (output_dir / "gold_survival_audit.md").write_text(write_gold_survival_audit_markdown(gold_survival_audit), encoding="utf-8")
    write_json(output_dir / "behavior_audit.json", behavior_audit)
    (output_dir / "behavior_audit.md").write_text(write_behavior_audit_markdown(behavior_audit), encoding="utf-8")
    write_json(output_dir / "code_behavior_audit.json", code_behavior_audit)
    (output_dir / "code_behavior_audit.md").write_text(write_code_behavior_audit_markdown(code_behavior_audit), encoding="utf-8")
    log_line("predictions.jsonl")
    log_line("metrics.json")
    log_line("report.md")
    log_line("error_overlap.json")
    log_line("error_overlap.md")
    log_line("gold_survival_audit.json")
    log_line("gold_survival_audit.md")
    log_line("behavior_audit.json")
    log_line("behavior_audit.md")
    log_line("code_behavior_audit.json")
    log_line("code_behavior_audit.md")
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


def run_future_aware_v2_proof_state_beam(
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
    audit_trace: list[dict[str, Any]] = []
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
                signals = future_aware_v2_fragment_signals(
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
                    start_entity_ids=example.start_entity_ids,
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
        if hop == 1:
            selected_states = diversify_first_hop_states(
                next_states,
                beam_width=beam_width,
                family_limit=int(FUTURE_AWARE_V2_CONSTANTS["first_hop_relation_family_limit"]),
            )
        else:
            selected_states = next_states[:beam_width]
        all_summaries = [
            summarize_future_aware_v2_state(state, rank)
            for rank, state in enumerate(next_states, start=1)
        ]
        selected_summaries = [
            summarize_future_aware_v2_state(state, rank)
            for rank, state in enumerate(selected_states, start=1)
        ]
        audit_trace.append(
            {
                "hop": hop,
                "all_candidate_states": all_summaries,
                "selected_states": selected_summaries,
                "constants": FUTURE_AWARE_V2_CONSTANTS,
            }
        )
        if debug_trace:
            trace.append(
                {
                    "hop": hop,
                    "top_candidate_states": all_summaries[:5],
                    "selected_states": selected_summaries[:5],
                    "constants": FUTURE_AWARE_V2_CONSTANTS,
                }
            )
        states = selected_states
    result = search_result(
        graph,
        states,
        example.gold_answer_ids,
        expansion_count,
        mode="future_aware_v2_proof_state_beam",
        debug_trace=trace,
    )
    result["audit_trace"] = audit_trace
    return result


def run_two_score_proof_state_beam(
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
    audit_trace: list[dict[str, Any]] = []
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
                signals = two_score_fragment_signals(
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
                    start_entity_ids=example.start_entity_ids,
                )
                relation_sequences = extend_relation_sequences(state.relation_sequences, steps)
                entity_sequences = extend_entity_sequences(graph, state.entity_sequences, steps)
                next_states.append(
                    SearchState(
                        frontier_ids={target_id},
                        score=state.score + signals["retention_delta"],
                        evidence=state.evidence + steps,
                        relation_sequences=relation_sequences,
                        entity_sequences=entity_sequences,
                        soft_signals=merge_signal_dicts(state.soft_signals, signals),
                    )
                )
        next_states.sort(key=lambda item: (-item.score, state_sort_key(item)))
        all_summaries = [
            summarize_two_score_state(state, rank)
            for rank, state in enumerate(next_states, start=1)
        ]
        selected_states = next_states[:beam_width]
        selected_summaries = [
            summarize_two_score_state(state, rank)
            for rank, state in enumerate(selected_states, start=1)
        ]
        audit_trace.append(
            {
                "hop": hop,
                "all_candidate_states": all_summaries,
                "selected_states": selected_summaries,
                "constants": TWO_SCORE_CONSTANTS,
            }
        )
        if debug_trace:
            trace.append(
                {
                    "hop": hop,
                    "top_candidate_states": all_summaries[:5],
                    "selected_states": selected_summaries[:5],
                    "constants": TWO_SCORE_CONSTANTS,
                }
            )
        states = selected_states
    result = search_result(
        graph,
        states,
        example.gold_answer_ids,
        expansion_count,
        mode="two_score_proof_state_beam",
        debug_trace=trace,
        answer_score_key="final_proof_score",
    )
    result["audit_trace"] = audit_trace
    return result


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


def future_aware_v2_fragment_signals(
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
    start_entity_ids: set[str],
) -> dict[str, float]:
    relation_text = " ".join(step["relation_id"].replace("_", " ") for step in state.evidence + steps)
    remaining_terms = remaining_question_terms(question, relation_text)
    current_relevance = max((float(step["relation_label_score"]) for step in steps), default=0.0)
    raw_future = future_satisfiability(
        graph=graph,
        question=question,
        remaining_terms=remaining_terms,
        entity_id=target_id,
        relation_cap=relation_cap,
        sample_entities=sample_entities,
    )
    future_capped = min(raw_future, float(FUTURE_AWARE_V2_CONSTANTS["future_cap"]))
    loop_penalty = strong_loop_penalty(state, steps, target_id, start_entity_ids, hop)
    drift = stronger_drift_penalty(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        steps=steps,
        remaining_terms=remaining_terms,
        future_satisfiability_score=raw_future,
        hop=hop,
    )
    role_gate = role_semantic_gate(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        remaining_terms=remaining_terms,
        relation_cap=relation_cap,
        sample_entities=sample_entities,
    )
    non_drift_gate = 0.35 if drift < 0.0 else 1.0
    non_loop_gate = 0.20 if loop_penalty < 0.0 else 1.0
    gated_future = future_capped * role_gate * non_drift_gate * non_loop_gate
    type_score = type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0
    progress = plausible_progress(question, graph, target_id, steps, hop)
    useful_convergence = useful_convergence_bonus_v2(
        graph=graph,
        target_id=target_id,
        steps=steps,
        future_satisfiability_score=gated_future,
        type_score=type_score,
    )
    surface_penalty = surface_convergence_penalty_v2(
        state=state,
        steps=steps,
        useful_convergence=useful_convergence,
        gated_future_bonus=gated_future,
    )
    redundancy = float(FUTURE_AWARE_V2_CONSTANTS["redundancy_penalty"]) if repeats_relation_pattern(state, steps) else 0.0
    branch_size = max((int(step["branch_size"]) for step in steps), default=1)
    noisy_penalty = float(FUTURE_AWARE_V2_CONSTANTS["noisy_branch_penalty"]) if branch_size >= noisy_branch_threshold else 0.0
    total = (
        float(FUTURE_AWARE_V2_CONSTANTS["current_relevance_weight"]) * current_relevance
        + float(FUTURE_AWARE_V2_CONSTANTS["gated_future_weight"]) * gated_future
        + useful_convergence
        + float(FUTURE_AWARE_V2_CONSTANTS["type_compatibility_weight"]) * type_score
        + progress
        + surface_penalty
        + loop_penalty
        + drift
        + redundancy
        + noisy_penalty
    )
    return {
        "current_relevance": current_relevance,
        "raw_future_bonus": raw_future,
        "future_bonus_capped": future_capped,
        "role_gate": role_gate,
        "non_drift_gate": non_drift_gate,
        "non_loop_gate": non_loop_gate,
        "gated_future_bonus": gated_future,
        "useful_convergence": useful_convergence,
        "type_compatibility": type_score,
        "progress": progress,
        "surface_convergence_penalty": surface_penalty,
        "loop_penalty": loop_penalty,
        "drift_penalty": drift,
        "redundancy_penalty": redundancy,
        "noisy_branch_penalty": noisy_penalty,
        "remaining_terms_count": float(len(remaining_terms)),
        "total_delta": total,
    }


def two_score_fragment_signals(
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
    start_entity_ids: set[str],
) -> dict[str, float]:
    relation_text = " ".join(step["relation_id"].replace("_", " ") for step in state.evidence + steps)
    remaining_terms = remaining_question_terms(question, relation_text)
    current_relevance = max((float(step["relation_label_score"]) for step in steps), default=0.0)
    raw_future = future_satisfiability(
        graph=graph,
        question=question,
        remaining_terms=remaining_terms,
        entity_id=target_id,
        relation_cap=relation_cap,
        sample_entities=sample_entities,
    )
    loop_signal = strong_loop_penalty(state, steps, target_id, start_entity_ids, hop)
    drift_signal = stronger_drift_penalty(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        steps=steps,
        remaining_terms=remaining_terms,
        future_satisfiability_score=raw_future,
        hop=hop,
    )
    role_gate = min(1.0, max(0.0, role_semantic_gate(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        remaining_terms=remaining_terms,
        relation_cap=relation_cap,
        sample_entities=sample_entities,
    )))
    non_drift_gate = 0.45 if drift_signal < 0.0 else 1.0
    non_loop_gate = 0.35 if loop_signal < 0.0 else 1.0
    gated_future = raw_future * role_gate * non_drift_gate * non_loop_gate
    future_retention_bonus = min(gated_future, float(TWO_SCORE_CONSTANTS["future_cap"]))
    answer_type_score = type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0
    soft_type_signal = float(TWO_SCORE_CONSTANTS["retention_type_weight"]) * answer_type_score
    soft_progress = plausible_progress(question, graph, target_id, steps, hop)
    soft_diversity_signal = two_score_diversity_signal(state, steps)
    branch_size = max((int(step["branch_size"]) for step in steps), default=1)
    soft_loop_penalty = float(TWO_SCORE_CONSTANTS["retention_soft_loop_penalty"]) if loop_signal < 0.0 else 0.0
    soft_drift_penalty = float(TWO_SCORE_CONSTANTS["retention_soft_drift_penalty"]) if drift_signal < 0.0 else 0.0
    noisy_penalty = float(TWO_SCORE_CONSTANTS["retention_noisy_branch_penalty"]) if branch_size >= noisy_branch_threshold else 0.0
    retention_delta = (
        float(TWO_SCORE_CONSTANTS["retention_current_relevance_weight"]) * current_relevance
        + future_retention_bonus
        + soft_progress
        + soft_type_signal
        + soft_diversity_signal
        + soft_loop_penalty
        + soft_drift_penalty
        + noisy_penalty
    )

    coverage = proof_role_coverage_signals(
        graph=graph,
        question=question,
        answer_type=answer_type,
        target_id=target_id,
        evidence=state.evidence + steps,
        steps=steps,
    )
    useful_convergence = two_score_useful_convergence(
        target_id=target_id,
        evidence=state.evidence + steps,
        steps=steps,
        proof_role_coverage=coverage["proof_role_coverage"],
        answer_type_score=coverage["answer_type_score"],
        future_retention_bonus=future_retention_bonus,
    )
    unresolved_need_penalty = float(TWO_SCORE_CONSTANTS["unresolved_need_penalty_weight"]) * coverage["unresolved_need_score"] if hop == 2 else 0.0
    hard_loop_penalty = two_score_hard_loop_penalty(
        state=state,
        steps=steps,
        target_id=target_id,
        start_entity_ids=start_entity_ids,
        proof_role_coverage=coverage["proof_role_coverage"],
        hop=hop,
    )
    semantic_level_drift_penalty = float(TWO_SCORE_CONSTANTS["semantic_level_drift_penalty"]) if (
        drift_signal < 0.0 and coverage["proof_role_coverage"] < 0.30 and coverage["answer_type_score"] == 0.0
    ) else 0.0
    surface_convergence_penalty_value = two_score_surface_convergence_penalty(
        state=state,
        steps=steps,
        useful_convergence=useful_convergence,
        proof_role_coverage=coverage["proof_role_coverage"],
        future_retention_bonus=future_retention_bonus,
    )
    redundancy_penalty = float(TWO_SCORE_CONSTANTS["redundancy_penalty"]) if (
        repeats_relation_pattern(state, steps) and coverage["proof_role_coverage"] < 0.40
    ) else 0.0
    final_proof_delta = (
        float(TWO_SCORE_CONSTANTS["final_role_coverage_weight"]) * coverage["proof_role_coverage"]
        + float(TWO_SCORE_CONSTANTS["final_answer_type_weight"]) * coverage["answer_type_score"]
        + float(TWO_SCORE_CONSTANTS["final_current_relevance_weight"]) * current_relevance
        + useful_convergence
        + unresolved_need_penalty
        + hard_loop_penalty
        + semantic_level_drift_penalty
        + surface_convergence_penalty_value
        + redundancy_penalty
    )
    return {
        "current_relevance": current_relevance,
        "raw_future": raw_future,
        "role_gate": role_gate,
        "non_drift_gate": non_drift_gate,
        "non_loop_gate": non_loop_gate,
        "future_retention_bonus": future_retention_bonus,
        "soft_progress": soft_progress,
        "soft_type_signal": soft_type_signal,
        "soft_diversity_signal": soft_diversity_signal,
        "soft_loop_penalty": soft_loop_penalty,
        "soft_drift_penalty": soft_drift_penalty,
        "noisy_branch_penalty": noisy_penalty,
        "retention_delta": retention_delta,
        "retention_score": retention_delta,
        "proof_role_coverage": coverage["proof_role_coverage"],
        "covered_need_score": coverage["covered_need_score"],
        "unresolved_need_score": coverage["unresolved_need_score"],
        "relation_role_coverage": coverage["relation_role_coverage"],
        "answer_type_score": coverage["answer_type_score"],
        "answer_type_compatibility": coverage["answer_type_score"],
        "useful_convergence": useful_convergence,
        "unresolved_need_penalty": unresolved_need_penalty,
        "hard_loop_penalty": hard_loop_penalty,
        "semantic_level_drift_penalty": semantic_level_drift_penalty,
        "surface_convergence_penalty": surface_convergence_penalty_value,
        "redundancy_penalty": redundancy_penalty,
        "final_proof_delta": final_proof_delta,
        "final_proof_score": final_proof_delta,
        "remaining_terms_count": float(len(remaining_terms)),
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


def two_score_diversity_signal(state: SearchState, steps: list[dict[str, Any]]) -> float:
    previous_families = {relation_family_from_text(str(step.get("relation_id", ""))) for step in state.evidence}
    current_families = {relation_family_from_text(str(step.get("relation_id", ""))) for step in steps}
    if not previous_families or not current_families:
        return 0.0
    if current_families - previous_families:
        return float(TWO_SCORE_CONSTANTS["retention_diversity_bonus"])
    return 0.0


def proof_role_coverage_signals(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    target_id: str,
    evidence: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> dict[str, float]:
    need_terms = tokenize_content(question)
    relation_terms = tokenize_content(" ".join(str(step.get("relation_id", "")).replace("_", " ") for step in evidence))
    entity_terms = tokenize_content(" ".join(
        " ".join([
            str(step.get("from_entity", "")),
            str(step.get("to_entity", "")),
            " ".join(str(type_name) for type_name in step.get("to_types", []) or []),
        ])
        for step in evidence
    ))
    target_terms = tokenize_content(" ".join([graph.entity_name(target_id), *graph.entity_type_names(target_id)]))
    relation_role_coverage = token_overlap_score(need_terms, relation_terms)
    entity_role_coverage = token_overlap_score(need_terms, entity_terms | target_terms)
    covered_need_score = min(1.0, 0.70 * relation_role_coverage + 0.30 * entity_role_coverage)
    unresolved_need_score = max(0.0, 1.0 - covered_need_score)
    answer_type_score = type_compatibility(graph, target_id, answer_type) if answer_type else 0.0
    if not answer_type and target_type_matches_question_hint(question, graph, target_id):
        answer_type_score = 0.5
    direct_step_match = max((float(step.get("relation_label_score", 0.0)) for step in steps), default=0.0)
    proof_role_coverage = min(
        1.0,
        0.50 * relation_role_coverage
        + 0.25 * covered_need_score
        + 0.15 * answer_type_score
        + 0.10 * min(1.0, direct_step_match),
    )
    return {
        "proof_role_coverage": proof_role_coverage,
        "covered_need_score": covered_need_score,
        "unresolved_need_score": unresolved_need_score,
        "relation_role_coverage": relation_role_coverage,
        "answer_type_score": answer_type_score,
    }


def target_type_matches_question_hint(question: str, graph: KnowledgeGraph, target_id: str) -> bool:
    q = question.casefold()
    target_text = entity_context_text(graph, target_id)
    hint_groups = {
        "occupation": ["occupation", "position", "profession", "job"],
        "legal form": ["legal form", "business"],
        "film": ["movie", "film", "visual artwork"],
        "music": ["music", "genre"],
        "county": ["county"],
        "city": ["city", "town"],
        "country": ["country", "sovereign state", "unitary state"],
        "award": ["award", "ceremony"],
    }
    for target_word, question_words in hint_groups.items():
        if target_word in target_text and any(word in q for word in question_words):
            return True
    return False


def two_score_useful_convergence(
    target_id: str,
    evidence: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    proof_role_coverage: float,
    answer_type_score: float,
    future_retention_bonus: float,
) -> float:
    if len(steps) < 2:
        return 0.0
    relation_pairs = {(step["relation_id"], step["direction"]) for step in steps}
    relation_names = {step["relation_id"] for step in steps}
    reaches_same_target = all(str(step.get("to_entity_id", "")) == str(target_id) for step in steps)
    relation_diverse = len(relation_names) > 1
    bidirectional_duplicate = len(relation_pairs) > len(relation_names)
    helps_role = proof_role_coverage >= 0.35 or answer_type_score > 0.0 or future_retention_bonus > 0.08
    if reaches_same_target and helps_role and (relation_diverse or not bidirectional_duplicate):
        return float(TWO_SCORE_CONSTANTS["final_useful_convergence_bonus"])
    return 0.0


def two_score_hard_loop_penalty(
    state: SearchState,
    steps: list[dict[str, Any]],
    target_id: str,
    start_entity_ids: set[str],
    proof_role_coverage: float,
    hop: int,
) -> float:
    evidence = state.evidence + steps
    returns_to_start = hop == 2 and target_id in start_entity_ids
    repeated_without_role = has_repeated_entity_cycle(evidence) and proof_role_coverage < 0.35
    same_relation_loop = same_relation_out_and_back(evidence) and proof_role_coverage < 0.35
    if returns_to_start or repeated_without_role or same_relation_loop:
        return float(TWO_SCORE_CONSTANTS["hard_loop_penalty"])
    return 0.0


def two_score_surface_convergence_penalty(
    state: SearchState,
    steps: list[dict[str, Any]],
    useful_convergence: float,
    proof_role_coverage: float,
    future_retention_bonus: float,
) -> float:
    if len(steps) < 2:
        return 0.0
    relation_names = [step["relation_id"] for step in steps]
    same_relation_only = len(set(relation_names)) == 1
    no_role_gain = proof_role_coverage < 0.25 and future_retention_bonus < 0.05 and useful_convergence <= 0.0
    if same_relation_only and no_role_gain:
        return float(TWO_SCORE_CONSTANTS["surface_convergence_penalty"])
    if repeats_relation_pattern(state, steps) and useful_convergence <= 0.0:
        return float(TWO_SCORE_CONSTANTS["surface_convergence_penalty"])
    return 0.0


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


def strong_loop_penalty(
    state: SearchState,
    steps: list[dict[str, Any]],
    target_id: str,
    start_entity_ids: set[str],
    hop: int,
) -> float:
    penalty = 0.0
    if has_inverse_loop(state, steps) or has_repeated_entity_cycle(state.evidence + steps):
        penalty += float(FUTURE_AWARE_V2_CONSTANTS["strong_loop_penalty"])
    if hop == 2 and target_id in start_entity_ids:
        penalty += float(FUTURE_AWARE_V2_CONSTANTS["start_entity_return_penalty"])
    if same_relation_out_and_back(state.evidence + steps):
        penalty += float(FUTURE_AWARE_V2_CONSTANTS["strong_loop_penalty"])
    return penalty


def has_repeated_entity_cycle(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False
    chain = [str(evidence[0].get("from_entity_id", ""))]
    chain.extend(str(step.get("to_entity_id", "")) for step in evidence)
    chain = [entity_id for entity_id in chain if entity_id]
    return len(chain) != len(set(chain))


def same_relation_out_and_back(evidence: list[dict[str, Any]]) -> bool:
    seen = set()
    for step in evidence:
        edge = (step.get("from_entity_id"), step.get("relation_id"), step.get("direction"), step.get("to_entity_id"))
        reverse_direction = "backward" if step.get("direction") == "forward" else "forward"
        reverse = (step.get("to_entity_id"), step.get("relation_id"), reverse_direction, step.get("from_entity_id"))
        if reverse in seen:
            return True
        seen.add(edge)
    return False


def role_semantic_gate(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    target_id: str,
    remaining_terms: set[str],
    relation_cap: int,
    sample_entities: int,
) -> float:
    q = question.casefold()
    target_text = entity_context_text(graph, target_id)
    future_labels = future_relation_labels(graph, target_id, relation_cap, sample_entities)
    future_text = " ".join(future_labels).casefold()
    if "legal form" in q or "legal-form" in q:
        return 1.0 if "legal form" in future_text else 0.35
    if any(word in q for word in ["occupation", "job", "position"]):
        if any(word in future_text for word in ["occupation", "position", "field of work", "work period"]):
            return 1.0
        if any(word in target_text for word in ["human", "person", "player", "athlete"]):
            return 0.85
        return 0.35
    if any(word in q for word in ["county", "city", "state", "province"]):
        if any(word in target_text for word in ["county", "city", "state", "province", "territorial"]):
            return 1.0
        if "country" in target_text and "country" not in q:
            return 0.35
        return 0.65
    if answer_type == "person":
        if any(word in target_text for word in ["human", "person"]):
            return 1.0
        if any(word in future_text for word in ["spouse", "place of birth", "occupation", "cast member", "participant", "member of sports team"]):
            return 0.85
        if generic_hub_text(target_text):
            return 0.40
    if answer_type == "organization":
        if any(word in target_text for word in ["organization", "company", "institution", "team"]):
            return 1.0
        if any(word in future_text for word in ["headquarters", "parent organization", "subsidiary", "legal form"]):
            return 0.85
    if generic_hub_text(target_text) and not question_needs_generic_family(q):
        return 0.45
    return 0.75


def stronger_drift_penalty(
    graph: KnowledgeGraph,
    question: str,
    answer_type: str,
    target_id: str,
    steps: list[dict[str, Any]],
    remaining_terms: set[str],
    future_satisfiability_score: float,
    hop: int,
) -> float:
    q = question.casefold()
    text = " ".join([entity_context_text(graph, target_id), " ".join(step["relation_id"].replace("_", " ") for step in steps)])
    family = relation_family_from_text(text)
    if family in {"geography_drift", "broad_location_or_country_branch"} and not question_needs_geography(q):
        return float(FUTURE_AWARE_V2_CONSTANTS["drift_penalty"])
    if family == "cast_or_film_drift" and not question_needs_film(q):
        return float(FUTURE_AWARE_V2_CONSTANTS["drift_penalty"])
    if family == "award_or_fame_drift" and not question_needs_award_or_fame(q):
        return float(FUTURE_AWARE_V2_CONSTANTS["drift_penalty"])
    if family == "organization_parent_subsidiary_loop" and not question_needs_org(q):
        return float(FUTURE_AWARE_V2_CONSTANTS["drift_penalty"])
    if hop == 2 and answer_type and type_compatibility(graph, target_id, answer_type) == 0.0 and future_satisfiability_score < 0.08:
        return float(FUTURE_AWARE_V2_CONSTANTS["drift_penalty"])
    return 0.0


def useful_convergence_bonus_v2(
    graph: KnowledgeGraph,
    target_id: str,
    steps: list[dict[str, Any]],
    future_satisfiability_score: float,
    type_score: float,
) -> float:
    if len(steps) < 2:
        return 0.0
    relation_pairs = {(step["relation_id"], step["direction"]) for step in steps}
    relation_names = {step["relation_id"] for step in steps}
    relation_diverse = len(relation_names) > 1
    not_inverse_duplicate = len(relation_pairs) == len(relation_names)
    if future_satisfiability_score > 0.08 or type_score > 0.0 or (relation_diverse and not_inverse_duplicate):
        return 0.14
    return 0.0


def surface_convergence_penalty_v2(
    state: SearchState,
    steps: list[dict[str, Any]],
    useful_convergence: float,
    gated_future_bonus: float,
) -> float:
    if len(steps) < 2:
        return 0.0
    if useful_convergence <= 0.0 or gated_future_bonus < 0.04:
        return float(FUTURE_AWARE_V2_CONSTANTS["surface_convergence_penalty"])
    relation_names = [step["relation_id"] for step in steps]
    if len(set(relation_names)) == 1:
        return float(FUTURE_AWARE_V2_CONSTANTS["surface_convergence_penalty"])
    return 0.0


def diversify_first_hop_states(states: list[SearchState], beam_width: int, family_limit: int) -> list[SearchState]:
    selected: list[SearchState] = []
    counts: Counter[str] = Counter()
    deferred: list[SearchState] = []
    for state in states:
        family = first_hop_relation_family(state)
        if counts[family] < family_limit:
            selected.append(state)
            counts[family] += 1
        else:
            deferred.append(state)
        if len(selected) >= beam_width:
            return selected
    for state in deferred:
        selected.append(state)
        if len(selected) >= beam_width:
            break
    return selected


def first_hop_relation_family(state: SearchState) -> str:
    for step in state.evidence:
        if step.get("hop") == 1:
            return relation_family_from_text(str(step.get("relation_id", "")))
    return "unknown"


def relation_family_from_text(text: str) -> str:
    value = text.casefold().replace("_", " ")
    if any(term in value for term in ["country", "capital", "shares border", "diplomatic", "administrative", "located in"]):
        return "geography_drift"
    if any(term in value for term in ["time zone", "location", "city", "state", "province"]):
        return "broad_location_or_country_branch"
    if any(term in value for term in ["cast member", "film", "movie", "director", "producer", "screenwriter", "composer"]):
        return "cast_or_film_drift"
    if any(term in value for term in ["award", "famous", "notable", "winner", "nominated"]):
        return "award_or_fame_drift"
    if any(term in value for term in ["subsidiary", "parent organization", "owner of", "owned by"]):
        return "organization_parent_subsidiary_loop"
    return "unknown"


def entity_context_text(graph: KnowledgeGraph, entity_id: str) -> str:
    return " ".join([graph.entity_name(entity_id), *graph.entity_type_names(entity_id)]).casefold()


def future_relation_labels(graph: KnowledgeGraph, entity_id: str, relation_cap: int, sample_entities: int) -> list[str]:
    return [
        relation.predicate.replace("_", " ")
        for relation in graph.candidate_relations([entity_id], cap=relation_cap, sample_entities=sample_entities)
    ]


def generic_hub_text(text: str) -> bool:
    return any(term in text for term in ["country", "sovereign state", "organization", "location", "continent", "time zone"])


def question_needs_generic_family(question: str) -> bool:
    return question_needs_geography(question) or question_needs_org(question)


def question_needs_geography(question: str) -> bool:
    question = question.casefold()
    return any(term in question for term in ["country", "city", "county", "state", "province", "capital", "border", "located", "place", "where"])


def question_needs_film(question: str) -> bool:
    question = question.casefold()
    return any(term in question for term in ["film", "movie", "cast", "director", "producer", "screenwriter", "composer", "visual artwork"])


def question_needs_award_or_fame(question: str) -> bool:
    question = question.casefold()
    return any(term in question for term in ["award", "winner", "famous", "notable", "nominated", "well-known"])


def question_needs_org(question: str) -> bool:
    question = question.casefold()
    return any(term in question for term in ["organization", "company", "subsidiary", "parent", "headquartered", "legal form", "institution", "publisher"])


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
        "occupation": ["occupation", "profession", "position"],
        "legal_form": ["legal form", "business"],
        "music": ["music", "genre"],
        "award": ["award", "ceremony"],
        "county": ["county"],
        "city": ["city", "town"],
        "country": ["country", "sovereign state", "unitary state"],
    }
    for needle in checks.get(answer_type, [answer_type]):
        if needle in type_names or needle in name:
            return 1.0
    return 0.0


def guess_answer_type(question: str) -> str:
    q = question.casefold()
    if q.startswith("who") or "which person" in q:
        return "person"
    if "occupation" in q or "profession" in q or "what position" in q or "which position" in q:
        return "occupation"
    if "legal-form" in q or "legal form" in q:
        return "legal_form"
    if "county" in q:
        return "county"
    if "which city" in q or "what city" in q or "capital city" in q or " town " in f" {q} ":
        return "city"
    if "which country" in q or "what country" in q or "sovereign state" in q or "unitary state" in q:
        return "country"
    if q.startswith("where"):
        return "location"
    if q.startswith("when") or "what year" in q or "date" in q:
        return "time"
    if "how many" in q or "number" in q:
        return "number"
    if "film" in q or "movie" in q:
        return "film"
    if "company" in q or "organization" in q or "team" in q:
        return "organization"
    if "music" in q or "genre" in q:
        return "music"
    if "award" in q or "ceremony" in q:
        return "award"
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
    answer_score_key: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for state in states:
        for entity_id in state.frontier_ids:
            answer_score = float(state.soft_signals.get(answer_score_key, state.score)) if answer_score_key else state.score
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
            row["score"] += answer_score
            row["best_path_score"] = max(row["best_path_score"], answer_score)
            row["paths"].append(
                {
                    "path_score": answer_score,
                    "retention_score": state.score,
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


def summarize_future_aware_v2_state(state: SearchState, rank: int) -> dict[str, Any]:
    target_entity = next(iter(state.frontier_ids), "")
    last_step = state.evidence[-1] if state.evidence else {}
    return {
        "rank": rank,
        "hop": state.evidence[-1].get("hop") if state.evidence else None,
        "candidate_evidence_state": " | ".join(step["readable"] for step in state.evidence),
        "fragments": [step["readable"] for step in state.evidence],
        "evidence": state.evidence,
        "target_entity": target_entity,
        "target_label": last_step.get("to_entity", ""),
        "current_relevance": state.soft_signals.get("current_relevance", 0.0),
        "raw_future_bonus": state.soft_signals.get("raw_future_bonus", 0.0),
        "future_bonus_capped": state.soft_signals.get("future_bonus_capped", 0.0),
        "role_gate": state.soft_signals.get("role_gate", 0.0),
        "non_drift_gate": state.soft_signals.get("non_drift_gate", 0.0),
        "non_loop_gate": state.soft_signals.get("non_loop_gate", 0.0),
        "gated_future_bonus": state.soft_signals.get("gated_future_bonus", 0.0),
        "useful_convergence": state.soft_signals.get("useful_convergence", 0.0),
        "type_compatibility": state.soft_signals.get("type_compatibility", 0.0),
        "progress": state.soft_signals.get("progress", 0.0),
        "surface_convergence_penalty": state.soft_signals.get("surface_convergence_penalty", 0.0),
        "loop_penalty": state.soft_signals.get("loop_penalty", 0.0),
        "drift_penalty": state.soft_signals.get("drift_penalty", 0.0),
        "redundancy_penalty": state.soft_signals.get("redundancy_penalty", 0.0),
        "noisy_branch_penalty": state.soft_signals.get("noisy_branch_penalty", 0.0),
        "final_score": state.score,
    }


def summarize_two_score_state(state: SearchState, rank: int) -> dict[str, Any]:
    target_entity = next(iter(state.frontier_ids), "")
    last_step = state.evidence[-1] if state.evidence else {}
    return {
        "rank": rank,
        "hop": state.evidence[-1].get("hop") if state.evidence else None,
        "candidate_evidence_state": " | ".join(step["readable"] for step in state.evidence),
        "fragments": [step["readable"] for step in state.evidence],
        "evidence": state.evidence,
        "target_entity": target_entity,
        "target_label": last_step.get("to_entity", ""),
        "current_relevance": state.soft_signals.get("current_relevance", 0.0),
        "raw_future": state.soft_signals.get("raw_future", 0.0),
        "role_gate": state.soft_signals.get("role_gate", 0.0),
        "non_drift_gate": state.soft_signals.get("non_drift_gate", 0.0),
        "non_loop_gate": state.soft_signals.get("non_loop_gate", 0.0),
        "future_retention_bonus": state.soft_signals.get("future_retention_bonus", 0.0),
        "soft_progress": state.soft_signals.get("soft_progress", 0.0),
        "soft_type_signal": state.soft_signals.get("soft_type_signal", 0.0),
        "soft_diversity_signal": state.soft_signals.get("soft_diversity_signal", 0.0),
        "soft_loop_penalty": state.soft_signals.get("soft_loop_penalty", 0.0),
        "soft_drift_penalty": state.soft_signals.get("soft_drift_penalty", 0.0),
        "noisy_branch_penalty": state.soft_signals.get("noisy_branch_penalty", 0.0),
        "retention_score": state.score,
        "proof_role_coverage": state.soft_signals.get("proof_role_coverage", 0.0),
        "covered_need_score": state.soft_signals.get("covered_need_score", 0.0),
        "unresolved_need_score": state.soft_signals.get("unresolved_need_score", 0.0),
        "relation_role_coverage": state.soft_signals.get("relation_role_coverage", 0.0),
        "answer_type_compatibility": state.soft_signals.get("answer_type_compatibility", 0.0),
        "useful_convergence": state.soft_signals.get("useful_convergence", 0.0),
        "unresolved_need_penalty": state.soft_signals.get("unresolved_need_penalty", 0.0),
        "hard_loop_penalty": state.soft_signals.get("hard_loop_penalty", 0.0),
        "semantic_level_drift_penalty": state.soft_signals.get("semantic_level_drift_penalty", 0.0),
        "surface_convergence_penalty": state.soft_signals.get("surface_convergence_penalty", 0.0),
        "redundancy_penalty": state.soft_signals.get("redundancy_penalty", 0.0),
        "final_proof_score": state.soft_signals.get("final_proof_score", 0.0),
    }


def build_prediction_row(
    graph: KnowledgeGraph,
    example: SmokeExample,
    baseline: dict[str, Any],
    proof_state: dict[str, Any],
    two_score: dict[str, Any],
    future_aware: dict[str, Any],
    future_aware_v2: dict[str, Any],
) -> dict[str, Any]:
    baseline_correct = baseline["hits_at_1"]
    proof_correct = proof_state["hits_at_1"]
    two_score_correct = two_score["hits_at_1"]
    future_correct = future_aware["hits_at_1"]
    future_v2_correct = future_aware_v2["hits_at_1"]
    if baseline_correct and proof_correct and two_score_correct:
        failure_type = "both_correct"
    elif two_score_correct and not proof_correct:
        failure_type = "two_score_correct"
    elif future_v2_correct and not future_correct and not proof_correct:
        failure_type = "future_aware_v2_correct"
    elif future_correct and not proof_correct:
        failure_type = "future_aware_correct"
    elif proof_correct:
        failure_type = "proof_state_correct"
    elif baseline_correct:
        failure_type = "baseline_correct"
    elif not baseline["gold_generated"] and not proof_state["gold_generated"] and not two_score["gold_generated"]:
        failure_type = "gold_not_generated"
    elif not baseline["candidate_answers"] or not proof_state["candidate_answers"] or not two_score["candidate_answers"]:
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
        "two_score_proof_state_beam": two_score,
        "future_aware_proof_state_beam": future_aware,
        "future_aware_v2_proof_state_beam": future_aware_v2,
        "failure_type": failure_type,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row["baseline_path_beam"] for row in rows]
    proof = [row["soft_proof_state_beam"] for row in rows]
    two_score = [row["two_score_proof_state_beam"] for row in rows]
    future = [row["future_aware_proof_state_beam"] for row in rows]
    future_v2 = [row["future_aware_v2_proof_state_beam"] for row in rows]
    future_v2_overlap = [error_overlap_case(row, future_key="future_aware_v2_proof_state_beam") for row in rows]
    two_score_overlap = [error_overlap_case(row, future_key="two_score_proof_state_beam") for row in rows]
    return {
        "number_of_selected_questions": len(rows),
        "baseline_hits_at_1": avg_bool(item["hits_at_1"] for item in baseline),
        "proof_state_hits_at_1": avg_bool(item["hits_at_1"] for item in proof),
        "two_score_hits_at_1": avg_bool(item["hits_at_1"] for item in two_score),
        "future_aware_hits_at_1": avg_bool(item["hits_at_1"] for item in future),
        "future_aware_v2_hits_at_1": avg_bool(item["hits_at_1"] for item in future_v2),
        "baseline_exact_match": avg_bool(item["exact_match"] for item in baseline),
        "proof_state_exact_match": avg_bool(item["exact_match"] for item in proof),
        "two_score_exact_match": avg_bool(item["exact_match"] for item in two_score),
        "future_aware_exact_match": avg_bool(item["exact_match"] for item in future),
        "future_aware_v2_exact_match": avg_bool(item["exact_match"] for item in future_v2),
        "baseline_final_answer_f1": avg(item["final_answer_f1"] for item in baseline),
        "proof_state_final_answer_f1": avg(item["final_answer_f1"] for item in proof),
        "two_score_final_answer_f1": avg(item["final_answer_f1"] for item in two_score),
        "future_aware_final_answer_f1": avg(item["final_answer_f1"] for item in future),
        "future_aware_v2_final_answer_f1": avg(item["final_answer_f1"] for item in future_v2),
        "baseline_gold_generated_rate": avg_bool(item["gold_generated"] for item in baseline),
        "proof_state_gold_generated_rate": avg_bool(item["gold_generated"] for item in proof),
        "two_score_gold_generated_rate": avg_bool(item["gold_generated"] for item in two_score),
        "future_aware_gold_generated_rate": avg_bool(item["gold_generated"] for item in future),
        "future_aware_v2_gold_generated_rate": avg_bool(item["gold_generated"] for item in future_v2),
        "average_candidate_count_baseline": avg(item["candidate_count"] for item in baseline),
        "average_candidate_count_proof_state": avg(item["candidate_count"] for item in proof),
        "average_candidate_count_two_score": avg(item["candidate_count"] for item in two_score),
        "average_candidate_count_future_aware": avg(item["candidate_count"] for item in future),
        "average_candidate_count_future_aware_v2": avg(item["candidate_count"] for item in future_v2),
        "average_expansion_count_baseline": avg(item["expansion_count"] for item in baseline),
        "average_expansion_count_proof_state": avg(item["expansion_count"] for item in proof),
        "average_expansion_count_two_score": avg(item["expansion_count"] for item in two_score),
        "average_expansion_count_future_aware": avg(item["expansion_count"] for item in future),
        "average_expansion_count_future_aware_v2": avg(item["expansion_count"] for item in future_v2),
        "average_final_result_size_baseline": avg(item["final_result_size"] for item in baseline),
        "average_final_result_size_proof_state": avg(item["final_result_size"] for item in proof),
        "average_final_result_size_two_score": avg(item["final_result_size"] for item in two_score),
        "average_final_result_size_future_aware": avg(item["final_result_size"] for item in future),
        "average_final_result_size_future_aware_v2": avg(item["final_result_size"] for item in future_v2),
        "proof_state_wins": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ),
        "baseline_wins": sum(
            1 for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "two_score_wins_over_baseline": sum(
            1 for row in rows
            if row["two_score_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ),
        "baseline_wins_over_two_score": sum(
            1 for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["two_score_proof_state_beam"]["final_answer_f1"]
        ),
        "two_score_wins_over_current_proof_state": sum(
            1 for row in rows
            if row["two_score_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "current_proof_state_wins_over_two_score": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["two_score_proof_state_beam"]["final_answer_f1"]
        ),
        "future_aware_wins_over_current_proof_state": sum(
            1 for row in rows
            if row["future_aware_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "current_proof_state_wins_over_future_aware": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["future_aware_proof_state_beam"]["final_answer_f1"]
        ),
        "future_v2_wins_over_future_v1": sum(
            1 for row in rows
            if row["future_aware_v2_proof_state_beam"]["final_answer_f1"] > row["future_aware_proof_state_beam"]["final_answer_f1"]
        ),
        "future_v1_wins_over_future_v2": sum(
            1 for row in rows
            if row["future_aware_proof_state_beam"]["final_answer_f1"] > row["future_aware_v2_proof_state_beam"]["final_answer_f1"]
        ),
        "future_v2_wins_over_current_proof_state": sum(
            1 for row in rows
            if row["future_aware_v2_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "current_proof_state_wins_over_future_v2": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["future_aware_v2_proof_state_beam"]["final_answer_f1"]
        ),
        "both_correct": sum(
            1 for row in rows
            if row["baseline_path_beam"]["hits_at_1"]
            and row["soft_proof_state_beam"]["hits_at_1"]
            and row["two_score_proof_state_beam"]["hits_at_1"]
        ),
        "both_fail": sum(
            1 for row in rows
            if not row["baseline_path_beam"]["hits_at_1"]
            and not row["soft_proof_state_beam"]["hits_at_1"]
            and not row["two_score_proof_state_beam"]["hits_at_1"]
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
        "future_v2_repeats_baseline_mistake": sum(
            1 for row in rows
            if error_overlap_case(row, future_key="future_aware_v2_proof_state_beam")["future_repeats_baseline_mistake"]
        ),
        "future_v2_same_first_hop_as_baseline": sum(
            1 for case in future_v2_overlap
            if case["same_first_hop_relation_as_baseline"]
        ),
        "future_v2_same_relation_sequence_as_baseline": sum(
            1 for case in future_v2_overlap
            if case["same_relation_sequence_as_baseline"]
        ),
        "future_v2_same_drift_family_as_baseline": sum(
            1 for case in future_v2_overlap
            if case["same_drift_family_as_baseline"]
        ),
        "future_v2_loses_current_proof_state_win": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["hits_at_1"] and not row["future_aware_v2_proof_state_beam"]["hits_at_1"]
        ),
        "future_v2_penalties_tiny_compared_with_positive_terms": sum(
            1 for case in future_v2_overlap
            if case["penalties_tiny_compared_with_positive_terms"]
        ),
        "two_score_repeats_baseline_mistake": sum(
            1 for case in two_score_overlap
            if case["future_repeats_baseline_mistake"]
        ),
        "two_score_same_first_hop_as_baseline": sum(
            1 for case in two_score_overlap
            if case["same_first_hop_relation_as_baseline"]
        ),
        "two_score_same_drift_family_as_baseline": sum(
            1 for case in two_score_overlap
            if case["same_drift_family_as_baseline"]
        ),
        "two_score_gold_generated_but_ranked_low": sum(
            1 for row in rows
            if row["two_score_proof_state_beam"]["gold_generated"] and not row["two_score_proof_state_beam"]["hits_at_1"]
        ),
        "failure_counts": dict(Counter(row["failure_type"] for row in rows)),
    }


def write_report(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Two-Score Proof-State Search Smoke Test",
        "",
        "This tests whether separating beam-retention score from final proof score improves simple KQA Pro two-hop search.",
        "",
        "No gold relation IDs, gold prefixes, relation cards, LLM constraint extraction, ToG/Freebase, or quantum-inspired scoring are used during search.",
        "",
        "The main comparison is `baseline_path_beam` vs current `soft_proof_state_beam` vs `two_score_proof_state_beam`.",
        "",
        "Two-score constants:",
        "",
        "```json",
        json.dumps(TWO_SCORE_CONSTANTS, indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Two-Score Wins Over Current Proof-State",
        "",
        *debug_section_rows(select_debug_rows(rows, "two_score_over_current"), limit=5),
        "## Current Proof-State Wins Over Two-Score",
        "",
        *debug_section_rows(select_debug_rows(rows, "current_over_two_score"), limit=5),
        "## Two-Score Wins Over Baseline",
        "",
        *debug_section_rows(select_debug_rows(rows, "two_score_over_baseline"), limit=5),
        "## Baseline Wins Over Two-Score",
        "",
        *debug_section_rows(select_debug_rows(rows, "baseline_over_two_score"), limit=5),
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
    for kind in [
        "two_score_over_current",
        "current_over_two_score",
        "two_score_over_baseline",
        "baseline_over_two_score",
        "future",
        "current_over_future",
        "surface_avoidance",
        "future_hurts",
        "proof",
        "baseline",
        "both_fail",
    ]:
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
        "two_score_correct": row["two_score_proof_state_beam"]["hits_at_1"],
        "future_aware_correct": row["future_aware_proof_state_beam"]["hits_at_1"],
        "future_aware_v2_correct": row["future_aware_v2_proof_state_beam"]["hits_at_1"],
        "baseline_top_answer": row["baseline_path_beam"]["top_answer"],
        "proof_state_top_answer": row["soft_proof_state_beam"]["top_answer"],
        "two_score_top_answer": row["two_score_proof_state_beam"]["top_answer"],
        "future_aware_top_answer": row["future_aware_proof_state_beam"]["top_answer"],
        "future_aware_v2_top_answer": row["future_aware_v2_proof_state_beam"]["top_answer"],
        "baseline_debug_trace": row["baseline_path_beam"].get("debug_trace", []),
        "proof_state_debug_trace": row["soft_proof_state_beam"].get("debug_trace", []),
        "two_score_debug_trace": row["two_score_proof_state_beam"].get("debug_trace", []),
        "future_aware_debug_trace": row["future_aware_proof_state_beam"].get("debug_trace", []),
        "future_aware_v2_debug_trace": row["future_aware_v2_proof_state_beam"].get("debug_trace", []),
        "two_score_constants": TWO_SCORE_CONSTANTS,
        "future_aware_v2_constants": FUTURE_AWARE_V2_CONSTANTS,
        "explanation": proof_state_choice_explanation(row),
        "two_score_explanation": two_score_choice_explanation(row),
        "future_aware_explanation": future_aware_choice_explanation(row),
        "future_aware_v2_explanation": future_aware_v2_choice_explanation(row),
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
        two_score = row["two_score_proof_state_beam"]
        future = row["future_aware_proof_state_beam"]
        future_v2 = row["future_aware_v2_proof_state_beam"]
        baseline_top = baseline["top_answer"] or {}
        proof_top = proof["top_answer"] or {}
        two_score_top = two_score["top_answer"] or {}
        future_top = future["top_answer"] or {}
        future_v2_top = future_v2["top_answer"] or {}
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
                f"Two-score top final state: {top_path_readable(two_score_top)}",
                "",
                f"Future-aware top final state: {top_path_readable(future_top)}",
                "",
                f"Future-aware v2 top final state: {top_path_readable(future_v2_top)}",
                "",
                f"Baseline correct: `{baseline['hits_at_1']}`",
                f"Proof-state correct: `{proof['hits_at_1']}`",
                f"Two-score correct: `{two_score['hits_at_1']}`",
                f"Future-aware correct: `{future['hits_at_1']}`",
                f"Future-aware v2 correct: `{future_v2['hits_at_1']}`",
                "",
                "### Baseline Hop Trace",
                "",
                *baseline_trace_lines(baseline.get("debug_trace", [])),
                "### Proof-State Hop Trace",
                "",
                *proof_trace_lines(proof.get("debug_trace", [])),
                "### Two-Score Proof-State Hop Trace",
                "",
                *two_score_trace_lines(two_score.get("debug_trace", [])),
                "### Future-Aware Proof-State Hop Trace",
                "",
                *future_aware_trace_lines(future.get("debug_trace", [])),
                "### Future-Aware V2 Proof-State Hop Trace",
                "",
                *future_aware_v2_trace_lines(future_v2.get("debug_trace", [])),
                "### Why Proof-State Chose This Over Baseline",
                "",
                *proof_state_choice_explanation(row),
                "### Why Two-Score Chose This",
                "",
                *two_score_choice_explanation(row),
                "### Why Future-Aware Chose This",
                "",
                *future_aware_choice_explanation(row),
                "### Why Future-Aware V2 Chose This",
                "",
                *future_aware_v2_choice_explanation(row),
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


def two_score_trace_lines(trace: list[dict[str, Any]]) -> list[str]:
    if not trace:
        return ["_No two-score proof-state trace captured._", ""]
    lines: list[str] = []
    for hop in trace:
        lines.append(f"#### Hop {hop['hop']}")
        lines.append("")
        lines.append("##### Retention Score")
        lines.append("")
        lines.append(
            "| Rank | Candidate evidence state | Current | Raw future | Role gate | Drift gate | Loop gate | Future keep | Progress | Type | Diversity | Loop soft | Drift soft | Noisy | Retention |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | {cur:.4f} | {future:.4f} | {role:.4f} | {drift_gate:.4f} | {loop_gate:.4f} | {keep:.4f} | {prog:.4f} | {typ:.4f} | {div:.4f} | {loop:.4f} | {drift:.4f} | {noisy:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_evidence_state", "")),
                    cur=float(state.get("current_relevance", 0.0)),
                    future=float(state.get("raw_future", 0.0)),
                    role=float(state.get("role_gate", 0.0)),
                    drift_gate=float(state.get("non_drift_gate", 0.0)),
                    loop_gate=float(state.get("non_loop_gate", 0.0)),
                    keep=float(state.get("future_retention_bonus", 0.0)),
                    prog=float(state.get("soft_progress", 0.0)),
                    typ=float(state.get("soft_type_signal", 0.0)),
                    div=float(state.get("soft_diversity_signal", 0.0)),
                    loop=float(state.get("soft_loop_penalty", 0.0)),
                    drift=float(state.get("soft_drift_penalty", 0.0)),
                    noisy=float(state.get("noisy_branch_penalty", 0.0)),
                    score=float(state.get("retention_score", 0.0)),
                )
            )
        lines.append("")
        lines.append("##### Final Proof Score")
        lines.append("")
        lines.append(
            "| Rank | Candidate evidence state | Proof role | Answer type | Current | Convergence | Unresolved | Hard loop | Semantic drift | Surface | Redundant | Final proof |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | {role:.4f} | {typ:.4f} | {cur:.4f} | {conv:.4f} | {unres:.4f} | {loop:.4f} | {drift:.4f} | {surf:.4f} | {red:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_evidence_state", "")),
                    role=float(state.get("proof_role_coverage", 0.0)),
                    typ=float(state.get("answer_type_compatibility", 0.0)),
                    cur=float(state.get("current_relevance", 0.0)),
                    conv=float(state.get("useful_convergence", 0.0)),
                    unres=float(state.get("unresolved_need_penalty", 0.0)),
                    loop=float(state.get("hard_loop_penalty", 0.0)),
                    drift=float(state.get("semantic_level_drift_penalty", 0.0)),
                    surf=float(state.get("surface_convergence_penalty", 0.0)),
                    red=float(state.get("redundancy_penalty", 0.0)),
                    score=float(state.get("final_proof_score", 0.0)),
                )
            )
        selected = hop.get("selected_states", [])
        if selected:
            lines.append("")
            lines.append(f"_Selected by retention score: {', '.join(str(item.get('rank')) for item in selected[:5])}_")
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


def future_aware_v2_trace_lines(trace: list[dict[str, Any]]) -> list[str]:
    if not trace:
        return ["_No future-aware v2 proof-state trace captured._", ""]
    lines: list[str] = []
    for hop in trace:
        lines.append(f"#### Hop {hop['hop']}")
        lines.append("")
        lines.append(
            "| Rank | Candidate evidence state | Current | Raw future | Future cap | Role gate | Drift gate | Loop gate | Gated future | Convergence | Type | Progress | Surface | Loop | Drift | Redundant | Noisy | Final score |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for state in hop.get("top_candidate_states", []):
            lines.append(
                "| {rank} | {path} | {cur:.4f} | {raw:.4f} | {cap:.4f} | {role:.4f} | {drift_gate:.4f} | {loop_gate:.4f} | {gated:.4f} | {conv:.4f} | {typ:.4f} | {prog:.4f} | {surf:.4f} | {loop:.4f} | {drift:.4f} | {red:.4f} | {noisy:.4f} | {score:.4f} |".format(
                    rank=state["rank"],
                    path=escape_md(state.get("candidate_evidence_state", "")),
                    cur=float(state.get("current_relevance", 0.0)),
                    raw=float(state.get("raw_future_bonus", 0.0)),
                    cap=float(state.get("future_bonus_capped", 0.0)),
                    role=float(state.get("role_gate", 0.0)),
                    drift_gate=float(state.get("non_drift_gate", 0.0)),
                    loop_gate=float(state.get("non_loop_gate", 0.0)),
                    gated=float(state.get("gated_future_bonus", 0.0)),
                    conv=float(state.get("useful_convergence", 0.0)),
                    typ=float(state.get("type_compatibility", 0.0)),
                    prog=float(state.get("progress", 0.0)),
                    surf=float(state.get("surface_convergence_penalty", 0.0)),
                    loop=float(state.get("loop_penalty", 0.0)),
                    drift=float(state.get("drift_penalty", 0.0)),
                    red=float(state.get("redundancy_penalty", 0.0)),
                    noisy=float(state.get("noisy_branch_penalty", 0.0)),
                    score=float(state.get("final_score", 0.0)),
                )
            )
        selected = hop.get("selected_states", [])
        if selected:
            lines.append("")
            lines.append(f"_Selected after v2 first-hop diversity/pruning: {', '.join(str(item.get('rank')) for item in selected[:5])}_")
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


def two_score_choice_explanation(row: dict[str, Any]) -> list[str]:
    two_top = row["two_score_proof_state_beam"].get("top_answer") or {}
    paths = two_top.get("paths", []) if two_top else []
    signals = paths[0].get("soft_signals", {}) if paths else {}
    retention_components = {
        key: float(signals.get(key, 0.0))
        for key in [
            "current_relevance",
            "future_retention_bonus",
            "soft_progress",
            "soft_type_signal",
            "soft_diversity_signal",
        ]
    }
    final_components = {
        key: float(signals.get(key, 0.0))
        for key in [
            "proof_role_coverage",
            "answer_type_compatibility",
            "current_relevance",
            "useful_convergence",
        ]
    }
    penalties = {
        key: float(signals.get(key, 0.0))
        for key in [
            "soft_loop_penalty",
            "soft_drift_penalty",
            "noisy_branch_penalty",
            "unresolved_need_penalty",
            "hard_loop_penalty",
            "semantic_level_drift_penalty",
            "surface_convergence_penalty",
            "redundancy_penalty",
        ]
    }
    strongest_retention = max(retention_components.items(), key=lambda item: item[1], default=("", 0.0))
    strongest_final = max(final_components.items(), key=lambda item: item[1], default=("", 0.0))
    lines = [
        f"- Strongest retention component: `{strongest_retention[0] or 'none'}` = `{strongest_retention[1]:.4f}`",
        f"- Strongest final-proof component: `{strongest_final[0] or 'none'}` = `{strongest_final[1]:.4f}`",
        f"- Retention score: `{float(signals.get('retention_score', paths[0].get('retention_score', 0.0) if paths else 0.0)):.4f}`",
        f"- Final proof score: `{float(signals.get('final_proof_score', paths[0].get('path_score', 0.0) if paths else 0.0)):.4f}`",
        f"- Future used for retention: `{float(signals.get('future_retention_bonus', 0.0)):.4f}`",
        f"- Unresolved need penalty: `{penalties.get('unresolved_need_penalty', 0.0):.4f}`",
        f"- Hard loop penalty: `{penalties.get('hard_loop_penalty', 0.0):.4f}`",
        f"- Semantic drift penalty: `{penalties.get('semantic_level_drift_penalty', 0.0):.4f}`",
    ]
    if row["two_score_proof_state_beam"]["hits_at_1"] and not row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: two-score beat current proof-state on this question.")
    elif row["soft_proof_state_beam"]["hits_at_1"] and not row["two_score_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: two-score hurt a case current proof-state got right.")
    elif row["two_score_proof_state_beam"]["hits_at_1"] and row["soft_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: both proof-state variants were correct.")
    else:
        lines.append("- Outcome: both proof-state variants failed.")
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


def future_aware_v2_choice_explanation(row: dict[str, Any]) -> list[str]:
    future_top = row["future_aware_v2_proof_state_beam"].get("top_answer") or {}
    paths = future_top.get("paths", []) if future_top else []
    signals = paths[0].get("soft_signals", {}) if paths else {}
    positive_components = {
        key: float(signals.get(key, 0.0))
        for key in ["current_relevance", "gated_future_bonus", "useful_convergence", "type_compatibility", "progress"]
    }
    negative_components = {
        key: float(signals.get(key, 0.0))
        for key in ["surface_convergence_penalty", "loop_penalty", "drift_penalty", "redundancy_penalty", "noisy_branch_penalty"]
    }
    strongest = max(positive_components.items(), key=lambda item: item[1], default=("", 0.0))
    lines = [
        f"- Strongest positive component: `{strongest[0] or 'none'}` = `{strongest[1]:.4f}`",
        f"- Raw future bonus: `{float(signals.get('raw_future_bonus', 0.0)):.4f}`",
        f"- Capped future bonus: `{float(signals.get('future_bonus_capped', 0.0)):.4f}`",
        f"- Role gate: `{float(signals.get('role_gate', 0.0)):.4f}`",
        f"- Non-drift gate: `{float(signals.get('non_drift_gate', 0.0)):.4f}`",
        f"- Non-loop gate: `{float(signals.get('non_loop_gate', 0.0)):.4f}`",
        f"- Gated future bonus: `{positive_components.get('gated_future_bonus', 0.0):.4f}`",
        f"- Loop penalty applied: `{negative_components.get('loop_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('loop_penalty', 0.0):.4f}`)",
        f"- Drift penalty applied: `{negative_components.get('drift_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('drift_penalty', 0.0):.4f}`)",
        f"- Surface convergence penalty applied: `{negative_components.get('surface_convergence_penalty', 0.0) < 0.0}` "
        f"(penalty `{negative_components.get('surface_convergence_penalty', 0.0):.4f}`)",
    ]
    if row["future_aware_v2_proof_state_beam"]["hits_at_1"] and not row["future_aware_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: future-aware v2 beat future-aware v1 on this question.")
    elif row["future_aware_proof_state_beam"]["hits_at_1"] and not row["future_aware_v2_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: future-aware v2 hurt a case future-aware v1 got right.")
    elif row["future_aware_v2_proof_state_beam"]["hits_at_1"] and row["future_aware_proof_state_beam"]["hits_at_1"]:
        lines.append("- Outcome: both future-aware variants were correct.")
    else:
        lines.append("- Outcome: both future-aware variants failed.")
    return lines


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def select_debug_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "two_score_over_current":
        selected = [
            row for row in rows
            if row["two_score_proof_state_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "current_over_two_score":
        selected = [
            row for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["two_score_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "two_score_over_baseline":
        selected = [
            row for row in rows
            if row["two_score_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ]
    elif kind == "baseline_over_two_score":
        selected = [
            row for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["two_score_proof_state_beam"]["final_answer_f1"]
        ]
    elif kind == "future":
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
        two_score_top = row["two_score_proof_state_beam"]["top_answer"] or {}
        future_top = row["future_aware_proof_state_beam"]["top_answer"] or {}
        future_v2_top = row["future_aware_v2_proof_state_beam"]["top_answer"] or {}
        baseline_path = top_path_readable(baseline_top)
        proof_path = top_path_readable(proof_top)
        two_score_path = top_path_readable(two_score_top)
        future_path = top_path_readable(future_top)
        future_v2_path = top_path_readable(future_v2_top)
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
                f"- Two-score top answer: `{two_score_top.get('answer_label', '')}`",
                f"- Two-score evidence: {two_score_path}",
                f"- Future-aware top answer: `{future_top.get('answer_label', '')}`",
                f"- Future-aware evidence: {future_path}",
                f"- Future-aware v2 top answer: `{future_v2_top.get('answer_label', '')}`",
                f"- Future-aware v2 evidence: {future_v2_path}",
                f"- Likely reason: {likely}",
                "",
            ]
        )
    return lines


def build_error_overlap(
    rows: list[dict[str, Any]],
    future_key: str = "future_aware_v2_proof_state_beam",
    diagnostic_label: str = "future_aware_v2_proof_state_beam",
) -> dict[str, Any]:
    cases_v2 = [error_overlap_case(row, future_key=future_key) for row in rows]
    cases_v1 = [error_overlap_case(row, future_key="future_aware_proof_state_beam") for row in rows]
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
    summary = {key: sum(1 for case in cases_v2 if case[key]) for key in summary_keys}
    summary["total_questions"] = len(cases_v2)
    summary["diagnostic_future_mode"] = diagnostic_label
    summary["diagnostic_constants"] = TWO_SCORE_CONSTANTS if future_key == "two_score_proof_state_beam" else FUTURE_AWARE_V2_CONSTANTS
    summary["drift_family_counts_baseline"] = dict(Counter(case["baseline_drift_family"] for case in cases_v2))
    summary["drift_family_counts_diagnostic_mode"] = dict(Counter(case["future_aware_drift_family"] for case in cases_v2))
    summary["future_v1_repeats_baseline_mistake"] = sum(1 for case in cases_v1 if case["future_repeats_baseline_mistake"])
    summary["future_v1_loses_current_proof_state_win"] = sum(1 for case in cases_v1 if case["future_loses_current_proof_state_win"])
    summary["future_v1_penalties_tiny_compared_with_positive_terms"] = sum(1 for case in cases_v1 if case["penalties_tiny_compared_with_positive_terms"])
    return {
        "summary": summary,
        "cases": cases_v2,
        "future_v1_cases": cases_v1,
        "examples": {
            "future_repeats_baseline_mistakes": select_overlap_examples(cases_v2, "future_repeats_baseline_mistake"),
            "future_avoids_baseline_mistakes": select_overlap_examples(cases_v2, "future_avoids_baseline_mistake"),
            "future_loses_to_current_proof_state": select_overlap_examples(cases_v2, "future_loses_current_proof_state_win"),
        },
    }


def error_overlap_case(row: dict[str, Any], future_key: str = "future_aware_proof_state_beam") -> dict[str, Any]:
    baseline = row["baseline_path_beam"]
    current = row["soft_proof_state_beam"]
    future = row[future_key]
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


V2_COMPONENT_KEYS = [
    "current_relevance",
    "raw_future_bonus",
    "future_bonus_capped",
    "role_gate",
    "non_drift_gate",
    "non_loop_gate",
    "gated_future_bonus",
    "useful_convergence",
    "type_compatibility",
    "progress",
    "surface_convergence_penalty",
    "loop_penalty",
    "drift_penalty",
    "redundancy_penalty",
    "noisy_branch_penalty",
    "final_score",
]

TWO_SCORE_COMPONENT_KEYS = [
    "current_relevance",
    "raw_future",
    "role_gate",
    "non_drift_gate",
    "non_loop_gate",
    "future_retention_bonus",
    "soft_progress",
    "soft_type_signal",
    "soft_diversity_signal",
    "soft_loop_penalty",
    "soft_drift_penalty",
    "noisy_branch_penalty",
    "retention_score",
    "proof_role_coverage",
    "covered_need_score",
    "unresolved_need_score",
    "relation_role_coverage",
    "answer_type_compatibility",
    "useful_convergence",
    "unresolved_need_penalty",
    "hard_loop_penalty",
    "semantic_level_drift_penalty",
    "surface_convergence_penalty",
    "redundancy_penalty",
    "final_proof_score",
]


def build_target_behavior_audit(
    rows: list[dict[str, Any]],
    gold_survival_audit: dict[str, Any],
    target_key: str,
    target_label: str,
) -> dict[str, Any]:
    survival_by_question = {case["question_id"]: case for case in gold_survival_audit.get("cases", [])}
    cases = [build_target_behavior_case(row, survival_by_question.get(row["question_id"], {}), target_key) for row in rows]
    label_counts = Counter(label for case in cases for label in case["behavior_labels"])
    changed_first_cases = [case for case in cases if not case["same_first_hop_relation_as_baseline"]]
    summary = {
        "diagnostic_mode": target_key,
        "target_label": target_label,
        "total_questions": len(cases),
        "behavior_label_counts": dict(label_counts),
        "target_changes_first_hop_away_from_baseline": len(changed_first_cases),
        "target_first_hop_change_helps": sum(
            1 for case in changed_first_cases
            if case["target_correct"] and not case["baseline_correct"]
        ),
        "target_first_hop_change_hurts": sum(
            1 for case in changed_first_cases
            if not case["target_correct"] and (case["baseline_correct"] or case["current_proof_state_correct"])
        ),
        "target_repeats_baseline_mistake": label_counts["repeated_baseline_mistake"],
        "target_changes_to_new_wrong_mistake": label_counts["changed_to_new_wrong_drift"],
        "target_over_penalizes_valid_convergence": label_counts["over_penalized_valid_convergence"],
        "target_under_penalizes_bad_drift": label_counts["under_penalized_generic_branch"],
        "target_picks_wrong_sibling_answer": label_counts["picked_wrong_sibling_answer"],
        "penalties_too_weak": label_counts["under_penalized_generic_branch"] + label_counts["under_penalized_loop"],
        "penalties_too_strong": label_counts["over_penalized_valid_convergence"] + label_counts["over_penalized_valid_bidirectional_evidence"],
    }
    return {"summary": summary, "cases": cases}


def build_target_behavior_case(row: dict[str, Any], survival: dict[str, Any], target_key: str) -> dict[str, Any]:
    overlap = error_overlap_case(row, future_key=target_key)
    baseline = row["baseline_path_beam"]
    current = row["soft_proof_state_beam"]
    target = row[target_key]
    target_top = target.get("top_answer") or {}
    breakdown = target_score_breakdown_from_candidate(target_top)
    labels: list[str] = []

    if target["hits_at_1"]:
        if not baseline["hits_at_1"] and not overlap["same_drift_family_as_baseline"]:
            labels.append("avoided_baseline_drift")
        if current["hits_at_1"] or baseline["hits_at_1"]:
            labels.append("kept_good_alternative")
        if breakdown["future_retention_bonus"] > 0.05:
            labels.append("preserved_useful_future_path")
    else:
        if target["gold_generated"]:
            labels.append("gold_generated_but_ranked_low")
        else:
            labels.append("gold_not_generated")
        if overlap["future_repeats_baseline_mistake"]:
            labels.append("repeated_baseline_mistake")
        elif not baseline["hits_at_1"]:
            labels.append("changed_to_new_wrong_drift")
        if current["hits_at_1"]:
            labels.append("killed_useful_future_path")
        if survival.get("gold_survival_stage") == "gold_answer_generated_but_ranked_low":
            labels.append("picked_wrong_sibling_answer")

    diagnoses = set(survival.get("diagnosis", []))
    if {
        "hard_loop_false_positive_on_gold",
        "surface_convergence_false_positive_on_gold",
        "semantic_drift_false_positive_on_gold",
    } & diagnoses:
        labels.append("over_penalized_valid_convergence")
    if "hard_loop_false_positive_on_gold" in diagnoses:
        labels.append("over_penalized_valid_bidirectional_evidence")
    if not target["hits_at_1"] and generic_bad_drift_family(overlap["future_aware_drift_family"]):
        if breakdown["target_penalty_to_positive_ratio"] < 0.35 or (
            breakdown["semantic_level_drift_penalty"] == 0.0 and breakdown["hard_loop_penalty"] == 0.0
        ):
            labels.append("under_penalized_generic_branch")
    if not target["hits_at_1"] and overlap["future_aware_drift_family"] in {"inverse_loop", "generic_relation_loop"}:
        if breakdown["hard_loop_penalty"] == 0.0:
            labels.append("under_penalized_loop")
    if not target["hits_at_1"] and (
        breakdown["future_retention_bonus"] >= 0.18 or breakdown["current_relevance"] >= 0.45
    ):
        labels.append("followed_surface_future_match")

    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "baseline_top_path": top_path_readable_plain(baseline.get("top_answer") or {}),
        "current_proof_state_top_path": top_path_readable_plain(current.get("top_answer") or {}),
        "target_top_path": top_path_readable_plain(target_top),
        "baseline_correct": baseline["hits_at_1"],
        "current_proof_state_correct": current["hits_at_1"],
        "target_correct": target["hits_at_1"],
        "target_drift_family": overlap["future_aware_drift_family"],
        "same_first_hop_relation_as_baseline": overlap["same_first_hop_relation_as_baseline"],
        "same_relation_sequence_as_baseline": overlap["same_relation_sequence_as_baseline"],
        "same_drift_family_as_baseline": overlap["same_drift_family_as_baseline"],
        "gold_survival_stage": survival.get("gold_survival_stage", "unknown"),
        "score_breakdown": {key: breakdown.get(key, 0.0) for key in TWO_SCORE_COMPONENT_KEYS + [
            "target_positive_score_sum",
            "target_negative_penalty_sum",
            "target_penalty_to_positive_ratio",
        ]},
        "behavior_labels": sorted(set(labels)) or ["unclear"],
    }


def build_target_gold_survival_audit(
    graph: KnowledgeGraph,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    target_key: str,
    target_label: str,
) -> dict[str, Any]:
    cases = [build_target_gold_survival_case(graph, row, args, target_key) for row in rows]
    stage_counts = Counter(case["gold_survival_stage"] for case in cases)
    diagnosis_counts = Counter(label for case in cases for label in case["diagnosis"])
    summary = {
        "diagnostic_mode": target_key,
        "target_label": target_label,
        "total_questions": len(cases),
        "stage_counts": dict(stage_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "number_of_pruning_failures": sum(1 for case in cases if "pruned" in case["gold_survival_stage"] or "removed" in case["gold_survival_stage"]),
        "number_of_ranking_failures": stage_counts["gold_answer_generated_but_ranked_low"],
        "number_of_first_hop_failures": sum(1 for case in cases if case["gold_survival_stage"] in {
            "gold_never_in_local_frontier",
            "gold_first_hop_available_but_pruned",
            "gold_first_hop_removed_by_diversity",
        }),
        "number_of_second_hop_failures": sum(1 for case in cases if case["gold_survival_stage"] in {
            "gold_first_hop_kept_but_second_hop_not_generated",
            "gold_second_hop_generated_but_pruned",
        }),
        "number_of_false_positive_penalty_failures": sum(
            1 for case in cases
            if {
                "semantic_drift_false_positive_on_gold",
                "hard_loop_false_positive_on_gold",
                "surface_convergence_false_positive_on_gold",
            } & set(case["diagnosis"])
        ),
        "number_of_bad_state_overreward_failures": sum(
            1 for case in cases
            if {
                "bad_state_future_retention_too_high",
                "bad_state_current_relevance_too_high",
                "bad_state_answer_type_too_high",
                "bad_state_proof_role_coverage_too_high",
            } & set(case["diagnosis"])
        ),
        "number_of_sibling_ambiguity_failures": diagnosis_counts["gold_sibling_ranked_lower"],
    }
    return {"summary": summary, "cases": cases}


def build_target_gold_survival_case(
    graph: KnowledgeGraph,
    row: dict[str, Any],
    args: argparse.Namespace,
    target_key: str,
) -> dict[str, Any]:
    target_result = row[target_key]
    gold_ids = set(str(entity_id) for entity_id in row["gold_answer_ids"])
    top_candidate = target_result.get("top_answer") or {}
    top_breakdown = target_score_breakdown_from_candidate(top_candidate)
    audit_trace = target_result.get("audit_trace", [])
    hop1_all = audit_states_for_hop(audit_trace, 1)
    hop1_selected = audit_selected_states_for_hop(audit_trace, 1)
    hop2_all = audit_states_for_hop(audit_trace, 2)
    final_gold_candidates = [candidate for candidate in target_result.get("candidate_answers", []) if candidate.get("is_gold")]
    final_gold_candidates.sort(key=lambda item: int(item.get("rank", 999999)))
    gold_hop2_states = [state for state in hop2_all if str(state.get("target_entity", "")) in gold_ids]
    best_gold_state = max(gold_hop2_states, key=lambda item: float(item.get("final_proof_score", 0.0)), default=None)
    best_final_gold = final_gold_candidates[0] if final_gold_candidates else None
    if best_gold_state is None and best_final_gold is not None:
        best_gold_state = target_state_summary_from_candidate(best_final_gold)

    selected_reaching_hop1 = [
        state for state in hop1_selected
        if state_can_reach_gold_next_hop(graph, row, state, args, top_k=args.top_k)
    ]
    generated_reaching_hop1 = [
        state for state in hop1_all
        if state_can_reach_gold_next_hop(graph, row, state, args, top_k=args.top_k)
    ]
    raw_reaching_hop1 = find_raw_reaching_hop1_states(graph, row, args)

    if target_result["hits_at_1"]:
        stage = "gold_answer_top1"
    elif final_gold_candidates:
        stage = "gold_answer_generated_but_ranked_low"
    elif gold_hop2_states:
        stage = "gold_second_hop_generated_but_pruned"
    elif selected_reaching_hop1:
        stage = "gold_first_hop_kept_but_second_hop_not_generated"
    elif generated_reaching_hop1:
        stage = "gold_first_hop_available_but_pruned"
    elif raw_reaching_hop1:
        stage = "gold_first_hop_available_but_pruned"
    elif not audit_trace:
        stage = "unclear"
    else:
        stage = "gold_never_in_local_frontier"

    gold_breakdown = target_score_breakdown_from_state(best_gold_state or {})
    comparison = target_score_component_comparison(top_breakdown, gold_breakdown)
    diagnosis = diagnose_target_gold_survival(
        row=row,
        stage=stage,
        target_result=target_result,
        top_breakdown=top_breakdown,
        gold_breakdown=gold_breakdown,
    )
    hop1_rank = gold_hop1_rank(hop1_all, best_gold_state)
    hop2_rank = int(best_gold_state.get("rank", 0)) if best_gold_state else None
    final_rank = int(best_final_gold.get("rank", 0)) if best_final_gold else None
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "gold_survival_stage": stage,
        "target_top_state": top_path_readable_plain(top_candidate),
        "target_top_answer": top_candidate.get("answer_label", ""),
        "target_top_retention_score": top_breakdown["retention_score"],
        "target_top_final_proof_score": top_breakdown["final_proof_score"],
        "best_gold_state": state_readable(best_gold_state) if best_gold_state else "",
        "hop_1_rank": hop1_rank,
        "hop_2_rank": hop2_rank,
        "final_rank": final_rank,
        "final_proof_score": gold_breakdown["final_proof_score"],
        "gold_candidate_states": [
            {
                "rank": state.get("rank"),
                "path": state_readable(state),
                "retention_score": float(state.get("retention_score", 0.0)),
                "final_proof_score": float(state.get("final_proof_score", 0.0)),
                "score_components": target_score_breakdown_from_state(state),
            }
            for state in gold_hop2_states[:10]
        ],
        "target_top_score_components": {key: top_breakdown.get(key, 0.0) for key in TWO_SCORE_COMPONENT_KEYS},
        "best_gold_score_components": {key: gold_breakdown.get(key, 0.0) for key in TWO_SCORE_COMPONENT_KEYS},
        "score_difference_top_minus_gold": comparison["score_difference_top_minus_gold"],
        "components_favoring_wrong_top": comparison["components_favoring_wrong_top"],
        "components_hurting_gold": comparison["components_hurting_gold"],
        "diagnosis": diagnosis,
    }


def build_behavior_audit(rows: list[dict[str, Any]], gold_survival_audit: dict[str, Any]) -> dict[str, Any]:
    survival_by_question = {case["question_id"]: case for case in gold_survival_audit.get("cases", [])}
    cases = []
    for row in rows:
        survival = survival_by_question.get(row["question_id"], {})
        cases.append(build_behavior_case(row, survival))
    label_counts = Counter(label for case in cases for label in case["behavior_labels"])
    changed_first_cases = [case for case in cases if not case["same_first_hop_relation_as_baseline"]]
    summary = {
        "total_questions": len(cases),
        "behavior_label_counts": dict(label_counts),
        "v2_changes_first_hop_away_from_baseline": len(changed_first_cases),
        "v2_first_hop_change_helps": sum(
            1 for case in changed_first_cases
            if case["future_aware_v2_correct"] and not case["baseline_correct"]
        ),
        "v2_first_hop_change_hurts": sum(
            1 for case in changed_first_cases
            if not case["future_aware_v2_correct"]
            and (case["baseline_correct"] or case["current_proof_state_correct"] or case["future_aware_v1_correct"])
        ),
        "v2_repeats_baseline_mistake": label_counts["repeated_baseline_mistake"],
        "v2_changes_to_new_wrong_mistake": label_counts["changed_to_new_wrong_drift"],
        "v2_over_penalizes_valid_convergence": label_counts["over_penalized_valid_convergence"],
        "v2_under_penalizes_bad_drift": label_counts["under_penalized_generic_branch"],
        "v2_picks_wrong_sibling_answer": label_counts["picked_wrong_sibling_answer"],
        "penalties_too_weak": label_counts["under_penalized_generic_branch"] + label_counts["under_penalized_loop"],
        "penalties_too_strong": label_counts["over_penalized_valid_convergence"] + label_counts["over_penalized_valid_bidirectional_evidence"],
    }
    return {
        "summary": summary,
        "cases": cases,
    }


def build_behavior_case(row: dict[str, Any], survival: dict[str, Any]) -> dict[str, Any]:
    overlap = error_overlap_case(row, future_key="future_aware_v2_proof_state_beam")
    baseline = row["baseline_path_beam"]
    current = row["soft_proof_state_beam"]
    future_v1 = row["future_aware_proof_state_beam"]
    future_v2 = row["future_aware_v2_proof_state_beam"]
    v2_top = future_v2.get("top_answer") or {}
    v2_breakdown = full_v2_score_breakdown_from_candidate(v2_top)
    labels: list[str] = []

    if future_v2["hits_at_1"]:
        if not baseline["hits_at_1"] and not overlap["same_drift_family_as_baseline"]:
            labels.append("avoided_baseline_drift")
        if current["hits_at_1"] or future_v1["hits_at_1"] or baseline["hits_at_1"]:
            labels.append("kept_good_alternative")
        if v2_breakdown["gated_future_bonus"] > 0.05:
            labels.append("preserved_useful_future_path")
    else:
        if future_v2["gold_generated"]:
            labels.append("gold_generated_but_ranked_low")
        else:
            labels.append("gold_not_generated")
        if overlap["future_repeats_baseline_mistake"]:
            labels.append("repeated_baseline_mistake")
        elif not baseline["hits_at_1"]:
            labels.append("changed_to_new_wrong_drift")
        if future_v1["hits_at_1"]:
            labels.append("killed_useful_future_path")
        if survival.get("gold_survival_stage") == "gold_answer_generated_but_ranked_low":
            labels.append("picked_wrong_sibling_answer")

    diagnoses = set(survival.get("diagnosis", []))
    if {
        "loop_penalty_false_positive_on_gold",
        "surface_convergence_false_positive_on_gold",
    } & diagnoses:
        labels.append("over_penalized_valid_convergence")
    if "loop_penalty_false_positive_on_gold" in diagnoses:
        labels.append("over_penalized_valid_bidirectional_evidence")
    if future_v2["hits_at_1"] is False and generic_bad_drift_family(overlap["future_aware_drift_family"]):
        if v2_breakdown["penalty_to_positive_ratio"] < 0.35 or (
            v2_breakdown["drift_penalty"] == 0.0 and v2_breakdown["loop_penalty"] == 0.0
        ):
            labels.append("under_penalized_generic_branch")
    if future_v2["hits_at_1"] is False and overlap["future_aware_drift_family"] in {"inverse_loop", "generic_relation_loop"}:
        if v2_breakdown["loop_penalty"] == 0.0:
            labels.append("under_penalized_loop")
    if future_v2["hits_at_1"] is False and (
        v2_breakdown["gated_future_bonus"] >= 0.18 or v2_breakdown["current_relevance"] >= 0.45
    ):
        labels.append("followed_surface_future_match")

    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "baseline_top_path": top_path_readable_plain(baseline.get("top_answer") or {}),
        "current_proof_state_top_path": top_path_readable_plain(current.get("top_answer") or {}),
        "future_aware_v1_top_path": top_path_readable_plain(future_v1.get("top_answer") or {}),
        "future_aware_v2_top_path": top_path_readable_plain(v2_top),
        "baseline_correct": baseline["hits_at_1"],
        "current_proof_state_correct": current["hits_at_1"],
        "future_aware_v1_correct": future_v1["hits_at_1"],
        "future_aware_v2_correct": future_v2["hits_at_1"],
        "future_aware_v2_drift_family": overlap["future_aware_drift_family"],
        "same_first_hop_relation_as_baseline": overlap["same_first_hop_relation_as_baseline"],
        "same_relation_sequence_as_baseline": overlap["same_relation_sequence_as_baseline"],
        "same_drift_family_as_baseline": overlap["same_drift_family_as_baseline"],
        "gold_survival_stage": survival.get("gold_survival_stage", "unknown"),
        "score_breakdown": {key: v2_breakdown.get(key, 0.0) for key in V2_COMPONENT_KEYS},
        "behavior_labels": sorted(set(labels)) or ["unclear"],
    }


def build_gold_survival_audit(graph: KnowledgeGraph, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    cases = [build_gold_survival_case(graph, row, args) for row in rows]
    stage_counts = Counter(case["gold_survival_stage"] for case in cases)
    diagnosis_counts = Counter(label for case in cases for label in case["diagnosis"])
    summary = {
        "total_questions": len(cases),
        "stage_counts": dict(stage_counts),
        "diagnosis_counts": dict(diagnosis_counts),
        "number_of_pruning_failures": sum(1 for case in cases if "pruned" in case["gold_survival_stage"] or "removed" in case["gold_survival_stage"]),
        "number_of_ranking_failures": stage_counts["gold_answer_generated_but_ranked_low"],
        "number_of_first_hop_failures": sum(1 for case in cases if case["gold_survival_stage"] in {
            "gold_never_in_local_frontier",
            "gold_first_hop_available_but_pruned",
            "gold_first_hop_removed_by_diversity",
        }),
        "number_of_second_hop_failures": sum(1 for case in cases if case["gold_survival_stage"] in {
            "gold_first_hop_kept_but_second_hop_not_generated",
            "gold_second_hop_generated_but_pruned",
        }),
        "number_of_false_positive_penalty_failures": sum(
            1 for case in cases
            if {
                "drift_penalty_false_positive_on_gold",
                "loop_penalty_false_positive_on_gold",
                "surface_convergence_false_positive_on_gold",
            } & set(case["diagnosis"])
        ),
        "number_of_bad_state_overreward_failures": sum(
            1 for case in cases
            if {
                "bad_state_future_bonus_too_high",
                "bad_state_current_relevance_too_high",
                "bad_state_type_compatibility_too_high",
            } & set(case["diagnosis"])
        ),
        "number_of_sibling_ambiguity_failures": diagnosis_counts["gold_sibling_ranked_lower"],
    }
    return {
        "summary": summary,
        "cases": cases,
    }


def build_gold_survival_case(graph: KnowledgeGraph, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    future_v2 = row["future_aware_v2_proof_state_beam"]
    gold_ids = set(str(entity_id) for entity_id in row["gold_answer_ids"])
    top_candidate = future_v2.get("top_answer") or {}
    top_breakdown = full_v2_score_breakdown_from_candidate(top_candidate)
    audit_trace = future_v2.get("audit_trace", [])
    hop1_all = audit_states_for_hop(audit_trace, 1)
    hop1_selected = audit_selected_states_for_hop(audit_trace, 1)
    hop2_all = audit_states_for_hop(audit_trace, 2)
    final_gold_candidates = [candidate for candidate in future_v2.get("candidate_answers", []) if candidate.get("is_gold")]
    final_gold_candidates.sort(key=lambda item: int(item.get("rank", 999999)))
    gold_hop2_states = [state for state in hop2_all if str(state.get("target_entity", "")) in gold_ids]
    best_gold_state = max(gold_hop2_states, key=lambda item: float(item.get("final_score", 0.0)), default=None)
    best_final_gold = final_gold_candidates[0] if final_gold_candidates else None
    if best_gold_state is None and best_final_gold is not None:
        best_gold_state = state_summary_from_candidate(best_final_gold)

    selected_reaching_hop1 = [
        state for state in hop1_selected
        if state_can_reach_gold_next_hop(graph, row, state, args, top_k=args.top_k)
    ]
    generated_reaching_hop1 = [
        state for state in hop1_all
        if state_can_reach_gold_next_hop(graph, row, state, args, top_k=args.top_k)
    ]
    raw_reaching_hop1 = find_raw_reaching_hop1_states(graph, row, args)

    if future_v2["hits_at_1"]:
        stage = "gold_answer_top1"
    elif final_gold_candidates:
        stage = "gold_answer_generated_but_ranked_low"
    elif gold_hop2_states:
        stage = "gold_second_hop_generated_but_pruned"
    elif selected_reaching_hop1:
        stage = "gold_first_hop_kept_but_second_hop_not_generated"
    elif generated_reaching_hop1:
        first_rank = min(int(state.get("rank", 999999)) for state in generated_reaching_hop1)
        selected_keys = {state_identity(state) for state in hop1_selected}
        generated_keys = {state_identity(state) for state in generated_reaching_hop1}
        if first_rank <= int(args.beam_width) and not (selected_keys & generated_keys):
            stage = "gold_first_hop_removed_by_diversity"
        else:
            stage = "gold_first_hop_available_but_pruned"
    elif raw_reaching_hop1:
        stage = "gold_first_hop_available_but_pruned"
    elif not audit_trace:
        stage = "unclear"
    else:
        stage = "gold_never_in_local_frontier"

    gold_breakdown = full_v2_score_breakdown_from_state(best_gold_state or {})
    comparison = score_component_comparison(top_breakdown, gold_breakdown)
    diagnosis = diagnose_gold_survival(
        row=row,
        stage=stage,
        top_candidate=top_candidate,
        best_gold_state=best_gold_state,
        top_breakdown=top_breakdown,
        gold_breakdown=gold_breakdown,
        generated_reaching_hop1=generated_reaching_hop1,
        hop1_selected=hop1_selected,
    )
    hop1_rank = gold_hop1_rank(hop1_all, best_gold_state)
    hop2_rank = int(best_gold_state.get("rank", 0)) if best_gold_state else None
    final_rank = int(best_final_gold.get("rank", 0)) if best_final_gold else None
    return {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_answers": row["gold_answers"],
        "gold_survival_stage": stage,
        "v2_top_state": top_path_readable_plain(top_candidate),
        "v2_top_answer": top_candidate.get("answer_label", ""),
        "v2_top_score": top_breakdown["final_score"],
        "best_gold_state": state_readable(best_gold_state) if best_gold_state else "",
        "hop_1_rank": hop1_rank,
        "hop_2_rank": hop2_rank,
        "final_rank": final_rank,
        "final_score": gold_breakdown["final_score"],
        "gold_candidate_states": [
            {
                "rank": state.get("rank"),
                "path": state_readable(state),
                "score": float(state.get("final_score", 0.0)),
                "score_components": full_v2_score_breakdown_from_state(state),
            }
            for state in gold_hop2_states[:10]
        ],
        "v2_top_score_components": {key: top_breakdown.get(key, 0.0) for key in V2_COMPONENT_KEYS},
        "best_gold_score_components": {key: gold_breakdown.get(key, 0.0) for key in V2_COMPONENT_KEYS},
        "score_difference_top_minus_gold": comparison["score_difference_top_minus_gold"],
        "components_favoring_wrong_top": comparison["components_favoring_wrong_top"],
        "components_hurting_gold": comparison["components_hurting_gold"],
        "diagnosis": diagnosis,
    }


def audit_states_for_hop(audit_trace: list[dict[str, Any]], hop: int) -> list[dict[str, Any]]:
    for item in audit_trace:
        if item.get("hop") == hop:
            return list(item.get("all_candidate_states", []) or item.get("top_candidate_states", []))
    return []


def audit_selected_states_for_hop(audit_trace: list[dict[str, Any]], hop: int) -> list[dict[str, Any]]:
    for item in audit_trace:
        if item.get("hop") == hop:
            return list(item.get("selected_states", []))
    return []


def state_identity(state: dict[str, Any]) -> str:
    return f"{state.get('target_entity', '')}|{state.get('candidate_evidence_state', '')}"


def state_can_reach_gold_next_hop(
    graph: KnowledgeGraph,
    row: dict[str, Any],
    state: dict[str, Any],
    args: argparse.Namespace,
    top_k: int | None,
) -> bool:
    source_id = str(state.get("target_entity", ""))
    if not source_id:
        return False
    gold_ids = set(str(entity_id) for entity_id in row["gold_answer_ids"])
    frontier = graph.candidate_relations([source_id], cap=args.relation_cap, sample_entities=args.sample_entities)
    ranked = rank_relations(row["question"], frontier)
    if top_k is not None:
        ranked = ranked[:top_k]
    for candidate in ranked:
        targets = relation_targets(graph, source_id, candidate, args.max_branch_entities)
        if gold_ids & set(targets):
            return True
    return False


def find_raw_reaching_hop1_states(graph: KnowledgeGraph, row: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for source_id in row["start_entity"].get("entity_ids", []):
        frontier = graph.candidate_relations([source_id], cap=args.relation_cap, sample_entities=args.sample_entities)
        for candidate in rank_relations(row["question"], frontier):
            targets = relation_targets(graph, source_id, candidate, args.max_branch_entities)
            for target_id in targets:
                state = {
                    "target_entity": target_id,
                    "target_label": graph.entity_name(target_id),
                    "candidate_evidence_state": f"{graph.entity_name(source_id)} --{candidate['relation_id']}[{candidate['direction']}]--> {graph.entity_name(target_id)}",
                    "rank": len(out) + 1,
                }
                if state_can_reach_gold_next_hop(graph, row, state, args, top_k=None):
                    out.append(state)
    return out


def gold_hop1_rank(hop1_all: list[dict[str, Any]], best_gold_state: dict[str, Any] | None) -> int | None:
    if not best_gold_state:
        return None
    evidence = best_gold_state.get("evidence", []) or []
    first_targets = {str(step.get("to_entity_id", "")) for step in evidence if step.get("hop") == 1}
    ranks = [
        int(state.get("rank", 999999))
        for state in hop1_all
        if str(state.get("target_entity", "")) in first_targets
    ]
    return min(ranks) if ranks else None


def state_summary_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    paths = candidate.get("paths", []) if candidate else []
    path = paths[0] if paths else {}
    return {
        "rank": candidate.get("rank"),
        "candidate_evidence_state": path.get("readable", ""),
        "evidence": path.get("evidence", []),
        "target_entity": candidate.get("answer_id", ""),
        "target_label": candidate.get("answer_label", ""),
        "final_score": path.get("path_score", candidate.get("best_path_score", candidate.get("score", 0.0))),
        **{key: path.get("soft_signals", {}).get(key, 0.0) for key in V2_COMPONENT_KEYS if key != "final_score"},
    }


def target_state_summary_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    paths = candidate.get("paths", []) if candidate else []
    path = paths[0] if paths else {}
    signals = path.get("soft_signals", {}) if path else {}
    return {
        "rank": candidate.get("rank"),
        "candidate_evidence_state": path.get("readable", ""),
        "evidence": path.get("evidence", []),
        "target_entity": candidate.get("answer_id", ""),
        "target_label": candidate.get("answer_label", ""),
        "retention_score": path.get("retention_score", signals.get("retention_score", 0.0)),
        "final_proof_score": path.get("path_score", signals.get("final_proof_score", candidate.get("score", 0.0))),
        **{key: signals.get(key, 0.0) for key in TWO_SCORE_COMPONENT_KEYS if key not in {"retention_score", "final_proof_score"}},
    }


def target_score_breakdown_from_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    paths = candidate.get("paths", []) if candidate else []
    path = paths[0] if paths else {}
    signals = path.get("soft_signals", {}) if path else {}
    out = {key: float(signals.get(key, 0.0)) for key in TWO_SCORE_COMPONENT_KEYS}
    out["retention_score"] = float(path.get("retention_score", signals.get("retention_score", 0.0)) or 0.0)
    out["final_proof_score"] = float(path.get("path_score", candidate.get("best_path_score", candidate.get("score", signals.get("final_proof_score", 0.0)))) or 0.0)
    out.update(target_penalty_ratio_fields(out))
    return out


def target_score_breakdown_from_state(state: dict[str, Any]) -> dict[str, float]:
    out = {key: float(state.get(key, 0.0) or 0.0) for key in TWO_SCORE_COMPONENT_KEYS}
    out["retention_score"] = float(state.get("retention_score", 0.0) or 0.0)
    out["final_proof_score"] = float(state.get("final_proof_score", 0.0) or 0.0)
    out.update(target_penalty_ratio_fields(out))
    return out


def target_penalty_ratio_fields(values: dict[str, float]) -> dict[str, float]:
    positive_keys = [
        "current_relevance",
        "future_retention_bonus",
        "soft_progress",
        "soft_type_signal",
        "soft_diversity_signal",
        "proof_role_coverage",
        "answer_type_compatibility",
        "useful_convergence",
    ]
    negative_keys = [
        "soft_loop_penalty",
        "soft_drift_penalty",
        "noisy_branch_penalty",
        "unresolved_need_penalty",
        "hard_loop_penalty",
        "semantic_level_drift_penalty",
        "surface_convergence_penalty",
        "redundancy_penalty",
    ]
    positive = sum(max(0.0, values.get(key, 0.0)) for key in positive_keys)
    negative = sum(abs(min(0.0, values.get(key, 0.0))) for key in negative_keys)
    return {
        "target_positive_score_sum": positive,
        "target_negative_penalty_sum": negative,
        "target_penalty_to_positive_ratio": negative / positive if positive else 0.0,
    }


def target_score_component_comparison(top: dict[str, float], gold: dict[str, float]) -> dict[str, Any]:
    diff = {key: float(top.get(key, 0.0)) - float(gold.get(key, 0.0)) for key in TWO_SCORE_COMPONENT_KEYS}
    favored_wrong = [
        {"component": key, "top_minus_gold": value}
        for key, value in sorted(diff.items(), key=lambda item: -item[1])
        if value > 0.05
    ][:5]
    penalty_keys = [
        "unresolved_need_penalty",
        "hard_loop_penalty",
        "semantic_level_drift_penalty",
        "surface_convergence_penalty",
        "redundancy_penalty",
        "soft_loop_penalty",
        "soft_drift_penalty",
        "noisy_branch_penalty",
    ]
    hurt_gold = [
        {"component": key, "gold_minus_top": float(gold.get(key, 0.0)) - float(top.get(key, 0.0))}
        for key in penalty_keys
        if float(gold.get(key, 0.0)) < float(top.get(key, 0.0))
    ]
    return {
        "score_difference_top_minus_gold": diff,
        "components_favoring_wrong_top": favored_wrong,
        "components_hurting_gold": hurt_gold,
    }


def diagnose_target_gold_survival(
    row: dict[str, Any],
    stage: str,
    target_result: dict[str, Any],
    top_breakdown: dict[str, float],
    gold_breakdown: dict[str, float],
) -> list[str]:
    labels: list[str] = []
    if gold_breakdown["hard_loop_penalty"] < 0.0:
        labels.append("hard_loop_false_positive_on_gold")
    if gold_breakdown["semantic_level_drift_penalty"] < 0.0:
        labels.append("semantic_drift_false_positive_on_gold")
    if gold_breakdown["surface_convergence_penalty"] < 0.0:
        labels.append("surface_convergence_false_positive_on_gold")
    if gold_breakdown["future_retention_bonus"] + 0.08 < top_breakdown["future_retention_bonus"]:
        labels.append("future_retention_too_low_on_gold")
    if not target_result["hits_at_1"]:
        if top_breakdown["future_retention_bonus"] > gold_breakdown["future_retention_bonus"] + 0.08:
            labels.append("bad_state_future_retention_too_high")
        if top_breakdown["current_relevance"] > gold_breakdown["current_relevance"] + 0.10:
            labels.append("bad_state_current_relevance_too_high")
        if top_breakdown["answer_type_compatibility"] > gold_breakdown["answer_type_compatibility"] + 0.50:
            labels.append("bad_state_answer_type_too_high")
        if top_breakdown["proof_role_coverage"] > gold_breakdown["proof_role_coverage"] + 0.15:
            labels.append("bad_state_proof_role_coverage_too_high")
        if stage == "gold_answer_generated_but_ranked_low":
            labels.append("gold_sibling_ranked_lower")
        if not guess_answer_type(row["question"]):
            labels.append("answer_type_wrong_or_missing")
        if top_breakdown["current_relevance"] >= max(top_breakdown["proof_role_coverage"], top_breakdown["answer_type_compatibility"]):
            labels.append("relation_label_similarity_dominated")
    if stage == "gold_first_hop_kept_but_second_hop_not_generated":
        labels.append("future_retention_too_low_on_gold")
    return sorted(set(labels)) or ["unknown"]


def full_v2_score_breakdown_from_candidate(candidate: dict[str, Any]) -> dict[str, float]:
    paths = candidate.get("paths", []) if candidate else []
    path = paths[0] if paths else {}
    signals = path.get("soft_signals", {}) if path else {}
    out = {key: float(signals.get(key, 0.0)) for key in V2_COMPONENT_KEYS}
    out["final_score"] = float(path.get("path_score", candidate.get("best_path_score", candidate.get("score", 0.0))) or 0.0)
    out.update(penalty_ratio_fields(out))
    return out


def full_v2_score_breakdown_from_state(state: dict[str, Any]) -> dict[str, float]:
    out = {key: float(state.get(key, 0.0) or 0.0) for key in V2_COMPONENT_KEYS}
    out["final_score"] = float(state.get("final_score", 0.0) or 0.0)
    out.update(penalty_ratio_fields(out))
    return out


def penalty_ratio_fields(values: dict[str, float]) -> dict[str, float]:
    positive_keys = ["current_relevance", "gated_future_bonus", "useful_convergence", "type_compatibility", "progress"]
    negative_keys = ["surface_convergence_penalty", "loop_penalty", "drift_penalty", "redundancy_penalty", "noisy_branch_penalty"]
    positive = sum(max(0.0, values.get(key, 0.0)) for key in positive_keys)
    negative = sum(abs(min(0.0, values.get(key, 0.0))) for key in negative_keys)
    return {
        "positive_score_sum": positive,
        "negative_penalty_sum": negative,
        "penalty_to_positive_ratio": negative / positive if positive else 0.0,
    }


def score_component_comparison(top: dict[str, float], gold: dict[str, float]) -> dict[str, Any]:
    diff = {key: float(top.get(key, 0.0)) - float(gold.get(key, 0.0)) for key in V2_COMPONENT_KEYS}
    favored_wrong = [
        {"component": key, "top_minus_gold": value}
        for key, value in sorted(diff.items(), key=lambda item: -item[1])
        if value > 0.05
    ][:5]
    penalty_keys = ["surface_convergence_penalty", "loop_penalty", "drift_penalty", "redundancy_penalty", "noisy_branch_penalty"]
    hurt_gold = [
        {"component": key, "gold_minus_top": float(gold.get(key, 0.0)) - float(top.get(key, 0.0))}
        for key in penalty_keys
        if float(gold.get(key, 0.0)) < float(top.get(key, 0.0))
    ]
    return {
        "score_difference_top_minus_gold": diff,
        "components_favoring_wrong_top": favored_wrong,
        "components_hurting_gold": hurt_gold,
    }


def diagnose_gold_survival(
    row: dict[str, Any],
    stage: str,
    top_candidate: dict[str, Any],
    best_gold_state: dict[str, Any] | None,
    top_breakdown: dict[str, float],
    gold_breakdown: dict[str, float],
    generated_reaching_hop1: list[dict[str, Any]],
    hop1_selected: list[dict[str, Any]],
) -> list[str]:
    labels: list[str] = []
    if stage == "gold_first_hop_removed_by_diversity":
        labels.append("diversity_pruned_gold_first_hop")
    if best_gold_state:
        if gold_breakdown["role_gate"] < max(0.75, top_breakdown["role_gate"] - 0.20):
            labels.append("role_gate_too_low_on_gold")
        if gold_breakdown["drift_penalty"] < 0.0:
            labels.append("drift_penalty_false_positive_on_gold")
        if gold_breakdown["loop_penalty"] < 0.0:
            labels.append("loop_penalty_false_positive_on_gold")
        if gold_breakdown["surface_convergence_penalty"] < 0.0:
            labels.append("surface_convergence_false_positive_on_gold")
        if gold_breakdown["gated_future_bonus"] + 0.08 < top_breakdown["gated_future_bonus"]:
            labels.append("future_bonus_too_low_on_gold")
    if not row["future_aware_v2_proof_state_beam"]["hits_at_1"]:
        if top_breakdown["gated_future_bonus"] > gold_breakdown["gated_future_bonus"] + 0.08:
            labels.append("bad_state_future_bonus_too_high")
        if top_breakdown["current_relevance"] > gold_breakdown["current_relevance"] + 0.10:
            labels.append("bad_state_current_relevance_too_high")
        if top_breakdown["type_compatibility"] > gold_breakdown["type_compatibility"] + 0.50:
            labels.append("bad_state_type_compatibility_too_high")
        if stage == "gold_answer_generated_but_ranked_low":
            labels.append("gold_sibling_ranked_lower")
        if not guess_answer_type(row["question"]):
            labels.append("answer_type_wrong_or_missing")
        if top_breakdown["current_relevance"] >= max(top_breakdown["gated_future_bonus"], top_breakdown["type_compatibility"]):
            labels.append("relation_label_similarity_dominated")
    if stage == "gold_first_hop_kept_but_second_hop_not_generated":
        labels.append("future_bonus_too_low_on_gold")
    if generated_reaching_hop1 and not {state_identity(state) for state in generated_reaching_hop1} & {state_identity(state) for state in hop1_selected}:
        labels.append("diversity_pruned_gold_first_hop")
    return sorted(set(labels)) or ["unknown"]


def generic_bad_drift_family(family: str) -> bool:
    return family in {
        "geography_drift",
        "broad_location_or_country_branch",
        "award_or_fame_drift",
        "cast_or_film_drift",
        "organization_parent_subsidiary_loop",
        "generic_relation_loop",
        "inverse_loop",
    }


def has_bidirectional_evidence(evidence: list[dict[str, Any]]) -> bool:
    seen = set()
    for step in evidence:
        relation = str(step.get("relation_id", ""))
        direction = str(step.get("direction", ""))
        reverse_direction = "backward" if direction == "forward" else "forward"
        edge = (step.get("from_entity_id"), relation, direction, step.get("to_entity_id"))
        reverse = (step.get("to_entity_id"), relation, reverse_direction, step.get("from_entity_id"))
        if reverse in seen:
            return True
        seen.add(edge)
    relation_dirs: dict[str, set[str]] = defaultdict(set)
    for step in evidence:
        relation_dirs[str(step.get("relation_id", ""))].add(str(step.get("direction", "")))
    return any({"forward", "backward"} <= directions for directions in relation_dirs.values())


def state_readable(state: dict[str, Any] | None) -> str:
    if not state:
        return ""
    return str(state.get("candidate_evidence_state") or state.get("readable") or "")


def write_behavior_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    target_label = summary.get("target_label", "Target")
    target_key = summary.get("diagnostic_mode", "target")
    lines = [
        f"# {target_label} Behavior Audit",
        "",
        "This is an offline diagnostic. It does not change search, scoring, prompts, or candidate generation.",
        f"Diagnostic mode: `{target_key}`.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Cases",
        "",
    ]
    for case in audit["cases"]:
        score = case["score_breakdown"]
        lines.extend(
            [
                f"### {case['question_id']} — {', '.join(case['behavior_labels'])}",
                "",
                f"- Question: {case['question']}",
                f"- Gold answer: {case['gold_answers']}",
                f"- {target_label} correct: `{case.get('target_correct', case.get('future_aware_v2_correct'))}`",
                f"- {target_label} drift family: `{case.get('target_drift_family', case.get('future_aware_v2_drift_family', 'unknown'))}`",
                f"- Gold survival stage: `{case['gold_survival_stage']}`",
                f"- Baseline top path: `{case['baseline_top_path']}`",
                f"- Current proof-state top path: `{case['current_proof_state_top_path']}`",
                f"- {target_label} top path: `{case.get('target_top_path', case.get('future_aware_v2_top_path', ''))}`",
                "",
                "| Component | Value |",
                "|---|---:|",
            ]
        )
        keys = TWO_SCORE_COMPONENT_KEYS if "final_proof_score" in score else V2_COMPONENT_KEYS
        for key in keys:
            lines.append(f"| `{key}` | {float(score.get(key, 0.0)):.4f} |")
        lines.append("")
    return "\n".join(lines)


def write_gold_survival_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit["summary"]
    target_label = summary.get("target_label", "Target")
    lines = [
        f"# {target_label} Gold Survival Audit",
        "",
        "This is an offline diagnostic. Gold answers are used only after search finishes to locate where useful paths died.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Cases",
        "",
    ]
    for case in audit["cases"]:
        lines.extend(
            [
                f"### {case['question_id']} — {case['gold_survival_stage']}",
                "",
                f"- Question: {case['question']}",
                f"- Gold answer: {case['gold_answers']}",
                f"- Diagnosis: {', '.join(case['diagnosis'])}",
                f"- {target_label} top state: `{case.get('target_top_state', case.get('v2_top_state', ''))}`",
                f"- Best gold state: `{case['best_gold_state']}`",
                f"- Hop 1 rank: `{case['hop_1_rank']}`",
                f"- Hop 2 rank: `{case['hop_2_rank']}`",
                f"- Final rank: `{case['final_rank']}`",
                f"- Gold final score: `{float(case.get('final_proof_score', case.get('final_score', 0.0))):.4f}`",
                "",
                "#### Components Favoring Wrong Top",
                "",
            ]
        )
        favored = case.get("components_favoring_wrong_top", [])
        if favored:
            for item in favored:
                lines.append(f"- `{item['component']}`: top minus gold = `{float(item['top_minus_gold']):.4f}`")
        else:
            lines.append("- _None._")
        lines.extend(["", "#### Components Hurting Gold", ""])
        hurt = case.get("components_hurting_gold", [])
        if hurt:
            for item in hurt:
                lines.append(f"- `{item['component']}`: gold minus top = `{float(item['gold_minus_top']):.4f}`")
        else:
            lines.append("- _None._")
        lines.extend(["", "#### Gold Candidates", ""])
        candidates = case.get("gold_candidate_states", [])
        if candidates:
            for item in candidates[:5]:
                score = item.get("score", item.get("final_proof_score", 0.0))
                lines.append(f"- rank `{item['rank']}` score `{float(score):.4f}`: `{item['path']}`")
        else:
            lines.append("- _No gold-reaching candidate state recorded._")
        lines.append("")
    return "\n".join(lines)


def build_two_score_code_behavior_audit(rows: list[dict[str, Any]], gold_survival_audit: dict[str, Any]) -> dict[str, Any]:
    states = collect_target_audit_states(rows, "two_score_proof_state_beam")
    breakouts = [target_score_breakdown_from_state(state) for state in states]
    positive_values = [item["target_positive_score_sum"] for item in breakouts]
    negative_values = [item["target_negative_penalty_sum"] for item in breakouts]
    ratio_values = [item["target_penalty_to_positive_ratio"] for item in breakouts if item["target_positive_score_sum"] > 0.0]
    role_values = [float(state.get("role_gate", 0.0)) for state in states]
    drift_gate_values = [float(state.get("non_drift_gate", 0.0)) for state in states]
    loop_gate_values = [float(state.get("non_loop_gate", 0.0)) for state in states]
    gate_product_values = [
        float(state.get("role_gate", 0.0))
        * float(state.get("non_drift_gate", 0.0))
        * float(state.get("non_loop_gate", 0.0))
        for state in states
    ]
    penalty_tiny_cases = sorted(
        [
            code_behavior_target_state_case(state, breakdown)
            for state, breakdown in zip(states, breakouts)
            if breakdown["target_positive_score_sum"] > 0.0 and breakdown["target_penalty_to_positive_ratio"] < 0.15
        ],
        key=lambda item: (-item["score_breakdown"]["target_positive_score_sum"], str(item["question_id"])),
    )[:10]
    penalty_strong_cases = sorted(
        [
            code_behavior_target_state_case(state, breakdown)
            for state, breakdown in zip(states, breakouts)
            if breakdown["target_positive_score_sum"] > 0.0 and breakdown["target_penalty_to_positive_ratio"] > 0.85
        ],
        key=lambda item: (-item["score_breakdown"]["target_penalty_to_positive_ratio"], str(item["question_id"])),
    )[:10]
    diversity_cases = [
        case
        for case in gold_survival_audit.get("cases", [])
        if case.get("gold_survival_stage") == "gold_first_hop_removed_by_diversity"
    ]
    return {
        "summary": {
            "diagnostic_mode": "two_score_proof_state_beam",
            "num_questions": len(rows),
            "num_audit_states": len(states),
            "average_positive_score_sum": avg(positive_values),
            "average_negative_penalty_sum": avg(negative_values),
            "average_penalty_to_positive_ratio": avg(ratio_values),
            "penalty_tiny_case_count": sum(1 for value in ratio_values if value < 0.15),
            "penalty_strong_case_count": sum(1 for value in ratio_values if value > 0.85),
            "role_gate_min": min(role_values) if role_values else 0.0,
            "role_gate_max": max(role_values) if role_values else 0.0,
            "non_drift_gate_values_observed": sorted({round(value, 6) for value in drift_gate_values}),
            "non_loop_gate_values_observed": sorted({round(value, 6) for value in loop_gate_values}),
            "max_gate_product_observed": max(gate_product_values) if gate_product_values else 0.0,
            "gates_can_amplify_observed": any(value > 1.0 for value in gate_product_values),
            "gold_removed_by_first_hop_diversity_count": len(diversity_cases),
        },
        "scorer_locations": {
            **scorer_locations(),
            "two_score_proof_state_beam": {
                "file": "rc_mex/run_proof_state_search_smoke.py",
                "search_function": "run_two_score_proof_state_beam",
                "scoring_function": "two_score_fragment_signals",
                "final_answer_ranker": "search_result(answer_score_key='final_proof_score')",
            },
        },
        "exact_formulas": two_score_formula_notes(),
        "order_of_operations": {
            **order_of_operations_notes(),
            "two_score_specific": [
                "Beam retention sorts by cumulative SearchState.score, which is cumulative retention_score.",
                "Final answer ranking calls search_result(..., answer_score_key='final_proof_score'), so grouped answers are ranked by final proof score.",
                "Future satisfiability appears in retention as future_retention_bonus but is not a direct positive term in final_proof_score.",
                "No hard first-hop diversity pruning is applied in two_score_proof_state_beam.",
            ],
        },
        "gate_behavior": two_score_gate_behavior_notes(states),
        "penalty_scale": {
            "average_positive_score_sum": avg(positive_values),
            "average_negative_penalty_sum": avg(negative_values),
            "average_penalty_to_positive_ratio": avg(ratio_values),
            "cases_where_penalties_are_tiny": penalty_tiny_cases,
            "cases_where_penalties_are_strong": penalty_strong_cases,
            "interpretation": (
                "These values are computed from two-score audit candidate states. Because soft signals are merged across hops, "
                "the reported state fields are accumulated values, but per-hop gate values are clamped before multiplication."
            ),
        },
        "first_hop_diversity": {
            "implementation": "No hard first-hop diversity pruning in two_score_proof_state_beam.",
            "relation_family_definition": "relation_family_from_text is used only for soft diversity/drift diagnostics.",
            "family_limit": None,
            "when_applied": "Not applied as pruning.",
            "candidate_order": "Candidates are sorted by retention score and normal beam_width is kept.",
            "can_remove_gold_before_hop2": False,
            "gold_removed_examples": [],
        },
        "mismatch_table": two_score_mismatch_table(),
    }


def collect_target_audit_states(rows: list[dict[str, Any]], target_key: str) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row in rows:
        target = row.get(target_key, {})
        for hop_trace in target.get("audit_trace", []):
            hop = hop_trace.get("hop")
            for state in hop_trace.get("all_candidate_states", []):
                item = dict(state)
                item["question_id"] = row.get("question_id", "")
                item["question"] = row.get("question", "")
                item["hop"] = hop
                states.append(item)
    return states


def code_behavior_target_state_case(state: dict[str, Any], breakdown: dict[str, float]) -> dict[str, Any]:
    return {
        "question_id": state.get("question_id", ""),
        "question": state.get("question", ""),
        "hop": state.get("hop"),
        "rank": state.get("rank"),
        "path": state_readable(state),
        "target_label": state.get("target_label", ""),
        "score_breakdown": {
            key: breakdown.get(key, 0.0)
            for key in [
                "current_relevance",
                "future_retention_bonus",
                "soft_progress",
                "soft_type_signal",
                "soft_diversity_signal",
                "proof_role_coverage",
                "answer_type_compatibility",
                "useful_convergence",
                "soft_loop_penalty",
                "soft_drift_penalty",
                "noisy_branch_penalty",
                "unresolved_need_penalty",
                "hard_loop_penalty",
                "semantic_level_drift_penalty",
                "surface_convergence_penalty",
                "redundancy_penalty",
                "target_positive_score_sum",
                "target_negative_penalty_sum",
                "target_penalty_to_positive_ratio",
                "retention_score",
                "final_proof_score",
            ]
        },
    }


def two_score_formula_notes() -> dict[str, dict[str, Any]]:
    formulas = scorer_formula_notes()
    formulas["two_score_proof_state_beam"] = {
        "retention_current_relevance": "current_relevance = max(step['relation_label_score'] for grouped steps)",
        "retention_raw_future": "raw_future = future_satisfiability(...)",
        "retention_gates": "role_gate is clamped to [0,1]; non_drift_gate = 0.45 if drift else 1.0; non_loop_gate = 0.35 if loop else 1.0",
        "future_retention_bonus": "min(raw_future * role_gate * non_drift_gate * non_loop_gate, TWO_SCORE_CONSTANTS['future_cap'])",
        "retention_score_delta": "0.50*current_relevance + future_retention_bonus + soft_progress + soft_type_signal + soft_diversity_signal + soft_loop_penalty + soft_drift_penalty + noisy_branch_penalty",
        "proof_role_coverage": "0.50*relation_role_coverage + 0.25*covered_need_score + 0.15*answer_type_score + 0.10*current step label score, capped at 1.0",
        "unresolved_need_penalty": "-0.25 * unresolved_need_score at hop 2",
        "final_proof_score_delta": "0.55*proof_role_coverage + 0.20*answer_type_compatibility + 0.25*current_relevance + useful_convergence + unresolved_need_penalty + hard_loop_penalty + semantic_level_drift_penalty + surface_convergence_penalty + redundancy_penalty",
        "final_answer_score": "search_result sums final_proof_score over retained paths to each answer; best final proof path is tie-breaker",
    }
    return formulas


def two_score_gate_behavior_notes(states: list[dict[str, Any]]) -> dict[str, Any]:
    role_values = sorted({round(float(state.get("role_gate", 0.0)), 6) for state in states})
    drift_gate_values = sorted({round(float(state.get("non_drift_gate", 0.0)), 6) for state in states})
    loop_gate_values = sorted({round(float(state.get("non_loop_gate", 0.0)), 6) for state in states})
    gate_products = [
        float(state.get("role_gate", 0.0))
        * float(state.get("non_drift_gate", 0.0))
        * float(state.get("non_loop_gate", 0.0))
        for state in states
    ]
    return {
        "role_gate": {
            "code_logic": "role_semantic_gate(...) clamped with min(1.0, max(0.0, value)) before use.",
            "possible_values_from_code": [0.35, 0.40, 0.45, 0.65, 0.75, 0.85, 1.0],
            "observed_accumulated_values": role_values,
        },
        "non_drift_gate": {
            "code_logic": "0.45 if semantic drift penalty is active, else 1.0.",
            "possible_values_from_code": [0.45, 1.0],
            "observed_accumulated_values": drift_gate_values,
        },
        "non_loop_gate": {
            "code_logic": "0.35 if loop signal is active, else 1.0.",
            "possible_values_from_code": [0.35, 1.0],
            "observed_accumulated_values": loop_gate_values,
        },
        "combined_gate": {
            "formula": "future_retention_bonus = min(raw_future * role_gate * non_drift_gate * non_loop_gate, 0.25)",
            "max_observed_accumulated_gate_product": max(gate_products) if gate_products else 0.0,
            "can_amplify_with_current_per_hop_code": False,
            "note": "Accumulated debug fields can exceed 1 across two hops, but per-hop gates are clamped before future_retention_bonus is computed.",
        },
    }


def two_score_mismatch_table() -> list[dict[str, str]]:
    return [
        {
            "intended_behavior": "Future satisfiability should keep branches alive but not prove answers.",
            "actual_code_behavior": "Future appears in retention_score via future_retention_bonus; final_proof_score has no direct future term.",
            "possible_issue": "If retention keeps too many surface-future branches, final proof score must still reject them using role coverage and penalties.",
        },
        {
            "intended_behavior": "Gates should only reduce future, not amplify.",
            "actual_code_behavior": "Per-hop gates are clamped to [0,1] and cap is applied after gating.",
            "possible_issue": "Accumulated debug values can look >1, so inspect future_retention_bonus rather than summed gate fields.",
        },
        {
            "intended_behavior": "Valid bidirectional evidence should not be punished automatically.",
            "actual_code_behavior": "Hard loop penalty only fires for returning to start, repeated entity cycles with low role coverage, or same-relation out-and-back with low role coverage.",
            "possible_issue": "Role coverage is still lexical, so valid evidence with weak lexical coverage can be falsely penalized.",
        },
        {
            "intended_behavior": "Diversity should preserve alternatives without killing gold.",
            "actual_code_behavior": "No hard first-hop diversity pruning is applied; only a small soft diversity bonus is used.",
            "possible_issue": "Without reserved diversity, high-scoring same-family states can still fill the beam.",
        },
        {
            "intended_behavior": "Final proof score should favor completed proof roles.",
            "actual_code_behavior": "Role coverage is approximated with relation-token coverage, entity/type-token coverage, and answer type.",
            "possible_issue": "This is still shallow and can confuse sibling answers or relation labels with similar surface forms.",
        },
    ]


def build_code_behavior_audit(rows: list[dict[str, Any]], gold_survival_audit: dict[str, Any]) -> dict[str, Any]:
    v2_states = collect_v2_audit_states(rows)
    breakouts = [full_v2_score_breakdown_from_state(state) for state in v2_states]
    positive_values = [item["positive_score_sum"] for item in breakouts]
    negative_values = [item["negative_penalty_sum"] for item in breakouts]
    ratio_values = [item["penalty_to_positive_ratio"] for item in breakouts if item["positive_score_sum"] > 0.0]
    role_values = [float(state.get("role_gate", 0.0)) for state in v2_states]
    drift_gate_values = [float(state.get("non_drift_gate", 0.0)) for state in v2_states]
    loop_gate_values = [float(state.get("non_loop_gate", 0.0)) for state in v2_states]
    gate_product_values = [
        float(state.get("role_gate", 0.0))
        * float(state.get("non_drift_gate", 0.0))
        * float(state.get("non_loop_gate", 0.0))
        for state in v2_states
    ]
    diversity_cases = [
        case
        for case in gold_survival_audit.get("cases", [])
        if case.get("gold_survival_stage") == "gold_first_hop_removed_by_diversity"
    ]
    penalty_tiny_cases = sorted(
        [
            code_behavior_state_case(state, breakdown)
            for state, breakdown in zip(v2_states, breakouts)
            if breakdown["positive_score_sum"] > 0.0 and breakdown["penalty_to_positive_ratio"] < 0.15
        ],
        key=lambda item: (-item["score_breakdown"]["positive_score_sum"], str(item["question_id"])),
    )[:10]
    penalty_strong_cases = sorted(
        [
            code_behavior_state_case(state, breakdown)
            for state, breakdown in zip(v2_states, breakouts)
            if breakdown["positive_score_sum"] > 0.0 and breakdown["penalty_to_positive_ratio"] > 0.85
        ],
        key=lambda item: (-item["score_breakdown"]["penalty_to_positive_ratio"], str(item["question_id"])),
    )[:10]
    return {
        "summary": {
            "num_questions": len(rows),
            "num_future_aware_v2_audit_states": len(v2_states),
            "average_positive_score_sum": avg(positive_values),
            "average_negative_penalty_sum": avg(negative_values),
            "average_penalty_to_positive_ratio": avg(ratio_values),
            "penalty_tiny_case_count": sum(1 for value in ratio_values if value < 0.15),
            "penalty_strong_case_count": sum(1 for value in ratio_values if value > 0.85),
            "role_gate_min": min(role_values) if role_values else 0.0,
            "role_gate_max": max(role_values) if role_values else 0.0,
            "non_drift_gate_values_observed": sorted({round(value, 6) for value in drift_gate_values}),
            "non_loop_gate_values_observed": sorted({round(value, 6) for value in loop_gate_values}),
            "max_gate_product_observed": max(gate_product_values) if gate_product_values else 0.0,
            "gates_can_amplify_observed": any(value > 1.0 for value in gate_product_values),
            "gold_removed_by_first_hop_diversity_count": len(diversity_cases),
        },
        "scorer_locations": scorer_locations(),
        "exact_formulas": scorer_formula_notes(),
        "order_of_operations": order_of_operations_notes(),
        "gate_behavior": gate_behavior_notes(v2_states),
        "penalty_scale": {
            "average_positive_score_sum": avg(positive_values),
            "average_negative_penalty_sum": avg(negative_values),
            "average_penalty_to_positive_ratio": avg(ratio_values),
            "cases_where_penalties_are_tiny": penalty_tiny_cases,
            "cases_where_penalties_are_strong": penalty_strong_cases,
            "interpretation": (
                "These values are computed from v2 audit candidate states. Because soft signals are merged across hops, "
                "the numbers are state-level accumulated components, not isolated per-edge terms."
            ),
        },
        "first_hop_diversity": {
            "implementation": "diversify_first_hop_states(next_states, beam_width, family_limit=FUTURE_AWARE_V2_CONSTANTS['first_hop_relation_family_limit'])",
            "relation_family_definition": "first_hop_relation_family(state) maps the first hop relation_id through relation_family_from_text().",
            "family_limit": FUTURE_AWARE_V2_CONSTANTS["first_hop_relation_family_limit"],
            "when_applied": "Only in future_aware_v2_proof_state_beam, after all hop-1 candidates are scored and sorted, before hop-1 beam retention and before hop 2 generation.",
            "candidate_order": "The first candidate per relation family is kept in descending score order; extra same-family candidates are deferred and only used if the beam is not full.",
            "can_remove_gold_before_hop2": True,
            "gold_removed_examples": [
                {
                    "question_id": case.get("question_id"),
                    "question": case.get("question"),
                    "gold_answer": case.get("gold_answers"),
                    "stage": case.get("gold_survival_stage"),
                    "diagnosis": case.get("diagnosis", []),
                    "v2_top_path": case.get("v2_top_state"),
                    "best_gold_path": case.get("best_gold_state"),
                }
                for case in diversity_cases[:10]
            ],
        },
        "mismatch_table": code_behavior_mismatch_table(),
    }


def collect_v2_audit_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for row in rows:
        question_id = row.get("question_id", "")
        question = row.get("question", "")
        v2 = row.get("future_aware_v2_proof_state_beam", {})
        for hop_trace in v2.get("audit_trace", []):
            hop = hop_trace.get("hop")
            for state in hop_trace.get("all_candidate_states", []):
                item = dict(state)
                item["question_id"] = question_id
                item["question"] = question
                item["hop"] = hop
                states.append(item)
    return states


def code_behavior_state_case(state: dict[str, Any], breakdown: dict[str, float]) -> dict[str, Any]:
    return {
        "question_id": state.get("question_id", ""),
        "question": state.get("question", ""),
        "hop": state.get("hop"),
        "rank": state.get("rank"),
        "path": state_readable(state),
        "target_label": state.get("target_label", ""),
        "score_breakdown": {
            key: breakdown.get(key, 0.0)
            for key in [
                "current_relevance",
                "gated_future_bonus",
                "useful_convergence",
                "type_compatibility",
                "progress",
                "surface_convergence_penalty",
                "loop_penalty",
                "drift_penalty",
                "redundancy_penalty",
                "noisy_branch_penalty",
                "positive_score_sum",
                "negative_penalty_sum",
                "penalty_to_positive_ratio",
                "final_score",
            ]
        },
    }


def scorer_locations() -> dict[str, dict[str, str]]:
    path = "rc_mex/run_proof_state_search_smoke.py"
    return {
        "baseline_path_beam": {
            "file": path,
            "search_function": "run_baseline_path_beam",
            "relation_ranker": "rank_relations",
            "final_answer_ranker": "search_result",
        },
        "soft_proof_state_beam": {
            "file": path,
            "search_function": "run_soft_proof_state_beam",
            "scoring_function": "soft_fragment_signals",
            "relation_ranker": "rank_relations",
            "final_answer_ranker": "search_result",
        },
        "future_aware_proof_state_beam": {
            "file": path,
            "search_function": "run_future_aware_proof_state_beam",
            "scoring_function": "future_aware_fragment_signals",
            "future_signal": "future_satisfiability",
            "final_answer_ranker": "search_result",
        },
        "future_aware_v2_proof_state_beam": {
            "file": path,
            "search_function": "run_future_aware_v2_proof_state_beam",
            "scoring_function": "future_aware_v2_fragment_signals",
            "future_signal": "future_satisfiability",
            "first_hop_diversity": "diversify_first_hop_states",
            "final_answer_ranker": "search_result",
        },
    }


def scorer_formula_notes() -> dict[str, dict[str, Any]]:
    return {
        "shared_relation_ranking": {
            "formula": "label_score = char_ngram_similarity(question, relation_label_with_direction) + 0.03 * min(1.0, relation.frequency / 5.0)",
            "relation_label_with_direction": "f\"{predicate.replace('_', ' ')} {direction}\"",
            "sort": "descending label_score, then relation_id, then direction",
        },
        "baseline_path_beam": {
            "state_delta": "candidate['label_score']",
            "state_score": "sum(label_score over selected hops)",
            "final_answer_score": "sum(state.score for all retained final states reaching that answer)",
            "best_path_score_tiebreak": "max(state.score for retained final states reaching that answer)",
            "unused_components": [
                "current_relevance",
                "raw_future_bonus",
                "future_bonus_capped",
                "role_gate",
                "non_drift_gate",
                "non_loop_gate",
                "gated_future_bonus",
                "convergence/useful_convergence",
                "type_compatibility",
                "progress",
                "surface_convergence_penalty",
                "loop_penalty",
                "drift_penalty",
                "redundancy_penalty",
                "noisy_branch_penalty",
            ],
        },
        "soft_proof_state_beam": {
            "best_label_similarity": "max(step['relation_label_score'] for step in grouped steps to same target)",
            "avg_label_similarity": "mean(step['relation_label_score'] for step in grouped steps to same target)",
            "convergence_bonus": "0.18 * max(0, len(steps) - 1)",
            "type_compatibility": "type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0",
            "progress": "plausible_progress(question, graph, target_id, steps, hop)",
            "uncertainty_floor": "0.03 if best_label < 0.08 and noisy_branch_penalty == 0.0 and redundancy_penalty == 0.0 else 0.0",
            "noisy_branch_penalty": "-0.18 if branch_size >= noisy_branch_threshold else 0.0",
            "redundancy_penalty": "-0.10 if repeats_relation_pattern(state, steps) else 0.0",
            "total_delta": "0.65*best_label + 0.20*avg_label + convergence_bonus + 0.16*type_compatibility + progress + uncertainty_floor + noisy_branch_penalty + redundancy_penalty",
            "state_score": "previous state.score + total_delta",
            "final_answer_score": "sum(state.score for retained final states reaching that answer)",
        },
        "future_aware_proof_state_beam": {
            "current_relevance": "max(step['relation_label_score'] for step in grouped steps)",
            "raw_future_bonus": "future_satisfiability(...), stored as future_satisfiability",
            "future_bonus_capped": "not used",
            "role_gate": "not used",
            "non_drift_gate": "not used",
            "non_loop_gate": "not used",
            "gated_future_bonus": "not used; raw future is weighted directly",
            "useful_convergence": "useful_convergence_bonus(...): 0.16 + 0.04*min(2, len(steps)-2) if len(steps)>=2 and future>0.12 or type_score>0 or relation labels differ; else 0",
            "type_compatibility": "type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0",
            "progress": "plausible_progress(question, graph, target_id, steps, hop)",
            "surface_convergence_penalty": "surface_convergence_penalty(...) can subtract 0.16, 0.18, 0.08, 0.08 depending on surface/no-future/loop/type mismatch rules",
            "loop_penalty": "not separately used; inverse loop is folded into surface_convergence_penalty",
            "drift_penalty": "drift_penalty(...): -0.14 for hop-2 answer-type mismatch with low future and low target match; -0.10 for generic drift with low future and low target match; else 0",
            "redundancy_penalty": "-0.12 if repeats_relation_pattern(state, steps) else 0.0",
            "noisy_branch_penalty": "-0.20 if branch_size >= noisy_branch_threshold else 0.0",
            "total_delta": "0.58*current_relevance + 0.34*future_satisfiability + useful_convergence + 0.14*type_compatibility + progress + surface_convergence_penalty + redundancy_penalty + drift_penalty + noisy_branch_penalty",
            "state_score": "previous state.score + total_delta",
            "final_answer_score": "sum(state.score for retained final states reaching that answer)",
        },
        "future_aware_v2_proof_state_beam": {
            "current_relevance": "max(step['relation_label_score'] for grouped steps)",
            "raw_future_bonus": "future_satisfiability(...)",
            "future_bonus_capped": "min(raw_future_bonus, FUTURE_AWARE_V2_CONSTANTS['future_cap'])",
            "role_gate": "role_semantic_gate(...), currently returns one of 0.35/0.40/0.45/0.65/0.75/0.85/1.0 depending on question/target/future labels",
            "non_drift_gate": "0.35 if stronger_drift_penalty(...) < 0.0 else 1.0",
            "non_loop_gate": "0.20 if strong_loop_penalty(...) < 0.0 else 1.0",
            "gated_future_bonus": "future_bonus_capped * role_gate * non_drift_gate * non_loop_gate",
            "useful_convergence": "useful_convergence_bonus_v2(...): 0.14 if len(steps)>=2 and (gated_future_bonus>0.08 or type_score>0 or relation_diverse and not inverse duplicate); else 0",
            "type_compatibility": "type_compatibility(graph, target_id, answer_type) if hop == 2 else 0.0",
            "progress": "plausible_progress(question, graph, target_id, steps, hop)",
            "surface_convergence_penalty": "surface_convergence_penalty_v2(...): -0.25 for multi-step convergence if useful_convergence<=0 or gated_future_bonus<0.04, or if all converging steps use same relation name; else 0",
            "loop_penalty": "strong_loop_penalty(...): adds -0.45 for inverse/repeated-entity loop, -0.45 for returning to start at hop 2, and -0.45 for same-relation out-and-back",
            "drift_penalty": "stronger_drift_penalty(...): -0.30 for detected geography/film/award/org drift not needed by question, or hop-2 type mismatch with raw_future<0.08; else 0",
            "redundancy_penalty": "-0.18 if repeats_relation_pattern(state, steps) else 0.0",
            "noisy_branch_penalty": "-0.30 if branch_size >= noisy_branch_threshold else 0.0",
            "total_delta": "0.55*current_relevance + 1.0*gated_future_bonus + useful_convergence + 0.16*type_compatibility + progress + surface_convergence_penalty + loop_penalty + drift_penalty + redundancy_penalty + noisy_branch_penalty",
            "state_score": "previous state.score + total_delta",
            "final_answer_score": "sum(state.score for retained final states reaching that answer)",
        },
        "future_satisfiability": {
            "formula": "min(1.0, 0.70*best_remaining + 0.30*best_question)",
            "best_remaining": "max token overlap between remaining question terms and each future relation label from target entity",
            "best_question": "max char_ngram_similarity(question, each future relation label from target entity)",
        },
    }


def order_of_operations_notes() -> dict[str, Any]:
    return {
        "candidate_generation": [
            "Initialize states from the gold topic entity/entities selected for this controlled smoke test.",
            "For each hop in [1, 2], for each retained state, iterate source entities in the current frontier.",
            "Call graph.candidate_relations([source_id], cap=relation_cap, sample_entities=sample_entities).",
            "Rank local relation+direction candidates with rank_relations(question, frontier).",
            "Expand only ranked[:top_k].",
            "For each selected relation+direction, call relation_targets(...), capped by max_branch_entities.",
        ],
        "score_computation": [
            "Baseline scores every target by adding the relation label_score.",
            "Proof-state modes group fragments by target_id, then compute one score delta per target evidence state.",
            "Score deltas are added to the previous state.score.",
            "soft_signals are merged by summing previous and current signal values.",
        ],
        "first_hop_diversity": [
            "Only future_aware_v2 applies first-hop diversity.",
            "It happens after all hop-1 candidates are scored and sorted.",
            "It happens before hop-1 beam retention and therefore before hop 2 candidate generation.",
        ],
        "pruning": [
            "Per-source relation pruning happens at ranked[:top_k].",
            "Per-relation target pruning happens in relation_targets(...)[ : max_branch_entities ].",
            "Beam pruning happens after scoring each hop.",
            "For v2 only, diversity pruning can remove same-family first-hop candidates before hop 2.",
        ],
        "gold_like_path_risk": [
            "A gold-like path can be removed before hop 2 if its first relation is outside top_k, its target is outside max_branch_entities, its state score falls outside beam_width, or v2 diversity defers it behind an already-full beam.",
            "A gold-like path can be generated at hop 2 but still ranked below another final answer because final answer ranking groups retained paths by answer and sums scores.",
        ],
        "final_answer_ranking": [
            "search_result groups retained final states by answer entity.",
            "candidate['score'] is the sum of state.score over retained paths to that answer.",
            "candidate['best_path_score'] is max state.score for that answer.",
            "Final candidates sort by -score, then -best_path_score, then answer label/id.",
            "So final answer ranking uses the same state scores as beam retention, but aggregates them by answer; it is not identical to selecting the single top final state.",
        ],
    }


def gate_behavior_notes(v2_states: list[dict[str, Any]]) -> dict[str, Any]:
    role_values = sorted({round(float(state.get("role_gate", 0.0)), 6) for state in v2_states})
    drift_gate_values = sorted({round(float(state.get("non_drift_gate", 0.0)), 6) for state in v2_states})
    loop_gate_values = sorted({round(float(state.get("non_loop_gate", 0.0)), 6) for state in v2_states})
    gate_products = [
        float(state.get("role_gate", 0.0))
        * float(state.get("non_drift_gate", 0.0))
        * float(state.get("non_loop_gate", 0.0))
        for state in v2_states
    ]
    return {
        "role_gate": {
            "code_logic": [
                "If question mentions legal form, return 1.0 if future relation labels include legal form else 0.35.",
                "If question asks occupation/job/position, return 1.0 for occupational future labels, 0.85 for person/player/athlete target text, else 0.35.",
                "If question asks county/city/state/province, return 1.0 for matching target text, 0.35 for country when question does not ask country, else 0.65.",
                "If answer_type == person, return 1.0 for human/person target text, 0.85 for person-like future labels, 0.40 for generic hubs.",
                "If answer_type == organization, return 1.0 for organization/company/institution/team target text, 0.85 for organization-like future labels.",
                "If target is a generic hub and question does not need generic geography/org family, return 0.45.",
                "Default return 0.75.",
            ],
            "possible_values_from_code": [0.35, 0.40, 0.45, 0.65, 0.75, 0.85, 1.0],
            "observed_values": role_values,
        },
        "non_drift_gate": {
            "code_logic": "0.35 if stronger_drift_penalty(...) is negative, else 1.0.",
            "possible_values_from_code": [0.35, 1.0],
            "observed_values": drift_gate_values,
        },
        "non_loop_gate": {
            "code_logic": "0.20 if strong_loop_penalty(...) is negative, else 1.0.",
            "possible_values_from_code": [0.20, 1.0],
            "observed_values": loop_gate_values,
        },
        "combined_gate": {
            "formula": "gated_future_bonus = min(raw_future_bonus, 0.25) * role_gate * non_drift_gate * non_loop_gate",
            "max_observed_gate_product": max(gate_products) if gate_products else 0.0,
            "can_amplify_with_current_code": False,
            "observed_amplification_above_cap": any(value > 1.0 for value in gate_products),
            "note": "Current gate values are <= 1, so they do not amplify above the capped future bonus. If any gate value is changed above 1 later, this formula would amplify rather than merely gate.",
        },
    }


def code_behavior_mismatch_table() -> list[dict[str, str]]:
    return [
        {
            "intended_behavior": "Future bonus should be gated.",
            "actual_code_behavior": "Future bonus is capped at 0.25 and multiplied by role_gate, non_drift_gate, and non_loop_gate. Current gate values are all <= 1.",
            "possible_issue": "No observed amplification today, but the formula would amplify if any gate value were later set above 1.",
        },
        {
            "intended_behavior": "Loop penalty should kill inverse loops.",
            "actual_code_behavior": "Loop penalties are additive negative terms. A loop can still survive if current relevance, gated future, convergence, type, or progress outweigh the penalty.",
            "possible_issue": "The penalty suppresses rather than forbids loops, so high positive lexical/future scores can still rank loop states highly.",
        },
        {
            "intended_behavior": "Convergence should be useful, not just surface-level.",
            "actual_code_behavior": "v2 gives useful_convergence only under gated future/type/diverse-relation conditions, then applies -0.25 if convergence lacks those signals or uses one relation name.",
            "possible_issue": "Valid bidirectional or same-relation evidence can be penalized when it is semantically valid but looks surface-redundant.",
        },
        {
            "intended_behavior": "First-hop diversity should preserve alternatives.",
            "actual_code_behavior": "v2 keeps at most one state per coarse relation family in score order until beam is full, then fills from deferred states only if space remains.",
            "possible_issue": "Multiple valid candidates in the same broad family can be removed before hop 2.",
        },
        {
            "intended_behavior": "Penalties should suppress bad drift.",
            "actual_code_behavior": "Drift detection is based on coarse string families and question keyword tests.",
            "possible_issue": "Bad generic branches can survive when not recognized by family rules; valid branches can be penalized when keywords misfire.",
        },
        {
            "intended_behavior": "Final answer should reflect the best proof.",
            "actual_code_behavior": "Final answer score sums all retained path scores to the same answer; best path score is only a tie-breaker.",
            "possible_issue": "An answer with several mediocre paths can outrank an answer with one better proof state.",
        },
        {
            "intended_behavior": "Future satisfiability should estimate useful remaining constraints.",
            "actual_code_behavior": "It uses lexical overlap between remaining question terms and one-hop future relation labels from the current target.",
            "possible_issue": "Surface future matches can reward wrong branches, while semantically correct but differently worded future relations may get low bonus.",
        },
    ]


def write_code_behavior_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    diagnostic_mode = summary.get("diagnostic_mode", "future_aware_v2_proof_state_beam")
    lines = [
        "# Code-Behavior Alignment Audit",
        "",
        "This audit describes the current scoring implementation as coded. It does not propose or apply a scorer change.",
        f"Diagnostic mode: `{diagnostic_mode}`.",
        "",
        "## Summary",
        "",
        f"- Questions: `{summary.get('num_questions', 0)}`",
        f"- Audit states: `{summary.get('num_audit_states', summary.get('num_future_aware_v2_audit_states', 0))}`",
        f"- Average positive score sum: `{float(summary.get('average_positive_score_sum', 0.0)):.4f}`",
        f"- Average negative penalty sum: `{float(summary.get('average_negative_penalty_sum', 0.0)):.4f}`",
        f"- Average penalty/positive ratio: `{float(summary.get('average_penalty_to_positive_ratio', 0.0)):.4f}`",
        f"- Gates amplify above cap in observed states: `{summary.get('gates_can_amplify_observed', False)}`",
        f"- Max observed gate product: `{float(summary.get('max_gate_product_observed', 0.0)):.4f}`",
        f"- Gold removed by first-hop diversity: `{summary.get('gold_removed_by_first_hop_diversity_count', 0)}`",
        "",
        "## Scorer Locations",
        "",
        "| Mode | File | Functions |",
        "|---|---|---|",
    ]
    for mode, location in audit.get("scorer_locations", {}).items():
        functions = ", ".join(f"`{value}`" for key, value in location.items() if key != "file")
        lines.append(f"| `{mode}` | `{location.get('file', '')}` | {functions} |")
    lines.extend(["", "## Exact Formulas From Code", ""])
    formulas = audit.get("exact_formulas", {})
    for mode in [
        "shared_relation_ranking",
        "baseline_path_beam",
        "soft_proof_state_beam",
        "future_aware_proof_state_beam",
        "future_aware_v2_proof_state_beam",
        "future_satisfiability",
    ]:
        lines.extend([f"### {mode}", ""])
        for key, value in formulas.get(mode, {}).items():
            if isinstance(value, list):
                lines.append(f"- `{key}`: {', '.join(f'`{item}`' for item in value)}")
            else:
                lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.extend(["## Order Of Operations", ""])
    for section, items in audit.get("order_of_operations", {}).items():
        lines.append(f"### {section}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(["## Gate Behavior", ""])
    gates = audit.get("gate_behavior", {})
    for gate_name in ["role_gate", "non_drift_gate", "non_loop_gate"]:
        gate = gates.get(gate_name, {})
        lines.extend([f"### {gate_name}", ""])
        logic = gate.get("code_logic", [])
        if isinstance(logic, list):
            for item in logic:
                lines.append(f"- {item}")
        else:
            lines.append(f"- {logic}")
        lines.append(f"- Possible values from code: `{gate.get('possible_values_from_code', [])}`")
        lines.append(f"- Observed values: `{gate.get('observed_values', [])}`")
        lines.append("")
    combined = gates.get("combined_gate", {})
    lines.extend([
        "### combined gate",
        "",
        f"- Formula: `{combined.get('formula', '')}`",
        f"- Max observed gate product: `{float(combined.get('max_observed_gate_product', 0.0)):.4f}`",
        f"- Can amplify with current code: `{combined.get('can_amplify_with_current_code', False)}`",
        f"- Observed amplification above cap: `{combined.get('observed_amplification_above_cap', False)}`",
        f"- Note: {combined.get('note', '')}",
        "",
        "## Penalty Scale",
        "",
    ])
    penalty = audit.get("penalty_scale", {})
    lines.extend([
        f"- Average positive score sum: `{float(penalty.get('average_positive_score_sum', 0.0)):.4f}`",
        f"- Average negative penalty sum: `{float(penalty.get('average_negative_penalty_sum', 0.0)):.4f}`",
        f"- Average penalty/positive ratio: `{float(penalty.get('average_penalty_to_positive_ratio', 0.0)):.4f}`",
        f"- Interpretation: {penalty.get('interpretation', '')}",
        "",
        "### Cases Where Penalties Are Tiny",
        "",
    ])
    for case in penalty.get("cases_where_penalties_are_tiny", [])[:10]:
        score = case.get("score_breakdown", {})
        ratio = score.get("penalty_to_positive_ratio", score.get("target_penalty_to_positive_ratio", 0.0))
        positive = score.get("positive_score_sum", score.get("target_positive_score_sum", 0.0))
        negative = score.get("negative_penalty_sum", score.get("target_negative_penalty_sum", 0.0))
        lines.append(
            f"- `{case.get('question_id')}` hop `{case.get('hop')}` rank `{case.get('rank')}` "
            f"ratio `{float(ratio):.4f}` "
            f"positive `{float(positive):.4f}` negative `{float(negative):.4f}`: "
            f"`{case.get('path', '')}`"
        )
    lines.extend(["", "### Cases Where Penalties Are Strong", ""])
    for case in penalty.get("cases_where_penalties_are_strong", [])[:10]:
        score = case.get("score_breakdown", {})
        ratio = score.get("penalty_to_positive_ratio", score.get("target_penalty_to_positive_ratio", 0.0))
        positive = score.get("positive_score_sum", score.get("target_positive_score_sum", 0.0))
        negative = score.get("negative_penalty_sum", score.get("target_negative_penalty_sum", 0.0))
        lines.append(
            f"- `{case.get('question_id')}` hop `{case.get('hop')}` rank `{case.get('rank')}` "
            f"ratio `{float(ratio):.4f}` "
            f"positive `{float(positive):.4f}` negative `{float(negative):.4f}`: "
            f"`{case.get('path', '')}`"
        )
    diversity = audit.get("first_hop_diversity", {})
    lines.extend([
        "",
        "## First-Hop Diversity",
        "",
        f"- Implementation: `{diversity.get('implementation', '')}`",
        f"- Relation family definition: {diversity.get('relation_family_definition', '')}",
        f"- Limit value: `{diversity.get('family_limit', '')}`",
        f"- When applied: {diversity.get('when_applied', '')}",
        f"- Candidate order: {diversity.get('candidate_order', '')}",
        f"- Can remove gold before hop 2: `{diversity.get('can_remove_gold_before_hop2', False)}`",
        "",
        "### Possible Gold/Useful Candidate Removals",
        "",
    ])
    for example in diversity.get("gold_removed_examples", [])[:10]:
        lines.append(
            f"- `{example.get('question_id')}` stage `{example.get('stage')}` diagnosis `{example.get('diagnosis', [])}`: "
            f"v2 top `{example.get('v2_top_path', '')}` vs gold `{example.get('best_gold_path', '')}`"
        )
    lines.extend([
        "",
        "## Mismatch Table",
        "",
        "| Intended behavior | Actual code behavior | Possible issue |",
        "|---|---|---|",
    ])
    for row in audit.get("mismatch_table", []):
        lines.append(
            f"| {row.get('intended_behavior', '')} | {row.get('actual_code_behavior', '')} | {row.get('possible_issue', '')} |"
        )
    return "\n".join(lines) + "\n"


def select_overlap_examples(cases: list[dict[str, Any]], flag: str, limit: int = 5) -> list[dict[str, Any]]:
    return [case for case in cases if case.get(flag)][:limit]


def write_error_overlap_markdown(error_overlap: dict[str, Any]) -> str:
    summary = error_overlap["summary"]
    diagnostic_mode = summary.get("diagnostic_future_mode", "diagnostic_mode")
    lines = [
        "# Error Overlap Diagnostic",
        "",
        f"This diagnostic compares `baseline_path_beam`, current `soft_proof_state_beam`, and `{diagnostic_mode}` on the same selected questions.",
        "",
        "It does not change search, scoring, prompts, or candidate generation.",
        "",
        "Diagnostic mode constants:",
        "",
        "```json",
        json.dumps(summary.get("diagnostic_constants", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Summary Counts",
        "",
        "| Diagnostic | Count |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, dict) or not isinstance(value, (bool, int, float)):
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
            f"**{diagnostic_mode}**",
            "",
            "```json",
            json.dumps(summary.get("drift_family_counts_diagnostic_mode", {}), indent=2, sort_keys=True),
            "```",
            "",
            f"## {diagnostic_mode} Repeats Baseline Mistakes",
            "",
            *overlap_example_lines(error_overlap["examples"]["future_repeats_baseline_mistakes"]),
            f"## {diagnostic_mode} Avoids Baseline Mistakes",
            "",
            *overlap_example_lines(error_overlap["examples"]["future_avoids_baseline_mistakes"]),
            f"## {diagnostic_mode} Loses To Current Proof-State",
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
                f"- Future-aware v2 top: `{case['future_aware_top_answer']}`",
                f"- Future-aware v2 path: `{case['future_aware_top_path']}`",
                f"- Same top as baseline: `{case['same_top_answer_as_baseline']}`",
                f"- Same first hop as baseline: `{case['same_first_hop_relation_as_baseline']}`",
                f"- Same second hop as baseline: `{case['same_second_hop_relation_as_baseline']}`",
                f"- Baseline drift family: `{case['baseline_drift_family']}`",
                f"- Future-aware v2 drift family: `{case['future_aware_drift_family']}`",
                "- Future-aware v2 score breakdown:",
                f"  - current_relevance: `{score['current_relevance']:.4f}`",
                f"  - raw_future_bonus: `{score['raw_future_bonus']:.4f}`",
                f"  - future_bonus_capped: `{score['future_bonus_capped']:.4f}`",
                f"  - role_gate: `{score['role_gate']:.4f}`",
                f"  - non_drift_gate: `{score['non_drift_gate']:.4f}`",
                f"  - non_loop_gate: `{score['non_loop_gate']:.4f}`",
                f"  - gated_future_bonus: `{score['gated_future_bonus']:.4f}`",
                f"  - future_bonus_total: `{score['future_bonus_total']:.4f}`",
                f"  - useful_convergence: `{score['useful_convergence']:.4f}`",
                f"  - surface_convergence_penalty: `{score['surface_convergence_penalty']:.4f}`",
                f"  - loop_penalty: `{score['loop_penalty']:.4f}`",
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
    if "final_proof_score" in signals:
        breakdown = target_score_breakdown_from_candidate(future_top)
        return {
            "current_relevance": breakdown["current_relevance"],
            "raw_future_bonus": breakdown["raw_future"],
            "future_bonus_capped": breakdown["future_retention_bonus"],
            "role_gate": breakdown["role_gate"],
            "non_drift_gate": breakdown["non_drift_gate"],
            "non_loop_gate": breakdown["non_loop_gate"],
            "gated_future_bonus": breakdown["future_retention_bonus"],
            "future_bonus_total": breakdown["future_retention_bonus"] + breakdown["useful_convergence"],
            "useful_convergence": breakdown["useful_convergence"],
            "surface_convergence_penalty": breakdown["surface_convergence_penalty"],
            "loop_penalty": breakdown["hard_loop_penalty"],
            "redundancy_penalty": breakdown["redundancy_penalty"],
            "drift_penalty": breakdown["semantic_level_drift_penalty"],
            "noisy_branch_penalty": breakdown["noisy_branch_penalty"],
            "positive_score_sum": breakdown["target_positive_score_sum"],
            "negative_penalty_sum": breakdown["target_negative_penalty_sum"],
            "penalty_to_positive_ratio": breakdown["target_penalty_to_positive_ratio"],
        }
    is_v2 = "gated_future_bonus" in signals
    current = float(signals.get("current_relevance", 0.0))
    raw_future = float(signals.get("raw_future_bonus", signals.get("future_satisfiability", 0.0)))
    capped_future = float(signals.get("future_bonus_capped", raw_future))
    role_gate = float(signals.get("role_gate", 1.0))
    non_drift_gate = float(signals.get("non_drift_gate", 1.0))
    non_loop_gate = float(signals.get("non_loop_gate", 1.0))
    gated_future = float(signals.get("gated_future_bonus", capped_future if is_v2 else raw_future))
    useful_convergence = float(signals.get("useful_convergence", 0.0))
    future_bonus_total = gated_future + useful_convergence if is_v2 else raw_future + useful_convergence
    surface = float(signals.get("surface_convergence_penalty", 0.0))
    loop = float(signals.get("loop_penalty", 0.0))
    redundancy = float(signals.get("redundancy_penalty", 0.0))
    drift = float(signals.get("drift_penalty", 0.0))
    noisy = float(signals.get("noisy_branch_penalty", 0.0))
    future_positive_key = "gated_future_bonus" if is_v2 else "future_satisfiability"
    positive_sum = sum(
        max(0.0, float(signals.get(key, 0.0)))
        for key in ["current_relevance", future_positive_key, "useful_convergence", "type_compatibility", "progress"]
    )
    negative_sum = sum(abs(min(0.0, value)) for value in [surface, loop, redundancy, drift, noisy])
    return {
        "current_relevance": current,
        "raw_future_bonus": raw_future,
        "future_bonus_capped": capped_future,
        "role_gate": role_gate,
        "non_drift_gate": non_drift_gate,
        "non_loop_gate": non_loop_gate,
        "gated_future_bonus": gated_future,
        "future_bonus_total": future_bonus_total,
        "useful_convergence": useful_convergence,
        "surface_convergence_penalty": surface,
        "loop_penalty": loop,
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
    two_score = row["two_score_proof_state_beam"]
    future = row["future_aware_proof_state_beam"]
    future_v2 = row["future_aware_v2_proof_state_beam"]
    baseline_top = baseline["top_answer"] or {}
    proof_top = proof["top_answer"] or {}
    two_score_top = two_score["top_answer"] or {}
    future_top = future["top_answer"] or {}
    future_v2_top = future_v2["top_answer"] or {}
    print(f"\nQuestion {index}/{total}", flush=True)
    print(f"Q: {row['question']}", flush=True)
    print(f"Gold answer: {row['gold_answers']}", flush=True)
    print(f"Start entity: {row['start_entity']['name']} {row['start_entity']['labels']}", flush=True)
    print(f"Baseline top answer: {baseline_top.get('answer_label', '<none>')}", flush=True)
    print(f"Proof-state top answer: {proof_top.get('answer_label', '<none>')}", flush=True)
    print(f"Two-score top answer: {two_score_top.get('answer_label', '<none>')}", flush=True)
    print(f"Future-aware top answer: {future_top.get('answer_label', '<none>')}", flush=True)
    print(f"Future-aware v2 top answer: {future_v2_top.get('answer_label', '<none>')}", flush=True)
    print(f"Gold generated baseline: {'yes' if baseline['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated proof-state: {'yes' if proof['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated two-score: {'yes' if two_score['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated future-aware: {'yes' if future['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated future-aware v2: {'yes' if future_v2['gold_generated'] else 'no'}", flush=True)
    print(f"Baseline top path: {top_path_readable(baseline_top)}", flush=True)
    print(f"Proof-state evidence: {top_path_readable(proof_top)}", flush=True)
    print(f"Two-score evidence: {top_path_readable(two_score_top)}", flush=True)
    print(f"Future-aware evidence: {top_path_readable(future_top)}", flush=True)
    print(f"Future-aware v2 evidence: {top_path_readable(future_v2_top)}", flush=True)
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
    parser.add_argument("--output", default="runs/proof_state_search_two_score")
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
