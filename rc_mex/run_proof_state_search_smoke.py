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

    log_stage(3, 4, "Running baseline path beam and soft proof-state beam")
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
        )
        row = build_prediction_row(graph, example, baseline, proof_state)
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
    log_line("predictions.jsonl")
    log_line("metrics.json")
    log_line("report.md")
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
                            soft_signals={"label_score": candidate["label_score"]},
                        )
                    )
        next_states.sort(key=lambda item: (-item.score, state_sort_key(item)))
        states = next_states[:beam_width]
    return search_result(graph, states, example.gold_answer_ids, expansion_count, mode="baseline_path_beam")


def run_soft_proof_state_beam(
    graph: KnowledgeGraph,
    example: SmokeExample,
    top_k: int,
    beam_width: int,
    relation_cap: int,
    sample_entities: int,
    max_branch_entities: int,
    noisy_branch_threshold: int,
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
        states = next_states[:beam_width]
    return search_result(graph, states, example.gold_answer_ids, expansion_count, mode="soft_proof_state_beam")


def rank_relations(question: str, frontier: list[Any]) -> list[dict[str, Any]]:
    ranked = []
    for relation in frontier:
        label = f"{relation.predicate.replace('_', ' ')} {relation.direction}"
        label_score = char_ngram_similarity(question, label)
        frequency_bonus = 0.03 * min(1.0, float(relation.frequency) / 5.0)
        ranked.append(
            {
                "relation_id": relation.predicate,
                "direction": relation.direction,
                "frequency": relation.frequency,
                "label_score": label_score + frequency_bonus,
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
    }


def build_prediction_row(
    graph: KnowledgeGraph,
    example: SmokeExample,
    baseline: dict[str, Any],
    proof_state: dict[str, Any],
) -> dict[str, Any]:
    baseline_correct = baseline["hits_at_1"]
    proof_correct = proof_state["hits_at_1"]
    if baseline_correct and proof_correct:
        failure_type = "both_correct"
    elif proof_correct:
        failure_type = "proof_state_correct"
    elif baseline_correct:
        failure_type = "baseline_correct"
    elif not baseline["gold_generated"] and not proof_state["gold_generated"]:
        failure_type = "gold_not_generated"
    elif not baseline["candidate_answers"] or not proof_state["candidate_answers"]:
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
        "failure_type": failure_type,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row["baseline_path_beam"] for row in rows]
    proof = [row["soft_proof_state_beam"] for row in rows]
    return {
        "number_of_selected_questions": len(rows),
        "baseline_hits_at_1": avg_bool(item["hits_at_1"] for item in baseline),
        "proof_state_hits_at_1": avg_bool(item["hits_at_1"] for item in proof),
        "baseline_exact_match": avg_bool(item["exact_match"] for item in baseline),
        "proof_state_exact_match": avg_bool(item["exact_match"] for item in proof),
        "baseline_final_answer_f1": avg(item["final_answer_f1"] for item in baseline),
        "proof_state_final_answer_f1": avg(item["final_answer_f1"] for item in proof),
        "baseline_gold_generated_rate": avg_bool(item["gold_generated"] for item in baseline),
        "proof_state_gold_generated_rate": avg_bool(item["gold_generated"] for item in proof),
        "average_candidate_count_baseline": avg(item["candidate_count"] for item in baseline),
        "average_candidate_count_proof_state": avg(item["candidate_count"] for item in proof),
        "average_expansion_count_baseline": avg(item["expansion_count"] for item in baseline),
        "average_expansion_count_proof_state": avg(item["expansion_count"] for item in proof),
        "average_final_result_size_baseline": avg(item["final_result_size"] for item in baseline),
        "average_final_result_size_proof_state": avg(item["final_result_size"] for item in proof),
        "proof_state_wins": sum(
            1 for row in rows
            if row["soft_proof_state_beam"]["final_answer_f1"] > row["baseline_path_beam"]["final_answer_f1"]
        ),
        "baseline_wins": sum(
            1 for row in rows
            if row["baseline_path_beam"]["final_answer_f1"] > row["soft_proof_state_beam"]["final_answer_f1"]
        ),
        "both_correct": sum(1 for row in rows if row["baseline_path_beam"]["hits_at_1"] and row["soft_proof_state_beam"]["hits_at_1"]),
        "both_fail": sum(1 for row in rows if not row["baseline_path_beam"]["hits_at_1"] and not row["soft_proof_state_beam"]["hits_at_1"]),
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
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
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


def select_debug_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "proof":
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
        baseline_path = top_path_readable(baseline_top)
        proof_path = top_path_readable(proof_top)
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
                f"- Likely reason: {likely}",
                "",
            ]
        )
    return lines


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
    baseline_top = baseline["top_answer"] or {}
    proof_top = proof["top_answer"] or {}
    print(f"\nQuestion {index}/{total}", flush=True)
    print(f"Q: {row['question']}", flush=True)
    print(f"Gold answer: {row['gold_answers']}", flush=True)
    print(f"Start entity: {row['start_entity']['name']} {row['start_entity']['labels']}", flush=True)
    print(f"Baseline top answer: {baseline_top.get('answer_label', '<none>')}", flush=True)
    print(f"Proof-state top answer: {proof_top.get('answer_label', '<none>')}", flush=True)
    print(f"Gold generated baseline: {'yes' if baseline['gold_generated'] else 'no'}", flush=True)
    print(f"Gold generated proof-state: {'yes' if proof['gold_generated'] else 'no'}", flush=True)
    print(f"Baseline top path: {top_path_readable(baseline_top)}", flush=True)
    print(f"Proof-state evidence: {top_path_readable(proof_top)}", flush=True)
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
    return parser.parse_args()


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
