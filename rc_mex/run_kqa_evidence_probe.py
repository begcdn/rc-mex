from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph


@dataclass
class ProbeExample:
    question_id: str
    question: str
    program_index: int
    start_entity_ids: set[str]
    start_entity_name: str
    gold_answer_ids: set[str]
    gold_answer_labels: list[str]


@dataclass
class PathState:
    entity_id: str
    entity_label: str
    score: float
    path: list[dict[str, Any]] = field(default_factory=list)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)

    log_stage(1, 4, f"Loading KQA Pro KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    log_line(f"Entities: {len(graph.entities)}")
    log_line(f"Relations: {len(graph.relations())}")

    log_stage(2, 4, f"Selecting simple 2-hop validation examples from {args.questions}")
    questions = load_json(args.questions)
    examples, selection_stats = select_examples(
        graph=graph,
        samples=questions,
        max_examples=args.max_examples,
        max_questions=args.max_questions,
    )
    log_line(f"Selected examples: {len(examples)}")
    log_line(f"Skipped unsupported: {selection_stats.get('unsupported_program', 0)}")
    log_line(f"Skipped empty gold execution: {selection_stats.get('empty_gold_execution', 0)}")

    log_stage(3, 4, "Generating frozen candidate path pools")
    rows = []
    for index, example in enumerate(examples, start=1):
        row = run_probe_for_question(
            graph=graph,
            example=example,
            top_k=args.top_k,
            beam_width=args.beam_width,
            relation_cap=args.relation_cap,
            sample_entities=args.sample_entities,
        )
        print_question_log(row, index, len(examples))
        rows.append(row)

    log_stage(4, 4, "Writing probe outputs")
    metrics = compute_metrics(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "selection_stats": selection_stats,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "candidate_paths.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    (output_dir / "report.md").write_text(write_report(metrics), encoding="utf-8")
    log_line("candidate_paths.jsonl")
    log_line("metrics.json")
    log_line("report.md")
    print(f"Wrote KQA evidence probe outputs to {output_dir}")


def select_examples(
    graph: KnowledgeGraph,
    samples: list[dict[str, Any]],
    max_examples: int,
    max_questions: int | None,
) -> tuple[list[ProbeExample], dict[str, int]]:
    stats: dict[str, int] = defaultdict(int)
    examples: list[ProbeExample] = []
    for program_index, sample in enumerate(samples):
        if max_questions is not None and stats["questions_seen"] >= max_questions:
            break
        stats["questions_seen"] += 1
        try:
            example = parse_simple_chain_sample(graph, sample, program_index)
        except ValueError as exc:
            stats[str(exc)] += 1
            continue
        examples.append(example)
        stats["examples_selected"] += 1
        if len(examples) >= max_examples:
            break
    return examples, dict(stats)


def parse_simple_chain_sample(graph: KnowledgeGraph, sample: dict[str, Any], program_index: int) -> ProbeExample:
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
    state = set(start_ids)
    for step in [first_relate, second_relate]:
        inputs = step.get("inputs", []) or []
        if len(inputs) < 2:
            raise ValueError("unsupported_program")
        state, _ = graph.follow(state, str(inputs[0]), str(inputs[1]), max_proofs=10000)
        if not state:
            raise ValueError("empty_gold_execution")
    return ProbeExample(
        question_id=str(sample.get("id") or sample.get("qid") or sample.get("ID") or f"val:{program_index}"),
        question=str(sample.get("question", "")),
        program_index=program_index,
        start_entity_ids=start_ids,
        start_entity_name=str(find_inputs[0]),
        gold_answer_ids=state,
        gold_answer_labels=entity_labels(graph, state),
    )


def run_probe_for_question(
    graph: KnowledgeGraph,
    example: ProbeExample,
    top_k: int,
    beam_width: int,
    relation_cap: int,
    sample_entities: int,
) -> dict[str, Any]:
    start_states = [
        PathState(
            entity_id=entity_id,
            entity_label=graph.entity_name(entity_id),
            score=0.0,
            path=[],
        )
        for entity_id in sorted(example.start_entity_ids)
    ]
    states = start_states
    all_final_states: list[PathState] = []
    hop_debug = []

    for hop in [1, 2]:
        next_states: list[PathState] = []
        for state in states:
            frontier = graph.candidate_relations([state.entity_id], cap=relation_cap, sample_entities=sample_entities)
            ranked = rank_relations(example.question, frontier)
            selected = ranked[:top_k]
            hop_debug.append(
                {
                    "hop": hop,
                    "source_entity_id": state.entity_id,
                    "source_entity": state.entity_label,
                    "frontier_size": len(frontier),
                    "selected_relations": [
                        {
                            "relation_id": candidate["relation_id"],
                            "direction": candidate["direction"],
                            "score": candidate["score"],
                            "frequency": candidate["frequency"],
                        }
                        for candidate in selected
                    ],
                }
            )
            for candidate in selected:
                expanded = expand_one_relation(graph, state, candidate, hop)
                next_states.extend(expanded)
        next_states.sort(key=lambda item: (-item.score, item.entity_label, item.entity_id))
        if hop == 1:
            states = next_states[:beam_width]
        else:
            all_final_states = next_states

    candidates = group_candidate_answers(graph, all_final_states, example.gold_answer_ids)
    gold_generated = any(candidate["is_gold"] for candidate in candidates)
    gold_rank = next((candidate["rank"] for candidate in candidates if candidate["is_gold"]), 0)
    gold_top1 = gold_rank == 1
    main_failure = classify_failure(candidates, gold_generated, gold_top1)
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
        "candidate_answers": candidates,
        "gold_answer_in_candidate_pool": gold_generated,
        "gold_answer_rank": gold_rank,
        "gold_answer_ranked_top1": gold_top1,
        "number_of_candidate_answers": len(candidates),
        "number_of_paths": sum(candidate["num_paths"] for candidate in candidates),
        "paths_per_answer": {candidate["answer_label"]: candidate["num_paths"] for candidate in candidates},
        "hop_debug": hop_debug,
        "main_failure": main_failure,
    }


def rank_relations(question: str, frontier: list[Any]) -> list[dict[str, Any]]:
    ranked = []
    for relation in frontier:
        relation_text = f"{relation.predicate.replace('_', ' ')} {relation.direction}"
        semantic_score = char_ngram_similarity(question, relation_text)
        frequency_bonus = 0.05 * min(1.0, float(relation.frequency) / 5.0)
        score = semantic_score + frequency_bonus
        ranked.append(
            {
                "relation_id": relation.predicate,
                "direction": relation.direction,
                "frequency": relation.frequency,
                "score": score,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["relation_id"], item["direction"]))
    return ranked


def expand_one_relation(graph: KnowledgeGraph, state: PathState, candidate: dict[str, Any], hop: int) -> list[PathState]:
    out = []
    for relation in graph.iter_relations(state.entity_id):
        if relation.get("predicate") != candidate["relation_id"]:
            continue
        if relation.get("direction") != candidate["direction"]:
            continue
        target_id = str(relation.get("object", ""))
        if not target_id:
            continue
        target_label = graph.entity_name(target_id)
        step = {
            "hop": hop,
            "from_entity_id": state.entity_id,
            "from_entity": state.entity_label,
            "relation_id": candidate["relation_id"],
            "direction": candidate["direction"],
            "to_entity_id": target_id,
            "to_entity": target_label,
            "relation_score": candidate["score"],
            "readable": f"{state.entity_label} --{candidate['relation_id']}[{candidate['direction']}]--> {target_label}",
        }
        out.append(
            PathState(
                entity_id=target_id,
                entity_label=target_label,
                score=state.score + float(candidate["score"]),
                path=state.path + [step],
            )
        )
    return out


def group_candidate_answers(graph: KnowledgeGraph, final_states: list[PathState], gold_answer_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for state in final_states:
        row = grouped.setdefault(
            state.entity_id,
            {
                "answer_id": state.entity_id,
                "answer_label": graph.entity_name(state.entity_id),
                "score": 0.0,
                "best_path_score": 0.0,
                "paths": [],
                "is_gold": state.entity_id in gold_answer_ids,
            },
        )
        row["score"] += state.score
        row["best_path_score"] = max(row["best_path_score"], state.score)
        row["paths"].append(
            {
                "path_score": state.score,
                "relation_sequence": [step["relation_id"] for step in state.path],
                "entity_sequence": [state.path[0]["from_entity"], *[step["to_entity"] for step in state.path]] if state.path else [state.entity_label],
                "steps": state.path,
                "readable": " | ".join(step["readable"] for step in state.path),
            }
        )
    candidates = list(grouped.values())
    for candidate in candidates:
        candidate["num_paths"] = len(candidate["paths"])
        candidate["path_diversity_summary"] = path_diversity(candidate["paths"])
    candidates.sort(key=lambda item: (-item["score"], -item["best_path_score"], item["answer_label"], item["answer_id"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def path_diversity(paths: list[dict[str, Any]]) -> dict[str, int]:
    relation_sequences = {" -> ".join(path["relation_sequence"]) for path in paths}
    first_hops = {path["relation_sequence"][0] for path in paths if path["relation_sequence"]}
    return {
        "unique_relation_sequences": len(relation_sequences),
        "unique_first_hop_relations": len(first_hops),
    }


def classify_failure(candidates: list[dict[str, Any]], gold_generated: bool, gold_top1: bool) -> str:
    if not candidates:
        return "no_candidates"
    if gold_top1:
        return "baseline_correct"
    if gold_generated:
        return "gold_ranked_low"
    return "gold_not_generated"


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not total:
        return {
            "total_questions": 0,
            "gold_generated_rate": 0.0,
            "gold_ranked_top1_rate": 0.0,
            "gold_generated_but_ranked_low_rate": 0.0,
            "multiple_paths_per_answer_rate": 0.0,
            "average_candidate_answers": 0.0,
            "average_paths_per_answer": 0.0,
            "failure_counts": {},
        }
    failure_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        failure_counts[row["main_failure"]] += 1
    total_candidate_answers = sum(row["number_of_candidate_answers"] for row in rows)
    total_paths = sum(row["number_of_paths"] for row in rows)
    return {
        "total_questions": total,
        "gold_generated_rate": avg_bool(row["gold_answer_in_candidate_pool"] for row in rows),
        "gold_ranked_top1_rate": avg_bool(row["gold_answer_ranked_top1"] for row in rows),
        "gold_generated_but_ranked_low_rate": avg_bool(
            row["gold_answer_in_candidate_pool"] and not row["gold_answer_ranked_top1"]
            for row in rows
        ),
        "multiple_paths_per_answer_rate": avg_bool(
            any(paths > 1 for paths in row["paths_per_answer"].values())
            for row in rows
        ),
        "gold_answer_multiple_paths_rate": avg_bool(
            any(candidate["is_gold"] and candidate["num_paths"] > 1 for candidate in row["candidate_answers"])
            for row in rows
        ),
        "average_candidate_answers": total_candidate_answers / total,
        "average_paths_per_answer": total_paths / max(1, total_candidate_answers),
        "failure_counts": dict(failure_counts),
    }


def write_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# KQA Pro Evidence Probe",
        "",
        "This minimal smoke test freezes a candidate path pool generated from KQA Pro local relation frontiers.",
        "",
        "It does not use gold relation steps or gold prefixes during search. Gold programs are used only to select simple 2-hop examples, get the start entity, and compute gold final answers.",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Bottleneck Checklist",
        "",
        f"1. How often is the gold answer generated? `{metrics['gold_generated_rate']:.3f}`",
        f"2. How often is it generated but ranked low? `{metrics['gold_generated_but_ranked_low_rate']:.3f}`",
        f"3. Are there multiple paths per answer? `{metrics['multiple_paths_per_answer_rate']:.3f}`",
        f"4. Do gold answers have multiple paths? `{metrics.get('gold_answer_multiple_paths_rate', 0.0):.3f}`",
        "5. Is answer scoring likely to help? It is promising only if gold generation is non-trivial and ranked-low cases exist.",
        "",
    ]
    return "\n".join(lines)


def print_question_log(row: dict[str, Any], index: int, total: int) -> None:
    print(f"\nQuestion {index}/{total}", flush=True)
    print(f"Q: {row['question']}", flush=True)
    print(f"Gold: {row['gold_answers']}", flush=True)
    print(f"Start entity: {row['start_entity']['name']} {row['start_entity']['labels']}", flush=True)
    print(f"Gold generated: {'yes' if row['gold_answer_in_candidate_pool'] else 'no'}", flush=True)
    print(f"Gold ranked top1: {'yes' if row['gold_answer_ranked_top1'] else 'no'}", flush=True)
    print("Top candidates:", flush=True)
    for candidate in row["candidate_answers"][:3]:
        print(
            f"{candidate['rank']}. answer={candidate['answer_label']} "
            f"score={candidate['score']:.4f} paths={candidate['num_paths']}",
            flush=True,
        )
    print(f"Main failure: {row['main_failure']}", flush=True)


def entity_labels(graph: KnowledgeGraph, entity_ids: set[str], limit: int = 20) -> list[str]:
    return [graph.entity_name(entity_id) for entity_id in sorted(entity_ids, key=lambda eid: graph.entity_name(eid))[:limit]]


def avg_bool(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


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
    parser = argparse.ArgumentParser(description="Minimal KQA Pro evidence/path aggregation smoke test.")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="runs/kqa_evidence_probe")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--relation-cap", type=int, default=30)
    parser.add_argument("--sample-entities", type=int, default=25)
    return parser.parse_args()


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
