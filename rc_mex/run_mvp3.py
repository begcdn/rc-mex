from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph
from cigr_d_mvp1.kopl import RelationGroundingInstance, extract_relation_grounding_instances

from .debug import graph_debug_stats
from .primitive_key import primitive_key
from .run_mvp2 import load_jsonl, load_samples


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)

    log_stage(1, 5, f"Loading KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    stats = graph_debug_stats(graph)
    log_line(f"Entities: {stats['entities']}")
    log_line(f"Relations: {stats['relations']}")
    log_line(f"Triples: {stats['triples']}")

    log_stage(2, 5, f"Loading questions from {args.questions}")
    questions = load_samples(args.questions)
    instances, extraction_stats = extract_relation_grounding_instances(
        samples=questions,
        graph=graph,
        split_name=args.split_name,
        max_instances=args.max_instances,
        max_questions=args.max_questions,
    )
    instances_by_id = {instance.instance_id: instance for instance in instances}
    log_line(f"Instances extracted: {len(instances)}")

    log_stage(3, 5, f"Loading MVP2 rankings from {args.retrieval_predictions}")
    prediction_rows = load_jsonl(args.retrieval_predictions)
    prediction_rows = filter_prediction_rows(
        prediction_rows,
        conditions=split_csv(args.conditions),
        variants=split_csv(args.card_variants),
        require_gold_available=args.require_gold_available,
        local_only=args.local_only,
        include_injected=args.include_injected,
    )
    log_line(f"Prediction rows loaded: {len(prediction_rows)}")

    log_stage(4, 5, "Executing top-1 and marginalized top-k relations")
    rows = []
    for prediction in prediction_rows:
        instance = instances_by_id.get(str(prediction.get("instance_id", "")))
        if not instance:
            continue
        row = evaluate_prediction(graph, instance, prediction, top_k=args.top_k, weight_scheme=args.weight_scheme)
        rows.append(row)
        if args.verbose:
            log_line(
                f"{row['instance_id']} {row['condition_id']}/{row['card_variant']} "
                f"top1_recall={row['top1_gold_recall']:.3f} "
                f"marginal_recall={row['marginal_gold_recall']:.3f}"
            )
    log_line(f"Execution rows: {len(rows)}")

    log_stage(5, 5, "Writing MVP3 reports")
    metrics = compute_mvp3_metrics(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "extraction_stats": extraction_stats,
        "n_prediction_rows": len(prediction_rows),
        "n_execution_rows": len(rows),
        "metrics": metrics,
    }
    write_jsonl(output_dir / "execution_predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    write_report(output_dir / "report.md", summary)
    log_line("execution_predictions.jsonl")
    log_line("metrics.json")
    log_line("report.md")
    print(f"Wrote RC-MEX MVP3 outputs to {output_dir}")


def filter_prediction_rows(
    rows: list[dict[str, Any]],
    conditions: list[str],
    variants: list[str],
    require_gold_available: bool,
    local_only: bool,
    include_injected: bool,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("condition_id") not in conditions:
            continue
        if row.get("card_variant") not in variants:
            continue
        if require_gold_available and not row.get("gold_card_id"):
            continue
        if local_only and not row.get("local_gold_in_candidate_pool"):
            continue
        if not include_injected and row.get("injected_gold"):
            continue
        if not row.get("ranked_card_ids"):
            continue
        out.append(row)
    return out


def evaluate_prediction(
    graph: KnowledgeGraph,
    instance: RelationGroundingInstance,
    prediction: dict[str, Any],
    top_k: int,
    weight_scheme: str,
) -> dict[str, Any]:
    gold_entities, gold_proofs = graph.follow(
        instance.current_entity_ids,
        instance.gold_predicate,
        instance.gold_direction,
    )
    ranked_card_ids = [str(card_id) for card_id in prediction.get("ranked_card_ids", []) if card_id]
    top_cards = [parse_card_id(card_id) for card_id in ranked_card_ids[:top_k]]
    top1 = top_cards[0] if top_cards else None
    top1_entities: set[str] = set()
    top1_relation_id = ""
    top1_direction = ""
    if top1:
        top1_relation_id = top1["relation_id"]
        top1_direction = top1["direction"]
        top1_entities, _ = graph.follow(instance.current_entity_ids, top1_relation_id, top1_direction)

    scores: dict[str, float] = defaultdict(float)
    relation_support = []
    for rank, parsed in enumerate(top_cards, start=1):
        relation_id = parsed["relation_id"]
        direction = parsed["direction"]
        entities, _ = graph.follow(instance.current_entity_ids, relation_id, direction)
        weight = rank_weight(rank, weight_scheme)
        for entity_id in entities:
            scores[entity_id] += weight
        relation_support.append(
            {
                "rank": rank,
                "relation_id": relation_id,
                "direction": direction,
                "weight": weight,
                "result_count": len(entities),
                "gold_overlap": len(entities & gold_entities),
            }
        )
    marginal_entities = set(scores)
    ranked_entities = sorted(scores.items(), key=lambda item: (-item[1], graph.entity_name(item[0]), item[0]))
    top_n = max(1, len(gold_entities))
    marginal_topn = {entity_id for entity_id, _ in ranked_entities[:top_n]}
    return {
        "instance_id": instance.instance_id,
        "question": instance.question,
        "condition_id": prediction.get("condition_id", ""),
        "card_variant": prediction.get("card_variant", ""),
        "frontier_mode": prediction.get("frontier_mode", ""),
        "local_gold_in_candidate_pool": bool(prediction.get("local_gold_in_candidate_pool")),
        "injected_gold": bool(prediction.get("injected_gold")),
        "gold_relation_id": instance.gold_predicate,
        "gold_direction": instance.gold_direction,
        "gold_result_count": len(gold_entities),
        "gold_result_examples": entity_examples(graph, gold_entities),
        "gold_execution_non_empty": bool(gold_entities),
        "top1_relation_id": top1_relation_id,
        "top1_direction": top1_direction,
        "top1_result_count": len(top1_entities),
        "top1_gold_recall": set_recall(gold_entities, top1_entities),
        "top1_precision_vs_gold": set_precision(gold_entities, top1_entities),
        "top1_jaccard": jaccard(gold_entities, top1_entities),
        "top1_any_gold": bool(gold_entities & top1_entities),
        "top1_exact_relation": primitive_key(top1_relation_id, top1_direction) == primitive_key(instance.gold_predicate, instance.gold_direction),
        "marginal_top_k": top_k,
        "marginal_result_count": len(marginal_entities),
        "marginal_gold_recall": set_recall(gold_entities, marginal_entities),
        "marginal_precision_vs_gold": set_precision(gold_entities, marginal_entities),
        "marginal_jaccard": jaccard(gold_entities, marginal_entities),
        "marginal_any_gold": bool(gold_entities & marginal_entities),
        "marginal_topn_gold_recall": set_recall(gold_entities, marginal_topn),
        "marginal_topn_precision_vs_gold": set_precision(gold_entities, marginal_topn),
        "marginal_top_entities": [
            {"entity": graph.entity_name(entity_id), "score": score}
            for entity_id, score in ranked_entities[:10]
        ],
        "relation_support": relation_support,
        "ranked_card_ids": ranked_card_ids,
        "gold_proofs": [
            {
                "subject": graph.entity_name(proof.subject_id),
                "predicate": proof.predicate,
                "object": graph.entity_name(proof.object_id),
                "direction": proof.direction,
            }
            for proof in gold_proofs[:10]
        ],
    }


def parse_card_id(card_id: str) -> dict[str, str]:
    parts = card_id.split("::")
    if len(parts) < 4:
        return {"condition_id": "", "card_variant": "", "relation_id": "", "direction": ""}
    return {
        "condition_id": parts[0],
        "card_variant": parts[1],
        "relation_id": parts[2],
        "direction": parts[3],
    }


def rank_weight(rank: int, scheme: str) -> float:
    if scheme == "uniform":
        return 1.0
    if scheme == "reciprocal_rank":
        return 1.0 / max(1, rank)
    if scheme == "softmax_rank":
        return math.exp(-float(rank - 1))
    raise ValueError(f"unknown weight scheme: {scheme}")


def entity_examples(graph: KnowledgeGraph, entity_ids: set[str], limit: int = 10) -> list[str]:
    return [graph.entity_name(entity_id) for entity_id in sorted(entity_ids)[:limit]]


def set_recall(gold: set[str], predicted: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & predicted) / len(gold)


def set_precision(gold: set[str], predicted: set[str]) -> float:
    if not predicted:
        return 0.0
    return len(gold & predicted) / len(predicted)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def compute_mvp3_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_id"], row["card_variant"])].append(row)
    return {
        f"{condition}/{variant}": group_metrics(group_rows)
        for (condition, variant), group_rows in sorted(groups.items())
    }


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "n_instances": float(len(rows)),
        "local_gold_in_candidate_pool_rate": average_bool([row["local_gold_in_candidate_pool"] for row in rows]),
        "injected_gold_rate": average_bool([row["injected_gold"] for row in rows]),
        "top1_exact_relation_rate": average_bool([row["top1_exact_relation"] for row in rows]),
        "top1_any_gold_rate": average_bool([row["top1_any_gold"] for row in rows]),
        "marginal_any_gold_rate": average_bool([row["marginal_any_gold"] for row in rows]),
        "top1_gold_recall": average([row["top1_gold_recall"] for row in rows]),
        "marginal_gold_recall": average([row["marginal_gold_recall"] for row in rows]),
        "marginal_recall_gain": average([row["marginal_gold_recall"] - row["top1_gold_recall"] for row in rows]),
        "top1_precision_vs_gold": average([row["top1_precision_vs_gold"] for row in rows]),
        "marginal_precision_vs_gold": average([row["marginal_precision_vs_gold"] for row in rows]),
        "top1_jaccard": average([row["top1_jaccard"] for row in rows]),
        "marginal_jaccard": average([row["marginal_jaccard"] for row in rows]),
        "marginal_topn_gold_recall": average([row["marginal_topn_gold_recall"] for row in rows]),
    }


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def average_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RC-MEX MVP3 Report",
        "",
        "MVP3 compares hard top-1 relation execution against top-k marginalized execution for one controlled relation slot.",
        "",
        "This is not full QA. It reuses MVP2 slot rankings and evaluates the next denotation from the gold-prefix state.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(
            {
                "n_prediction_rows": summary["n_prediction_rows"],
                "n_execution_rows": summary["n_execution_rows"],
                "args": summary["args"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Metrics",
        "",
    ]
    for group, metrics in sorted(summary["metrics"].items()):
        lines.extend([f"### {group}", "", "```json"])
        lines.append(json.dumps(metrics, indent=2, sort_keys=True))
        lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-MEX MVP3 marginalized execution diagnostics.")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--retrieval-predictions", required=True, help="MVP2 retrieval_predictions.jsonl")
    parser.add_argument("--output", default="runs/rc_mex_mvp3")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--max-instances", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--conditions", default="B1,B2")
    parser.add_argument("--card-variants", default="contrastive_hard")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--weight-scheme",
        choices=["uniform", "reciprocal_rank", "softmax_rank"],
        default="reciprocal_rank",
    )
    parser.add_argument("--require-gold-available", action="store_true", default=True)
    parser.add_argument("--include-injected", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Evaluate only rows where gold was in the local frontier.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
