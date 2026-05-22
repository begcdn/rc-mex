from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, load_json, write_json, write_jsonl
from .judges import (
    CommandClient,
    HashEmbeddingRanker,
    LLMListwiseRanker,
    MockLLMClient,
    OllamaClient,
    OpenAICompatibleClient,
    RandomRanker,
)
from .kg import KnowledgeGraph, RelationCandidate
from .kopl import RelationGroundingInstance, extract_relation_grounding_instances
from .metrics import RankingRecord, add_metric_cis, ranking_metrics
from .witness import WitnessCard, build_witness_cards


CORE_CONDITIONS = {
    "normal_real": {"label_mode": "normal", "witness_mode": "real"},
    "anon_rel_real": {"label_mode": "anonymous", "witness_mode": "real"},
    "anon_rel_anon_entities_types": {
        "label_mode": "anonymous",
        "witness_mode": "anon_entities_types",
    },
}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = ensure_dir(args.output)

    kb = load_json(args.kb)
    samples = load_json(args.questions)
    graph = KnowledgeGraph(kb)

    instances, extraction_stats = extract_relation_grounding_instances(
        samples=samples,
        graph=graph,
        split_name=args.split_name,
        max_instances=args.max_instances,
        max_questions=args.max_questions,
    )

    client = make_llm_client(args)
    rankers = {
        "random": RandomRanker(args.seed),
        "embedding_schema": HashEmbeddingRanker(),
        "schema_llm": LLMListwiseRanker("schema_llm", client),
        "witness_llm": LLMListwiseRanker("witness_llm", client),
    }

    condition_names = split_csv(args.conditions)
    method_names = split_csv(args.methods)
    all_rankings: list[dict[str, Any]] = []
    all_instance_rows: list[dict[str, Any]] = []
    metrics_by_condition: dict[str, Any] = {}

    for condition_name in condition_names:
        condition = CORE_CONDITIONS[condition_name]
        condition_records: dict[str, list[RankingRecord]] = {name: [] for name in method_names}
        candidate_present = 0
        relation_present = 0
        candidate_counts = []
        skipped_no_candidates = 0

        for instance in instances:
            candidates = graph.candidate_relations(
                instance.current_entity_ids,
                cap=args.candidate_cap,
                sample_entities=args.sample_current_entities,
            )
            if args.force_include_gold:
                candidates = maybe_force_include_gold(
                    candidates,
                    instance,
                    graph,
                    args.candidate_cap,
                    args.sample_current_entities,
                )
            if not candidates:
                skipped_no_candidates += 1
                continue

            candidate_counts.append(len(candidates))
            candidate_keys = {candidate.key for candidate in candidates}
            if (instance.gold_predicate, instance.gold_direction) in candidate_keys:
                candidate_present += 1
            if any(predicate == instance.gold_predicate for predicate, _ in candidate_keys):
                relation_present += 1

            cards = build_witness_cards(
                graph=graph,
                current_entity_ids=instance.current_entity_ids,
                candidates=candidates,
                label_mode=condition["label_mode"],
                witness_mode=condition["witness_mode"],
                returned_sample_size=args.returned_sample_size,
            )
            rng.shuffle(cards)
            card_by_id = {card.candidate_id: card for card in cards}
            all_instance_rows.append(instance_row(instance, candidates, cards, condition_name, graph))

            for method_name in method_names:
                ranker = rankers[method_name]
                evidence = evidence_for_method(method_name)
                result = ranker.rank(instance.question, cards, evidence=evidence)
                ranked_pairs = [
                    card_by_id[cid].key for cid in result.ranked_candidate_ids if cid in card_by_id
                ]
                llm_calls = 1 if method_name.endswith("_llm") else 0
                record = RankingRecord(
                    instance_id=instance.instance_id,
                    gold_predicate=instance.gold_predicate,
                    gold_direction=instance.gold_direction,
                    ranked_pairs=ranked_pairs,
                    llm_calls=llm_calls,
                    prompt_tokens=result.prompt_tokens_estimate,
                    completion_tokens=result.completion_tokens_estimate,
                    latency_seconds=result.latency_seconds,
                )
                condition_records[method_name].append(record)
                all_rankings.append(
                    {
                        "condition": condition_name,
                        "method": method_name,
                        "instance_id": instance.instance_id,
                        "gold_predicate": instance.gold_predicate,
                        "gold_direction": instance.gold_direction,
                        "ranking": result.ranked_candidate_ids,
                        "ranked_pairs": ranked_pairs,
                        "raw_output": result.raw_output if args.save_raw_outputs else "",
                    }
                )

        denom = max(1, len(candidate_counts))
        metrics_by_condition[condition_name] = {
            "candidate_pool": {
                "n_instances_with_candidates": len(candidate_counts),
                "skipped_no_candidates": skipped_no_candidates,
                "gold_relation_present_rate": relation_present / denom,
                "gold_relation_direction_present_rate": candidate_present / denom,
                "avg_candidate_count": sum(candidate_counts) / denom,
            },
            "methods": {},
        }
        for method_name, records in condition_records.items():
            method_metrics = ranking_metrics(records)
            metrics_by_condition[condition_name]["methods"][method_name] = add_metric_cis(
                records,
                method_metrics,
                samples=args.bootstrap_samples,
                seed=args.seed,
            )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "extraction": extraction_stats,
        "conditions": metrics_by_condition,
        "notes": [
            "Gold programs are used only to create controlled relation-grounding instances.",
            "The mock LLM backend is for smoke tests only, not a valid experiment.",
        ],
    }
    write_json(output_dir / "metrics.json", summary)
    write_jsonl(output_dir / "instances.jsonl", all_instance_rows)
    write_jsonl(output_dir / "rankings.jsonl", all_rankings)
    if args.position_bias_instances > 0:
        position_bias = run_position_bias_checks(
            args=args,
            graph=graph,
            instances=instances[: args.position_bias_instances],
            condition_names=condition_names,
            method_names=[method for method in method_names if method.endswith("_llm")],
            client=client,
        )
        write_json(output_dir / "position_bias.json", position_bias)
    write_report(output_dir / "report.md", summary)
    print(f"Wrote MVP1 outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIGR-D MVP1 execution-witness grounding.")
    parser.add_argument("--kb", required=True, help="Path to KQA Pro kb.json")
    parser.add_argument("--questions", required=True, help="Path to KQA Pro train/val JSON")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--output", default="runs/mvp1")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=200)
    parser.add_argument("--candidate-cap", type=int, default=50)
    parser.add_argument("--sample-current-entities", type=int, default=100)
    parser.add_argument("--returned-sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--conditions", default="normal_real,anon_rel_real,anon_rel_anon_entities_types")
    parser.add_argument("--methods", default="random,embedding_schema,schema_llm,witness_llm")
    parser.add_argument(
        "--judge-backend",
        choices=["mock", "ollama", "openai-compatible", "command"],
        default="mock",
    )
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--judge-command", default=None)
    parser.add_argument("--save-raw-outputs", action="store_true")
    parser.add_argument(
        "--position-bias-instances",
        type=int,
        default=0,
        help="Optional: rerank this many instances with multiple candidate-order permutations.",
    )
    parser.add_argument("--position-bias-trials", type=int, default=3)
    parser.add_argument(
        "--force-include-gold",
        action="store_true",
        help="Diagnostic mode only: include the gold pair if it appears outside the capped candidate list.",
    )
    args = parser.parse_args()
    for condition in split_csv(args.conditions):
        if condition not in CORE_CONDITIONS:
            raise SystemExit(f"unknown condition: {condition}")
    valid_methods = {"random", "embedding_schema", "schema_llm", "witness_llm"}
    for method in split_csv(args.methods):
        if method not in valid_methods:
            raise SystemExit(f"unknown method: {method}")
    return args


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_llm_client(args: argparse.Namespace):
    if args.judge_backend == "mock":
        return MockLLMClient()
    if args.judge_backend == "ollama":
        return OllamaClient(model=args.model, host=args.ollama_host)
    if args.judge_backend == "openai-compatible":
        return OpenAICompatibleClient(
            model=args.model,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
        )
    if args.judge_backend == "command":
        if not args.judge_command:
            raise SystemExit("--judge-command is required for command backend")
        return CommandClient(args.judge_command)
    raise SystemExit(f"unknown judge backend: {args.judge_backend}")


def evidence_for_method(method_name: str) -> str:
    if method_name == "witness_llm":
        return "witness"
    return "schema"


def maybe_force_include_gold(
    candidates: list[RelationCandidate],
    instance: RelationGroundingInstance,
    graph: KnowledgeGraph,
    cap: int,
    sample_entities: int,
) -> list[RelationCandidate]:
    if (instance.gold_predicate, instance.gold_direction) in {candidate.key for candidate in candidates}:
        return candidates
    all_candidates = graph.candidate_relations(
        instance.current_entity_ids,
        cap=1_000_000,
        sample_entities=sample_entities,
    )
    gold = [
        candidate
        for candidate in all_candidates
        if candidate.key == (instance.gold_predicate, instance.gold_direction)
    ]
    if not gold:
        return candidates
    if len(candidates) < cap:
        return candidates + gold[:1]
    return candidates[:-1] + gold[:1]


def run_position_bias_checks(
    args: argparse.Namespace,
    graph: KnowledgeGraph,
    instances: list[RelationGroundingInstance],
    condition_names: list[str],
    method_names: list[str],
    client: Any,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    rankers = {
        "schema_llm": LLMListwiseRanker("schema_llm", client),
        "witness_llm": LLMListwiseRanker("witness_llm", client),
    }
    for condition_name in condition_names:
        condition = CORE_CONDITIONS[condition_name]
        for instance in instances:
            candidates = graph.candidate_relations(
                instance.current_entity_ids,
                cap=args.candidate_cap,
                sample_entities=args.sample_current_entities,
            )
            if not candidates:
                continue
            base_cards = build_witness_cards(
                graph=graph,
                current_entity_ids=instance.current_entity_ids,
                candidates=candidates,
                label_mode=condition["label_mode"],
                witness_mode=condition["witness_mode"],
                returned_sample_size=args.returned_sample_size,
            )
            for method_name in method_names:
                top_pairs = []
                top_ids = []
                for trial in range(args.position_bias_trials):
                    rng = random.Random(f"{args.seed}:{condition_name}:{instance.instance_id}:{method_name}:{trial}")
                    cards = list(base_cards)
                    rng.shuffle(cards)
                    by_id = {card.candidate_id: card for card in cards}
                    result = rankers[method_name].rank(
                        instance.question,
                        cards,
                        evidence=evidence_for_method(method_name),
                    )
                    top_id = result.ranked_candidate_ids[0] if result.ranked_candidate_ids else ""
                    top_ids.append(top_id)
                    top_pairs.append(by_id[top_id].key if top_id in by_id else ("", ""))
                most_common_pair, count = Counter(top_pairs).most_common(1)[0]
                results.append(
                    {
                        "condition": condition_name,
                        "method": method_name,
                        "instance_id": instance.instance_id,
                        "top_ids": top_ids,
                        "top_pairs": top_pairs,
                        "unique_top_pairs": len(set(top_pairs)),
                        "top_pair_agreement": count / max(1, len(top_pairs)),
                        "most_common_top_pair": most_common_pair,
                    }
                )
    if not results:
        return {"enabled": True, "n": 0, "avg_top_pair_agreement": 0.0, "rows": []}
    return {
        "enabled": True,
        "n": len(results),
        "trials": args.position_bias_trials,
        "avg_top_pair_agreement": sum(row["top_pair_agreement"] for row in results) / len(results),
        "rows": results,
    }


def instance_row(
    instance: RelationGroundingInstance,
    candidates: list[RelationCandidate],
    cards: list[WitnessCard],
    condition_name: str,
    graph: KnowledgeGraph,
) -> dict[str, Any]:
    return {
        "condition": condition_name,
        "instance_id": instance.instance_id,
        "question": instance.question,
        "step_index": instance.step_index,
        "current_entities": [
            {"id": entity_id, "name": graph.entity_name(entity_id)}
            for entity_id in sorted(instance.current_entity_ids)[:20]
        ],
        "gold_predicate": instance.gold_predicate,
        "gold_direction": instance.gold_direction,
        "candidate_count": len(candidates),
        "candidate_pairs": [candidate.key for candidate in candidates],
        "cards": [card.witness_text() for card in cards],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CIGR-D MVP1 Report",
        "",
        "This report is generated by the MVP1 runner.",
        "",
        "## Extraction",
        "",
        "```json",
        json.dumps(summary["extraction"], indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
    ]
    for condition_name, condition in summary["conditions"].items():
        lines.extend([f"### {condition_name}", "", "Candidate pool:", "", "```json"])
        lines.append(json.dumps(condition["candidate_pool"], indent=2, sort_keys=True))
        lines.extend(["```", "", "Methods:", ""])
        for method_name, metrics in condition["methods"].items():
            lines.extend([f"#### {method_name}", "", "```json"])
            lines.append(json.dumps(metrics, indent=2, sort_keys=True))
            lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
