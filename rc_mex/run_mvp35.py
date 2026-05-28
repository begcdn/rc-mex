from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.judges import estimate_tokens
from cigr_d_mvp1.kg import KnowledgeGraph
from cigr_d_mvp1.kopl import RelationGroundingInstance, extract_relation_grounding_instances

from .debug import graph_debug_stats, is_metadata_relation, parse_metadata_patterns
from .evidence import CONDITIONS
from .oracle import LLMClient, make_client, parse_json_object
from .primitive_key import key_from_instance, primitive_key
from .run_mvp2 import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OPENAI_MODEL,
    RankingResult,
    build_candidate_cards,
    card_id,
    card_search_text,
    char_ngram_similarity,
    filter_cards,
    index_cards,
    load_jsonl,
    load_samples,
    parse_ranking,
)


@dataclass
class RelationLabelContext:
    relation_ids: list[str]
    entity_map: dict[str, str]

    @classmethod
    def from_cards(cls, cards: list[dict[str, Any]]) -> "RelationLabelContext":
        return cls(
            relation_ids=sorted({str(card.get("relation_id", "")) for card in cards if card.get("relation_id")}),
            entity_map={},
        )

    def relation_display(self, relation_id: str, condition_id: str) -> str:
        if condition_id == "A":
            return relation_id
        if condition_id in {"B1", "B2", "B3"}:
            try:
                return f"R_{self.relation_ids.index(relation_id) + 1:04d}"
            except ValueError:
                return "R_UNKNOWN"
        if condition_id == "C":
            if len(self.relation_ids) <= 1:
                return "misleading_relation"
            try:
                index = self.relation_ids.index(relation_id)
            except ValueError:
                return "misleading_relation"
            return self.relation_ids[(index + 1) % len(self.relation_ids)]
        return relation_id

    def entity_display(self, entity_id: str) -> str:
        if entity_id not in self.entity_map:
            self.entity_map[entity_id] = f"ENTITY_{len(self.entity_map) + 1:04d}"
        return self.entity_map[entity_id]


class LabelRanker:
    def __init__(self, client: LLMClient, max_retries: int = 2):
        self.client = client
        self.max_retries = max_retries

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        prompt = build_label_prompt(prompt_payload, candidates)
        valid_ids = {candidate["candidate_id"] for candidate in candidates}
        start = time.time()
        last_output = ""
        for _ in range(self.max_retries + 1):
            try:
                last_output = self.client.complete(prompt)
            except RuntimeError as exc:
                last_output = f"CLIENT_ERROR: {exc}"
                continue
            ranked = parse_ranking(last_output, valid_ids)
            if ranked:
                ranked += [candidate["candidate_id"] for candidate in candidates if candidate["candidate_id"] not in ranked]
                return RankingResult(
                    ranked_card_ids=ranked,
                    raw_output=last_output,
                    prompt_tokens=estimate_tokens(prompt),
                    completion_tokens=estimate_tokens(last_output),
                    latency_seconds=time.time() - start,
                )
        return RankingResult(
            ranked_card_ids=[candidate["candidate_id"] for candidate in candidates],
            raw_output=last_output,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(last_output),
            latency_seconds=time.time() - start,
        )


class MockLabelRanker(LabelRanker):
    def __init__(self):
        self.client = None  # type: ignore[assignment]
        self.max_retries = 0

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        question = str(prompt_payload.get("question", ""))
        scored = []
        for candidate in candidates:
            text = f"{candidate.get('visible_relation', '')} {candidate.get('direction', '')}"
            scored.append((char_ngram_similarity(question, text), candidate["candidate_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking = [candidate_id for _, candidate_id in scored]
        return RankingResult(ranking, json.dumps({"ranking": ranking}), 0, 0, 0.0)


class MockCardRanker:
    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        question = str(prompt_payload.get("question", ""))
        scored = []
        for candidate in candidates:
            scored.append((char_ngram_similarity(question, card_search_text(candidate["card"])), candidate["candidate_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking = [candidate_id for _, candidate_id in scored]
        return RankingResult(ranking, json.dumps({"ranking": ranking}), 0, 0, 0.0)


class CardPromptRanker:
    def __init__(self, client: LLMClient):
        from .run_mvp2 import CardRanker

        self.ranker = CardRanker(client)

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        return self.ranker.rank(prompt_payload, candidates)


class CardBlueprintRanker:
    """CoG-style soft-prior reranker using relation cards instead of blueprints."""

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        question = str(prompt_payload.get("question", ""))
        current_types = type_set(prompt_payload.get("current_entities", []))
        scored = []
        for candidate in candidates:
            card = candidate["card"]
            question_score = char_ngram_similarity(question, card_search_text(card))
            label_score = char_ngram_similarity(question, str(candidate.get("visible_relation", "")))
            domain_score = type_overlap_score(current_types, set(card.get("domain_types", []) or []))
            output_score = output_compatibility_score(candidate)
            final_score = 0.60 * question_score + 0.15 * label_score + 0.15 * domain_score + 0.10 * output_score
            scored.append((final_score, candidate["candidate_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking = [candidate_id for _, candidate_id in scored]
        return RankingResult(
            ranked_card_ids=ranking,
            raw_output=json.dumps({"ranking": ranking, "method": "relation_card_blueprint"}),
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=0.0,
        )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    metadata_patterns = parse_metadata_patterns(args.metadata_relation_patterns)

    log_stage(1, 6, f"Loading KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    stats = graph_debug_stats(graph)
    log_line(f"Entities: {stats['entities']}")
    log_line(f"Relations: {stats['relations']}")
    log_line(f"Triples: {stats['triples']}")

    log_stage(2, 6, f"Loading relation cards from {args.cards}")
    cards = filter_cards(
        load_jsonl(args.cards),
        conditions=split_csv(args.conditions),
        variants=split_csv(args.card_variants),
        exclude_metadata=args.exclude_metadata_relations,
        metadata_patterns=metadata_patterns,
    )
    card_index = index_cards(cards)
    label_context = RelationLabelContext.from_cards(cards)
    log_line(f"Cards loaded: {len(cards)}")
    log_line(f"Condition/variant groups: {len(card_index)}")

    log_stage(3, 6, f"Extracting controlled slots from {args.questions}")
    questions = load_samples(args.questions)
    instances, extraction_stats = extract_relation_grounding_instances(
        samples=questions,
        graph=graph,
        split_name=args.split_name,
        max_instances=args.max_instances,
        max_questions=args.max_questions,
    )
    log_line(f"Instances created: {len(instances)}")

    log_stage(4, 6, "Preparing rankers")
    if args.oracle_backend == "mock":
        label_ranker: Any = MockLabelRanker()
        card_ranker: Any = MockCardRanker()
    else:
        client = make_client(
            backend=args.oracle_backend,
            model=args.model,
            ollama_host=args.ollama_host,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            command=args.oracle_command,
        )
        label_ranker = LabelRanker(client)
        card_ranker = CardPromptRanker(client)
    card_blueprint_ranker = CardBlueprintRanker()

    log_stage(5, 6, "Ranking names vs relation cards")
    rows: list[dict[str, Any]] = []
    for instance in instances:
        relation_frontier = graph.candidate_relations(
            instance.current_entity_ids,
            cap=args.relation_cap,
            sample_entities=args.sample_entities,
        )
        for (condition_id, variant), cards_by_relation in sorted(card_index.items()):
            candidates = build_candidate_cards(cards_by_relation, relation_frontier, args.candidate_card_cap)
            if not candidates:
                continue
            gold_card = cards_by_relation.get(key_from_instance(instance))
            gold_card_id = card_id(gold_card) if gold_card else ""
            if args.covered_only and not gold_card_id:
                continue
            attach_visible_relation_labels(candidates, condition_id, label_context)
            prompt_payload = {
                "question": instance.question,
                "current_entities": render_current_entities_for_condition(
                    graph,
                    instance.current_entity_ids,
                    condition_id,
                    label_context,
                    args.entity_sample_size,
                ),
            }
            for method, ranker in [
                ("relation_label", label_ranker),
                ("relation_card", card_ranker),
                ("relation_card_blueprint", card_blueprint_ranker),
            ]:
                result = ranker.rank(prompt_payload, candidates)
                row = evaluate_ranked_candidates(
                    graph=graph,
                    instance=instance,
                    condition_id=condition_id,
                    variant=variant,
                    method=method,
                    candidates=candidates,
                    ranked_candidate_ids=result.ranked_card_ids,
                    gold_card_id=gold_card_id,
                    result=result,
                    save_raw=args.save_raw_outputs,
                )
                rows.append(row)
        if args.verbose:
            log_line(f"{instance.instance_id}: wrote {len(rows)} rows so far")

    log_stage(6, 6, "Writing MVP3.5 reports")
    metrics = compute_metrics(rows)
    debug = build_debug_examples(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "extraction_stats": extraction_stats,
        "n_rows": len(rows),
        "metrics": metrics,
    }
    write_jsonl(output_dir / "comparison_predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    write_json(output_dir / "debug_examples.json", debug)
    write_debug_report(output_dir / "debug_examples.md", debug)
    write_report(output_dir / "report.md", summary)
    log_line("comparison_predictions.jsonl")
    log_line("metrics.json")
    log_line("debug_examples.json")
    log_line("debug_examples.md")
    log_line("report.md")
    print(f"Wrote RC-MEX MVP3.5 outputs to {output_dir}")


def attach_visible_relation_labels(candidates: list[dict[str, Any]], condition_id: str, context: RelationLabelContext) -> None:
    for candidate in candidates:
        candidate["visible_relation"] = context.relation_display(str(candidate.get("relation_id", "")), condition_id)


def render_current_entities_for_condition(
    graph: KnowledgeGraph,
    entity_ids: set[str],
    condition_id: str,
    context: RelationLabelContext,
    limit: int,
) -> list[dict[str, Any]]:
    condition = CONDITIONS.get(condition_id, CONDITIONS["A"])
    rows = []
    for entity_id in sorted(entity_ids)[:limit]:
        entity = graph.entity_name(entity_id) if condition.entity_mode == "real" else context.entity_display(entity_id)
        rows.append(
            {
                "entity": entity,
                "types": graph.entity_type_names(entity_id) if condition.include_types else [],
            }
        )
    return rows


def build_label_prompt(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    candidate_text = "\n".join(
        f"{candidate['candidate_id']}. relation_label={candidate.get('visible_relation', '')}; "
        f"direction={candidate.get('direction', '')}"
        for candidate in candidates
    )
    return (
        "RC-MEX MVP3.5 RELATION-LABEL BASELINE\n"
        "Choose the candidate relation whose label and direction best match the next relation required by the question.\n"
        "Do not use any hidden gold relation. Return only JSON as: {\"ranking\": [\"C001\", \"C002\"]}.\n\n"
        f"Question:\n{payload['question']}\n\n"
        f"Current entity examples:\n{json.dumps(payload['current_entities'], ensure_ascii=False, indent=2)}\n\n"
        f"Candidate relation labels:\n{candidate_text}\n\n"
        "Rank every candidate ID from best to worst."
    )


def type_set(current_entities: list[dict[str, Any]]) -> set[str]:
    out = set()
    for entity in current_entities:
        for type_name in entity.get("types", []) or []:
            out.add(str(type_name).casefold())
    return out


def type_overlap_score(left: set[str], right: set[str]) -> float:
    right = {str(value).casefold() for value in right if value}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def output_compatibility_score(candidate: dict[str, Any]) -> float:
    frequency = float(candidate.get("frontier_frequency", 0) or 0)
    return min(1.0, frequency / 5.0)


def evaluate_ranked_candidates(
    graph: KnowledgeGraph,
    instance: RelationGroundingInstance,
    condition_id: str,
    variant: str,
    method: str,
    candidates: list[dict[str, Any]],
    ranked_candidate_ids: list[str],
    gold_card_id: str,
    result: RankingResult,
    save_raw: bool,
) -> dict[str, Any]:
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    ranked_candidates = [candidates_by_id[candidate_id] for candidate_id in ranked_candidate_ids if candidate_id in candidates_by_id]
    ranked_card_ids = [candidate["card_id"] for candidate in ranked_candidates]
    gold_entities, _ = graph.follow(instance.current_entity_ids, instance.gold_predicate, instance.gold_direction)
    top1_entities = execute_candidate(graph, instance, ranked_candidates[:1])
    top3_entities = execute_candidate(graph, instance, ranked_candidates[:3])
    gold_rank = rank_of(gold_card_id, ranked_card_ids)
    return {
        "instance_id": instance.instance_id,
        "question": instance.question,
        "condition_id": condition_id,
        "card_variant": variant,
        "method": method,
        "gold_relation_id": instance.gold_predicate,
        "gold_direction": instance.gold_direction,
        "gold_card_id": gold_card_id,
        "gold_in_candidate_pool": bool(gold_card_id and any(candidate["card_id"] == gold_card_id for candidate in candidates)),
        "gold_rank": gold_rank,
        "candidate_count": len(candidates),
        "ranked_card_ids": ranked_card_ids,
        "top1_card_id": ranked_card_ids[0] if ranked_card_ids else "",
        "gold_relation_in_top1": 0 < gold_rank <= 1,
        "gold_relation_in_top3": 0 < gold_rank <= 3,
        "gold_relation_in_top5": 0 < gold_rank <= 5,
        "gold_result_count": len(gold_entities),
        "top1_result_count": len(top1_entities),
        "top3_result_count": len(top3_entities),
        "top1_gold_recall": recall(gold_entities, top1_entities),
        "top3_gold_recall": recall(gold_entities, top3_entities),
        "top1_precision": precision(gold_entities, top1_entities),
        "top3_precision": precision(gold_entities, top3_entities),
        "top1_f1": f1(precision(gold_entities, top1_entities), recall(gold_entities, top1_entities)),
        "top3_f1": f1(precision(gold_entities, top3_entities), recall(gold_entities, top3_entities)),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_seconds": result.latency_seconds,
        "raw_output": result.raw_output if save_raw else "",
    }


def execute_candidate(graph: KnowledgeGraph, instance: RelationGroundingInstance, candidates: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for candidate in candidates:
        entities, _ = graph.follow(
            instance.current_entity_ids,
            str(candidate.get("relation_id", "")),
            str(candidate.get("direction", "")),
        )
        out.update(entities)
    return out


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_id"], row["card_variant"], row["method"])].append(row)
    metrics = {
        f"{condition}/{variant}/{method}": metric_group(group_rows)
        for (condition, variant, method), group_rows in sorted(groups.items())
    }
    metrics["robustness_drop"] = robustness_drop(metrics)
    return metrics


def metric_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "n_instances": float(len(rows)),
        "candidate_recall": average_bool([row["gold_in_candidate_pool"] for row in rows]),
        "recall_at_1": average_bool([0 < row["gold_rank"] <= 1 for row in rows]),
        "recall_at_3": average_bool([0 < row["gold_rank"] <= 3 for row in rows]),
        "mrr": average([1.0 / row["gold_rank"] for row in rows if row["gold_rank"]]),
        "top1_gold_recall": average([row["top1_gold_recall"] for row in rows]),
        "top3_gold_recall": average([row["top3_gold_recall"] for row in rows]),
        "top1_precision": average([row["top1_precision"] for row in rows]),
        "top3_precision": average([row["top3_precision"] for row in rows]),
        "top1_f1": average([row["top1_f1"] for row in rows]),
        "top3_f1": average([row["top3_f1"] for row in rows]),
        "average_top1_result_size": average([float(row["top1_result_count"]) for row in rows]),
        "average_top3_result_size": average([float(row["top3_result_count"]) for row in rows]),
        "avg_prompt_tokens": average([float(row["prompt_tokens"]) for row in rows]),
        "avg_latency_seconds": average([float(row["latency_seconds"]) for row in rows]),
    }


def robustness_drop(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for method in {"relation_label", "relation_card"}:
        normal = metrics.get(f"A/contrastive_hard/{method}", {})
        for condition in ["B1", "B2", "C"]:
            other = metrics.get(f"{condition}/contrastive_hard/{method}", {})
            if normal and other:
                out[f"{method}:A_to_{condition}:recall_at_1_drop"] = normal.get("recall_at_1", 0.0) - other.get("recall_at_1", 0.0)
                out[f"{method}:A_to_{condition}:top1_f1_drop"] = normal.get("top1_f1", 0.0) - other.get("top1_f1", 0.0)
    return out


def build_debug_examples(rows: list[dict[str, Any]], limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(row["instance_id"], row["condition_id"])][row["method"]] = row
    buckets = {
        "cards_beat_names": [],
        "names_beat_cards": [],
        "misleading_names_fail_cards_succeed": [],
        "both_fail": [],
    }
    for (_instance_id, condition), methods in paired.items():
        name = methods.get("relation_label")
        card = methods.get("relation_card")
        if not name or not card:
            continue
        name_ok = name["gold_relation_in_top1"]
        card_ok = card["gold_relation_in_top1"]
        example = {
            "condition_id": condition,
            "instance_id": card["instance_id"],
            "question": card["question"],
            "gold": f"{card['gold_relation_id']}/{card['gold_direction']}",
            "name_top1": name["top1_card_id"],
            "card_top1": card["top1_card_id"],
            "name_gold_rank": name["gold_rank"],
            "card_gold_rank": card["gold_rank"],
            "name_top1_f1": name["top1_f1"],
            "card_top1_f1": card["top1_f1"],
        }
        if card_ok and not name_ok:
            buckets["cards_beat_names"].append(example)
            if condition == "C":
                buckets["misleading_names_fail_cards_succeed"].append(example)
        elif name_ok and not card_ok:
            buckets["names_beat_cards"].append(example)
        elif not name_ok and not card_ok:
            buckets["both_fail"].append(example)
    return {key: value[:limit] for key, value in buckets.items()}


def write_debug_report(path: Path, debug: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# MVP3.5 Debug Examples", ""]
    for section, examples in debug.items():
        lines.extend([f"## {section}", ""])
        if not examples:
            lines.extend(["_None_", ""])
            continue
        for example in examples:
            lines.extend(
                [
                    f"- `{example['instance_id']}` {example['condition_id']}: {example['question']}",
                    f"  - Gold: `{example['gold']}`",
                    f"  - Name top1: `{example['name_top1']}` rank={example['name_gold_rank']} F1={example['name_top1_f1']:.3f}",
                    f"  - Card top1: `{example['card_top1']}` rank={example['card_gold_rank']} F1={example['card_top1_f1']:.3f}",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RC-MEX MVP3.5 Report",
        "",
        "MVP3.5 compares raw relation labels/IDs against relation cards under the controlled MVP2 slot setup.",
        "",
        "This is not full KGQA and does not claim top-k graph search as novel. It tests whether relation cards are a more robust semantic interface than raw schema labels.",
        "",
        "## Metrics",
        "",
    ]
    for group, metrics in sorted(summary["metrics"].items()):
        lines.extend([f"### {group}", "", "```json", json.dumps(metrics, indent=2, sort_keys=True), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def rank_of(gold_card_id: str, ranked_card_ids: list[str]) -> int:
    if not gold_card_id:
        return 0
    try:
        return ranked_card_ids.index(gold_card_id) + 1
    except ValueError:
        return 0


def recall(gold: set[str], predicted: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & predicted) / len(gold)


def precision(gold: set[str], predicted: set[str]) -> float:
    if not predicted:
        return 0.0
    return len(gold & predicted) / len(predicted)


def f1(p: float, r: float) -> float:
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def average_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-MEX MVP3.5 card-vs-name retrieval/execution comparison.")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--output", default="runs/rc_mex_mvp35")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--max-instances", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--relation-cap", type=int, default=30)
    parser.add_argument("--sample-entities", type=int, default=25)
    parser.add_argument("--candidate-card-cap", type=int, default=30)
    parser.add_argument("--entity-sample-size", type=int, default=8)
    parser.add_argument("--conditions", default="A,B1,B2,C")
    parser.add_argument("--card-variants", default="contrastive_hard")
    parser.add_argument("--covered-only", action="store_true")
    parser.add_argument("--exclude-metadata-relations", action="store_true")
    parser.add_argument("--metadata-relation-patterns", default=None)
    parser.add_argument("--oracle-backend", choices=["mock", "ollama", "openai", "openai-compatible", "command"], default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--oracle-command", default=None)
    parser.add_argument("--save-raw-outputs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_OPENAI_MODEL if args.oracle_backend == "openai" else DEFAULT_LOCAL_MODEL
    return args


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
