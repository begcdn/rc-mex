from __future__ import annotations

import argparse
import json
import random
import re
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
from .oracle import LLMClient, make_client, parse_json_object


DEFAULT_LOCAL_MODEL = "llama3:8b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class RankingResult:
    ranked_card_ids: list[str]
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


class CardRanker:
    def __init__(self, client: LLMClient, max_retries: int = 2):
        self.client = client
        self.max_retries = max_retries

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        prompt = build_ranking_prompt(prompt_payload, candidates)
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


class MockCardRanker(CardRanker):
    def __init__(self):
        self.client = None  # type: ignore[assignment]
        self.max_retries = 0

    def rank(self, prompt_payload: dict[str, Any], candidates: list[dict[str, Any]]) -> RankingResult:
        question = str(prompt_payload.get("question", ""))
        scored = []
        for candidate in candidates:
            text = card_search_text(candidate["card"])
            scored.append((char_ngram_similarity(question, text), candidate["candidate_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return RankingResult(
            ranked_card_ids=[candidate_id for _, candidate_id in scored],
            raw_output=json.dumps({"ranking": [candidate_id for _, candidate_id in scored]}),
            prompt_tokens=0,
            completion_tokens=0,
            latency_seconds=0.0,
        )


def main() -> None:
    args = parse_args()
    metadata_patterns = parse_metadata_patterns(args.metadata_relation_patterns)
    output_dir = ensure_dir(args.output)

    log_stage(1, 6, f"Loading KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    stats = graph_debug_stats(graph)
    log_line(f"Entities: {stats['entities']}")
    log_line(f"Concepts: {stats['concepts']}")
    log_line(f"Relations: {stats['relations']}")
    log_line(f"Triples: {stats['triples']}")

    log_stage(2, 6, f"Loading relation cards from {args.cards}")
    all_cards = load_jsonl(args.cards)
    all_cards = filter_cards(
        all_cards,
        conditions=split_csv(args.conditions),
        variants=split_csv(args.card_variants),
        exclude_metadata=args.exclude_metadata_relations,
        metadata_patterns=metadata_patterns,
    )
    card_index = index_cards(all_cards)
    log_line(f"Cards loaded: {len(all_cards)}")
    log_line(f"Condition/variant groups: {len(card_index)}")

    log_stage(3, 6, f"Extracting gold relation slots from {args.questions}")
    questions = load_samples(args.questions)
    instances, extraction_stats = extract_relation_grounding_instances(
        samples=questions,
        graph=graph,
        split_name=args.split_name,
        max_instances=args.max_instances,
        max_questions=args.max_questions,
    )
    log_line(f"Questions seen: {extraction_stats['questions_seen']}")
    log_line(f"Relation steps seen: {extraction_stats['relation_steps_seen']}")
    log_line(f"Instances created: {extraction_stats['instances_created']}")
    log_line(f"Unsupported prefixes: {extraction_stats['unsupported_prefix']}")

    if args.oracle_backend == "mock":
        ranker: CardRanker = MockCardRanker()
    else:
        client = make_client(
            backend=args.oracle_backend,
            model=args.model,
            ollama_host=args.ollama_host,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            command=args.oracle_command,
        )
        ranker = CardRanker(client)

    log_stage(4, 6, "Ranking candidate relation cards")
    rows: list[dict[str, Any]] = []
    for instance in instances:
        relation_frontier = graph.candidate_relations(
            instance.current_entity_ids,
            cap=args.relation_cap,
            sample_entities=args.sample_entities,
        )
        if args.verbose:
            log_line(
                f"{instance.instance_id}: frontier={len(relation_frontier)} "
                f"gold={instance.gold_predicate}/{instance.gold_direction}"
            )
        for (condition_id, variant), cards_by_relation in sorted(card_index.items()):
            candidates = build_candidate_cards(
                cards_by_relation=cards_by_relation,
                relation_frontier=relation_frontier,
                candidate_card_cap=args.candidate_card_cap,
            )
            gold_key = (instance.gold_predicate, instance.gold_direction)
            gold_card = cards_by_relation.get(gold_key)
            gold_card_id = card_id(gold_card) if gold_card else ""
            gold_in_pool = bool(gold_card_id and any(candidate["card_id"] == gold_card_id for candidate in candidates))
            row = base_prediction_row(instance, condition_id, variant, candidates, gold_card_id, gold_in_pool)

            if not candidates:
                row["skip_reason"] = "empty_candidate_pool"
                rows.append(row)
                continue
            if not gold_card:
                row["skip_reason"] = "gold_card_missing_from_library"
                rows.append(row)
                continue
            if not gold_in_pool:
                row["skip_reason"] = "gold_card_not_in_frontier"
                rows.append(row)
                continue

            prompt_payload = {
                "question": instance.question,
                "current_entities": render_current_entities(graph, instance.current_entity_ids, args.entity_sample_size),
                "gold_prefix_note": "The current entities are produced by the gold prefix. Choose the next relation-card grounding.",
            }
            result = ranker.rank(prompt_payload, candidates)
            ranked_card_ids = [
                candidates_by_id(candidates)[candidate_id]["card_id"]
                for candidate_id in result.ranked_card_ids
                if candidate_id in candidates_by_id(candidates)
            ]
            row.update(
                {
                    "ranked_card_ids": ranked_card_ids,
                    "top_card_id": ranked_card_ids[0] if ranked_card_ids else "",
                    "top_relation_id": relation_id_from_card_id(ranked_card_ids[0]) if ranked_card_ids else "",
                    "top_direction": direction_from_card_id(ranked_card_ids[0]) if ranked_card_ids else "",
                    "gold_rank": rank_of(gold_card_id, ranked_card_ids),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_seconds": result.latency_seconds,
                    "raw_output": result.raw_output if args.save_raw_outputs else "",
                }
            )
            rows.append(row)

    log_line(f"Prediction rows: {len(rows)}")

    log_stage(5, 6, "Computing metrics")
    metrics = compute_retrieval_metrics(rows)
    for group, group_metrics in sorted(metrics.items()):
        log_line(
            f"{group}: candidate_recall={group_metrics['candidate_recall']:.3f} "
            f"R@1={group_metrics['recall_at_1']:.3f} "
            f"R@5={group_metrics['recall_at_5']:.3f} "
            f"MRR={group_metrics['mrr']:.3f}"
        )

    log_stage(6, 6, "Writing reports")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "extraction_stats": extraction_stats,
        "n_cards": len(all_cards),
        "n_instances": len(instances),
        "n_prediction_rows": len(rows),
        "metrics": metrics,
    }
    write_jsonl(output_dir / "retrieval_predictions.jsonl", rows)
    write_json(output_dir / "metrics.json", summary)
    write_report(output_dir / "report.md", summary)
    log_line("retrieval_predictions.jsonl")
    log_line("metrics.json")
    log_line("report.md")
    print(f"Wrote RC-MEX MVP2 outputs to {output_dir}")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "questions", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise SystemExit(f"unsupported questions file format: {path}")


def filter_cards(
    cards: list[dict[str, Any]],
    conditions: list[str],
    variants: list[str],
    exclude_metadata: bool,
    metadata_patterns: list[str],
) -> list[dict[str, Any]]:
    out = []
    for card in cards:
        if card.get("condition_id") not in conditions:
            continue
        if card.get("card_variant") not in variants:
            continue
        if exclude_metadata and is_metadata_relation(str(card.get("relation_id", "")), metadata_patterns):
            continue
        out.append(card)
    return out


def index_cards(cards: list[dict[str, Any]]) -> dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]]:
    index: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for card in cards:
        group = (str(card.get("condition_id", "")), str(card.get("card_variant", "")))
        relation_key = (str(card.get("relation_id", "")), str(card.get("direction", "")))
        index[group][relation_key] = card
    return dict(index)


def build_candidate_cards(
    cards_by_relation: dict[tuple[str, str], dict[str, Any]],
    relation_frontier: list[Any],
    candidate_card_cap: int,
) -> list[dict[str, Any]]:
    candidates = []
    for relation in relation_frontier:
        card = cards_by_relation.get((relation.predicate, relation.direction))
        if not card:
            continue
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1:03d}",
                "card_id": card_id(card),
                "relation_id": card.get("relation_id", ""),
                "direction": card.get("direction", ""),
                "frontier_frequency": relation.frequency,
                "card": card,
            }
        )
        if len(candidates) >= candidate_card_cap:
            break
    return candidates


def base_prediction_row(
    instance: RelationGroundingInstance,
    condition_id: str,
    variant: str,
    candidates: list[dict[str, Any]],
    gold_card_id: str,
    gold_in_pool: bool,
) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "question": instance.question,
        "program_index": instance.program_index,
        "step_index": instance.step_index,
        "condition_id": condition_id,
        "card_variant": variant,
        "gold_relation_id": instance.gold_predicate,
        "gold_direction": instance.gold_direction,
        "gold_card_id": gold_card_id,
        "gold_in_candidate_pool": gold_in_pool,
        "candidate_count": len(candidates),
        "candidate_card_ids": [candidate["card_id"] for candidate in candidates],
        "ranked_card_ids": [],
        "top_card_id": "",
        "top_relation_id": "",
        "top_direction": "",
        "gold_rank": 0,
        "skip_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_seconds": 0.0,
        "raw_output": "",
    }


def build_ranking_prompt(payload: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    candidate_text = "\n\n".join(format_candidate(candidate) for candidate in candidates)
    return (
        "RC-MEX MVP2 SLOT-TO-CARD RETRIEVAL TASK\n"
        "You are selecting the knowledge-graph relation card that best matches the next relation needed by the question.\n"
        "The current entities were produced by a gold prefix. Choose the candidate card whose exact predicate should be applied next.\n"
        "Do not rank by general relatedness. Prefer the card whose predicate and argument direction best match the question.\n"
        "Return only JSON as: {\"ranking\": [\"C001\", \"C002\", \"C003\"]}.\n\n"
        f"Question:\n{payload['question']}\n\n"
        f"Current entity examples:\n{json.dumps(payload['current_entities'], ensure_ascii=False, indent=2)}\n\n"
        f"Candidate relation cards:\n{candidate_text}\n\n"
        "Rank every candidate ID from best to worst."
    )


def format_candidate(candidate: dict[str, Any]) -> str:
    card = candidate["card"]
    generated = card.get("generated", {}) or {}
    parts = [
        f"{candidate['candidate_id']}.",
        f"Description: {card.get('description', '')}",
        f"Argument 1 role: {generated.get('argument_1_role', '')}",
        f"Argument 2 role: {generated.get('argument_2_role', '')}",
        f"Direction explanation: {generated.get('valid_direction_explanation', '')}",
        f"Positive rule: {generated.get('positive_rule', '')}",
        f"Negative rule: {generated.get('negative_rule', '')}",
        f"Domain types: {', '.join(card.get('domain_types', []) or []) or '<hidden>'}",
        f"Range types: {', '.join(card.get('range_types', []) or []) or '<hidden>'}",
        f"Positive examples: {json.dumps(card.get('positive_examples_train', [])[:3], ensure_ascii=False)}",
        f"Hard-negative examples: {json.dumps(card.get('hard_negative_examples_train', [])[:3], ensure_ascii=False)}",
    ]
    return "\n".join(parts)


def render_current_entities(graph: KnowledgeGraph, entity_ids: set[str], limit: int) -> list[dict[str, Any]]:
    rows = []
    for entity_id in sorted(entity_ids)[:limit]:
        rows.append(
            {
                "entity": graph.entity_name(entity_id),
                "types": graph.entity_type_names(entity_id),
            }
        )
    return rows


def parse_ranking(output: str, valid_ids: set[str]) -> list[str]:
    data = parse_json_object(output)
    ranking = data.get("ranking", []) if data else []
    out: list[str] = []
    if isinstance(ranking, list):
        for candidate_id in ranking:
            candidate_id = str(candidate_id)
            if candidate_id in valid_ids and candidate_id not in out:
                out.append(candidate_id)
    if out:
        return out
    for candidate_id in re.findall(r"C\d{3}", output):
        if candidate_id in valid_ids and candidate_id not in out:
            out.append(candidate_id)
    return out


def candidates_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in candidates}


def card_id(card: dict[str, Any] | None) -> str:
    if not card:
        return ""
    return "::".join(
        [
            str(card.get("condition_id", "")),
            str(card.get("card_variant", "")),
            str(card.get("relation_id", "")),
            str(card.get("direction", "")),
        ]
    )


def relation_id_from_card_id(value: str) -> str:
    parts = value.split("::")
    return parts[2] if len(parts) >= 4 else ""


def direction_from_card_id(value: str) -> str:
    parts = value.split("::")
    return parts[3] if len(parts) >= 4 else ""


def rank_of(gold_card_id: str, ranked_card_ids: list[str]) -> int:
    if not gold_card_id:
        return 0
    try:
        return ranked_card_ids.index(gold_card_id) + 1
    except ValueError:
        return 0


def compute_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_id"], row["card_variant"])].append(row)
    out: dict[str, Any] = {}
    for key, group_rows in sorted(groups.items()):
        condition_id, variant = key
        out[f"{condition_id}/{variant}"] = group_metrics(group_rows)
    return out


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    candidate_rows = [row for row in rows if row["gold_card_id"]]
    ranked_rows = [row for row in rows if row.get("gold_rank", 0)]
    return {
        "n_instances": float(len(rows)),
        "n_gold_card_available": float(len(candidate_rows)),
        "n_ranked_instances": float(len(ranked_rows)),
        "missing_gold_card_rate": average_bool([not bool(row["gold_card_id"]) for row in rows]),
        "candidate_recall": average_bool([bool(row["gold_in_candidate_pool"]) for row in candidate_rows]),
        "recall_at_1": recall_at(ranked_rows, 1),
        "recall_at_3": recall_at(ranked_rows, 3),
        "recall_at_5": recall_at(ranked_rows, 5),
        "mrr": average([1.0 / row["gold_rank"] for row in ranked_rows if row["gold_rank"]]),
        "avg_candidate_count": average([float(row["candidate_count"]) for row in rows]),
        "avg_prompt_tokens": average([float(row.get("prompt_tokens", 0.0) or 0.0) for row in ranked_rows]),
        "avg_completion_tokens": average([float(row.get("completion_tokens", 0.0) or 0.0) for row in ranked_rows]),
        "avg_latency_seconds": average([float(row.get("latency_seconds", 0.0) or 0.0) for row in ranked_rows]),
    }


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if 0 < int(row["gold_rank"]) <= k) / len(rows)


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
        "# RC-MEX MVP2 Report",
        "",
        "MVP2 evaluates whether question slots retrieve the correct reusable relation cards.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(
            {
                "n_cards": summary["n_cards"],
                "n_instances": summary["n_instances"],
                "n_prediction_rows": summary["n_prediction_rows"],
                "extraction_stats": summary["extraction_stats"],
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


def card_search_text(card: dict[str, Any]) -> str:
    generated = card.get("generated", {}) or {}
    return " ".join(
        str(value)
        for value in [
            card.get("description", ""),
            generated.get("positive_rule", ""),
            generated.get("negative_rule", ""),
            generated.get("argument_1_role", ""),
            generated.get("argument_2_role", ""),
            " ".join(card.get("domain_types", []) or []),
            " ".join(card.get("range_types", []) or []),
        ]
    )


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
    normalized = f"  {text.casefold()}  "
    out: dict[str, int] = {}
    for index in range(max(1, len(normalized) - n + 1)):
        gram = normalized[index : index + n]
        out[gram] = out.get(gram, 0) + 1
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-MEX MVP2 slot-to-card retrieval.")
    parser.add_argument("--kb", required=True, help="Path to KQA Pro-style kb.json")
    parser.add_argument("--questions", required=True, help="Path to KQA Pro train/val JSON")
    parser.add_argument("--cards", required=True, help="Path to MVP1 relation_cards.jsonl")
    parser.add_argument("--output", default="runs/rc_mex_mvp2")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--max-instances", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--relation-cap", type=int, default=30)
    parser.add_argument("--sample-entities", type=int, default=25)
    parser.add_argument("--candidate-card-cap", type=int, default=30)
    parser.add_argument("--entity-sample-size", type=int, default=8)
    parser.add_argument("--conditions", default="A,B1,B2,B3,C")
    parser.add_argument("--card-variants", default="contrastive_hard,random_negative,name_only")
    parser.add_argument(
        "--oracle-backend",
        choices=["mock", "ollama", "openai", "openai-compatible", "command"],
        default="ollama",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--oracle-command", default=None)
    parser.add_argument("--save-raw-outputs", action="store_true")
    parser.add_argument("--exclude-metadata-relations", action="store_true")
    parser.add_argument("--metadata-relation-patterns", default=None)
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
    random.seed(17)
    main()
