from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph
from cigr_d_mvp1.kopl import extract_relation_grounding_instances

from .debug import graph_debug_stats, is_metadata_relation, parse_metadata_patterns
from .evidence import CONDITIONS, RenderContext
from .oracle import CardGenerator, make_client, prompt_templates_markdown
from .primitive_key import PrimitiveKey, key_from_instance, primitive_key
from .run_mvp1 import CARD_VARIANTS, DEFAULT_LOCAL_MODEL, DEFAULT_OPENAI_MODEL, build_card, split_csv
from .sampling import PrimitiveSamples, sample_for_primitive
from .schema import Primitive, inventory_primitives


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    metadata_patterns = parse_metadata_patterns(args.metadata_relation_patterns)

    log_stage(1, 5, f"Loading KG from {args.kb}")
    graph = KnowledgeGraph(load_json(args.kb))
    stats = graph_debug_stats(graph)
    log_line(f"Entities: {stats['entities']}")
    log_line(f"Relations: {stats['relations']}")
    log_line(f"Triples: {stats['triples']}")

    log_stage(2, 5, f"Extracting target relation slots from {args.programs}")
    samples = load_samples(args.programs)
    instances, extraction_stats = extract_relation_grounding_instances(
        samples=samples,
        graph=graph,
        split_name=args.split_name,
        max_instances=args.max_slots,
        max_questions=args.max_questions,
    )
    target_keys = ordered_unique_keys([key_from_instance(instance) for instance in instances])
    if args.exclude_metadata_relations:
        target_keys = [
            key for key in target_keys
            if not is_metadata_relation(key.relation_id, metadata_patterns)
        ]
    log_line(f"Relation slots extracted: {len(instances)}")
    log_line(f"Unique target primitive keys: {len(target_keys)}")

    log_stage(3, 5, "Building primitive inventory and selecting targets")
    primitives = inventory_primitives(graph, min_examples=args.min_examples)
    primitive_by_key = {primitive_key(primitive.relation_id, primitive.direction): primitive for primitive in primitives}
    target_primitives = [primitive_by_key[key] for key in target_keys if key in primitive_by_key]
    missing_keys = [key for key in target_keys if key not in primitive_by_key]
    log_line(f"Target primitives found in KG inventory: {len(target_primitives)}")
    log_line(f"Missing target primitive keys: {len(missing_keys)}")
    if not target_primitives:
        raise SystemExit("No target primitives found. Check relation/direction normalization and input programs.")

    context = RenderContext.from_primitives(target_primitives)
    client = make_client(
        backend=args.oracle_backend,
        model=args.model,
        ollama_host=args.ollama_host,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        command=args.oracle_command,
    )
    generator = CardGenerator(client)

    log_stage(4, 5, "Generating targeted relation cards")
    condition_ids = split_csv(args.conditions)
    variants = split_csv(args.card_variants)
    card_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for primitive in target_primitives:
        samples_for_primitive = sample_for_primitive(
            graph=graph,
            target=primitive,
            primitives=primitives,
            train_positives=args.train_positives,
            heldout_positives=args.heldout_positives,
            train_negatives=args.train_negatives,
            heldout_negatives=args.heldout_negatives,
            random_negatives=args.random_negatives,
            seed=args.seed,
        )
        target_rows.append(target_summary(primitive, samples_for_primitive))
        for condition_id in condition_ids:
            condition = CONDITIONS[condition_id]
            for variant in variants:
                card = build_card(
                    graph=graph,
                    context=context,
                    condition=condition,
                    primitive=primitive,
                    samples=samples_for_primitive,
                    card_variant=variant,
                    generator=generator,
                )
                card_json = card.to_json()
                card_json["condition_id"] = condition_id
                card_rows.append(card_json)
                log_line(
                    f"Generated {primitive.relation_id}/{primitive.direction} "
                    f"{condition_id}/{variant}: {card.description}"
                )
    log_line(f"Cards generated: {len(card_rows)}")

    log_stage(5, 5, "Writing outputs")
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "extraction_stats": extraction_stats,
        "n_instances": len(instances),
        "n_target_keys": len(target_keys),
        "n_target_primitives": len(target_primitives),
        "n_missing_keys": len(missing_keys),
        "missing_keys": [key.to_json() for key in missing_keys],
        "n_cards": len(card_rows),
    }
    write_jsonl(output_dir / "relation_cards.jsonl", card_rows)
    write_jsonl(output_dir / "target_primitives.jsonl", target_rows)
    write_json(output_dir / "target_summary.json", summary)
    (output_dir / "prompts_used.md").write_text(prompt_templates_markdown("not_applicable"), encoding="utf-8")
    log_line("relation_cards.jsonl")
    log_line("target_primitives.jsonl")
    log_line("target_summary.json")
    print(f"Wrote targeted RC-MEX cards to {output_dir}")


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "questions", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise SystemExit(f"unsupported programs file format: {path}")


def ordered_unique_keys(keys: list[PrimitiveKey]) -> list[PrimitiveKey]:
    out = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def target_summary(primitive: Primitive, samples: PrimitiveSamples) -> dict[str, Any]:
    return {
        "primitive_id": primitive.primitive_id,
        "relation_id": primitive.relation_id,
        "direction": primitive.direction,
        "normalized_key": primitive_key(primitive.relation_id, primitive.direction).to_json(),
        "n_examples": primitive.cardinality,
        "n_positive_train": len(samples.positive_train),
        "n_positive_heldout": len(samples.positive_heldout),
        "n_hard_negative_train": len(samples.hard_negative_train),
        "n_hard_negative_heldout": len(samples.hard_negative_heldout),
        "n_random_negative_heldout": len(samples.random_negative_heldout),
        "hard_negative_source_ids": samples.hard_negative_source_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted RC-MEX relation cards from gold KoPL Relate slots.")
    parser.add_argument("--kb", required=True)
    parser.add_argument("--programs", required=True, help="KQA Pro train/val JSON with gold programs")
    parser.add_argument("--output", default="runs/rc_mex_cards_for_mvp2")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--max-slots", type=int, default=100)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--min-examples", type=int, default=4)
    parser.add_argument("--train-positives", type=int, default=4)
    parser.add_argument("--heldout-positives", type=int, default=4)
    parser.add_argument("--train-negatives", type=int, default=4)
    parser.add_argument("--heldout-negatives", type=int, default=4)
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--conditions", default="B1,B2")
    parser.add_argument("--card-variants", default="contrastive_hard")
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
    parser.add_argument("--exclude-metadata-relations", action="store_true")
    parser.add_argument("--metadata-relation-patterns", default=None)
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_OPENAI_MODEL if args.oracle_backend == "openai" else DEFAULT_LOCAL_MODEL
    for condition in split_csv(args.conditions):
        if condition not in CONDITIONS:
            raise SystemExit(f"unknown condition: {condition}")
    for variant in split_csv(args.card_variants):
        if variant not in CARD_VARIANTS:
            raise SystemExit(f"unknown card variant: {variant}")
    return args


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


if __name__ == "__main__":
    main()
