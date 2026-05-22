from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph

from .cards import RelationCard
from .debug import (
    graph_debug_stats,
    is_metadata_relation,
    pair_text,
    parse_metadata_patterns,
    prediction_result,
    short_diagnosis,
    type_text,
    write_debug_artifacts,
)
from .evidence import CONDITIONS, EvidenceCondition, RenderContext, render_example, render_type_summary
from .metrics import compute_metrics
from .oracle import CardGenerator, PairClassifier, make_client
from .sampling import PrimitiveSamples, sample_for_primitive
from .schema import Primitive, RelationExample, inventory_primitives


CARD_VARIANTS = {"contrastive_hard", "random_negative", "name_only"}
DEFAULT_LOCAL_MODEL = "llama3:8b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


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

    log_stage(2, 6, "Building primitive inventory")
    primitives = inventory_primitives(graph, min_examples=args.min_examples)
    before_filter = len(primitives)
    if args.exclude_metadata_relations:
        primitives = [
            primitive for primitive in primitives
            if not is_metadata_relation(primitive.relation_id, metadata_patterns)
        ]
    primitives = primitives[: args.max_primitives] if args.max_primitives else primitives
    context = RenderContext.from_primitives(primitives)
    log_line(f"Found {len(primitives)} relation+direction primitives")
    if args.exclude_metadata_relations:
        log_line(f"Excluded metadata-looking primitives: {before_filter - len(primitives)}")
    if primitives:
        example = primitives[0]
        log_line(f'Example: {example.primitive_id} | relation="{example.relation_id}" | direction={example.direction}')

    log_stage(3, 6, "Sampling examples")
    primitive_samples: dict[str, PrimitiveSamples] = {}
    sample_rows: list[dict[str, Any]] = []
    for primitive in primitives:
        samples = sample_for_primitive(
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
        primitive_samples[primitive.primitive_id] = samples
        sample_rows.append(sample_summary(primitive, samples))
        if args.verbose:
            log_line(
                f"Primitive {primitive.primitive_id}: "
                f"{len(samples.positive_train)} train positives, "
                f"{len(samples.positive_heldout)} heldout positives"
            )
            log_line(
                f"Hard negatives: {len(samples.hard_negative_train)} train, "
                f"{len(samples.hard_negative_heldout)} heldout"
            )
            log_line(f"Random negatives: {len(samples.random_negative_heldout)} heldout")

    client = make_client(
        backend=args.oracle_backend,
        model=args.model,
        ollama_host=args.ollama_host,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
        command=args.oracle_command,
    )
    generator = CardGenerator(client)
    validator = PairClassifier(client, include_mock_label=args.oracle_backend == "mock")

    card_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    condition_ids = split_csv(args.conditions)
    variants = split_csv(args.card_variants)
    card_jobs: list[tuple[Primitive, PrimitiveSamples, EvidenceCondition, str, RelationCard]] = []

    log_stage(4, 6, "Generating relation cards")
    for primitive in primitives:
        samples = primitive_samples[primitive.primitive_id]
        if not samples.positive_train or not samples.positive_heldout:
            if args.verbose:
                log_line(f"Skipping {primitive.primitive_id}: missing train or heldout positives")
            continue
        for condition_id in condition_ids:
            condition = CONDITIONS[condition_id]
            for variant in variants:
                card = build_card(
                    graph=graph,
                    context=context,
                    condition=condition,
                    primitive=primitive,
                    samples=samples,
                    card_variant=variant,
                    generator=generator,
                )
                card_json = card.to_json()
                card_json["condition_id"] = condition_id
                card_rows.append(card_json)
                card_jobs.append((primitive, samples, condition, variant, card))
                if args.verbose:
                    visible_relation = context.relation_display(primitive.relation_id, condition)
                    log_line(f"Condition: {condition_id} | Variant: {variant}")
                    log_line(f"Primitive: {primitive.primitive_id} | relation={visible_relation} | direction={primitive.direction}")
                    log_line(f"Card description: {card.description}")
                    log_line(f"Confidence: {card.confidence:.3f}")

    log_line(f"Generated {len(card_rows)} cards")

    log_stage(5, 6, "Validating cards")
    validation_print_counts: dict[str, int] = {}
    for primitive, samples, condition, variant, card in card_jobs:
        for validation in validation_examples(graph, context, condition, samples):
            prediction = validator.classify(
                card=card,
                pair=validation["pair"],
                expected_label=validation["expected_label"],
            )
            row = {
                "primitive_id": primitive.primitive_id,
                "relation_id": primitive.relation_id,
                "direction": primitive.direction,
                "condition_id": condition.condition_id,
                "card_variant": variant,
                "category": validation["category"],
                "expected_label": validation["expected_label"],
                "pair": validation["pair"],
                "predicted_satisfies": prediction.satisfies,
                "predicted_direction_correct": prediction.direction_correct,
                "confidence": prediction.confidence,
                "prompt_tokens": prediction.prompt_tokens_estimate,
                "completion_tokens": prediction.completion_tokens_estimate,
                "latency_seconds": prediction.latency_seconds,
                "result": "",
                "diagnosis": "",
                "raw_output": prediction.raw_output if args.save_raw_outputs else "",
            }
            row["result"] = prediction_result(row)
            row["diagnosis"] = short_diagnosis(row)
            prediction_rows.append(row)
            count = validation_print_counts.get(primitive.primitive_id, 0)
            if args.verbose and count < args.debug_examples_per_primitive:
                log_line(f"Pair: {pair_text(validation['pair'])}")
                log_line(f"Types: {type_text(validation['pair'])}")
                log_line(f"Expected: {str(validation['expected_label']).upper()}")
                log_line(f"Predicted: {str(prediction.satisfies).upper()}")
                log_line(f"Category: {validation['category']}")
                log_line(f"Result: {row['result']}")
                validation_print_counts[primitive.primitive_id] = count + 1
    log_line(f"Validated {len(prediction_rows)} pairs")

    log_stage(6, 6, "Writing reports")
    metrics = compute_metrics(card_rows, prediction_rows)
    debug_data = write_debug_artifacts(
        output_dir=output_dir,
        cards=card_rows,
        predictions=prediction_rows,
        metrics=metrics,
        metadata_patterns=metadata_patterns,
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "n_primitives_inventoried": len(primitives),
        "n_cards": len(card_rows),
        "n_validation_predictions": len(prediction_rows),
        "n_primitive_metric_rows": len(debug_data["primitive_metrics"]),
        "metrics": metrics,
        "notes": [
            "Mock backend is for smoke tests only, not valid experimental evidence.",
            "MVP1 validates cards by held-out pair classification, not by whether cards sound plausible.",
        ],
    }
    write_jsonl(output_dir / "relation_cards.jsonl", card_rows)
    write_jsonl(output_dir / "validation_predictions.jsonl", prediction_rows)
    write_jsonl(output_dir / "primitive_samples.jsonl", sample_rows)
    write_json(output_dir / "metrics.json", summary)
    write_report(output_dir / "report.md", summary)
    for filename in [
        "relation_cards.jsonl",
        "validation_predictions.jsonl",
        "primitive_samples.jsonl",
        "metrics.json",
        "report.md",
        "examples_summary.json",
        "primitive_metrics.jsonl",
        "debug_examples.md",
        "debug_report.html",
    ]:
        log_line(filename)
    print(f"Wrote RC-MEX MVP1 outputs to {output_dir}")


def build_card(
    graph: KnowledgeGraph,
    context: RenderContext,
    condition: EvidenceCondition,
    primitive: Primitive,
    samples: PrimitiveSamples,
    card_variant: str,
    generator: CardGenerator,
) -> RelationCard:
    domain_types, range_types = render_type_summary(graph, primitive, condition)
    positive_train = [render_example(graph, context, condition, example) for example in samples.positive_train]
    positive_heldout = [render_example(graph, context, condition, example) for example in samples.positive_heldout]
    hard_train = [render_example(graph, context, condition, example) for example in samples.hard_negative_train]
    hard_heldout = [render_example(graph, context, condition, example) for example in samples.hard_negative_heldout]
    random_heldout = [render_example(graph, context, condition, example) for example in samples.random_negative_heldout]
    swapped_heldout = [render_example(graph, context, condition, example) for example in samples.swapped_direction_heldout]
    negative_train = []
    if card_variant == "contrastive_hard":
        negative_train = hard_train
    elif card_variant == "random_negative":
        negative_train = [render_example(graph, context, condition, example) for example in samples.random_negative_heldout[: len(hard_train)]]
    elif card_variant == "name_only":
        positive_train = []
        negative_train = []
    else:
        raise ValueError(f"unknown card variant: {card_variant}")

    payload = {
        "card_variant": card_variant,
        "visible_relation": context.relation_display(primitive.relation_id, condition),
        "direction": primitive.direction,
        "domain_types": domain_types,
        "range_types": range_types,
        "positive_examples": positive_train,
        "negative_examples": negative_train,
    }
    generated = generator.generate(payload)
    opaque_reason = generated.opaque_reason if generated.opaque else ""
    return RelationCard(
        primitive_id=primitive.primitive_id,
        relation_id=primitive.relation_id,
        direction=primitive.direction,
        card_variant=card_variant,
        obfuscation_mode=condition.obfuscation_mode,
        entity_evidence_mode=condition.entity_evidence_mode,
        positive_examples_train=positive_train,
        positive_examples_heldout=positive_heldout,
        hard_negative_examples_train=hard_train,
        hard_negative_examples_heldout=hard_heldout,
        random_negative_examples_heldout=random_heldout,
        swapped_direction_examples_heldout=swapped_heldout,
        description=generated.predicate_description,
        domain_types=domain_types,
        range_types=range_types,
        argument_direction=generated.direction,
        confidence=generated.confidence,
        opaque_reason=opaque_reason,
        generated={
            "predicate_description": generated.predicate_description,
            "argument_1_role": generated.argument_1_role,
            "argument_2_role": generated.argument_2_role,
            "domain": generated.domain,
            "range": generated.range,
            "direction": generated.direction,
            "confidence": generated.confidence,
            "opaque": generated.opaque,
            "opaque_reason": generated.opaque_reason,
            "raw_output": generated.raw_output,
        },
    )


def validation_examples(
    graph: KnowledgeGraph,
    context: RenderContext,
    condition: EvidenceCondition,
    samples: PrimitiveSamples,
) -> list[dict[str, Any]]:
    rows = []
    rows.extend(make_validation_rows(graph, context, condition, samples.positive_heldout, "positive", True))
    rows.extend(make_validation_rows(graph, context, condition, samples.hard_negative_heldout, "hard_negative", False))
    rows.extend(make_validation_rows(graph, context, condition, samples.random_negative_heldout, "random_negative", False))
    rows.extend(make_validation_rows(graph, context, condition, samples.swapped_direction_heldout, "swapped_direction", False))
    return rows


def make_validation_rows(
    graph: KnowledgeGraph,
    context: RenderContext,
    condition: EvidenceCondition,
    examples: list[RelationExample],
    category: str,
    expected_label: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "expected_label": expected_label,
            "pair": render_example(graph, context, condition, example),
        }
        for example in examples
    ]


def sample_summary(primitive: Primitive, samples: PrimitiveSamples) -> dict[str, Any]:
    return {
        "primitive_id": primitive.primitive_id,
        "relation_id": primitive.relation_id,
        "direction": primitive.direction,
        "n_examples": primitive.cardinality,
        "n_positive_train": len(samples.positive_train),
        "n_positive_heldout": len(samples.positive_heldout),
        "n_hard_negative_train": len(samples.hard_negative_train),
        "n_hard_negative_heldout": len(samples.hard_negative_heldout),
        "n_random_negative_heldout": len(samples.random_negative_heldout),
        "n_swapped_direction_heldout": len(samples.swapped_direction_heldout),
        "hard_negative_source_ids": samples.hard_negative_source_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RC-MEX MVP1 relation-card induction.")
    parser.add_argument("--kb", required=True, help="Path to KQA Pro-style kb.json")
    parser.add_argument("--output", default="runs/rc_mex_mvp1")
    parser.add_argument("--max-primitives", type=int, default=20)
    parser.add_argument("--min-examples", type=int, default=4)
    parser.add_argument("--train-positives", type=int, default=4)
    parser.add_argument("--heldout-positives", type=int, default=4)
    parser.add_argument("--train-negatives", type=int, default=4)
    parser.add_argument("--heldout-negatives", type=int, default=4)
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
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
    parser.add_argument("--verbose", action="store_true", help="Print primitive-level debug progress.")
    parser.add_argument(
        "--debug-examples-per-primitive",
        type=int,
        default=3,
        help="Number of validation examples to print per primitive when --verbose is enabled.",
    )
    parser.add_argument(
        "--exclude-metadata-relations",
        action="store_true",
        help="Skip metadata-looking primitives such as external IDs, URLs, and source relations.",
    )
    parser.add_argument(
        "--metadata-relation-patterns",
        default=None,
        help="Comma-separated regex patterns used to label or exclude metadata-looking relations.",
    )
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


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def log_stage(index: int, total: int, message: str) -> None:
    print(f"[{index}/{total}] {message}", flush=True)


def log_line(message: str) -> None:
    print(f"      {message}", flush=True)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RC-MEX MVP1 Report",
        "",
        "MVP1 validates relation cards by held-out pair classification.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(
            {
                "n_primitives_inventoried": summary["n_primitives_inventoried"],
                "n_cards": summary["n_cards"],
                "n_validation_predictions": summary["n_validation_predictions"],
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


if __name__ == "__main__":
    main()
