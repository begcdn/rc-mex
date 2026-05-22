from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph

from .cards import RelationCard
from .evidence import CONDITIONS, EvidenceCondition, RenderContext, render_example, render_type_summary
from .metrics import compute_metrics
from .oracle import CardGenerator, PairClassifier, make_client
from .sampling import PrimitiveSamples, sample_for_primitive
from .schema import Primitive, RelationExample, inventory_primitives


CARD_VARIANTS = {"contrastive_hard", "random_negative", "name_only"}
DEFAULT_LOCAL_MODEL = "llama3.1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    graph = KnowledgeGraph(load_json(args.kb))
    primitives = inventory_primitives(graph, min_examples=args.min_examples)
    primitives = primitives[: args.max_primitives] if args.max_primitives else primitives
    context = RenderContext.from_primitives(primitives)
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
    sample_rows: list[dict[str, Any]] = []
    condition_ids = split_csv(args.conditions)
    variants = split_csv(args.card_variants)

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
        sample_rows.append(sample_summary(primitive, samples))
        if not samples.positive_train or not samples.positive_heldout:
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
                for validation in validation_examples(graph, context, condition, samples):
                    prediction = validator.classify(
                        card=card,
                        pair=validation["pair"],
                        expected_label=validation["expected_label"],
                    )
                    prediction_rows.append(
                        {
                            "primitive_id": primitive.primitive_id,
                            "relation_id": primitive.relation_id,
                            "direction": primitive.direction,
                            "condition_id": condition_id,
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
                            "raw_output": prediction.raw_output if args.save_raw_outputs else "",
                        }
                    )

    metrics = compute_metrics(card_rows, prediction_rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "n_primitives_inventoried": len(primitives),
        "n_cards": len(card_rows),
        "n_validation_predictions": len(prediction_rows),
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
        default="mock",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--oracle-command", default=None)
    parser.add_argument("--save-raw-outputs", action="store_true")
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
