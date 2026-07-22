from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import delexicalize_question, read_jsonl, write_jsonl
from .model import generate_joint_questions, load_seq2seq


def relation_keys(row: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    path = row["positive_path"]
    kg = path.get("kg", row.get("kg", "unknown"))
    return tuple((kg, hop["relation"], hop["direction"]) for hop in path["hops"])


def derive_generalization_slices(
    train_rows: list[dict[str, Any]],
    evaluation_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    training_paths = []
    for row in train_rows:
        training_paths.append({"positive_path": row["positive_path"]})
        training_paths.extend(
            {"positive_path": path}
            for path in [
                *row.get("negative_paths", []),
                *row.get("contrast_only_negative_paths", []),
            ]
        )
    train_relations = {key for row in training_paths for key in relation_keys(row)}
    train_compositions = {relation_keys(row) for row in training_paths}

    unique_rows: dict[str, dict[str, Any]] = {}
    for rows in evaluation_rows.values():
        for row in rows:
            unique_rows.setdefault(row["example_id"], row)

    strict_unseen_relation = []
    strict_unseen_composition = []
    for row in unique_rows.values():
        keys = relation_keys(row)
        if any(key not in train_relations for key in keys):
            strict_unseen_relation.append(row)
        elif len(keys) > 1 and keys not in train_compositions:
            strict_unseen_composition.append(row)

    slices = {
        **evaluation_rows,
        "strict_unseen_relation": strict_unseen_relation,
        "strict_unseen_composition": strict_unseen_composition,
    }
    coverage = {
        "train_examples": len(train_rows),
        "train_paths_seen_by_loss": len(training_paths),
        "train_relation_direction_keys": len(train_relations),
        "train_compositions": len(train_compositions),
        "unique_evaluation_examples": len(unique_rows),
        "slice_examples": {name: len(rows) for name, rows in slices.items()},
        "strict_definition": {
            "unseen_relation": "at least one KG-scoped relation+direction key absent from training",
            "unseen_composition": (
                "all KG-scoped relation+direction keys seen in training, but the ordered multi-hop "
                "composition absent from training"
            ),
        },
    }
    return slices, coverage


def evaluate_slice(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    model_device: Any,
    encoder: SentenceTransformer,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths: list[dict[str, Any]] = []
    locations: list[tuple[int, str, bool]] = []
    for row_index, row in enumerate(rows):
        paths.append(row["positive_path"])
        locations.append((row_index, "positive", True))
        for negative in row.get("negative_paths", []):
            paths.append(negative)
            locations.append((row_index, negative.get("negative_type", "negative"), False))

    generated = generate_joint_questions(
        model, tokenizer, paths, model_device, batch_size=batch_size
    )
    references = [
        delexicalize_question(rows[row_index]["question"], path["anchor"])
        for path, (row_index, _, _) in zip(paths, locations, strict=True)
    ]
    intents = [
        delexicalize_question(question, path["anchor"])
        for question, path in zip(generated, paths, strict=True)
    ]
    reference_embeddings = encoder.encode(
        references, batch_size=batch_size, normalize_embeddings=True
    )
    intent_embeddings = encoder.encode(
        intents, batch_size=batch_size, normalize_embeddings=True
    )
    similarities = np.sum(reference_embeddings * intent_embeddings, axis=1).tolist()

    candidates_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path, question, intent, similarity, (row_index, category, is_positive) in zip(
        paths, generated, intents, similarities, locations, strict=True
    ):
        candidates_by_row[row_index].append(
            {
                "category": category,
                "is_positive": is_positive,
                "generated_question": question,
                "generated_intent": intent,
                "semantic_similarity": float(similarity),
                "path": path,
            }
        )

    predictions = []
    category_results: dict[str, list[bool]] = defaultdict(list)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    for row_index, row in enumerate(rows):
        candidates = candidates_by_row[row_index]
        positive_score = candidates[0]["semantic_similarity"]
        negatives = candidates[1:]
        positive_scores.append(positive_score)
        negative_scores.extend(item["semantic_similarity"] for item in negatives)
        for item in negatives:
            category_results[item["category"]].append(
                positive_score > item["semantic_similarity"]
            )
        beats_all = all(positive_score > item["semantic_similarity"] for item in negatives)
        margin = (
            positive_score - max(item["semantic_similarity"] for item in negatives)
            if negatives
            else 0.0
        )
        margins.append(margin)
        predictions.append(
            {
                "example_id": row["example_id"],
                "question": row["question"],
                "kg": row.get("kg", row["positive_path"].get("kg")),
                "relation_sequence": list(relation_keys(row)),
                "positive_beats_all_negatives": beats_all,
                "positive_margin_over_best_negative": margin,
                "candidates": candidates,
            }
        )

    pair_count = sum(len(values) for values in category_results.values())
    metrics = {
        "examples": len(rows),
        "candidate_pairs": pair_count,
        "positive_beats_all_negatives": (
            sum(row["positive_beats_all_negatives"] for row in predictions) / len(predictions)
            if predictions
            else 0.0
        ),
        "pairwise_accuracy": (
            sum(sum(values) for values in category_results.values()) / pair_count
            if pair_count
            else 0.0
        ),
        "mean_positive_similarity": float(np.mean(positive_scores)) if positive_scores else 0.0,
        "mean_negative_similarity": float(np.mean(negative_scores)) if negative_scores else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "by_negative_type": {
            category: {
                "pairs": len(values),
                "pairwise_accuracy": sum(values) / len(values),
            }
            for category, values in sorted(category_results.items())
        },
    }
    return metrics, predictions


def evaluate_faithful_generalization(
    train_data: Path,
    evaluation_data: Path,
    model_path: str,
    semantic_model: str,
    output: Path,
    split_names: list[str],
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    train_rows = read_jsonl(train_data)
    evaluation_rows = {
        name: read_jsonl(evaluation_data / f"{name}.jsonl") for name in split_names
    }
    slices, coverage = derive_generalization_slices(train_rows, evaluation_rows)

    model, tokenizer, model_device = load_seq2seq(model_path, device)
    encoder = SentenceTransformer(semantic_model, local_files_only=True)
    metrics: dict[str, Any] = {
        "model": model_path,
        "semantic_model": semantic_model,
        "device": str(model_device),
        "training_relative_coverage": coverage,
        "slices": {},
    }
    prediction_rows = []
    for name, rows in slices.items():
        print(f"Evaluating {name}: {len(rows)} examples", flush=True)
        slice_metrics, slice_predictions = evaluate_slice(
            rows, model, tokenizer, model_device, encoder, batch_size
        )
        metrics["slices"][name] = slice_metrics
        prediction_rows.extend({"slice": name, **row} for row in slice_predictions)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", prediction_rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# Faithful Inverse Generalization",
        "",
        "Strict relation and composition slices are derived from the actual training corpus, not old split names.",
        "",
        "| Slice | Examples | Gold beats all | Pair accuracy | Gold similarity | Margin |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics["slices"].items():
        lines.append(
            f"| {name} | {values['examples']} | {values['positive_beats_all_negatives']:.3f} | "
            f"{values['pairwise_accuracy']:.3f} | {values['mean_positive_similarity']:.3f} | "
            f"{values['mean_margin']:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
