from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import delexicalize_question, read_jsonl, render_path, write_jsonl
from .metrics import ranking_metrics, rouge_l, token_f1
from .model import (
    generate_joint_questions,
    generate_questions,
    informative_answer_type,
    load_joint_ranker,
    load_seq2seq,
    score_direct_relevance,
    score_joint_relevance,
    score_questions_given_paths,
    type_compatibility_scores,
)


def flatten_candidates(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]], list[tuple[int, int]]]:
    questions: list[str] = []
    paths: list[dict[str, Any]] = []
    locations: list[tuple[int, int]] = []
    for row_index, row in enumerate(rows):
        if "candidates" not in row:
            candidates = [{"path": row["positive_path"], "is_positive": True, "negative_type": "positive"}]
            candidates.extend(
                {"path": path, "is_positive": True, "negative_type": "alternate_positive"}
                for path in row.get("alternate_positive_paths", [])
            )
            candidates.extend(
                {"path": path, "is_positive": False, "negative_type": path["negative_type"]}
                for path in row["negative_paths"]
            )
            row["candidates"] = candidates
        candidates = row["candidates"]
        for candidate_index, candidate in enumerate(candidates):
            questions.append(row["question"])
            paths.append(candidate["path"])
            locations.append((row_index, candidate_index))
    return questions, paths, locations


def attach_scores(
    rows: list[dict[str, Any]],
    locations: list[tuple[int, int]],
    scores: list[float],
    key: str,
) -> None:
    for (row_index, candidate_index), score in zip(locations, scores, strict=True):
        rows[row_index]["candidates"][candidate_index][key] = float(score)


def direct_similarity_scores(
    encoder: SentenceTransformer,
    questions: list[str],
    paths: list[dict[str, Any]],
    batch_size: int,
) -> list[float]:
    question_embeddings = encoder.encode(questions, batch_size=batch_size, normalize_embeddings=True)
    path_embeddings = encoder.encode(
        [render_path(path, include_instruction=False) for path in paths],
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    return np.sum(question_embeddings * path_embeddings, axis=1).tolist()


def generation_metrics(
    references: list[str],
    predictions: list[str],
    encoder: SentenceTransformer,
) -> dict[str, float]:
    if not references:
        return {"examples": 0, "token_f1": 0.0, "rouge_l": 0.0, "semantic_similarity": 0.0}
    reference_embeddings = encoder.encode(references, normalize_embeddings=True)
    prediction_embeddings = encoder.encode(predictions, normalize_embeddings=True)
    similarities = np.sum(reference_embeddings * prediction_embeddings, axis=1)
    return {
        "examples": len(references),
        "token_f1": sum(token_f1(pred, ref) for pred, ref in zip(predictions, references)) / len(references),
        "rouge_l": sum(rouge_l(pred, ref) for pred, ref in zip(predictions, references)) / len(references),
        "semantic_similarity": float(np.mean(similarities)),
    }


def similarity_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"examples": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "examples": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "at_least_0_6": float(np.mean(array >= 0.6)),
        "at_least_0_7": float(np.mean(array >= 0.7)),
        "at_least_0_8": float(np.mean(array >= 0.8)),
        "at_least_0_9": float(np.mean(array >= 0.9)),
    }


def evaluate_gold_generation(
    data_dir: Path,
    output: Path,
    model_path: str,
    splits: list[str],
    device: str = "auto",
    batch_size: int = 16,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Measure only gold-path-to-question generation; no candidates or ranking."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, tokenizer, model_device = load_seq2seq(model_path, device)
    comparators = {
        "bge_small": SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True),
        "minilm": SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        ),
    }
    metrics: dict[str, Any] = {
        "model": model_path,
        "device": str(model_device),
        "test": "gold_path_to_question_generation",
        "candidate_search_used": False,
        "path_ranking_used": False,
        "entity_names_masked_for_generation_and_similarity": True,
        "splits": {},
    }
    predictions = []
    for split in splits:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        if limit_per_split:
            rows = rows[:limit_per_split]
        paths = [row["positive_path"] for row in rows]
        print(f"Generating {len(paths)} gold-path questions for {split}", flush=True)
        generated = generate_joint_questions(
            model,
            tokenizer,
            paths,
            model_device,
            batch_size=max(1, batch_size // 2),
        )
        references = [
            delexicalize_question(row["question"], path["anchor"])
            for row, path in zip(rows, paths, strict=True)
        ]
        generated_intents = [
            delexicalize_question(question, path["anchor"])
            for question, path in zip(generated, paths, strict=True)
        ]
        type_indices = [
            index
            for index, path in enumerate(paths)
            if informative_answer_type(path.get("answer_type", ""))
        ]
        type_questions = [references[index] for index in type_indices]
        generated_type_questions = [generated_intents[index] for index in type_indices]
        positive_types = [paths[index]["answer_type"] for index in type_indices]
        reference_type_scores = type_compatibility_scores(
            model,
            tokenizer,
            type_questions,
            positive_types,
            model_device,
            batch_size=batch_size,
        )
        generated_type_scores = type_compatibility_scores(
            model,
            tokenizer,
            generated_type_questions,
            positive_types,
            model_device,
            batch_size=batch_size,
        )
        negative_type_items = []
        for index in type_indices:
            wrong = next(
                (
                    candidate.get("answer_type", "")
                    for candidate in rows[index].get("negative_paths", [])
                    if candidate.get("negative_type") == "wrong_answer_type"
                    and informative_answer_type(candidate.get("answer_type", ""))
                    and candidate.get("answer_type", "").casefold()
                    != paths[index]["answer_type"].casefold()
                ),
                None,
            )
            if wrong is not None:
                negative_type_items.append((index, wrong))
        negative_type_scores = type_compatibility_scores(
            model,
            tokenizer,
            [references[index] for index, _ in negative_type_items],
            [answer_type for _, answer_type in negative_type_items],
            model_device,
            batch_size=batch_size,
        )
        split_metrics: dict[str, Any] = {
            "examples": len(rows),
            "token_f1": float(
                np.mean(
                    [token_f1(prediction, reference) for prediction, reference in zip(generated_intents, references, strict=True)]
                )
            ) if rows else 0.0,
            "rouge_l": float(
                np.mean(
                    [rouge_l(prediction, reference) for prediction, reference in zip(generated_intents, references, strict=True)]
                )
            ) if rows else 0.0,
            "by_hops": {},
            "answer_type_compatibility": {
                "informative_positive_examples": len(type_indices),
                "reference_question_positive_accuracy": float(
                    np.mean([score >= 0.5 for score in reference_type_scores])
                ) if reference_type_scores else 0.0,
                "generated_question_positive_accuracy": float(
                    np.mean([score >= 0.5 for score in generated_type_scores])
                ) if generated_type_scores else 0.0,
                "wrong_type_examples": len(negative_type_scores),
                "wrong_type_rejection_accuracy": float(
                    np.mean([score < 0.5 for score in negative_type_scores])
                ) if negative_type_scores else 0.0,
            },
        }
        comparator_scores: dict[str, list[float]] = {}
        for name, comparator in comparators.items():
            reference_embeddings = comparator.encode(
                references, batch_size=batch_size, normalize_embeddings=True
            )
            generated_embeddings = comparator.encode(
                generated_intents, batch_size=batch_size, normalize_embeddings=True
            )
            scores = np.sum(reference_embeddings * generated_embeddings, axis=1).tolist()
            comparator_scores[name] = scores
            split_metrics[name] = similarity_distribution(scores)
        for hop_count in sorted({len(path["hops"]) for path in paths}):
            indices = [index for index, path in enumerate(paths) if len(path["hops"]) == hop_count]
            split_metrics["by_hops"][str(hop_count)] = {
                name: similarity_distribution([scores[index] for index in indices])
                for name, scores in comparator_scores.items()
            }
        metrics["splits"][split] = split_metrics
        type_score_by_index = {
            index: {
                "reference_question": reference_type_scores[position],
                "generated_question": generated_type_scores[position],
            }
            for position, index in enumerate(type_indices)
        }
        negative_score_by_index = {
            index: {"answer_type": answer_type, "score": negative_type_scores[position]}
            for position, (index, answer_type) in enumerate(negative_type_items)
        }
        for index, (row, path, question, reference, generated_intent) in enumerate(
            zip(rows, paths, generated, references, generated_intents, strict=True)
        ):
            predictions.append(
                {
                    "split": split,
                    "example_id": row["example_id"],
                    "gold_path": path,
                    "reference_question": row["question"],
                    "reference_intent": reference,
                    "generated_question": question,
                    "generated_intent": generated_intent,
                    "semantic_similarity": {
                        name: scores[index] for name, scores in comparator_scores.items()
                    },
                    "answer_type_compatibility": type_score_by_index.get(index),
                    "wrong_answer_type_probe": negative_score_by_index.get(index),
                }
            )

    metrics["elapsed_seconds"] = time.time() - started
    write_jsonl(output / "gold_path_generations.jsonl", predictions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# Gold Path to Question Generation",
        "",
        "This test supplies only an annotated correct path, generates its question, masks the topic entity in both texts, and measures semantic similarity. It performs no candidate search or path ranking.",
        "",
        "| Split | N | BGE mean | >=0.8 | Generated/type agree | Wrong type rejected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in metrics["splits"].items():
        bge = row["bge_small"]
        lines.append(
            f"| {split} | {row['examples']} | {bge['mean']:.3f} | "
            f"{bge['at_least_0_8']:.3f} | "
            f"{row['answer_type_compatibility']['generated_question_positive_accuracy']:.3f} | "
            f"{row['answer_type_compatibility']['wrong_type_rejection_accuracy']:.3f} |"
        )
    lines.extend(["", "## Lowest-Similarity Generations", ""])
    for row in sorted(
        predictions, key=lambda item: item["semantic_similarity"]["bge_small"]
    )[:20]:
        relations = " -> ".join(
            f"{hop['relation']} ({hop['direction']})" for hop in row["gold_path"]["hops"]
        )
        lines.extend(
            [
                f"### {row['example_id']} ({row['semantic_similarity']['bge_small']:.3f})",
                f"- Path: {relations}",
                f"- Reference: {row['reference_question']}",
                f"- Generated: {row['generated_question']}",
                "",
            ]
        )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def generated_question_similarity_scores(
    encoder: SentenceTransformer,
    questions: list[str],
    generated_questions: list[str],
    paths: list[dict[str, Any]],
    batch_size: int,
) -> list[float]:
    reference_intents = [
        delexicalize_question(question, path["anchor"])
        for question, path in zip(questions, paths, strict=True)
    ]
    generated_intents = [
        delexicalize_question(question, path["anchor"])
        for question, path in zip(generated_questions, paths, strict=True)
    ]
    question_embeddings = encoder.encode(
        reference_intents, batch_size=batch_size, normalize_embeddings=True
    )
    generated_embeddings = encoder.encode(
        generated_intents, batch_size=batch_size, normalize_embeddings=True
    )
    return np.sum(question_embeddings * generated_embeddings, axis=1).tolist()


def evaluate(
    data_dir: Path,
    output: Path,
    trained_model: str,
    base_model: str,
    splits: list[str],
    device: str = "auto",
    batch_size: int = 16,
    limit_per_split: int | None = None,
    generation_examples: int = 64,
    include_pretrained: bool = True,
    direct_model: str | None = None,
    joint_model: str | None = None,
    ranker_model: str | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {"splits": {}, "models": {}}
    started = time.time()
    # Decoding every candidate is an explanatory ablation, not the primary ranker.
    # Keep it to genuinely small pools; executable graph frontiers can contain
    # tens of thousands of paths even in a modest question sample.
    generated_ranking_candidate_limit = 5_000
    training_metadata_path = Path(trained_model).parent / "training.json"
    if training_metadata_path.exists():
        metrics["trained_model"] = json.loads(training_metadata_path.read_text(encoding="utf-8"))

    for split in splits:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        if limit_per_split:
            rows = rows[:limit_per_split]
        questions, paths, locations = flatten_candidates(rows)
        direct_scores = direct_similarity_scores(encoder, questions, paths, batch_size)
        attach_scores(rows, locations, direct_scores, "direct_similarity")
        all_rows[split] = rows
        metrics["splits"][split] = {
            "examples": len(rows),
            "direct_similarity": ranking_metrics(rows, "direct_similarity"),
        }

    systems = [("inverse_trained", trained_model)]
    if include_pretrained:
        systems.insert(0, ("inverse_pretrained", base_model))
    for system_name, model_path in systems:
        print(f"Loading {system_name}: {model_path}", flush=True)
        model, tokenizer, model_device = load_seq2seq(model_path, device)
        metrics["models"][system_name] = {"path": model_path, "device": str(model_device)}
        for split, rows in all_rows.items():
            print(f"Scoring {split} with {system_name} ({len(rows)} examples)", flush=True)
            questions, paths, locations = flatten_candidates(rows)
            scores = score_questions_given_paths(
                model, tokenizer, questions, paths, model_device, batch_size=batch_size
            )
            attach_scores(rows, locations, scores, system_name)
            split_metrics = ranking_metrics(rows, system_name)
            sample = rows[: min(generation_examples, len(rows))]
            if system_name == "inverse_trained" and len(paths) <= generated_ranking_candidate_limit:
                generated_all = generate_questions(
                    model,
                    tokenizer,
                    paths,
                    model_device,
                    batch_size=max(1, batch_size // 2),
                    num_beams=1,
                )
                generated_scores = generated_question_similarity_scores(
                    encoder, questions, generated_all, paths, batch_size
                )
                attach_scores(rows, locations, generated_scores, "inverse_generated_similarity")
                for (row_index, candidate_index), generated_question in zip(
                    locations, generated_all, strict=True
                ):
                    rows[row_index]["candidates"][candidate_index]["generated_question"] = generated_question
                metrics["splits"][split]["inverse_generated_similarity"] = ranking_metrics(
                    rows, "inverse_generated_similarity"
                )
                generated = [row["candidates"][0]["generated_question"] for row in sample]
            else:
                generated = generate_questions(
                    model,
                    tokenizer,
                    [row["positive_path"] for row in sample],
                    model_device,
                    batch_size=max(1, batch_size // 2),
                )
                if system_name == "inverse_trained":
                    metrics["splits"][split]["generated_ranking_skipped"] = {
                        "candidate_paths": len(paths),
                        "limit": generated_ranking_candidate_limit,
                        "reason": "candidate pool too large for decoding every path",
                    }
            for row, question in zip(sample, generated, strict=True):
                row.setdefault("generated_questions", {})[system_name] = question
            split_metrics["generation"] = generation_metrics(
                [row["question"] for row in sample], generated, encoder
            )
            metrics["splits"][split][system_name] = split_metrics
        del model
        gc.collect()

    if direct_model:
        print(f"Loading direct_trained: {direct_model}", flush=True)
        model, tokenizer, model_device = load_seq2seq(direct_model, device)
        metrics["models"]["direct_trained"] = {"path": direct_model, "device": str(model_device)}
        for split, rows in all_rows.items():
            print(f"Scoring {split} with direct_trained ({len(rows)} examples)", flush=True)
            questions, paths, locations = flatten_candidates(rows)
            scores = score_direct_relevance(
                model, tokenizer, questions, paths, model_device, batch_size=batch_size
            )
            attach_scores(rows, locations, scores, "direct_trained")
            metrics["splits"][split]["direct_trained"] = ranking_metrics(rows, "direct_trained")
        del model
        gc.collect()

    for system_name, model_path in (
        ("joint_ranker", joint_model),
        ("ranker_only", ranker_model),
    ):
        if not model_path:
            continue
        print(f"Loading {system_name}: {model_path}", flush=True)
        model, tokenizer, model_device = load_joint_ranker(model_path, device)
        metrics["models"][system_name] = {"path": model_path, "device": str(model_device)}
        for split, rows in all_rows.items():
            print(f"Scoring {split} with {system_name} ({len(rows)} examples)", flush=True)
            questions, paths, locations = flatten_candidates(rows)
            scores = score_joint_relevance(
                model, tokenizer, questions, paths, model_device, batch_size=batch_size
            )
            attach_scores(rows, locations, scores, system_name)
            split_metrics = ranking_metrics(rows, system_name)
            if system_name == "joint_ranker":
                sample = rows[: min(generation_examples, len(rows))]
                generated = generate_joint_questions(
                    model,
                    tokenizer,
                    [row["positive_path"] for row in sample],
                    model_device,
                    batch_size=max(1, batch_size // 2),
                )
                for row, question in zip(sample, generated, strict=True):
                    row.setdefault("generated_questions", {})[system_name] = question
                split_metrics["generation"] = generation_metrics(
                    [row["question"] for row in sample], generated, encoder
                )
            metrics["splits"][split][system_name] = split_metrics
        del model
        gc.collect()

    metrics["elapsed_seconds"] = time.time() - started
    prediction_rows = []
    for split, rows in all_rows.items():
        for row in rows:
            prediction_rows.append(
                {
                    "split": split,
                    "example_id": row["example_id"],
                    "question": row["question"],
                    "kg": row["kg"],
                    "positive_path": row["positive_path"],
                    "generated_questions": row.get("generated_questions", {}),
                    "candidates": row["candidates"],
                }
            )
    write_jsonl(output / "predictions.jsonl", prediction_rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "report.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics


def render_report(metrics: dict[str, Any]) -> str:
    regime = metrics.get("trained_model", {}).get("regime", "unknown")
    if regime == "multi_kg":
        training_note = (
            "The trained model saw KQA Pro and WebQSP training paths. Test questions are unseen; "
            "the KQA relation/composition splits remain excluded from training."
        )
    else:
        training_note = "The trained model saw KQA Pro training paths only. WebQSP is an unseen Freebase schema test."
    lines = [
        "# Inverse Verifier Evaluation",
        "",
        training_note,
        "No answer entity name is in the path input. Synthetic splits use controlled corruptions; the executable split uses only graph paths.",
        "",
        "| Split | System | R@1 | MRR | Pair accuracy | Gen semantic |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, split_metrics in metrics["splits"].items():
        direct = split_metrics["direct_similarity"]
        lines.append(
            f"| {split} | direct MiniLM | {direct['recall_at_1']:.3f} | {direct['mrr']:.3f} | "
            f"{direct['pairwise_accuracy']:.3f} | n/a |"
        )
        trained_direct = split_metrics.get("direct_trained")
        if trained_direct:
            lines.append(
                f"| {split} | matched direct T5 | {trained_direct['recall_at_1']:.3f} | "
                f"{trained_direct['mrr']:.3f} | {trained_direct['pairwise_accuracy']:.3f} | n/a |"
            )
        for system, label in (("ranker_only", "ranker-only encoder"), ("joint_ranker", "joint inverse + ranker")):
            if system not in split_metrics:
                continue
            row = split_metrics[system]
            generation = row.get("generation", {}).get("semantic_similarity")
            generation_text = f"{generation:.3f}" if generation is not None else "n/a"
            lines.append(
                f"| {split} | {label} | {row['recall_at_1']:.3f} | {row['mrr']:.3f} | "
                f"{row['pairwise_accuracy']:.3f} | {generation_text} |"
            )
        for system in ("inverse_pretrained", "inverse_trained"):
            if system not in split_metrics:
                continue
            row = split_metrics[system]
            lines.append(
                f"| {split} | {system} | {row['recall_at_1']:.3f} | {row['mrr']:.3f} | "
                f"{row['pairwise_accuracy']:.3f} | {row['generation']['semantic_similarity']:.3f} |"
            )
        generated = split_metrics.get("inverse_generated_similarity")
        if generated:
            lines.append(
                f"| {split} | generated-question similarity | {generated['recall_at_1']:.3f} | "
                f"{generated['mrr']:.3f} | {generated['pairwise_accuracy']:.3f} | n/a |"
            )
    lines.extend(["", "## Hard-Negative Accuracy", ""])
    for split, split_metrics in metrics["splits"].items():
        lines.append(f"### {split}")
        for system in (
            "direct_similarity",
            "inverse_pretrained",
            "inverse_trained",
            "inverse_generated_similarity",
            "direct_trained",
            "ranker_only",
            "joint_ranker",
        ):
            if system not in split_metrics:
                continue
            values = split_metrics[system]["pairwise_by_negative_type"]
            rendered = ", ".join(f"{name}={score:.3f}" for name, score in values.items())
            lines.append(f"- **{system}:** {rendered}")
        lines.append("")
    return "\n".join(lines)
