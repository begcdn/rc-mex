from __future__ import annotations

import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import (
    ENTITY_PLACEHOLDER,
    load_relation_glossary,
    read_jsonl,
    render_path,
)
from .metrics import ranking_metrics
from .model import best_device, generate_joint_questions, load_seq2seq


COMPARATOR_INPUT_MODES = (
    "question_generated",
    "question_path",
    "question_generated_path",
)
MAX_SEQUENCE_LENGTH = 512


def _candidate_key(path: dict[str, Any]) -> str:
    return json.dumps(path, sort_keys=True, ensure_ascii=False)


def candidate_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a faithful-training row into one multi-positive candidate set."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        path: dict[str, Any],
        is_positive: bool,
        negative_type: str,
        answer_entity: str | None,
    ) -> None:
        key = _candidate_key(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "path": path,
                "is_positive": is_positive,
                "negative_type": negative_type,
                "answer_entity": answer_entity or path.get("answer_entity") or "",
            }
        )

    add(
        row["positive_path"],
        True,
        "positive",
        row.get("positive_answer_entity"),
    )
    for path in row.get("alternate_positive_paths", []):
        add(path, True, "alternate_positive", path.get("answer_entity"))
    for path in row.get("negative_paths", []):
        add(
            path,
            False,
            path.get("negative_type", "negative"),
            path.get("answer_entity"),
        )
    if not any(candidate["is_positive"] for candidate in candidates):
        raise ValueError(f"{row.get('example_id', '<unknown>')} has no positive path")
    return candidates


def comparator_path_text(
    path: dict[str, Any],
    answer_entity: str,
    relation_glossary: dict[str, dict[str, Any]] | None = None,
) -> str:
    text = render_path(
        path,
        include_instruction=False,
        mask_anchor=False,
        relation_glossary=relation_glossary,
    )
    endpoint = answer_entity or "unknown"
    return f"{text}\nCandidate endpoint: {endpoint}"


def comparator_input_text(
    original_question: str,
    generated_question: str,
    path_text: str,
    mode: str,
) -> str:
    if mode not in COMPARATOR_INPUT_MODES:
        raise ValueError(f"unknown comparator input mode: {mode}")
    sections = [f"[ORIGINAL QUESTION]\n{original_question}"]
    if mode in {"question_generated", "question_generated_path"}:
        sections.append(f"[GENERATED QUESTION]\n{generated_question}")
    if mode in {"question_path", "question_generated_path"}:
        sections.append(f"[CANDIDATE PATH]\n{path_text}")
    return "\n\n".join(sections)


def listwise_multi_positive_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_offsets: list[tuple[int, int]],
) -> torch.Tensor:
    """Negative log probability assigned to any valid path in each candidate set."""
    losses = []
    for start, stop in group_offsets:
        group_scores = logits[start:stop]
        group_labels = labels[start:stop].bool()
        if not group_labels.any():
            raise ValueError("every comparator group must contain a positive candidate")
        losses.append(
            torch.logsumexp(group_scores, dim=0)
            - torch.logsumexp(group_scores[group_labels], dim=0)
        )
    return torch.stack(losses).mean()


def materialize_comparator_data(
    source: Path,
    generator_model: str,
    output: Path,
    batch_size: int = 8,
    device_name: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    """Generate the questions the comparator will actually observe at inference."""
    output.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        glossary = load_relation_glossary(
            Path(generator_model) / "relation_glossary.json"
        )
        rows_by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(source):
            candidates = []
            for candidate in row["candidates"]:
                path = candidate["path"]
                answer_entity = (
                    candidate.get("answer_entity")
                    or path.get("answer_entity")
                    or ""
                )
                candidates.append(
                    {
                        "path": path,
                        "is_positive": candidate["is_positive"],
                        "negative_type": candidate.get(
                            "negative_type",
                            candidate.get("category", "negative"),
                        ),
                        "answer_entity": answer_entity,
                        "generated_question": candidate["generated_question"],
                        "path_text": comparator_path_text(
                            path, answer_entity, glossary
                        ),
                    }
                )
            rows_by_slice[row.get("slice", "test")].append(
                {
                    "example_id": row.get("example_id", ""),
                    "kg": row.get("kg", ""),
                    "original_question": row.get(
                        "question", row.get("original_question", "")
                    ),
                    "candidates": candidates,
                }
            )
        manifest: dict[str, Any] = {
            "generator_model": generator_model,
            "source": str(source),
            "source_kind": "existing_generator_predictions",
            "splits": {},
        }
        for split, rows in sorted(rows_by_slice.items()):
            destination = output / f"{split}.jsonl"
            with destination.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["splits"][split] = {
                "candidate_sets": len(rows),
                "candidates": sum(len(row["candidates"]) for row in rows),
                "positive_candidates": sum(
                    candidate["is_positive"]
                    for row in rows
                    for candidate in row["candidates"]
                ),
            }
            print(
                f"Materialized {split}: {len(rows)} sets / "
                f"{manifest['splits'][split]['candidates']} candidates",
                flush=True,
            )
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    model, tokenizer, device = load_seq2seq(generator_model, device_name)
    glossary = getattr(model, "_relation_glossary", None)
    manifest: dict[str, Any] = {
        "generator_model": generator_model,
        "source": str(source),
        "splits": {},
    }

    for source_name, output_name in (
        ("train_faithful.jsonl", "train.jsonl"),
        ("dev_faithful.jsonl", "dev.jsonl"),
    ):
        rows = read_jsonl(source / source_name)
        if limit is not None:
            rows = rows[:limit]
        specs = [candidate_specs(row) for row in rows]
        flat_paths = [
            candidate["path"]
            for candidates in specs
            for candidate in candidates
        ]
        generated = generate_joint_questions(
            model,
            tokenizer,
            flat_paths,
            device,
            batch_size=batch_size,
            max_new_tokens=96,
        )
        cursor = 0
        materialized = []
        category_counts: dict[str, int] = defaultdict(int)
        for row, candidates in zip(rows, specs, strict=True):
            original = row["question"].replace(
                ENTITY_PLACEHOLDER, row["positive_path"]["anchor"]
            )
            output_candidates = []
            for candidate in candidates:
                generated_question = generated[cursor]
                cursor += 1
                category_counts[candidate["negative_type"]] += 1
                output_candidates.append(
                    {
                        **candidate,
                        "generated_question": generated_question,
                        "path_text": comparator_path_text(
                            candidate["path"],
                            candidate["answer_entity"],
                            glossary,
                        ),
                    }
                )
            materialized.append(
                {
                    "example_id": row.get("example_id", ""),
                    "kg": row.get("kg", ""),
                    "original_question": original,
                    "candidates": output_candidates,
                }
            )
        destination = output / output_name
        with destination.open("w", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][output_name.removesuffix(".jsonl")] = {
            "candidate_sets": len(materialized),
            "candidates": cursor,
            "positive_candidates": sum(
                candidate["is_positive"]
                for row in materialized
                for candidate in row["candidates"]
            ),
            "categories": dict(sorted(category_counts.items())),
        }
        print(
            f"Materialized {output_name}: {len(materialized)} sets / "
            f"{cursor} candidates",
            flush=True,
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class ComparatorDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class ComparatorCollator:
    def __init__(self, tokenizer: Any, mode: str):
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        texts, labels = [], []
        offsets = []
        metadata = []
        for row in rows:
            start = len(texts)
            for candidate in row["candidates"]:
                texts.append(
                    comparator_input_text(
                        row["original_question"],
                        candidate["generated_question"],
                        candidate["path_text"],
                        self.mode,
                    )
                )
                labels.append(candidate["is_positive"])
                metadata.append((row, candidate))
            offsets.append((start, len(texts)))
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "labels": torch.tensor(labels, dtype=torch.bool),
            "group_offsets": offsets,
            "metadata": metadata,
        }


def _move_encoded(encoded: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def score_comparator_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    mode: str,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    collator = ComparatorCollator(tokenizer, mode)
    scored_rows = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = collator(batch_rows)
        logits = model(
            **_move_encoded(batch["encoded"], device)
        ).logits.float().reshape(-1)
        cursor = 0
        for row in batch_rows:
            candidates = []
            for candidate in row["candidates"]:
                candidates.append({**candidate, "cross_encoder_score": float(logits[cursor])})
                cursor += 1
            scored_rows.append({**row, "candidates": candidates})
    return scored_rows


def _dev_metrics(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    mode: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    scored = score_comparator_rows(
        model, tokenizer, rows, mode, device, batch_size
    )
    return ranking_metrics(scored, "cross_encoder_score")


def train_comparator(
    data: Path,
    output: Path,
    base_model: str,
    mode: str,
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    device_name: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    if mode not in COMPARATOR_INPUT_MODES:
        raise ValueError(f"unknown comparator input mode: {mode}")
    random.seed(17)
    torch.manual_seed(17)
    device = best_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        local_files_only=True,
    ).to(device)
    train_rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")
    if limit is not None:
        train_rows = train_rows[:limit]
        dev_rows = dev_rows[: max(1, min(limit, len(dev_rows)))]
    train_dataset = ComparatorDataset(data / "train.jsonl")
    train_dataset.rows = train_rows
    collator = ComparatorCollator(tokenizer, mode)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(len(loader) * epochs, 1)
    warmup_steps = max(round(total_steps * 0.06), 1)

    def learning_rate_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max((total_steps - step) / max(total_steps - warmup_steps, 1), 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model"
    best_key = (-1.0, -1.0)
    history = []
    global_step = 0
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(
                    **_move_encoded(batch["encoded"], device)
                ).logits.float().reshape(-1)
                loss = listwise_multi_positive_loss(
                    logits,
                    batch["labels"].to(device),
                    batch["group_offsets"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss)
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(
                    f"epoch {epoch}/{epochs} step {step}/{len(loader)} "
                    f"listwise_loss={running_loss / step:.3f}",
                    flush=True,
                )
        metrics = _dev_metrics(
            model, tokenizer, dev_rows, mode, device, batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_listwise_loss": running_loss / max(len(loader), 1),
                **metrics,
            }
        )
        print(
            f"dev epoch {epoch}: R@1={metrics['recall_at_1']:.3f} "
            f"MRR={metrics['mrr']:.3f} pair={metrics['pairwise_accuracy']:.3f}",
            flush=True,
        )
        key = (metrics["recall_at_1"], metrics["mrr"])
        if key > best_key:
            best_key = key
            if checkpoint.exists():
                shutil.rmtree(checkpoint)
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            (checkpoint / "comparator_config.json").write_text(
                json.dumps(
                    {
                        "input_mode": mode,
                        "base_model": base_model,
                        "max_sequence_length": MAX_SEQUENCE_LENGTH,
                        "loss": "listwise_multi_positive_softmax",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    run = {
        "data": str(data),
        "base_model": base_model,
        "input_mode": mode,
        "train_candidate_sets": len(train_rows),
        "dev_candidate_sets": len(dev_rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_dev_recall_at_1": best_key[0],
        "best_dev_mrr": best_key[1],
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output / "training.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    return run


def load_comparator(
    model_path: str,
    device_name: str = "auto",
) -> tuple[Any, Any, torch.device, str]:
    device = best_device(device_name)
    config = json.loads(
        (Path(model_path) / "comparator_config.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    return model, tokenizer, device, config["input_mode"]


def _cosine_scores(
    rows: list[dict[str, Any]],
    semantic_model: str,
    device_name: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        semantic_model,
        device=str(best_device(device_name)),
        local_files_only=True,
    )
    texts = []
    for row in rows:
        for candidate in row["candidates"]:
            texts.extend([row["original_question"], candidate["generated_question"]])
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cursor = 0
    scored = []
    for row in rows:
        candidates = []
        for candidate in row["candidates"]:
            score = float(np.dot(embeddings[cursor], embeddings[cursor + 1]))
            cursor += 2
            candidates.append({**candidate, "cosine_similarity": score})
        scored.append({**row, "candidates": candidates})
    return scored


def evaluate_comparator(
    data: Path,
    model_path: str,
    output: Path,
    semantic_model: str | None = None,
    split: str = "dev",
    batch_size: int = 8,
    device_name: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(data / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    model, tokenizer, device, mode = load_comparator(model_path, device_name)
    scored = score_comparator_rows(
        model, tokenizer, rows, mode, device, batch_size
    )
    metrics: dict[str, Any] = {
        "split": split,
        "input_mode": mode,
        "cross_encoder": ranking_metrics(scored, "cross_encoder_score"),
    }
    if semantic_model:
        cosine_rows = _cosine_scores(
            rows, semantic_model, device_name, batch_size
        )
        metrics["cosine_similarity"] = ranking_metrics(
            cosine_rows, "cosine_similarity"
        )
        cosine_by_id = {
            row["example_id"]: row["candidates"] for row in cosine_rows
        }
        for row in scored:
            for candidate, cosine_candidate in zip(
                row["candidates"],
                cosine_by_id[row["example_id"]],
                strict=True,
            ):
                candidate["cosine_similarity"] = cosine_candidate[
                    "cosine_similarity"
                ]

    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    report = [
        "# Path Comparator Evaluation",
        "",
        f"- Split: {split}",
        f"- Input mode: `{mode}`",
        f"- Candidate sets: {len(rows)}",
        "",
        "| Method | Recall@1 | MRR | Pair accuracy |",
        "|---|---:|---:|---:|",
    ]
    methods = [("Cross-encoder", metrics["cross_encoder"])]
    if "cosine_similarity" in metrics:
        methods.append(("Cosine similarity", metrics["cosine_similarity"]))
    for name, values in methods:
        report.append(
            f"| {name} | {values['recall_at_1']:.3f} | "
            f"{values['mrr']:.3f} | {values['pairwise_accuracy']:.3f} |"
        )
    (output / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return metrics
