from __future__ import annotations

import json
import gc
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from safetensors import safe_open
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import (
    ENTITY_PLACEHOLDER,
    delexicalize_question,
    load_relation_glossary,
    read_jsonl,
    render_path,
)


UNINFORMATIVE_ANSWER_TYPES = {
    "",
    "answer",
    "answer entity",
    "entity",
    "intermediate entity",
    "thing",
    "unknown",
}


def best_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator_backbone(model_path: str) -> Any:
    """Load fine-tuned generators without losing their separate output weights."""
    path = Path(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    fine_tuned = (path / "joint_ranker.json").exists() or (
        path / "type_aware_generator.json"
    ).exists()
    weights_path = path / "model.safetensors"
    if fine_tuned and weights_path.exists():
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            if "lm_head.weight" in weights.keys():
                saved_output = weights.get_tensor("lm_head.weight")
                model.get_output_embeddings().weight = nn.Parameter(saved_output)
                model.config.tie_word_embeddings = False
    embedded_glossary = path / "relation_glossary.json"
    model._relation_glossary = (
        load_relation_glossary(embedded_glossary) if embedded_glossary.exists() else {}
    )
    return model


def normalized_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    losses = F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    mask = labels.ne(-100)
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class ContrastivePathDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], seed: int = 17):
        self.rows = rows
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        negatives = row["negative_paths"]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        positives = [row["positive_path"], *row.get("alternate_positive_paths", [])]
        positive = positives[rng.randrange(len(positives))]
        negative = negatives[rng.randrange(len(negatives))]
        style = "schema" if rng.random() < 0.35 else "natural"
        return {
            "positive_source": render_path(positive, style=style),
            "negative_source": render_path(negative, style=style),
            "question": row["question"],
            "negative_type": negative["negative_type"],
        }


def direct_prompt(question: str, path: dict[str, Any]) -> str:
    return (
        f"Question: {question}\n"
        f"Candidate knowledge-graph path:\n{render_path(path, include_instruction=False)}\n"
        "Does this exact path answer the question? Answer yes or no:"
    )


def rank_prompt(question: str, path: dict[str, Any]) -> str:
    masked_question = delexicalize_question(question, path["anchor"])
    return (
        f"Question intent: {masked_question}\n"
        f"Candidate knowledge-graph path:\n"
        f"{render_path(path, include_instruction=False, mask_anchor=True)}"
    )


class JointPathDataset(Dataset):
    """Matched examples for ranker-only and joint inverse/ranking training."""

    def __init__(self, rows: list[dict[str, Any]], seed: int = 17):
        self.rows = rows
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        positives = [row["positive_path"], *row.get("alternate_positive_paths", [])]
        positive = positives[rng.randrange(len(positives))]
        negative = row["negative_paths"][rng.randrange(len(row["negative_paths"]))]
        style = "schema" if rng.random() < 0.35 else "natural"
        return {
            "generation_source": render_path(positive, style=style, mask_anchor=True),
            "generation_target": delexicalize_question(row["question"], positive["anchor"]),
            "positive_rank_source": rank_prompt(row["question"], positive),
            "negative_rank_source": rank_prompt(row["question"], negative),
            "negative_type": negative["negative_type"],
        }


class FaithfulInverseDataset(Dataset):
    """Generation targets for every path plus gold-vs-negative likelihood pairs."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        seed: int = 17,
        relation_glossary: dict[str, dict[str, Any]] | None = None,
    ):
        self.rows = rows
        self.seed = seed
        self.epoch = 0
        self.relation_glossary = relation_glossary

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        if row.get("bidirectional_pair"):
            paired = row["negative_paths"][0]
            if self.epoch % 2 == 0:
                positive = row["positive_path"]
                negative = paired
                target = row["question"]
            else:
                positive = paired
                negative = row["positive_path"]
                target = paired["question"]
            return {
                "generation_source": render_path(
                    positive, mask_anchor=True, relation_glossary=self.relation_glossary
                ),
                "generation_target": target,
                "positive_source": render_path(
                    positive, mask_anchor=True, relation_glossary=self.relation_glossary
                ),
                "negative_source": render_path(
                    negative, mask_anchor=True, relation_glossary=self.relation_glossary
                ),
                "contrast_target": target,
                "negative_type": "executable_opposite_direction",
            }
        positive = row["positive_path"]
        natural_negative = row["negative_paths"][rng.randrange(len(row["negative_paths"]))]
        direction_negatives = row.get("contrast_only_negative_paths", [])
        if direction_negatives and self.epoch % 2 == 0:
            contrast_negative = direction_negatives[
                rng.randrange(len(direction_negatives))
            ]
        else:
            contrast_negative = natural_negative
        generate_negative = (self.epoch + index) % 2 == 1
        generation_path = natural_negative if generate_negative else positive
        generation_target = (
            natural_negative["question"] if generate_negative else row["question"]
        )
        return {
            "generation_source": render_path(
                generation_path,
                mask_anchor=True,
                relation_glossary=self.relation_glossary,
            ),
            "generation_target": generation_target,
            "positive_source": render_path(
                positive, mask_anchor=True, relation_glossary=self.relation_glossary
            ),
            "negative_source": render_path(
                contrast_negative,
                mask_anchor=True,
                relation_glossary=self.relation_glossary,
            ),
            "contrast_target": row["question"],
            "negative_type": contrast_negative["negative_type"],
        }


def informative_answer_type(answer_type: str) -> bool:
    return answer_type.strip().casefold() not in UNINFORMATIVE_ANSWER_TYPES


def type_compatibility_prompt(question: str, answer_type: str) -> str:
    return (
        "Decide only whether the requested answer in the question can have the "
        "candidate semantic type.\n"
        f"Question: {question}\n"
        f"Candidate answer type: {answer_type}\n"
        "Compatible answer type (yes or no):"
    )


class TypeAwareGeneratorDataset(Dataset):
    """Pure question generation plus an auxiliary answer-type consistency task.

    The auxiliary task shapes the same seq2seq model during fine-tuning.  The
    path-to-question task still emits only a natural-language question, and no
    independent path-ranking head is created.
    """

    def __init__(self, rows: list[dict[str, Any]], seed: int = 17):
        self.rows = rows
        self.seed = seed
        self.epoch = 0
        self.type_rows = [
            index
            for index, row in enumerate(rows)
            if informative_answer_type(row["positive_path"].get("answer_type", ""))
        ]
        self.type_inventory = sorted(
            {
                row["positive_path"]["answer_type"].strip()
                for row in rows
                if informative_answer_type(row["positive_path"].get("answer_type", ""))
            }
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows) + len(self.type_rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        if index < len(self.rows):
            row = self.rows[index]
            rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
            positives = [row["positive_path"], *row.get("alternate_positive_paths", [])]
            path = positives[rng.randrange(len(positives))]
            style = "schema" if rng.random() < 0.35 else "natural"
            return {
                "source": render_path(path, style=style, mask_anchor=True),
                "target": delexicalize_question(row["question"], path["anchor"]),
                "task": "question_generation",
            }

        row_index = self.type_rows[index - len(self.rows)]
        row = self.rows[row_index]
        positive_type = row["positive_path"]["answer_type"].strip()
        rng = random.Random(
            self.seed + 10_000_019 + self.epoch * max(len(self.type_rows), 1) + row_index
        )
        use_positive = (self.epoch + row_index) % 2 == 0
        answer_type = positive_type
        target = "yes"
        if not use_positive:
            wrong_types = [
                path.get("answer_type", "").strip()
                for path in row.get("negative_paths", [])
                if path.get("negative_type") == "wrong_answer_type"
                and informative_answer_type(path.get("answer_type", ""))
                and path.get("answer_type", "").strip().casefold() != positive_type.casefold()
            ]
            if not wrong_types:
                wrong_types = [
                    value
                    for value in self.type_inventory
                    if value.casefold() != positive_type.casefold()
                ]
            if wrong_types:
                answer_type = wrong_types[rng.randrange(len(wrong_types))]
                target = "no"
        question = delexicalize_question(row["question"], row["positive_path"]["anchor"])
        return {
            "source": type_compatibility_prompt(question, answer_type),
            "target": target,
            "task": "answer_type_compatibility",
        }


@dataclass
class Seq2SeqTaskCollator:
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["source"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=[item["target"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            **encoded,
            "labels": labels,
            "batch_size": len(batch),
            "tasks": [item["task"] for item in batch],
        }


@dataclass
class JointBatchCollator:
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        generation = self.tokenizer(
            [item["generation_source"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            text_target=[item["generation_target"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        targets[targets == self.tokenizer.pad_token_id] = -100
        rank = self.tokenizer(
            [item["positive_rank_source"] for item in batch]
            + [item["negative_rank_source"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        return {
            "generation_input_ids": generation["input_ids"],
            "generation_attention_mask": generation["attention_mask"],
            "generation_labels": targets,
            "rank_input_ids": rank["input_ids"],
            "rank_attention_mask": rank["attention_mask"],
            "batch_size": len(batch),
            "negative_types": [item["negative_type"] for item in batch],
        }


@dataclass
class FaithfulInverseCollator:
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        generation = self.tokenizer(
            [item["generation_source"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        generation_labels = self.tokenizer(
            text_target=[item["generation_target"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        generation_labels[generation_labels == self.tokenizer.pad_token_id] = -100

        contrast = self.tokenizer(
            [item["positive_source"] for item in batch]
            + [item["negative_source"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        contrast_targets = [item["contrast_target"] for item in batch] * 2
        contrast_labels = self.tokenizer(
            text_target=contrast_targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        contrast_labels[contrast_labels == self.tokenizer.pad_token_id] = -100
        return {
            "generation_input_ids": generation["input_ids"],
            "generation_attention_mask": generation["attention_mask"],
            "generation_labels": generation_labels,
            "contrast_input_ids": contrast["input_ids"],
            "contrast_attention_mask": contrast["attention_mask"],
            "contrast_labels": contrast_labels,
            "batch_size": len(batch),
            "negative_types": [item["negative_type"] for item in batch],
        }


class JointInverseRanker(nn.Module):
    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator
        hidden_size = int(generator.config.d_model)
        self.rank_head = nn.Linear(hidden_size, 1)

    def rank(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.generator.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.rank_head(pooled).squeeze(-1)

    def save_checkpoint(self, output: Path, tokenizer: Any, objective: str) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.generator.save_pretrained(output)
        tokenizer.save_pretrained(output)
        torch.save(self.rank_head.state_dict(), output / "rank_head.pt")
        (output / "joint_ranker.json").write_text(
            json.dumps({"objective": objective, "hidden_size": self.rank_head.in_features}, indent=2),
            encoding="utf-8",
        )


class DirectPathDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], seed: int = 17):
        self.rows = rows
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * len(self.rows) + index)
        positives = [row["positive_path"], *row.get("alternate_positive_paths", [])]
        positive = positives[rng.randrange(len(positives))]
        negative = row["negative_paths"][rng.randrange(len(row["negative_paths"]))]
        return {
            "positive_source": direct_prompt(row["question"], positive),
            "negative_source": direct_prompt(row["question"], negative),
        }


@dataclass
class DirectBatchCollator:
    tokenizer: Any
    max_source_length: int

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        sources = [item["positive_source"] for item in batch] + [item["negative_source"] for item in batch]
        targets = ["yes"] * len(batch) + ["no"] * len(batch)
        encoded = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(text_target=targets, padding=True, return_tensors="pt")["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {**encoded, "labels": labels, "batch_size": len(batch)}


@dataclass
class BatchCollator:
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Any]:
        sources = [item["positive_source"] for item in batch] + [item["negative_source"] for item in batch]
        targets = [item["question"] for item in batch] * 2
        encoded = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        target = self.tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = target["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            **encoded,
            "labels": labels,
            "batch_size": len(batch),
            "negative_types": [item["negative_type"] for item in batch],
        }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


@torch.no_grad()
def evaluate_loss(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    totals = {"generation_nll": 0.0, "rank_loss": 0.0, "pair_accuracy": 0.0, "examples": 0}
    for batch in loader:
        size = batch["batch_size"]
        tensors = batch_to_device(batch, device)
        outputs = model(**tensors)
        nll = normalized_sequence_nll(outputs.logits, tensors["labels"])
        positive_nll, negative_nll = nll[:size], nll[size:]
        rank_loss = F.softplus((positive_nll - negative_nll + margin) / temperature) * temperature
        totals["generation_nll"] += positive_nll.sum().item()
        totals["rank_loss"] += rank_loss.sum().item()
        totals["pair_accuracy"] += positive_nll.lt(negative_nll).sum().item()
        totals["examples"] += size
    count = max(int(totals["examples"]), 1)
    return {key: value / count for key, value in totals.items() if key != "examples"} | {"examples": count}


@torch.no_grad()
def evaluate_joint_loader(
    model: JointInverseRanker,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    temperature: float,
    include_generation: bool,
) -> dict[str, float]:
    model.eval()
    totals = {"generation_nll": 0.0, "rank_loss": 0.0, "pair_accuracy": 0.0, "examples": 0}
    for batch in loader:
        size = batch["batch_size"]
        tensors = batch_to_device(batch, device)
        rank_scores = model.rank(tensors["rank_input_ids"], tensors["rank_attention_mask"])
        positive_scores, negative_scores = rank_scores[:size], rank_scores[size:]
        rank_loss = (
            F.softplus((negative_scores - positive_scores + margin) / temperature) * temperature
        )
        if include_generation:
            generated = model.generator(
                input_ids=tensors["generation_input_ids"],
                attention_mask=tensors["generation_attention_mask"],
                labels=tensors["generation_labels"],
            )
            generation_nll = normalized_sequence_nll(generated.logits, tensors["generation_labels"])
            totals["generation_nll"] += generation_nll.sum().item()
        totals["rank_loss"] += rank_loss.sum().item()
        totals["pair_accuracy"] += positive_scores.gt(negative_scores).sum().item()
        totals["examples"] += size
    count = max(int(totals["examples"]), 1)
    return {key: value / count for key, value in totals.items() if key != "examples"} | {"examples": count}


@torch.no_grad()
def evaluate_type_aware_loader(
    model: Any,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "generation_nll": 0.0,
        "generation_examples": 0,
        "type_nll": 0.0,
        "type_examples": 0,
        "type_correct": 0,
    }
    for batch in loader:
        tensors = batch_to_device(batch, device)
        outputs = model(**tensors)
        nll = normalized_sequence_nll(outputs.logits, tensors["labels"])
        predicted_ids = outputs.logits.argmax(dim=-1)
        predicted = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        targets = tokenizer.batch_decode(
            tensors["labels"].masked_fill(tensors["labels"].eq(-100), tokenizer.pad_token_id),
            skip_special_tokens=True,
        )
        for index, task in enumerate(batch["tasks"]):
            if task == "question_generation":
                totals["generation_nll"] += nll[index].item()
                totals["generation_examples"] += 1
            else:
                totals["type_nll"] += nll[index].item()
                totals["type_examples"] += 1
                totals["type_correct"] += int(
                    predicted[index].strip().casefold() == targets[index].strip().casefold()
                )
    generation_count = max(int(totals["generation_examples"]), 1)
    type_count = max(int(totals["type_examples"]), 1)
    return {
        "generation_nll": totals["generation_nll"] / generation_count,
        "type_nll": totals["type_nll"] / type_count,
        "type_accuracy": totals["type_correct"] / type_count,
        "examples": generation_count + type_count,
    }


@torch.no_grad()
def evaluate_faithful_loader(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    temperature: float,
) -> dict[str, float]:
    model.eval()
    totals = {"generation_nll": 0.0, "rank_loss": 0.0, "pair_accuracy": 0.0, "examples": 0}
    for batch in loader:
        size = batch["batch_size"]
        tensors = batch_to_device(batch, device)
        generated = model(
            input_ids=tensors["generation_input_ids"],
            attention_mask=tensors["generation_attention_mask"],
            labels=tensors["generation_labels"],
        )
        generation_nll = normalized_sequence_nll(
            generated.logits, tensors["generation_labels"]
        )
        contrasted = model(
            input_ids=tensors["contrast_input_ids"],
            attention_mask=tensors["contrast_attention_mask"],
            labels=tensors["contrast_labels"],
        )
        contrast_nll = normalized_sequence_nll(
            contrasted.logits, tensors["contrast_labels"]
        )
        positive_nll, negative_nll = contrast_nll[:size], contrast_nll[size:]
        rank_loss = (
            F.softplus((positive_nll - negative_nll + margin) / temperature) * temperature
        )
        totals["generation_nll"] += generation_nll.sum().item()
        totals["rank_loss"] += rank_loss.sum().item()
        totals["pair_accuracy"] += positive_nll.lt(negative_nll).sum().item()
        totals["examples"] += size
    count = max(int(totals["examples"]), 1)
    return {key: value / count for key, value in totals.items() if key != "examples"} | {
        "examples": count
    }


def train_model(
    train_path: Path,
    dev_path: Path,
    output: Path,
    base_model: str,
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    rank_weight: float = 1.0,
    margin: float = 0.2,
    temperature: float = 0.2,
    max_source_length: int = 512,
    max_target_length: int = 96,
    seed: int = 17,
    device_name: str = "auto",
    limit: int | None = None,
    regime: str = "kqa_only",
    objective: str = "inverse",
    relation_glossary_path: Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = best_device(device_name)
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    if limit:
        train_rows = train_rows[:limit]
        dev_rows = dev_rows[: max(16, limit // 10)]
    if not train_rows or not dev_rows:
        raise ValueError("training and development splits must both be non-empty")
    embedded_glossary_path = Path(base_model) / "relation_glossary.json"
    effective_glossary_path = relation_glossary_path
    if effective_glossary_path is None and embedded_glossary_path.exists():
        effective_glossary_path = embedded_glossary_path
    if objective == "faithful_inverse" and effective_glossary_path is None:
        raise ValueError(
            "faithful_inverse requires --relation-glossary, unless the base checkpoint "
            "already embeds relation_glossary.json"
        )
    relation_glossary = load_relation_glossary(effective_glossary_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    generator = load_generator_backbone(base_model)
    is_joint_ranker = objective in {"joint", "ranker"}
    model: Any = JointInverseRanker(generator).to(device) if is_joint_ranker else generator.to(device)
    inherited_rank_head = Path(base_model) / "rank_head.pt"
    if is_joint_ranker and inherited_rank_head.exists():
        state = torch.load(inherited_rank_head, map_location="cpu", weights_only=True)
        model.rank_head.load_state_dict(state)
    if objective == "inverse":
        train_dataset = ContrastivePathDataset(train_rows, seed)
        dev_dataset = ContrastivePathDataset(dev_rows, seed + 1)
        collator = BatchCollator(tokenizer, max_source_length, max_target_length)
    elif objective == "direct":
        train_dataset = DirectPathDataset(train_rows, seed)
        dev_dataset = DirectPathDataset(dev_rows, seed + 1)
        collator = DirectBatchCollator(tokenizer, max_source_length)
    elif objective in {"joint", "ranker"}:
        train_dataset = JointPathDataset(train_rows, seed)
        dev_dataset = JointPathDataset(dev_rows, seed + 1)
        collator = JointBatchCollator(tokenizer, max_source_length, max_target_length)
    elif objective == "type_aware_generator":
        train_dataset = TypeAwareGeneratorDataset(train_rows, seed)
        dev_dataset = TypeAwareGeneratorDataset(dev_rows, seed + 1)
        collator = Seq2SeqTaskCollator(tokenizer, max_source_length, max_target_length)
    elif objective == "faithful_inverse":
        train_dataset = FaithfulInverseDataset(train_rows, seed, relation_glossary)
        dev_dataset = FaithfulInverseDataset(dev_rows, seed + 1, relation_glossary)
        collator = FaithfulInverseCollator(tokenizer, max_source_length, max_target_length)
    else:
        raise ValueError(f"unknown objective: {objective}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_pair_accuracy = -1.0
    best_dev_loss = float("inf")
    started = time.time()
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_generation = running_rank = examples_seen = 0.0
        for step, batch in enumerate(train_loader, 1):
            size = batch["batch_size"]
            tensors = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if is_joint_ranker:
                rank_scores = model.rank(tensors["rank_input_ids"], tensors["rank_attention_mask"])
                positive_scores, negative_scores = rank_scores[:size], rank_scores[size:]
                rank_loss = (
                    F.softplus((negative_scores - positive_scores + margin) / temperature)
                    * temperature
                ).mean()
                if objective == "joint":
                    generated = model.generator(
                        input_ids=tensors["generation_input_ids"],
                        attention_mask=tensors["generation_attention_mask"],
                        labels=tensors["generation_labels"],
                    )
                    generation_loss = normalized_sequence_nll(
                        generated.logits, tensors["generation_labels"]
                    ).mean()
                    loss = generation_loss + rank_weight * rank_loss
                else:
                    generation_loss = torch.zeros((), device=device)
                    loss = rank_loss
            else:
                if objective == "faithful_inverse":
                    generated = model(
                        input_ids=tensors["generation_input_ids"],
                        attention_mask=tensors["generation_attention_mask"],
                        labels=tensors["generation_labels"],
                    )
                    generation_loss = normalized_sequence_nll(
                        generated.logits, tensors["generation_labels"]
                    ).mean()
                    contrasted = model(
                        input_ids=tensors["contrast_input_ids"],
                        attention_mask=tensors["contrast_attention_mask"],
                        labels=tensors["contrast_labels"],
                    )
                    contrast_nll = normalized_sequence_nll(
                        contrasted.logits, tensors["contrast_labels"]
                    )
                    positive_nll, negative_nll = contrast_nll[:size], contrast_nll[size:]
                    rank_loss = (
                        F.softplus(
                            (positive_nll - negative_nll + margin) / temperature
                        )
                        * temperature
                    ).mean()
                    loss = generation_loss + rank_weight * rank_loss
                else:
                    outputs = model(**tensors)
                    nll = normalized_sequence_nll(outputs.logits, tensors["labels"])
            if objective == "inverse":
                positive_nll, negative_nll = nll[:size], nll[size:]
                generation_loss = positive_nll.mean()
                rank_loss = (
                    F.softplus((positive_nll - negative_nll + margin) / temperature) * temperature
                ).mean()
                loss = generation_loss + rank_weight * rank_loss
            elif objective == "direct":
                generation_loss = nll.mean()
                rank_loss = torch.zeros((), device=device)
                loss = generation_loss
            elif objective == "type_aware_generator":
                generation_loss = nll.mean()
                rank_loss = torch.zeros((), device=device)
                loss = generation_loss
            elif objective == "faithful_inverse":
                pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_generation += generation_loss.item() * size
            running_rank += rank_loss.item() * size
            examples_seen += size
            if step == 1 or step % 25 == 0 or step == len(train_loader):
                loss_name = "task_nll" if objective == "type_aware_generator" else "gen_nll"
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} "
                    f"{loss_name}={running_generation / examples_seen:.3f} "
                    f"rank_loss={running_rank / examples_seen:.3f}",
                    flush=True,
                )
        if objective == "inverse":
            dev_metrics = evaluate_loss(model, dev_loader, device, margin, temperature)
        elif objective == "direct":
            direct_scores = evaluate_direct_pairs(
                model,
                tokenizer,
                dev_rows,
                device,
                batch_size=batch_size,
                max_source_length=max_source_length,
            )
            dev_metrics = {
                "generation_nll": running_generation / examples_seen,
                "rank_loss": 0.0,
                "pair_accuracy": direct_scores,
                "examples": len(dev_rows),
            }
        elif objective == "type_aware_generator":
            dev_metrics = evaluate_type_aware_loader(
                model, dev_loader, tokenizer, device
            )
            dev_metrics["pair_accuracy"] = dev_metrics["type_accuracy"]
        elif objective == "faithful_inverse":
            dev_metrics = evaluate_faithful_loader(
                model, dev_loader, device, margin, temperature
            )
        else:
            dev_metrics = evaluate_joint_loader(
                model,
                dev_loader,
                device,
                margin,
                temperature,
                include_generation=objective == "joint",
            )
            dev_metrics["pair_accuracy"] = evaluate_joint_pairs(
                model,
                tokenizer,
                dev_rows,
                device,
                batch_size,
                max_source_length,
            )
        epoch_row = {
            "epoch": epoch + 1,
            "train_generation_nll": running_generation / examples_seen,
            "train_rank_loss": running_rank / examples_seen,
            **{f"dev_{key}": value for key, value in dev_metrics.items()},
        }
        history.append(epoch_row)
        print(
            f"dev epoch {epoch + 1}: nll={dev_metrics['generation_nll']:.3f} "
            f"pair_accuracy={dev_metrics['pair_accuracy']:.3f}",
            flush=True,
        )
        selection_loss = (
            dev_metrics["generation_nll"] + dev_metrics.get("rank_loss", 0.0)
            if objective == "faithful_inverse"
            else dev_metrics["generation_nll"] + dev_metrics.get("type_nll", 0.0)
        )
        improved = (
            selection_loss < best_dev_loss
            if objective in {"type_aware_generator", "faithful_inverse"}
            else dev_metrics["pair_accuracy"] > best_pair_accuracy
        )
        if improved:
            best_pair_accuracy = dev_metrics["pair_accuracy"]
            best_dev_loss = selection_loss
            if is_joint_ranker:
                model.save_checkpoint(output / "model", tokenizer, objective)
            else:
                model.save_pretrained(output / "model")
                tokenizer.save_pretrained(output / "model")
                if objective == "type_aware_generator":
                    (output / "model" / "type_aware_generator.json").write_text(
                        json.dumps(
                            {
                                "objective": objective,
                                "tasks": [
                                    "path_to_question",
                                    "question_answer_type_compatibility",
                                ],
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    run = {
        "base_model": base_model,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "rank_weight": rank_weight,
        "margin": margin,
        "temperature": temperature,
        "seed": seed,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_task_examples": len(train_dataset),
        "dev_task_examples": len(dev_dataset),
        "elapsed_seconds": time.time() - started,
        "best_dev_pair_accuracy": best_pair_accuracy,
        "best_dev_selection_loss": best_dev_loss,
        "regime": regime,
        "objective": objective,
        "input_contract": "grounded_relation_semantics_v1" if relation_glossary else "legacy",
        "relation_glossary_entries": len(relation_glossary),
        "history": history,
    }
    if effective_glossary_path:
        destination_glossary = output / "model" / "relation_glossary.json"
        if Path(effective_glossary_path).resolve() != destination_glossary.resolve():
            shutil.copyfile(effective_glossary_path, destination_glossary)
    (output / "training.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


@torch.no_grad()
def score_direct_relevance(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    max_source_length: int = 256,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_questions = questions[start : start + batch_size]
        sources = [direct_prompt(question, path) for question, path in zip(batch_questions, batch_paths)]
        encoded = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        ).to(device)
        size = len(sources)
        targets = tokenizer(
            text_target=["yes"] * size + ["no"] * size,
            padding=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        targets[targets == tokenizer.pad_token_id] = -100
        doubled = {key: value.repeat(2, 1) for key, value in encoded.items()}
        outputs = model(**doubled, labels=targets)
        nll = normalized_sequence_nll(outputs.logits, targets)
        yes_nll, no_nll = nll[:size], nll[size:]
        scores.extend((no_nll - yes_nll).cpu().tolist())
    return scores


@torch.no_grad()
def score_joint_relevance(
    model: JointInverseRanker,
    tokenizer: Any,
    questions: list[str],
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    max_source_length: int = 256,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_questions = questions[start : start + batch_size]
        encoded = tokenizer(
            [rank_prompt(question, path) for question, path in zip(batch_questions, batch_paths)],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        ).to(device)
        scores.extend(model.rank(encoded["input_ids"], encoded["attention_mask"]).cpu().tolist())
    return scores


@torch.no_grad()
def evaluate_joint_pairs(
    model: JointInverseRanker,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_source_length: int,
) -> float:
    questions: list[str] = []
    paths: list[dict[str, Any]] = []
    comparisons: list[tuple[list[int], list[int]]] = []
    for row in rows:
        positive_indices, negative_indices = [], []
        for positive in [row["positive_path"], *row.get("alternate_positive_paths", [])]:
            positive_indices.append(len(paths))
            questions.append(row["question"])
            paths.append(positive)
        for negative in row["negative_paths"]:
            negative_indices.append(len(paths))
            questions.append(row["question"])
            paths.append(negative)
        comparisons.append((positive_indices, negative_indices))
    scores = score_joint_relevance(
        model, tokenizer, questions, paths, device, batch_size, max_source_length
    )
    correct = total = 0
    for positive_indices, negative_indices in comparisons:
        positive_score = max(scores[index] for index in positive_indices)
        for negative_index in negative_indices:
            correct += positive_score > scores[negative_index]
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def generate_joint_questions(
    model: Any,
    tokenizer: Any,
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    relation_glossary: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    generations: list[str] = []
    model.eval()
    backbone = model.generator if isinstance(model, JointInverseRanker) else model
    glossary = relation_glossary
    if glossary is None:
        glossary = getattr(backbone, "_relation_glossary", None)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        encoded = tokenizer(
            [
                render_path(path, mask_anchor=True, relation_glossary=glossary)
                for path in batch_paths
            ],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        output_ids = backbone.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=1)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for question, path in zip(decoded, batch_paths, strict=True):
            generations.append(question.replace(ENTITY_PLACEHOLDER, path["anchor"]))
    return generations


@torch.no_grad()
def score_question_likelihood(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    max_source_length: int = 512,
    max_target_length: int = 96,
    relation_glossary: dict[str, dict[str, Any]] | None = None,
) -> list[float]:
    """Return length-normalized log-likelihood of each question given its path."""
    scores: list[float] = []
    model.eval()
    backbone = model.generator if isinstance(model, JointInverseRanker) else model
    glossary = relation_glossary
    if glossary is None:
        glossary = getattr(backbone, "_relation_glossary", None)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_questions = questions[start : start + batch_size]
        encoded = tokenizer(
            [
                render_path(path, mask_anchor=True, relation_glossary=glossary)
                for path in batch_paths
            ],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        ).to(device)
        target = tokenizer(
            text_target=batch_questions,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = target["input_ids"].to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        output = backbone(**encoded, labels=labels)
        scores.extend((-normalized_sequence_nll(output.logits, labels)).cpu().tolist())
    return scores


def load_joint_ranker(
    model_path: str,
    device_name: str = "auto",
) -> tuple[JointInverseRanker, Any, torch.device]:
    device = best_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    generator = load_generator_backbone(model_path)
    model = JointInverseRanker(generator)
    state = torch.load(Path(model_path) / "rank_head.pt", map_location="cpu", weights_only=True)
    model.rank_head.load_state_dict(state)
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def evaluate_direct_pairs(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_source_length: int,
) -> float:
    questions, paths, comparisons = [], [], []
    for row in rows:
        positives = [row["positive_path"], *row.get("alternate_positive_paths", [])]
        positive_indices = []
        for positive in positives:
            positive_indices.append(len(paths))
            questions.append(row["question"])
            paths.append(positive)
        negative_indices = []
        for negative in row["negative_paths"]:
            negative_indices.append(len(paths))
            questions.append(row["question"])
            paths.append(negative)
        comparisons.append((positive_indices, negative_indices))
    scores = score_direct_relevance(
        model, tokenizer, questions, paths, device, batch_size, max_source_length
    )
    correct = total = 0
    for positive_indices, negative_indices in comparisons:
        positive_score = max(scores[index] for index in positive_indices)
        for negative_index in negative_indices:
            correct += positive_score > scores[negative_index]
            total += 1
    return correct / max(total, 1)


@torch.no_grad()
def score_questions_given_paths(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
    max_source_length: int = 256,
    max_target_length: int = 96,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_questions = questions[start : start + batch_size]
        sources = [render_path(path) for path in batch_paths]
        encoded = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        ).to(device)
        targets = tokenizer(
            text_target=batch_questions,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )["input_ids"].to(device)
        targets[targets == tokenizer.pad_token_id] = -100
        outputs = model(**encoded, labels=targets)
        nll = normalized_sequence_nll(outputs.logits, targets)
        scores.extend((-nll).cpu().tolist())
    return scores


@torch.no_grad()
def generate_questions(
    model: Any,
    tokenizer: Any,
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    num_beams: int = 2,
) -> list[str]:
    generations: list[str] = []
    model.eval()
    for start in range(0, len(paths), batch_size):
        sources = [render_path(path) for path in paths[start : start + batch_size]]
        encoded = tokenizer(sources, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, num_beams=num_beams)
        generations.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return generations


@torch.no_grad()
def type_compatibility_scores(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    answer_types: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_source_length: int = 256,
) -> list[float]:
    """Return P(yes) for the auxiliary question/type compatibility task."""
    if len(questions) != len(answer_types):
        raise ValueError("questions and answer_types must have equal length")
    backbone = model.generator if isinstance(model, JointInverseRanker) else model
    scores: list[float] = []
    for start in range(0, len(questions), batch_size):
        prompts = [
            type_compatibility_prompt(question, answer_type)
            for question, answer_type in zip(
                questions[start : start + batch_size],
                answer_types[start : start + batch_size],
                strict=True,
            )
        ]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        candidate_losses = []
        for target in ("yes", "no"):
            labels = tokenizer(
                text_target=[target] * len(prompts),
                padding=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            output = backbone(**encoded, labels=labels)
            candidate_losses.append(normalized_sequence_nll(output.logits, labels))
        yes_nll, no_nll = candidate_losses
        probabilities = torch.sigmoid(no_nll - yes_nll)
        scores.extend(probabilities.cpu().tolist())
    return scores


def load_seq2seq(model_path: str, device_name: str = "auto") -> tuple[Any, Any, torch.device]:
    device = best_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = load_generator_backbone(model_path).to(device)
    return model, tokenizer, device
