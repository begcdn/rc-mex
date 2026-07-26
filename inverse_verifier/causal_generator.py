from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import (
    ENTITY_PLACEHOLDER,
    load_relation_glossary,
    read_jsonl,
    render_path,
)


SYSTEM_PROMPT = (
    "Convert the exact grounded knowledge-graph path into one natural-language "
    "question. Preserve every fact, relation meaning, direction, intermediate "
    "dependency, and the requested answer role. Do not add facts. Output only "
    "the question."
)
MAX_SEQUENCE_LENGTH = 1024
EFFECTIVE_BATCH_SIZE = 16


def flatten_path_question_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every validated path with the question that path actually means."""
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        candidates = [
            (row["positive_path"], row["question"]),
            *[
                (path, row["question"])
                for path in row.get("alternate_positive_paths", [])
            ],
            *[
                (path, path["question"])
                for path in row.get("negative_paths", [])
                if path.get("question")
            ],
        ]
        for path, question in candidates:
            key = json.dumps([path, question], sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"path": path, "question": question})
    return pairs


def causal_generation_prompt(
    tokenizer: Any,
    path: dict[str, Any],
    relation_glossary: dict[str, dict[str, Any]],
) -> list[int]:
    grounded_path = render_path(
        path,
        include_instruction=False,
        mask_anchor=True,
        relation_glossary=relation_glossary,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": grounded_path},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


class CausalInverseDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        tokenizer: Any,
        relation_glossary: dict[str, dict[str, Any]],
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.relation_glossary = relation_glossary

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        pair = self.pairs[index]
        prompt_ids = causal_generation_prompt(
            self.tokenizer, pair["path"], self.relation_glossary
        )
        target_ids = self.tokenizer(
            pair["question"] + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        if len(prompt_ids) + len(target_ids) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"causal generator example {index} exceeds {MAX_SEQUENCE_LENGTH} tokens"
            )
        return {
            "input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
        }


class CausalInverseCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        input_ids, attention_mask, labels = [], [], []
        for row in rows:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@torch.no_grad()
def evaluate_causal_nll(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = total_tokens = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        token_count = batch["labels"].ne(-100).sum().item()
        total_loss += output.loss.item() * token_count
        total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def train_causal_inverse_generator(
    train_path: Path,
    dev_path: Path,
    output: Path,
    base_model: str,
    relation_glossary_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
    limit: int | None,
) -> dict[str, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "causal_inverse requires PEFT; install the project with `pip install -e '.[llm]'`"
        ) from exc

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("causal_inverse training currently requires a CUDA GPU")

    torch.manual_seed(seed)
    random.seed(seed)
    glossary = load_relation_glossary(relation_glossary_path)
    train_pairs = flatten_path_question_pairs(read_jsonl(train_path))
    dev_pairs = flatten_path_question_pairs(read_jsonl(dev_path))
    if limit:
        train_pairs = train_pairs[:limit]
        dev_pairs = dev_pairs[: max(16, limit // 10)]

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    ).to(device)
    model.enable_input_require_grads()

    collator = CausalInverseCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        CausalInverseDataset(train_pairs, tokenizer, glossary),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        CausalInverseDataset(dev_pairs, tokenizer, glossary),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    accumulation_steps = max(1, math.ceil(EFFECTIVE_BATCH_SIZE / batch_size))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = max(1, updates_per_epoch * epochs)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_updates // 20),
        num_training_steps=total_updates,
    )

    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_dev_nll = float("inf")
    started = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = running_tokens = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            output_row = model(**batch)
            (output_row.loss / accumulation_steps).backward()
            token_count = batch["labels"].ne(-100).sum().item()
            running_loss += output_row.loss.item() * token_count
            running_tokens += token_count
            if step % accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{len(train_loader)} "
                    f"token_nll={running_loss / max(running_tokens, 1):.3f}",
                    flush=True,
                )

        dev_nll = evaluate_causal_nll(model, dev_loader, device)
        epoch_row = {
            "epoch": epoch + 1,
            "train_token_nll": running_loss / max(running_tokens, 1),
            "dev_token_nll": dev_nll,
        }
        history.append(epoch_row)
        print(f"dev epoch {epoch + 1}: token_nll={dev_nll:.3f}", flush=True)
        if dev_nll < best_dev_nll:
            best_dev_nll = dev_nll
            model.save_pretrained(output / "model")
            tokenizer.save_pretrained(output / "model")
        gc.collect()
        torch.cuda.empty_cache()

    shutil.copyfile(
        relation_glossary_path, output / "model" / "relation_glossary.json"
    )
    run = {
        "architecture": "causal_lora",
        "base_model": base_model,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * accumulation_steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "train_rows": len(read_jsonl(train_path)),
        "dev_rows": len(read_jsonl(dev_path)),
        "train_path_question_pairs": len(train_pairs),
        "dev_path_question_pairs": len(dev_pairs),
        "input_contract": "grounded_relation_semantics_v2",
        "relation_glossary_entries": len(glossary),
        "best_dev_token_nll": best_dev_nll,
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output / "training.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def load_causal_inverse_generator(
    model_path: str, device: torch.device
) -> tuple[Any, Any]:
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("loading causal_inverse requires PEFT") from exc

    adapter = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        adapter.base_model_name_or_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, model_path).to(device).eval()
    model._inverse_causal = True
    glossary_path = Path(model_path) / "relation_glossary.json"
    model._relation_glossary = load_relation_glossary(glossary_path)
    return model, tokenizer


@torch.no_grad()
def generate_causal_questions(
    model: Any,
    tokenizer: Any,
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    glossary = model._relation_glossary
    generations: list[str] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        prompts = [
            causal_generation_prompt(tokenizer, path, glossary)
            for path in batch_paths
        ]
        width = max(len(prompt) for prompt in prompts)
        input_ids = torch.tensor(
            [
                [tokenizer.pad_token_id] * (width - len(prompt)) + prompt
                for prompt in prompts
            ],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [
                [0] * (width - len(prompt)) + [1] * len(prompt)
                for prompt in prompts
            ],
            dtype=torch.long,
            device=device,
        )
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            output_ids[:, width:], skip_special_tokens=True
        )
        for question, path in zip(decoded, batch_paths, strict=True):
            generations.append(
                question.strip().replace(ENTITY_PLACEHOLDER, path["anchor"])
            )
    return generations


@torch.no_grad()
def score_causal_question_likelihood(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    paths: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> list[float]:
    glossary = model._relation_glossary
    scores: list[float] = []
    collator = CausalInverseCollator(tokenizer.pad_token_id)
    for start in range(0, len(paths), batch_size):
        rows = []
        for question, path in zip(
            questions[start : start + batch_size],
            paths[start : start + batch_size],
            strict=True,
        ):
            prompt = causal_generation_prompt(tokenizer, path, glossary)
            target = tokenizer(
                question + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"]
            rows.append(
                {
                    "input_ids": prompt + target,
                    "labels": [-100] * len(prompt) + target,
                }
            )
        batch = {key: value.to(device) for key, value in collator(rows).items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits[:, :-1]
        labels = batch["labels"][:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(labels.shape)
        mask = labels.ne(-100)
        nll = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        scores.extend((-nll).cpu().tolist())
    return scores
