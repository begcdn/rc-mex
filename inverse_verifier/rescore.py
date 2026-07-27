"""Re-score a finished run with a different comparator.

The pipeline's candidate log already stores every (original question, generated
question) pair it produced, and comparators score pairs independently, so
swapping the comparator needs no GPU pass over the graph -- only a rescoring of
text pairs. That makes it cheap to ask the question the architecture rests on:
does an off-the-shelf model that has never seen a knowledge graph do better than
the comparator trained here?

A general paraphrase or entailment model beating the trained one would mean the
architecture is sound and the training was the weak link. None of them beating it
would mean choosing among ~98 near-identical questions is intrinsically hard, and
no amount of extra training data fixes that.

The searcher's own score is never used. Selection is by comparator score alone,
which is the only way the number measures the verifier rather than the proposer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .data import read_jsonl, write_jsonl
from .selector import (
    answer_metrics,
    answer_set_key,
    gold_equivalent_answer_sets,
    has_answerable_endpoint,
)


SCORER_KINDS = ("cross_encoder", "nli_bidirectional", "bi_encoder")


def _device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_pairs(
    pairs: list[tuple[str, str]],
    model_name: str,
    kind: str,
    device: str = "auto",
    batch_size: int = 64,
) -> list[float]:
    if kind not in SCORER_KINDS:
        raise ValueError(f"unknown scorer kind: {kind}")
    resolved = _device(device)

    if kind == "bi_encoder":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=resolved)
        left = model.encode([a for a, _ in pairs], batch_size=batch_size, normalize_embeddings=True)
        right = model.encode([b for _, b in pairs], batch_size=batch_size, normalize_embeddings=True)
        return [float((left[i] * right[i]).sum()) for i in range(len(pairs))]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(resolved).eval()

    def run(batch: list[tuple[str, str]]) -> torch.Tensor:
        encoded = tokenizer(
            [a for a, _ in batch],
            [b for _, b in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(resolved)
        with torch.no_grad():
            return model(**encoded).logits.float().cpu()

    if kind == "cross_encoder":
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            logits = run(pairs[start : start + batch_size])
            # Single-logit relevance heads and 2-way paraphrase heads both occur.
            scores.extend(
                (logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]).tolist()
            )
        return scores

    # Bidirectional entailment: equivalence means each question implies the other,
    # which is the strict reading the manual judging used. NLI models are trained on
    # far more data than paraphrase models, so this is the stronger general prior.
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    entail = labels.get("entailment")
    if entail is None:
        raise ValueError(f"{model_name} has no entailment label: {model.config.id2label}")
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        forward = run(batch).log_softmax(dim=-1)[:, entail]
        backward = run([(b, a) for a, b in batch]).log_softmax(dim=-1)[:, entail]
        scores.extend((forward + backward).tolist())
    return scores


def rescore_run(
    predictions: Path,
    model_name: str,
    kind: str,
    output: Path | None = None,
    device: str = "auto",
    batch_size: int = 64,
    endpoint_filter: bool = True,
) -> dict[str, Any]:
    started = time.time()
    rows = read_jsonl(predictions)
    pairs: list[tuple[str, str]] = []
    for row in rows:
        for candidate in row["candidate_log"]:
            pairs.append((row["question"], candidate["generated_question"]))
    scores = score_pairs(pairs, model_name, kind, device, batch_size)

    cursor = 0
    gold_path = gold_equivalent = 0
    totals = {"exact_match": 0.0, "f1": 0.0, "has_correct_answer": 0.0}
    selections = []
    for row in rows:
        candidates = []
        for candidate in row["candidate_log"]:
            candidates.append({**candidate, "rescore": scores[cursor]})
            cursor += 1
        pool = candidates
        if endpoint_filter:
            pool = [c for c in candidates if has_answerable_endpoint(c)] or candidates
        best = max(pool, key=lambda c: c["rescore"])

        gold_sequences = {tuple(s) for s in row.get("gold_sequences", [])}
        equivalent = gold_equivalent_answer_sets(candidates, gold_sequences)
        is_gold = tuple(best["relation_sequence"]) in gold_sequences
        gold_path += is_gold
        gold_equivalent += is_gold or answer_set_key(best["answers"]) in equivalent
        metrics = answer_metrics(best["answers"], row.get("gold_answers", []))
        for key in totals:
            totals[key] += metrics[key]
        selections.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "relation_sequence": best["relation_sequence"],
                "generated_question": best["generated_question"],
                "rescore": best["rescore"],
                "matches_gold_path": is_gold,
                "answers": best["answers"][:20],
            }
        )

    count = max(len(rows), 1)
    result = {
        "predictions": str(predictions),
        "model": model_name,
        "scorer": kind,
        "endpoint_filter": endpoint_filter,
        "questions": len(rows),
        "candidate_pairs": len(pairs),
        "selected_gold_path_accuracy": gold_path / count,
        "selected_gold_equivalent_accuracy": gold_equivalent / count,
        **{f"answer_{k}": v / count for k, v in totals.items()},
        "elapsed_seconds": time.time() - started,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_jsonl(output / "selections.jsonl", selections)
    return result
