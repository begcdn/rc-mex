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


SCORER_KINDS = (
    "cross_encoder",
    "nli_bidirectional",
    "bi_encoder",
    "qwen3_reranker",
    "bge_gemma",
)

# Instruction-following rerankers take the relevance criterion in words, so the
# rubric used in the manual judging can be stated directly rather than learned.
# It is deliberately permissive about surface form: the hand labelling counted
# extra words, a named answer type, plural/singular and partial names as matches,
# and only reversed direction, a different relation, or a wrong added description
# as mismatches.
EQUIVALENCE_INSTRUCTION = (
    "Given a user's question, find the question that asks for the same thing. "
    "Judge meaning, not wording. Two questions still match if one is longer, "
    "names the type of answer it wants (\"which author\" for \"who\"), uses "
    "singular where the other uses plural, or uses a partial or alternative name "
    "for the same entity. They do not match if a relation runs in the opposite "
    "direction, if a different relation is used, or if a description is added that "
    "is wrong or that changes which things would be returned."
)


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

    if kind == "qwen3_reranker":
        return _score_qwen3(pairs, model_name, resolved, batch_size)

    if kind == "bge_gemma":
        return _score_bge_gemma(pairs, model_name, resolved, batch_size)

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


def _score_qwen3(
    pairs: list[tuple[str, str]],
    model_name: str,
    device: str,
    batch_size: int,
    instruction: str = EQUIVALENCE_INSTRUCTION,
) -> list[float]:
    """Score with a Qwen3 reranker, which answers yes/no rather than emitting a logit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based "
        "on the Query and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        texts = [
            f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{suffix}"
            for query, doc in pairs[start : start + batch_size]
        ]
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).log_softmax(dim=1)
        scores.extend(pair[:, 1].cpu().tolist())
    return scores


def _score_bge_gemma(
    pairs: list[tuple[str, str]],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int = 512,
) -> list[float]:
    """Score with BAAI's LLM-based reranker, which uses its own prompt layout.

    Format is taken from the model card: the query prefixed "A: ", the passage
    prefixed "B: ", the instruction appended last, and the score read from the
    logit of the "Yes" token at the final position. Feeding it the Qwen3 chat
    template instead would still produce numbers, which is why the layout is
    reproduced rather than guessed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = (
        "Given a query A and a passage B, determine whether the passage contains an "
        "answer to the query by providing a prediction of either 'Yes' or 'No'."
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()
    yes_loc = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sep_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]

    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        items = []
        for query, passage in pairs[start : start + batch_size]:
            query_ids = tokenizer(
                f"A: {query}",
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True,
            )["input_ids"]
            passage_ids = tokenizer(
                f"B: {passage}",
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )["input_ids"]
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_ids,
                sep_ids + passage_ids,
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_ids + prompt_ids
            item["attention_mask"] = [1] * len(item["input_ids"])
            items.append(item)
        encoded = tokenizer.pad(
            items,
            padding=True,
            max_length=max_length + len(sep_ids) + len(prompt_ids),
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded, return_dict=True).logits[:, -1, yes_loc]
        scores.extend(logits.view(-1).float().cpu().tolist())
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
        # Every candidate's score, so independently-scored runs can be combined
        # afterwards without re-running any model.
        cursor = 0
        dumped = []
        for row in rows:
            scored = []
            for candidate in row["candidate_log"]:
                scored.append(
                    {
                        "relation_sequence": candidate["relation_sequence"],
                        "score": scores[cursor],
                        "matches_gold_path": candidate["matches_gold_path"],
                    }
                )
                cursor += 1
            dumped.append({"question_id": row["question_id"], "candidates": scored})
        write_jsonl(output / "scores.jsonl", dumped)
    return result


def rebuild_paths(predictions: list[dict[str, Any]], graphs: Path) -> None:
    """Fill in each candidate's path by re-executing its relations on the graph.

    Runs before the path field was logged do not carry it, and the path-only and
    combined comparator modes need it. Re-executing the stored relation sequence
    reproduces exactly what the pipeline built, so no run has to be repeated.
    """
    from .retrieval import LocalQuestionGraph, materialize_path

    by_id = {row["question_id"]: row for row in predictions}
    with graphs.open(encoding="utf-8") as handle:
        for line in handle:
            graph_row = json.loads(line)
            row = by_id.get(graph_row["id"])
            if row is None:
                continue
            graph = LocalQuestionGraph(graph_row.get("graph", []))
            anchors = graph_row.get("q_entity", [])
            for candidate in row["candidate_log"]:
                if candidate.get("path"):
                    continue
                for anchor in anchors:
                    path, answers, _ = materialize_path(
                        graph, anchor, tuple(candidate["relation_sequence"])
                    )
                    if answers:
                        candidate["path"] = path
                        break


def rescore_with_comparator(
    predictions: Path,
    model_path: str,
    output: Path | None = None,
    graphs: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    endpoint_filter: bool = True,
) -> dict[str, Any]:
    """Re-rank a run with a trained comparator checkpoint, in whichever mode it stores."""
    from .comparator import (
        comparator_answer_evidence,
        comparator_path_text,
        load_comparator,
        score_comparator_rows,
    )
    from .data import unlabeled_answer_count

    started = time.time()
    rows = read_jsonl(predictions)
    model, tokenizer, resolved, mode = load_comparator(model_path, device)
    if mode != "question_generated" and graphs is not None:
        rebuild_paths(rows, graphs)

    scored_rows = []
    for row in rows:
        candidates = []
        for candidate in row["candidate_log"]:
            path = candidate.get("path") or {}
            answers = candidate.get("answers", [])
            candidates.append(
                {
                    **candidate,
                    "is_positive": False,
                    "negative_type": "unlabeled",
                    "path_text": comparator_path_text(path, ", ".join(answers[:10]))
                    if path
                    else "",
                    "answer_evidence": candidate.get("answer_evidence")
                    or comparator_answer_evidence(
                        answers, path.get("answer_type"), unlabeled_answer_count(answers)
                    ),
                }
            )
        scored_rows.append(
            {
                "example_id": row["question_id"],
                "original_question": row["question"],
                "candidates": candidates,
            }
        )
    scored = score_comparator_rows(
        model, tokenizer, scored_rows, mode, resolved, batch_size
    )

    gold_path = gold_equivalent = 0
    totals = {"exact_match": 0.0, "f1": 0.0, "has_correct_answer": 0.0}
    picks = []
    for row, result in zip(rows, scored, strict=True):
        candidates = result["candidates"]
        pool = candidates
        if endpoint_filter:
            pool = [c for c in candidates if has_answerable_endpoint(c)] or candidates
        best = max(pool, key=lambda c: c["cross_encoder_score"])
        gold_sequences = {tuple(s) for s in row.get("gold_sequences", [])}
        equivalent = gold_equivalent_answer_sets(candidates, gold_sequences)
        is_gold = tuple(best["relation_sequence"]) in gold_sequences
        gold_path += is_gold
        gold_equivalent += is_gold or answer_set_key(best["answers"]) in equivalent
        metrics = answer_metrics(best["answers"], row.get("gold_answers", []))
        for key in totals:
            totals[key] += metrics[key]
        # Per-question picks, so two runs can be compared on the same questions
        # rather than only by their averages.
        picks.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "relation_sequence": best["relation_sequence"],
                "generated_question": best["generated_question"],
                "matches_gold_path": is_gold,
                "exact_match": metrics["exact_match"],
                "f1": metrics["f1"],
                "answers": best["answers"][:20],
            }
        )

    count = max(len(rows), 1)
    result = {
        "predictions": str(predictions),
        "model": model_path,
        "input_mode": mode,
        "questions": len(rows),
        "selected_gold_path_accuracy": gold_path / count,
        "selected_gold_equivalent_accuracy": gold_equivalent / count,
        **{f"answer_{k}": v / count for k, v in totals.items()},
        "elapsed_seconds": time.time() - started,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_jsonl(output / "picks.jsonl", picks)
    return result
