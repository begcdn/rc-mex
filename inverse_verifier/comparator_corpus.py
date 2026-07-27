"""Build comparator training data from real pipeline candidates.

The deployed comparator was trained on synthetic negatives -- added hops, missing
hops, sibling relations, direction flips -- and then asked to rank real proposals.
Its dominant live failure, choosing a semantically adjacent relation over the
right one, is not one of those categories at all, so training and inference see
different distributions. Its synthetic dev set scores 0.991 and cannot separate
input modes; on real candidates it picks the annotated path 41% of the time.

This module converts a `verify` run into listwise training data made of the
candidates the pipeline actually produces. Gold is used to label, which is
allowed during training, and never to select.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .comparator import comparator_answer_evidence, comparator_path_text
from .data import read_jsonl, unlabeled_answer_count, write_jsonl


DEV_FRACTION = 0.1
HARD_NEGATIVES = 12
RANDOM_NEGATIVES = 4
MAX_POSITIVES = 4


def answer_key(answers: list[str]) -> tuple[str, ...]:
    return tuple(sorted({answer.casefold().strip() for answer in answers if answer.strip()}))


def label_candidates(
    candidates: list[dict[str, Any]],
    gold_answers: list[str],
    gold_sequences: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Mark a candidate positive when it answers the question, however it got there.

    Labelling by annotated-path identity alone is wrong in both directions: 15 of
    90 audited annotated paths do not answer their own question, and Freebase
    reaches the same answers by several equally valid routes. A candidate counts
    as positive when it returns exactly the gold answer set, or when it is an
    annotated path.
    """
    gold = answer_key(gold_answers)
    labelled = []
    for candidate in candidates:
        is_gold_path = tuple(candidate["relation_sequence"]) in gold_sequences
        answers_match = bool(gold) and answer_key(candidate.get("answers", [])) == gold
        labelled.append(
            {
                **candidate,
                "is_positive": bool(is_gold_path or answers_match),
                "negative_type": (
                    "positive"
                    if is_gold_path
                    else "answer_equivalent"
                    if answers_match
                    else "proposed"
                ),
            }
        )
    return labelled


def sample_candidates(
    labelled: list[dict[str, Any]],
    rng: random.Random,
    hard: int = HARD_NEGATIVES,
    random_count: int = RANDOM_NEGATIVES,
    max_positives: int = MAX_POSITIVES,
) -> list[dict[str, Any]] | None:
    """Keep a trainable slice of a ~98-candidate set.

    Every group must contain a positive for the listwise loss, so groups without
    one are dropped rather than forced. Negatives are drawn hardest-first, by the
    incumbent comparator's own score: those are the candidates it currently
    prefers over the right answer, which is precisely what it has to learn to
    reject. A few random negatives keep the easy region represented.
    """
    positives = [c for c in labelled if c["is_positive"]]
    negatives = [c for c in labelled if not c["is_positive"]]
    if not positives:
        return None
    if len(positives) > max_positives:
        positives = sorted(positives, key=lambda c: -c.get("score", 0.0))[:max_positives]

    ranked = sorted(negatives, key=lambda c: -c.get("score", 0.0))
    chosen = ranked[:hard]
    remaining = ranked[hard:]
    if remaining and random_count:
        chosen += rng.sample(remaining, min(random_count, len(remaining)))
    return positives + chosen


def build_comparator_corpus(
    predictions: Path,
    output: Path,
    seed: int = 17,
    dev_fraction: float = DEV_FRACTION,
    hard: int = HARD_NEGATIVES,
    random_count: int = RANDOM_NEGATIVES,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = read_jsonl(predictions)
    built: list[dict[str, Any]] = []
    dropped = 0
    category: Counter[str] = Counter()

    for row in rows:
        gold_sequences = {tuple(sequence) for sequence in row.get("gold_sequences", [])}
        labelled = label_candidates(
            row.get("candidate_log", []), row.get("gold_answers", []), gold_sequences
        )
        sampled = sample_candidates(labelled, rng, hard, random_count)
        if sampled is None:
            dropped += 1
            continue
        candidates = []
        for candidate in sampled:
            answers = candidate.get("answers", [])
            path = candidate.get("path") or {}
            category[candidate["negative_type"]] += 1
            candidates.append(
                {
                    "path": path,
                    "is_positive": candidate["is_positive"],
                    "negative_type": candidate["negative_type"],
                    "answer_entity": answers[0] if answers else "",
                    "generated_question": candidate["generated_question"],
                    "path_text": comparator_path_text(path, ", ".join(answers[:10]))
                    if path
                    else "",
                    "answer_evidence": candidate.get("answer_evidence")
                    or comparator_answer_evidence(
                        answers, path.get("answer_type"), unlabeled_answer_count(answers)
                    ),
                }
            )
        built.append(
            {
                "example_id": row["question_id"],
                "kg": "webqsp",
                "original_question": row["question"],
                "candidates": candidates,
            }
        )

    rng.shuffle(built)
    split = max(1, round(len(built) * dev_fraction)) if built else 0
    dev, train = built[:split], built[split:]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "dev.jsonl", dev)

    manifest = {
        "source": str(predictions),
        "source_questions": len(rows),
        "candidate_sets": len(built),
        "dropped_no_positive": dropped,
        "train_candidate_sets": len(train),
        "dev_candidate_sets": len(dev),
        "candidates": sum(len(row["candidates"]) for row in built),
        "positive_candidates": sum(
            candidate["is_positive"] for row in built for candidate in row["candidates"]
        ),
        "category_counts": dict(sorted(category.items())),
        "negative_sampling": {"hard": hard, "random": random_count},
        "labelling": "annotated path or exact gold answer-set match",
        "seed": seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
