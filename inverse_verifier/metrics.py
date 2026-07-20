from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def token_f1(prediction: str, reference: str) -> float:
    predicted, gold = Counter(tokens(prediction)), Counter(tokens(reference))
    overlap = sum((predicted & gold).values())
    if not predicted or not gold:
        return float(predicted == gold)
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    left, right = tokens(prediction), tokens(reference)
    if not left or not right:
        return float(left == right)
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(left), lcs / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def ranking_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    reciprocal_ranks = []
    top1_credit = 0.0
    strict_top1 = 0
    top_ties = 0
    pair_total = pair_correct = 0
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        candidates = row["candidates"]
        positive_score = max(candidate[score_key] for candidate in candidates if candidate["is_positive"])
        higher = sum(candidate[score_key] > positive_score for candidate in candidates)
        tied = [candidate for candidate in candidates if abs(candidate[score_key] - positive_score) <= 1e-12]
        tied_positives = sum(candidate["is_positive"] for candidate in tied)
        expected_first_positive_rank = higher + (len(tied) + 1) / (tied_positives + 1)
        reciprocal_ranks.append(1.0 / expected_first_positive_rank)
        top_score = max(candidate[score_key] for candidate in candidates)
        top = [candidate for candidate in candidates if abs(candidate[score_key] - top_score) <= 1e-12]
        positive_top = sum(candidate["is_positive"] for candidate in top)
        top1_credit += positive_top / len(top)
        strict_top1 += positive_top > 0 and positive_top == len(top)
        top_ties += len(top) > 1
        for candidate in candidates:
            if candidate["is_positive"]:
                continue
            correct = float(positive_score > candidate[score_key])
            pair_total += 1
            pair_correct += correct
            by_type[candidate["negative_type"]].append(correct)
    total = max(len(rows), 1)
    return {
        "examples": len(rows),
        "recall_at_1": top1_credit / total,
        "strict_recall_at_1": strict_top1 / total,
        "top_score_tie_rate": top_ties / total,
        "mrr": sum(reciprocal_ranks) / total,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "pairwise_by_negative_type": {
            key: sum(values) / len(values) for key, values in sorted(by_type.items())
        },
    }
