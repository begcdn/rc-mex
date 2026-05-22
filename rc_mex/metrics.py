from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_metrics(cards: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[(row["condition_id"], row["card_variant"])].append(row)

    card_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        card_groups[(card["condition_id"], card["card_variant"])].append(card)

    out: dict[str, Any] = {}
    for key in sorted(set(groups) | set(card_groups)):
        condition_id, variant = key
        rows = groups.get(key, [])
        group_cards = card_groups.get(key, [])
        out[f"{condition_id}/{variant}"] = {
            **classification_metrics(rows),
            "n_cards": len(group_cards),
            "opaque_rate": opaque_rate(group_cards),
            "avg_card_confidence": average([float(card.get("confidence", 0.0) or 0.0) for card in group_cards]),
        }
    return out


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "n_validation_pairs": 0.0,
            "positive_accuracy": 0.0,
            "hard_negative_rejection_accuracy": 0.0,
            "random_negative_rejection_accuracy": 0.0,
            "swapped_direction_rejection_accuracy": 0.0,
            "f1": 0.0,
            "direction_accuracy": 0.0,
            "avg_validation_confidence": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_latency_seconds": 0.0,
        }
    positives = [row for row in rows if row["expected_label"]]
    hard = [row for row in rows if row["category"] == "hard_negative"]
    random_rows = [row for row in rows if row["category"] == "random_negative"]
    swapped = [row for row in rows if row["category"] == "swapped_direction"]
    tp = sum(1 for row in rows if row["expected_label"] and row["predicted_satisfies"])
    fp = sum(1 for row in rows if not row["expected_label"] and row["predicted_satisfies"])
    fn = sum(1 for row in rows if row["expected_label"] and not row["predicted_satisfies"])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    direction_rows = positives + swapped
    return {
        "n_validation_pairs": float(len(rows)),
        "positive_accuracy": accuracy(positives, expected_prediction=True),
        "hard_negative_rejection_accuracy": accuracy(hard, expected_prediction=False),
        "random_negative_rejection_accuracy": accuracy(random_rows, expected_prediction=False),
        "swapped_direction_rejection_accuracy": accuracy(swapped, expected_prediction=False),
        "f1": f1,
        "direction_accuracy": direction_accuracy(direction_rows),
        "avg_validation_confidence": average([float(row.get("confidence", 0.0) or 0.0) for row in rows]),
        "avg_prompt_tokens": average([float(row.get("prompt_tokens", 0.0) or 0.0) for row in rows]),
        "avg_completion_tokens": average([float(row.get("completion_tokens", 0.0) or 0.0) for row in rows]),
        "avg_latency_seconds": average([float(row.get("latency_seconds", 0.0) or 0.0) for row in rows]),
    }


def accuracy(rows: list[dict[str, Any]], expected_prediction: bool) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row["predicted_satisfies"]) is expected_prediction) / len(rows)


def direction_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        if row["category"] == "positive":
            correct += int(bool(row["predicted_direction_correct"]))
        elif row["category"] == "swapped_direction":
            correct += int(not bool(row["predicted_satisfies"]))
    return correct / len(rows)


def opaque_rate(cards: list[dict[str, Any]]) -> float:
    if not cards:
        return 0.0
    return sum(1 for card in cards if card.get("opaque_reason")) / len(cards)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
