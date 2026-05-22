from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Callable


@dataclass
class RankingRecord:
    instance_id: str
    gold_predicate: str
    gold_direction: str
    ranked_pairs: list[tuple[str, str]]
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0


def ranking_metrics(records: list[RankingRecord]) -> dict[str, float]:
    if not records:
        return {}
    return {
        "n": float(len(records)),
        "relation_recall@1": mean(metric_relation_recall_at(record, 1) for record in records),
        "relation_recall@3": mean(metric_relation_recall_at(record, 3) for record in records),
        "relation_recall@5": mean(metric_relation_recall_at(record, 5) for record in records),
        "pair_recall@1": mean(metric_pair_recall_at(record, 1) for record in records),
        "pair_recall@3": mean(metric_pair_recall_at(record, 3) for record in records),
        "pair_recall@5": mean(metric_pair_recall_at(record, 5) for record in records),
        "mrr_pair": mean(metric_pair_mrr(record) for record in records),
        "relation_accuracy_ignore_direction": mean(metric_relation_recall_at(record, 1) for record in records),
        "direction_accuracy_given_gold_relation": mean(metric_direction_given_relation(record) for record in records),
        "avg_llm_calls": mean(record.llm_calls for record in records),
        "avg_prompt_tokens": mean(record.prompt_tokens for record in records),
        "avg_completion_tokens": mean(record.completion_tokens for record in records),
        "avg_latency_seconds": mean(record.latency_seconds for record in records),
    }


def metric_relation_recall_at(record: RankingRecord, k: int) -> float:
    return float(any(predicate == record.gold_predicate for predicate, _ in record.ranked_pairs[:k]))


def metric_pair_recall_at(record: RankingRecord, k: int) -> float:
    return float((record.gold_predicate, record.gold_direction) in record.ranked_pairs[:k])


def metric_pair_mrr(record: RankingRecord) -> float:
    gold = (record.gold_predicate, record.gold_direction)
    for idx, pair in enumerate(record.ranked_pairs, start=1):
        if pair == gold:
            return 1.0 / idx
    return 0.0


def metric_direction_given_relation(record: RankingRecord) -> float:
    for predicate, direction in record.ranked_pairs:
        if predicate == record.gold_predicate:
            return float(direction == record.gold_direction)
    return 0.0


def bootstrap_ci(
    records: list[RankingRecord],
    metric_fn: Callable[[list[RankingRecord]], float],
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not records:
        return (0.0, 0.0, 0.0)
    point = metric_fn(records)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sample = [records[rng.randrange(len(records))] for _ in records]
        values.append(metric_fn(sample))
    values.sort()
    lo_idx = max(0, int((alpha / 2) * samples) - 1)
    hi_idx = min(samples - 1, int((1 - alpha / 2) * samples))
    return point, values[lo_idx], values[hi_idx]


def add_metric_cis(
    records: list[RankingRecord],
    metrics: dict[str, float],
    samples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    ci_metrics = {
        "relation_recall@1": lambda rs: mean(metric_relation_recall_at(record, 1) for record in rs),
        "relation_recall@5": lambda rs: mean(metric_relation_recall_at(record, 5) for record in rs),
        "pair_recall@1": lambda rs: mean(metric_pair_recall_at(record, 1) for record in rs),
        "pair_recall@5": lambda rs: mean(metric_pair_recall_at(record, 5) for record in rs),
        "mrr_pair": lambda rs: mean(metric_pair_mrr(record) for record in rs),
    }
    out: dict[str, float | list[float]] = dict(metrics)
    for name, fn in ci_metrics.items():
        point, lo, hi = bootstrap_ci(records, fn, samples=samples, seed=seed)
        out[f"{name}_ci95"] = [point, lo, hi]
    return out
