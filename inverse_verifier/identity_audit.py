"""Identity-hop audit for name-collapsed graphs.

The WebQSP graphs this project executes over store entity display names rather
than Freebase machine ids, so distinct entities sharing a name collapse into one
node and the relation between them becomes a self-loop. Traversing such a hop is
a no-op, and any candidate path containing one is execution-equivalent to a
shorter path.

This module measures how much that matters:

  ``audit_predictions``  collapse reflexive hops in an existing candidate pool,
                         merge candidates that become identical, re-select, and
                         compare answer metrics before and after. No retraining.

Reflexivity is judged per hop against the node set it actually acts on, which is
the conservative definition: it only drops a hop when traversal leaves the node
set unchanged in this graph.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from .retrieval import LocalQuestionGraph


def strict_reduce(graph: LocalQuestionGraph, topic: str, path: tuple[str, ...]) -> tuple[str, ...]:
    """Drop hops that leave the current node set unchanged."""
    nodes = {topic}
    kept: list[str] = []
    for edge in path:
        following: set[str] = set()
        for node in nodes:
            following.update(t for candidate, t in graph.adjacency.get(node, []) if candidate == edge)
        if not following:
            return tuple(kept) + tuple(path[len(kept):])
        if following == nodes:
            continue
        kept.append(edge)
        nodes = following
    return tuple(kept)


def answer_scores(predicted: Iterable[str], gold: Iterable[str]) -> tuple[float, float]:
    """Exact match and F1 over answer sets."""
    predicted_set, gold_set = set(predicted), set(gold)
    exact = float(predicted_set == gold_set and bool(gold_set))
    if not predicted_set or not gold_set:
        return exact, 0.0
    overlap = len(predicted_set & gold_set)
    if not overlap:
        return exact, 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return exact, 2 * precision * recall / (precision + recall)


def load_graphs(graph_path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with graph_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                rows[row["id"]] = row
                if len(rows) == len(wanted):
                    break
    return rows


def stream_predictions(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def audit_predictions(predictions_path: Path, graph_path: Path) -> dict[str, Any]:
    """Reducibility by supervision label, and path match under collapsed comparison.

    Deliberately does NOT report a before/after answer metric from re-selecting a
    merged pool. Two such comparisons are invariant by construction and measure
    nothing: removing a no-op cannot change the answer set, and merging duplicates
    while keeping each group's highest score always preserves the global argmax.
    Whether cleaned paths would change a *trained* model's behaviour is a separate
    question this function cannot answer.
    """
    predictions = list(stream_predictions(predictions_path))
    graphs = load_graphs(graph_path, {row["question_id"] for row in predictions})

    counts: Counter = Counter()
    exact_match = collapsed_match = 0
    flips: list[dict[str, Any]] = []
    scored = 0

    for row in predictions:
        graph_row = graphs.get(row["question_id"])
        candidates = row.get("candidate_log") or []
        if not graph_row or not candidates or len(graph_row["q_entity"]) != 1:
            continue
        graph = LocalQuestionGraph(graph_row["graph"])
        topic = graph_row["q_entity"][0]
        gold_answers = set(row.get("gold_answers") or [])
        scored += 1

        for candidate in candidates:
            path = tuple(candidate["relation_sequence"])
            reducible = strict_reduce(graph, topic, path) != path
            annotated = bool(candidate.get("matches_gold_path"))
            denotation = bool(gold_answers) and set(candidate.get("answers") or []) == gold_answers
            for population, member in (
                ("all", True),
                ("annotated_positive", annotated),
                ("denotation_positive", denotation),
                ("denotation_only_positive", denotation and not annotated),
            ):
                if member:
                    counts[population] += 1
                    if reducible:
                        counts[f"{population}_reducible"] += 1

        # path match, exact sequence versus collapsed on both sides
        gold_sequences = {tuple(s) for s in row.get("gold_sequences") or []}
        gold_collapsed = {strict_reduce(graph, topic, s) for s in gold_sequences}
        pick = max(candidates, key=lambda c: c["score"])
        selected = tuple(pick["relation_sequence"])
        selected_collapsed = strict_reduce(graph, topic, selected)
        hit_exact = selected in gold_sequences
        hit_collapsed = selected_collapsed in gold_collapsed
        exact_match += hit_exact
        collapsed_match += hit_collapsed
        if hit_exact != hit_collapsed:
            flips.append(
                {
                    "question": row["question"],
                    "selected": list(selected),
                    "collapsed": list(selected_collapsed),
                    "exact": hit_exact,
                    "collapsed_hit": hit_collapsed,
                }
            )

    return {
        "questions": scored,
        "populations": {
            name: {
                "n": counts[name],
                "reducible": counts[f"{name}_reducible"],
                "rate": counts[f"{name}_reducible"] / counts[name] if counts[name] else 0.0,
            }
            for name in ("all", "annotated_positive", "denotation_positive", "denotation_only_positive")
        },
        "path_match": {
            "exact": exact_match / scored if scored else 0.0,
            "collapsed": collapsed_match / scored if scored else 0.0,
        },
        "flips": flips,
    }


def format_report(result: dict[str, Any]) -> str:
    lines = [f"questions: {result['questions']}", "",
             f"{'population':28s} {'n':>7s} {'reducible':>10s} {'rate':>8s}"]
    for name, cell in result["populations"].items():
        lines.append(f"{name:28s} {cell['n']:7d} {cell['reducible']:10d} {cell['rate']:8.2%}")
    match = result["path_match"]
    lines += [
        "",
        f"path match, exact sequence : {match['exact']:.3f}",
        f"path match, collapsed      : {match['collapsed']:.3f}",
        f"flips: {len(result['flips'])}",
    ]
    return "\n".join(lines)
