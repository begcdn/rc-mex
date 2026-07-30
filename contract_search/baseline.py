from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .program import answer_set_f1
from .search import (
    LexicalScorer,
    TransformerPairScorer,
    search_question,
    program_record,
    serialize_program,
)
from .substrate import GNNRAGDataset


def run_baseline(
    substrate: Path,
    output: Path,
    scorer_model: str | None,
    device: str,
    limit: int | None = None,
) -> dict[str, Any]:
    scorer = (
        TransformerPairScorer(scorer_model, device)
        if scorer_model
        else LexicalScorer()
    )
    reported = scorer_model is not None
    rows = []
    for index, question in enumerate(GNNRAGDataset(substrate, "dev")):
        if limit is not None and index >= limit:
            break
        before_calls = scorer.calls
        result = search_question(question, scorer)
        predicted = result.state.answers if result.state else frozenset()
        f1 = answer_set_f1(predicted, question.gold_answers)
        rows.append(
            {
                "id": question.question_id,
                "question": question.question,
                "gold_answers": list(question.gold_answers),
                "predicted_answers": sorted(predicted),
                "exact": set(predicted) == set(question.gold_answers),
                "f1": f1,
                "program": (
                    serialize_program(result.state.program, question.topic_entities)
                    if result.state
                    else None
                ),
                "program_state": (
                    program_record(result.state.program) if result.state else None
                ),
                "stopped_by_finish": result.stopped_by_finish,
                "cap_reached": result.cap_reached,
                "scored_expansions": result.scored_expansions,
                "scorer_calls": scorer.calls - before_calls,
                "trace": result.trace,
            }
        )
        if index == 0 or (index + 1) % 25 == 0:
            print(f"baseline: {index + 1} questions", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    metrics = {
        "dataset": substrate.name.casefold(),
        "split": "gnnrag_dev",
        "substrate": "gnnrag_mid_keyed",
        "questions": len(rows),
        "reported_result": reported,
        "scorer": scorer_model or "lexical_smoke_only",
        "exact": sum(row["exact"] for row in rows) / len(rows) if rows else 0.0,
        "mean_f1": sum(row["f1"] for row in rows) / len(rows) if rows else 0.0,
        "mean_scored_expansions": (
            sum(row["scored_expansions"] for row in rows) / len(rows) if rows else 0.0
        ),
        "total_scorer_calls": sum(row["scorer_calls"] for row in rows),
        "finish_rate": (
            sum(row["stopped_by_finish"] for row in rows) / len(rows) if rows else 0.0
        ),
        "cap_rate": (
            sum(row["cap_reached"] for row in rows) / len(rows) if rows else 0.0
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
