from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .openai_naturalize import (
    OpenAIBatchClient,
    run_chat_records_sync,
)


JUDGE_SYSTEM_PROMPT = """You are labeling semantic equivalence for a question-comparison benchmark.

For every candidate, decide whether it asks for exactly the same answer set as the original question.
Equivalent wording may use different syntax or explicitly unfold a multi-hop relation.

Mark false if any meaningful difference changes the requested answers, including:
- reversed subject/object or relation direction;
- a missing or extra condition/hop;
- a different relation;
- a different order or intermediate dependency;
- a different requested answer role or answer type.

Judge only the text. You are not shown a knowledge-graph path and must not infer one.
If two candidate questions express the same intent, both may be true.
If a candidate is awkward but has one unambiguous interpretation, judge that interpretation.

Return JSON with an items array. Each item must contain:
id, equivalent (boolean), issue, confidence (0 to 1), and reason.
Use issue values: equivalent, direction, missing_constraint, extra_constraint,
wrong_relation, wrong_order, wrong_answer_role, ambiguous, or other."""


def semantic_judge_record(
    row: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    candidates = [
        {
            "id": f"candidate-{index}",
            "question": candidate["generated_question"],
        }
        for index, candidate in enumerate(row["candidates"])
    ]
    return {
        "custom_id": row["example_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_question": row["original_question"],
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
    }


def apply_semantic_labels(
    rows: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored, unscorable = [], []
    issue_counts: Counter[str] = Counter()
    path_positive_rejected = 0
    path_negative_accepted = 0
    candidates_total = 0

    for row in rows:
        judgment = judgments.get(row["example_id"])
        if judgment is None:
            raise ValueError(f"missing semantic judgment for {row['example_id']}")
        items = judgment.get("items")
        if not isinstance(items, list):
            raise ValueError(f"invalid semantic judgment for {row['example_id']}")
        by_id = {item.get("id"): item for item in items}
        candidates = []
        for index, candidate in enumerate(row["candidates"]):
            item = by_id.get(f"candidate-{index}")
            if item is None or not isinstance(item.get("equivalent"), bool):
                raise ValueError(
                    f"missing candidate-{index} judgment for {row['example_id']}"
                )
            equivalent = item["equivalent"]
            path_positive = bool(candidate["is_positive"])
            candidates_total += 1
            issue_counts[str(item.get("issue", "other"))] += 1
            path_positive_rejected += path_positive and not equivalent
            path_negative_accepted += not path_positive and equivalent
            candidates.append(
                {
                    **candidate,
                    "path_is_positive": path_positive,
                    "is_positive": equivalent,
                    "semantic_judgment": {
                        "issue": item.get("issue", "other"),
                        "confidence": item.get("confidence"),
                        "reason": item.get("reason", ""),
                    },
                }
            )
        labeled = {**row, "candidates": candidates}
        if any(candidate["is_positive"] for candidate in candidates):
            scored.append(labeled)
        else:
            unscorable.append(labeled)

    summary = {
        "candidate_sets": len(rows),
        "scorable_candidate_sets": len(scored),
        "no_equivalent_candidate_sets": len(unscorable),
        "candidates": candidates_total,
        "path_positive_rejected_as_not_equivalent": path_positive_rejected,
        "path_negative_accepted_as_equivalent": path_negative_accepted,
        "path_semantic_label_disagreement_rate": (
            path_positive_rejected + path_negative_accepted
        )
        / max(candidates_total, 1),
        "semantic_issue_counts": dict(sorted(issue_counts.items())),
    }
    return scored, unscorable, summary


def parse_semantic_results(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    judgments: dict[str, dict[str, Any]] = {}
    errors = []
    for path in paths:
        for line in path.open(encoding="utf-8"):
            result = json.loads(line)
            try:
                content = result["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                judgments[result["custom_id"]] = json.loads(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                errors.append(
                    {
                        "custom_id": result.get("custom_id"),
                        "parse_error": str(exc),
                    }
                )
    return judgments, errors


def build_semantic_benchmark(
    data: Path,
    output: Path,
    model: str = "gpt-4o-2024-11-20",
    workers: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(data / "test_kqa_val.jsonl")
    if limit is not None:
        rows = rows[:limit]
    output.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for semantic judgment")
    client = OpenAIBatchClient(api_key)
    records = [semantic_judge_record(row, model) for row in rows]
    result_files = run_chat_records_sync(
        records,
        output / ".semantic_judge",
        client,
        "semantic equivalence judgment",
        workers=workers,
    )
    judgments, errors = parse_semantic_results(result_files)
    if errors:
        raise RuntimeError(
            f"{len(errors)} semantic judgments could not be parsed; "
            f"inspect {output / '.semantic_judge'}"
        )

    scored, unscorable, overall = apply_semantic_labels(rows, judgments)
    scored_by_id = {row["example_id"]: row for row in scored}
    split_summaries: dict[str, Any] = {}
    for split_path in sorted(data.glob("*.jsonl")):
        split = split_path.stem
        ids = {row["example_id"] for row in read_jsonl(split_path)}
        split_rows = [
            scored_by_id[row["example_id"]]
            for row in rows
            if row["example_id"] in ids and row["example_id"] in scored_by_id
        ]
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_summaries[split] = {
            "source_candidate_sets": sum(row["example_id"] in ids for row in rows),
            "scorable_candidate_sets": len(split_rows),
        }

    with (output / "unscorable.jsonl").open("w", encoding="utf-8") as handle:
        for row in unscorable:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "judge_model": model,
        "label_definition": (
            "original and candidate questions request exactly the same answer set"
        ),
        "path_labels_hidden_from_judge": True,
        "overall": overall,
        "splits": split_summaries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
