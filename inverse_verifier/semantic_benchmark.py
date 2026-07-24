from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .openai_naturalize import (
    OpenAIBatchClient,
    run_chat_records_sync,
)


JUDGE_SYSTEM_PROMPT = """You are labeling strict semantic equivalence for a question-comparison benchmark.

For every candidate, decide whether it asks for exactly the same answer set as the original question.
Equivalent wording may use different syntax or explicitly unfold a multi-hop relation.

Before deciding, compare:
1. the requested answer role and its semantic type;
2. every relation and its subject/object direction;
3. every intermediate dependency;
4. every modifier, restriction, tense, and quantifier.

Mark false if any difference could change the answer set, including:
- reversed subject/object or relation direction;
- a missing or extra condition/hop;
- a different relation;
- a different order or intermediate dependency;
- a different requested answer role or answer type.
- broader or narrower wording;
- present tense versus "has ever";
- a specific relation weakened to generic association.

Examples of non-equivalence:
- "signed to label X" is not equivalent to "associated with label X";
- "which movie" is not equivalent to "which visual artwork";
- "who resides in X" is not equivalent to "who resides or has resided in X".

Judge only the text. You are not shown a knowledge-graph path and must not infer one.
If two candidate questions express the same intent, both may be true.
Do not repair a candidate using the original question or assume a dataset-specific
meaning that is not expressed in its words. If awkward wording has one literal,
unambiguous interpretation, judge that literal interpretation.

Return JSON with an items array. Each item must contain:
id, equivalent (boolean), issue, confidence (0 to 1), and reason.
Use issue values: equivalent, direction, missing_constraint, extra_constraint,
wrong_relation, wrong_order, wrong_answer_role, broader_or_narrower,
tense_or_quantifier, ambiguous, or other."""

ADJUDICATOR_SYSTEM_PROMPT = """You are independently adjudicating whether two questions request exactly the same answer set.

Use a difference-first procedure:
1. State the answer role and answer type requested by each question.
2. State each relation, its direction, and each constraint in each question.
3. List every meaning difference that could change the answer set.
4. Mark equivalent=true only when no such difference remains.

Important distinctions:
- A symmetric relation such as spouse, partner, sibling, or shares-a-border-with
  does not change meaning merely because its surface wording reverses the entities.
- Present-tense relational wording such as "which event follows X" can express the
  same timeless relation as "which event came after X"; do not invent a temporal
  restriction unless the words actually impose one.
- A specific relation is not equivalent to a generic association.
- Headquarters location, founding location, operating area, and representation are
  different relations unless the wording explicitly makes them the same.
- A broader or narrower answer type is not equivalent when it can change the answer set.
- Entity-name changes, missing dependencies, and extra dependencies are not equivalent.
- Do not repair either question or infer a hidden knowledge-graph relation.

Return one JSON object containing:
equivalent (boolean), answer_role_comparison, relation_comparison,
constraint_comparison, differences (array of strings), issue, confidence (0 to 1),
and reason.
Use issue values: equivalent, direction, missing_constraint, extra_constraint,
wrong_relation, wrong_order, wrong_answer_role, broader_or_narrower,
tense_or_quantifier, ambiguous, or other."""


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


def load_semantic_benchmark_rows(data: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(data / "test_kqa_val.jsonl")
    rows.extend(read_jsonl(data / "unscorable.jsonl"))
    return sorted(rows, key=lambda row: row["example_id"])


def select_adjudication_candidates(
    rows: list[dict[str, Any]],
    agreement_sample: int = 30,
    seed: int = 17,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    agreements = []
    for row in rows:
        for index, candidate in enumerate(row["candidates"]):
            semantic_positive = bool(candidate["is_positive"])
            path_positive = bool(candidate["path_is_positive"])
            issue = candidate.get("semantic_judgment", {}).get("issue")
            item = {
                "example_id": row["example_id"],
                "candidate_index": index,
                "original_question": row["original_question"],
                "candidate_question": candidate["generated_question"],
                "first_label": semantic_positive,
                "first_judgment": candidate.get("semantic_judgment", {}),
                "path_is_positive": path_positive,
                "negative_type": candidate.get("negative_type"),
            }
            key = (row["example_id"], index)
            if semantic_positive != path_positive:
                item["selection_reason"] = "path_semantic_disagreement"
                selected[key] = item
            elif issue == "ambiguous":
                item["selection_reason"] = "ambiguous"
                selected[key] = item
            else:
                item["selection_reason"] = "agreement_sample"
                agreements.append(item)

    generator = random.Random(seed)
    generator.shuffle(agreements)
    for item in agreements[:agreement_sample]:
        key = (item["example_id"], item["candidate_index"])
        selected.setdefault(key, item)
    return sorted(
        selected.values(),
        key=lambda item: (item["example_id"], item["candidate_index"]),
    )


def semantic_adjudication_record(
    item: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    return {
        "custom_id": adjudication_id(item["example_id"], item["candidate_index"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": ADJUDICATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_question": item["original_question"],
                            "candidate_question": item["candidate_question"],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
    }


def adjudication_id(example_id: str, candidate_index: int) -> str:
    return f"{example_id.replace(':', '-')}-candidate-{candidate_index}"


def apply_adjudication(
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adjudication_rows = []
    disputed_keys = set()
    reason_counts: Counter[str] = Counter()
    for item in selected:
        key = (item["example_id"], item["candidate_index"])
        judgment = judgments.get(adjudication_id(*key))
        if judgment is None or not isinstance(judgment.get("equivalent"), bool):
            raise ValueError(f"invalid adjudication judgment for {adjudication_id(*key)}")
        second_label = judgment["equivalent"]
        agreed = second_label == item["first_label"]
        reason_counts[item["selection_reason"]] += 1
        if not agreed:
            disputed_keys.add(key)
        adjudication_rows.append(
            {
                **item,
                "second_label": second_label,
                "judges_agree": agreed,
                "second_judgment": judgment,
            }
        )

    clean, disputed = [], []
    for row in rows:
        candidates = []
        row_disputed = False
        for index, candidate in enumerate(row["candidates"]):
            is_disputed = (row["example_id"], index) in disputed_keys
            row_disputed = row_disputed or is_disputed
            candidates.append({**candidate, "adjudication_disputed": is_disputed})
        labeled = {**row, "candidates": candidates}
        (disputed if row_disputed else clean).append(labeled)

    summary = {
        "candidate_sets": len(rows),
        "candidates_adjudicated": len(adjudication_rows),
        "first_second_agreements": sum(
            item["judges_agree"] for item in adjudication_rows
        ),
        "first_second_disagreements": len(disputed_keys),
        "candidate_sets_excluded_as_disputed": len(disputed),
        "clean_candidate_sets": len(clean),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
    }
    return clean, disputed, adjudication_rows, summary


def parse_adjudication_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    judgments: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.open(encoding="utf-8"):
            result = json.loads(line)
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            judgments[result["custom_id"]] = json.loads(content)
    return judgments


def adjudicate_semantic_benchmark(
    data: Path,
    output: Path,
    model: str = "gpt-4o-2024-11-20",
    workers: int = 3,
    agreement_sample: int = 30,
    seed: int = 17,
) -> dict[str, Any]:
    rows = load_semantic_benchmark_rows(data)
    selected = select_adjudication_candidates(rows, agreement_sample, seed)
    output.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for semantic adjudication")
    client = OpenAIBatchClient(api_key)
    result_files = run_chat_records_sync(
        [semantic_adjudication_record(item, model) for item in selected],
        output / ".semantic_adjudicator",
        client,
        "semantic equivalence adjudication",
        workers=workers,
    )
    judgments = parse_adjudication_results(result_files)
    clean, disputed, adjudication_rows, overall = apply_adjudication(
        rows, selected, judgments
    )
    clean_by_id = {row["example_id"]: row for row in clean}

    split_summaries = {}
    for split_path in sorted(data.glob("*.jsonl")):
        if split_path.stem == "unscorable":
            continue
        ids = {row["example_id"] for row in read_jsonl(split_path)}
        split_rows = [
            row
            for example_id, row in clean_by_id.items()
            if example_id in ids
            and any(candidate["is_positive"] for candidate in row["candidates"])
        ]
        with (output / split_path.name).open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_summaries[split_path.stem] = {
            "source_candidate_sets": len(ids),
            "clean_scorable_candidate_sets": len(split_rows),
        }

    with (output / "adjudication.jsonl").open("w", encoding="utf-8") as handle:
        for row in adjudication_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "disputed.jsonl").open("w", encoding="utf-8") as handle:
        for row in disputed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_manifest = json.loads(
        (data / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "first_judge_model": source_manifest.get("judge_model"),
        "adjudicator_model": model,
        "method": "difference-first independent adjudication",
        "disputed_candidate_sets_excluded": True,
        "overall": overall,
        "splits": split_summaries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


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
