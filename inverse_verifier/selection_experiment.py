"""Controlled experiments for path supervision and representation disagreement."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from .comparator import comparator_answer_evidence, comparator_path_text
from .data import read_jsonl, unlabeled_answer_count, write_jsonl
from .selector import answer_metrics, answer_set_key, has_answerable_endpoint


SUPERVISION_MODES = ("annotated", "denotation", "annotated_or_denotation")
PATH_AUDIT_LABELS = (
    "direct_intent",
    "reliable_alternative",
    "correlated_shortcut",
    "entity_specific_coincidence",
    "unrelated",
    "unclear",
)
DEFAULT_HARD_NEGATIVES = 12
DEFAULT_RANDOM_NEGATIVES = 4
DEFAULT_DEV_FRACTION = 0.1


def _candidate_pool_fingerprint(rows: list[dict[str, Any]]) -> str:
    projection = [
        [row["example_id"], row["selected_indices"]]
        for row in rows
    ]
    payload = json.dumps(
        projection, ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _gold_answer_key(row: dict[str, Any]) -> tuple[str, ...]:
    return answer_set_key(row.get("gold_answers", []))


def _candidate_flags(
    candidate: dict[str, Any],
    gold_sequences: set[tuple[str, ...]],
    gold_answers: tuple[str, ...],
) -> tuple[bool, bool]:
    annotated = tuple(candidate["relation_sequence"]) in gold_sequences
    denotation = bool(gold_answers) and answer_set_key(
        candidate.get("answers", [])
    ) == gold_answers
    return annotated, denotation


def _candidate_record(
    candidate: dict[str, Any],
    source_index: int,
    annotated: bool,
    denotation: bool,
    mode: str,
) -> dict[str, Any]:
    path = candidate.get("path") or {}
    if not path:
        raise ValueError(
            "fixed-pool supervision needs materialized candidate paths; "
            "use a predictions run whose candidate_log includes path"
        )
    answers = list(candidate.get("answers", []))
    positive = {
        "annotated": annotated,
        "denotation": denotation,
        "annotated_or_denotation": annotated or denotation,
    }[mode]
    source_category = (
        "annotated_and_denotation"
        if annotated and denotation
        else "annotated_only"
        if annotated
        else "denotation_only"
        if denotation
        else "distractor"
    )
    return {
        "source_candidate_index": source_index,
        "path": path,
        "is_positive": positive,
        "negative_type": "positive" if positive else source_category,
        "source_category": source_category,
        "is_annotated_path": annotated,
        "is_exact_denotation_match": denotation,
        "answer_entity": answers[0] if answers else "",
        "generated_question": candidate["generated_question"],
        "path_text": comparator_path_text(path, ", ".join(answers[:10])),
        "answer_evidence": candidate.get("answer_evidence")
        or comparator_answer_evidence(
            answers,
            path.get("answer_type"),
            unlabeled_answer_count(answers),
        ),
    }


def build_fixed_supervision_study(
    predictions: Path,
    output: Path,
    seed: int = 17,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    hard_negatives: int = DEFAULT_HARD_NEGATIVES,
    random_negatives: int = DEFAULT_RANDOM_NEGATIVES,
) -> dict[str, Any]:
    """Build label ablations over one candidate universe and one data split.

    Every retained question must expose at least one annotated path and at least
    one exact-denotation path. The candidate pool is selected once from the union
    of those positives plus fixed hard/random negatives, then copied unchanged
    across all supervision modes. Only ``is_positive`` changes.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    rng = random.Random(seed)
    source_rows = read_jsonl(predictions)
    base_rows: list[dict[str, Any]] = []
    excluded = {
        "missing_annotated_candidate": 0,
        "missing_denotation_candidate": 0,
    }

    for row in source_rows:
        gold_sequences = {
            tuple(sequence) for sequence in row.get("gold_sequences", [])
        }
        gold_answers = _gold_answer_key(row)
        candidates = row.get("candidate_log", [])
        flags = [
            _candidate_flags(candidate, gold_sequences, gold_answers)
            for candidate in candidates
        ]
        if not any(annotated for annotated, _ in flags):
            excluded["missing_annotated_candidate"] += 1
            continue
        if not any(denotation for _, denotation in flags):
            excluded["missing_denotation_candidate"] += 1
            continue

        required = [
            index
            for index, (annotated, denotation) in enumerate(flags)
            if annotated or denotation
        ]
        distractors = [
            index
            for index, (annotated, denotation) in enumerate(flags)
            if not annotated and not denotation
        ]
        distractors.sort(
            key=lambda index: -float(candidates[index].get("score", 0.0))
        )
        selected = required + distractors[:hard_negatives]
        remaining = distractors[hard_negatives:]
        if remaining and random_negatives:
            selected += rng.sample(
                remaining, min(random_negatives, len(remaining))
            )
        selected = list(dict.fromkeys(selected))
        base_rows.append(
            {
                "example_id": row["question_id"],
                "kg": "webqsp",
                "original_question": row["question"],
                "source_candidates": candidates,
                "flags": flags,
                "selected_indices": selected,
            }
        )

    rng.shuffle(base_rows)
    dev_size = max(1, round(len(base_rows) * dev_fraction)) if base_rows else 0
    split_rows = {"dev": base_rows[:dev_size], "train": base_rows[dev_size:]}
    output.mkdir(parents=True, exist_ok=True)

    mode_summaries: dict[str, Any] = {}
    for mode in SUPERVISION_MODES:
        mode_dir = output / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_counts = {
            "candidate_sets": 0,
            "candidates": 0,
            "positives": 0,
            "annotated_only_candidates": 0,
            "denotation_only_candidates": 0,
        }
        for split, rows in split_rows.items():
            rendered = []
            for row in rows:
                candidates = []
                for index in row["selected_indices"]:
                    annotated, denotation = row["flags"][index]
                    candidates.append(
                        _candidate_record(
                            row["source_candidates"][index],
                            index,
                            annotated,
                            denotation,
                            mode,
                        )
                    )
                if not any(candidate["is_positive"] for candidate in candidates):
                    raise AssertionError(
                        f"{row['example_id']} has no positive under {mode}"
                    )
                rendered.append(
                    {
                        "example_id": row["example_id"],
                        "kg": row["kg"],
                        "original_question": row["original_question"],
                        "candidates": candidates,
                    }
                )
            write_jsonl(mode_dir / f"{split}.jsonl", rendered)
            mode_counts[f"{split}_candidate_sets"] = len(rendered)
            mode_counts["candidate_sets"] += len(rendered)
            mode_counts["candidates"] += sum(
                len(row["candidates"]) for row in rendered
            )
            mode_counts["positives"] += sum(
                candidate["is_positive"]
                for row in rendered
                for candidate in row["candidates"]
            )
            mode_counts["annotated_only_candidates"] += sum(
                candidate["source_category"] == "annotated_only"
                for row in rendered
                for candidate in row["candidates"]
            )
            mode_counts["denotation_only_candidates"] += sum(
                candidate["source_category"] == "denotation_only"
                for row in rendered
                for candidate in row["candidates"]
            )
        mode_summaries[mode] = mode_counts

    manifest = {
        "source": str(predictions),
        "source_questions": len(source_rows),
        "comparison_questions": len(base_rows),
        "comparison_population": (
            "questions with both an annotated candidate and an exact-denotation "
            "candidate in the fixed source pool"
        ),
        "candidate_selection": (
            "union of all annotated/denotation candidates plus fixed hard and "
            "random distractors; identical across supervision modes"
        ),
        "supervision_modes": list(SUPERVISION_MODES),
        "excluded": excluded,
        "hard_negatives": hard_negatives,
        "random_negatives": random_negatives,
        "dev_fraction": dev_fraction,
        "seed": seed,
        "candidate_pool_fingerprints": {
            split: _candidate_pool_fingerprint(rows)
            for split, rows in split_rows.items()
        },
        "modes": mode_summaries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def export_answer_equivalent_path_audit(
    predictions: Path,
    output: Path,
    graphs: Path | None = None,
) -> dict[str, Any]:
    """Export raw paths that reach the gold answer through non-annotated routes."""
    rows = read_jsonl(predictions)
    if any(
        not candidate.get("path")
        for row in rows
        for candidate in row.get("candidate_log", [])
    ):
        if graphs is None:
            raise ValueError(
                "candidate paths are missing; provide --graphs to reconstruct them"
            )
        from .rescore import rebuild_paths

        rebuild_paths(rows, graphs)

    audit_rows = []
    questions = set()
    for row in rows:
        gold_sequences = {
            tuple(sequence) for sequence in row.get("gold_sequences", [])
        }
        gold_answers = _gold_answer_key(row)
        if not gold_answers:
            continue
        for index, candidate in enumerate(row.get("candidate_log", [])):
            annotated, denotation = _candidate_flags(
                candidate, gold_sequences, gold_answers
            )
            if annotated or not denotation:
                continue
            path = candidate.get("path") or {}
            if not path:
                raise ValueError(
                    f"could not reconstruct candidate {index} for "
                    f"{row['question_id']}"
                )
            answers = list(candidate.get("answers", []))
            questions.add(row["question_id"])
            audit_rows.append(
                {
                    "question_id": row["question_id"],
                    "source_candidate_index": index,
                    "question": row["question"],
                    "gold_answers": row.get("gold_answers", []),
                    "relation_sequence": candidate["relation_sequence"],
                    "path": path,
                    "path_text": comparator_path_text(
                        path, ", ".join(answers[:10])
                    ),
                    "generated_question": candidate.get(
                        "generated_question", ""
                    ),
                    "answers": answers,
                    "path_label": None,
                    "generator_faithfulness": None,
                    "notes": "",
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidates.jsonl", audit_rows)
    manifest = {
        "source": str(predictions),
        "graphs": str(graphs) if graphs else None,
        "questions": len(questions),
        "path_instances": len(audit_rows),
        "deduplicated": False,
        "unit_of_analysis": "executed candidate path instance",
        "path_labels": list(PATH_AUDIT_LABELS),
        "generator_faithfulness_labels": ["faithful", "unfaithful", "unclear"],
        "warning": (
            "path_label judges raw path meaning; generator_faithfulness separately "
            "judges whether the generated question describes that path"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_path_audit_markdown(audit_rows, output / "audit.md")
    return manifest


def _write_path_audit_markdown(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Raw Path Intent Audit",
        "",
        "These candidates return exactly the gold answer but do not match the "
        "annotated relation sequence. Judge the raw path and its verbalization "
        "separately.",
        "",
        "Path labels: `direct_intent`, `reliable_alternative`, "
        "`correlated_shortcut`, `entity_specific_coincidence`, `unrelated`, "
        "`unclear`.",
        "",
        "Generator labels: `faithful`, `unfaithful`, `unclear`.",
        "",
    ]
    current_question = None
    for row in rows:
        if row["question_id"] != current_question:
            current_question = row["question_id"]
            lines.extend(
                [
                    f"## `{row['question_id']}`",
                    "",
                    f"Question: {row['question']}",
                    "",
                    "Gold answers: " + ", ".join(row["gold_answers"]),
                    "",
                ]
            )
        lines.extend(
            [
                f"### Candidate {row['source_candidate_index']}",
                "",
                "```text",
                row["path_text"],
                "```",
                "",
                f"Generated question: {row['generated_question']}",
                "",
                "- Path label:",
                "- Generator faithfulness:",
                "- Notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _average_ranks(scores: list[float]) -> list[float]:
    ordered = sorted(range(len(scores)), key=lambda index: -scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ordered):
        stop = cursor + 1
        while stop < len(ordered) and math.isclose(
            scores[ordered[stop]],
            scores[ordered[cursor]],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            stop += 1
        rank = (cursor + 1 + stop) / 2
        for index in ordered[cursor:stop]:
            ranks[index] = rank
        cursor = stop
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank vectors must have equal nonzero length")
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else 0.0


def _load_score_run(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    scores_path = path / "scores.jsonl"
    if not metrics_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(
            f"{path} must contain metrics.json and scores.jsonl; rescore the "
            "fixed predictions with the current code"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": path.name,
        "path": str(path),
        "metrics": metrics,
        "scores": {
            row["question_id"]: [
                float(candidate["score"]) for candidate in row["candidates"]
            ]
            for row in read_jsonl(scores_path)
        },
    }


def _select_from_scores(
    row: dict[str, Any],
    scores: list[float],
    endpoint_filter: bool,
) -> dict[str, Any]:
    candidates = row["candidate_log"]
    if len(candidates) != len(scores):
        raise ValueError(
            f"{row['question_id']} has {len(candidates)} candidates but "
            f"{len(scores)} scores"
        )
    eligible = list(range(len(candidates)))
    if endpoint_filter:
        filtered = [
            index
            for index, candidate in enumerate(candidates)
            if has_answerable_endpoint(candidate)
        ]
        if filtered:
            eligible = filtered
    selected_index = max(eligible, key=lambda index: scores[index])
    selected = candidates[selected_index]
    result = answer_metrics(selected.get("answers", []), row.get("gold_answers", []))
    return {
        "candidate_index": selected_index,
        "relation_sequence": selected.get("relation_sequence", []),
        "generated_question": selected.get("generated_question", ""),
        "answers": selected.get("answers", []),
        "score": scores[selected_index],
        "margin": (
            scores[selected_index]
            - max(scores[index] for index in eligible if index != selected_index)
            if len(eligible) > 1
            else 0.0
        ),
        "exact_match": result["exact_match"],
        "f1": result["f1"],
    }


def _run_selections(
    source_rows: list[dict[str, Any]],
    run: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    endpoint_filter = bool(run["metrics"].get("endpoint_filter", True))
    selections = {}
    for row in source_rows:
        question_id = row["question_id"]
        if question_id not in run["scores"]:
            raise ValueError(f"{run['name']} has no scores for {question_id}")
        selections[question_id] = _select_from_scores(
            row, run["scores"][question_id], endpoint_filter
        )
    return selections


def _pair_metrics(
    source_rows: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    pair_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_selections = _run_selections(source_rows, left)
    right_selections = _run_selections(source_rows, right)
    cases = []
    rank_correlations = []
    counts = {
        "both_correct": 0,
        "left_only_correct": 0,
        "right_only_correct": 0,
        "neither_correct": 0,
        "same_candidate": 0,
        "same_answer": 0,
        "agreement_correct": 0,
        "disagreement_left_correct": 0,
        "disagreement_right_correct": 0,
        "disagreement_oracle_correct": 0,
    }
    for row in source_rows:
        question_id = row["question_id"]
        left_pick = left_selections[question_id]
        right_pick = right_selections[question_id]
        left_correct = bool(left_pick["exact_match"])
        right_correct = bool(right_pick["exact_match"])
        same_candidate = (
            left_pick["candidate_index"] == right_pick["candidate_index"]
        )
        same_answer = answer_set_key(left_pick["answers"]) == answer_set_key(
            right_pick["answers"]
        )
        counts["same_candidate"] += same_candidate
        counts["same_answer"] += same_answer
        if left_correct and right_correct:
            counts["both_correct"] += 1
        elif left_correct:
            counts["left_only_correct"] += 1
        elif right_correct:
            counts["right_only_correct"] += 1
        else:
            counts["neither_correct"] += 1
        if same_candidate:
            counts["agreement_correct"] += left_correct
        else:
            counts["disagreement_left_correct"] += left_correct
            counts["disagreement_right_correct"] += right_correct
            counts["disagreement_oracle_correct"] += left_correct or right_correct

        left_scores = left["scores"][question_id]
        right_scores = right["scores"][question_id]
        rank_correlations.append(
            _correlation(_average_ranks(left_scores), _average_ranks(right_scores))
        )
        cases.append(
            {
                "pair_kind": pair_kind,
                "left_run": left["name"],
                "right_run": right["name"],
                "question_id": question_id,
                "question": row["question"],
                "gold_answers": row.get("gold_answers", []),
                "same_candidate": same_candidate,
                "same_answer": same_answer,
                "left": left_pick,
                "right": right_pick,
            }
        )

    total = max(len(source_rows), 1)
    disagreements = total - counts["same_candidate"]
    pair = {
        "pair_kind": pair_kind,
        "left_run": left["name"],
        "right_run": right["name"],
        "questions": len(source_rows),
        "left_exact_match": (
            counts["both_correct"] + counts["left_only_correct"]
        )
        / total,
        "right_exact_match": (
            counts["both_correct"] + counts["right_only_correct"]
        )
        / total,
        "oracle_exact_match": (
            counts["both_correct"]
            + counts["left_only_correct"]
            + counts["right_only_correct"]
        )
        / total,
        "oracle_gain_over_better_arm": (
            min(counts["left_only_correct"], counts["right_only_correct"]) / total
        ),
        "same_candidate_rate": counts["same_candidate"] / total,
        "same_answer_rate": counts["same_answer"] / total,
        "agreement_exact_match": (
            counts["agreement_correct"] / counts["same_candidate"]
            if counts["same_candidate"]
            else None
        ),
        "disagreement_left_exact_match": (
            counts["disagreement_left_correct"] / disagreements
            if disagreements
            else None
        ),
        "disagreement_right_exact_match": (
            counts["disagreement_right_correct"] / disagreements
            if disagreements
            else None
        ),
        "disagreement_oracle_exact_match": (
            counts["disagreement_oracle_correct"] / disagreements
            if disagreements
            else None
        ),
        "mean_candidate_rank_correlation": mean(rank_correlations),
        **counts,
    }
    return pair, cases


def _mean_pair_metric(pairs: list[dict[str, Any]], key: str) -> float | None:
    values = [pair[key] for pair in pairs if pair.get(key) is not None]
    return mean(values) if values else None


def audit_view_runs(
    predictions: Path,
    generated_runs: list[Path],
    path_runs: list[Path],
    output: Path,
    path_labels: Path | None = None,
) -> dict[str, Any]:
    """Compare cross-view diversity with the same-view ensemble null."""
    if not generated_runs or not path_runs:
        raise ValueError("at least one generated-question and one path run are required")
    source_rows = read_jsonl(predictions)
    generated = [_load_score_run(path) for path in generated_runs]
    path = [_load_score_run(run_path) for run_path in path_runs]

    pair_specs: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    pair_specs.extend(
        (left, right, "generated_same_view")
        for index, left in enumerate(generated)
        for right in generated[index + 1 :]
    )
    pair_specs.extend(
        (left, right, "path_same_view")
        for index, left in enumerate(path)
        for right in path[index + 1 :]
    )
    pair_specs.extend(
        (left, right, "cross_view") for left in generated for right in path
    )

    pairs, cases = [], []
    for left, right, kind in pair_specs:
        pair, pair_cases = _pair_metrics(source_rows, left, right, kind)
        pairs.append(pair)
        cases.extend(pair_cases)

    by_kind = {}
    for kind in ("generated_same_view", "path_same_view", "cross_view"):
        selected = [pair for pair in pairs if pair["pair_kind"] == kind]
        by_kind[kind] = {
            "pairs": len(selected),
            "mean_oracle_gain_over_better_arm": _mean_pair_metric(
                selected, "oracle_gain_over_better_arm"
            ),
            "mean_same_candidate_rate": _mean_pair_metric(
                selected, "same_candidate_rate"
            ),
            "mean_candidate_rank_correlation": _mean_pair_metric(
                selected, "mean_candidate_rank_correlation"
            ),
            "mean_agreement_exact_match": _mean_pair_metric(
                selected, "agreement_exact_match"
            ),
            "mean_disagreement_oracle_exact_match": _mean_pair_metric(
                selected, "disagreement_oracle_exact_match"
            ),
        }

    cross_pairs = [pair for pair in pairs if pair["pair_kind"] == "cross_view"]
    within_pairs = [pair for pair in pairs if pair["pair_kind"] != "cross_view"]
    target = min(
        cross_pairs,
        key=lambda pair: (
            abs(pair["left_exact_match"] - pair["right_exact_match"]),
            -max(pair["left_exact_match"], pair["right_exact_match"]),
        ),
    )
    control = None
    if within_pairs:
        target_mean = (target["left_exact_match"] + target["right_exact_match"]) / 2
        target_gap = abs(target["left_exact_match"] - target["right_exact_match"])
        control = min(
            within_pairs,
            key=lambda pair: (
                abs(
                    (pair["left_exact_match"] + pair["right_exact_match"]) / 2
                    - target_mean
                ),
                abs(
                    abs(pair["left_exact_match"] - pair["right_exact_match"])
                    - target_gap
                ),
            ),
        )

    conclusion = {
        "matched_cross_view_pair": target,
        "closest_same_view_control": control,
        "cross_view_excess_oracle_gain": (
            target["oracle_gain_over_better_arm"]
            - control["oracle_gain_over_better_arm"]
            if control
            else None
        ),
        "interpretation_gate": (
            "Cross-view complementarity is interesting only if its oracle gain "
            "exceeds the closest same-view control and later predicts independently "
            "audited raw-path spuriousness."
        ),
    }
    metrics = {
        "predictions": str(predictions),
        "questions": len(source_rows),
        "generated_runs": [run["path"] for run in generated],
        "path_runs": [run["path"] for run in path],
        "summary_by_pair_kind": by_kind,
        "comparison": conclusion,
        "pairs": pairs,
    }
    output.mkdir(parents=True, exist_ok=True)
    if path_labels is not None:
        metrics["spurious_path_test"] = _audit_spurious_path_signal(
            source_rows,
            generated,
            path,
            target,
            path_labels,
            output / "path_label_cases.jsonl",
        )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_jsonl(output / "cases.jsonl", cases)
    _write_view_report(metrics, output / "report.md")
    return metrics


def _pairwise_auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_value in positive:
        for negative_value in negative:
            if positive_value > negative_value:
                wins += 1.0
            elif math.isclose(
                positive_value,
                negative_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _audit_spurious_path_signal(
    source_rows: list[dict[str, Any]],
    generated_runs: list[dict[str, Any]],
    path_runs: list[dict[str, Any]],
    target_pair: dict[str, Any],
    labels_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    """Test whether raw-path preference is enriched for audited spurious paths."""
    runs = {run["name"]: run for run in generated_runs + path_runs}
    generated = runs[target_pair["left_run"]]
    path = runs[target_pair["right_run"]]
    source_by_id = {row["question_id"]: row for row in source_rows}
    valid_labels = {"direct_intent", "reliable_alternative"}
    spurious_labels = {
        "correlated_shortcut",
        "entity_specific_coincidence",
        "unrelated",
    }
    cases = []
    skipped = {"unlabeled": 0, "unclear": 0, "missing_question": 0}

    for label_row in read_jsonl(labels_path):
        label = label_row.get("path_label")
        if not label:
            skipped["unlabeled"] += 1
            continue
        if label == "unclear":
            skipped["unclear"] += 1
            continue
        if label not in valid_labels | spurious_labels:
            raise ValueError(f"unknown path label: {label}")
        question_id = label_row["question_id"]
        source = source_by_id.get(question_id)
        if source is None:
            skipped["missing_question"] += 1
            continue
        candidate_index = int(label_row["source_candidate_index"])
        generated_ranks = _average_ranks(generated["scores"][question_id])
        path_ranks = _average_ranks(path["scores"][question_id])
        if candidate_index >= len(generated_ranks):
            raise ValueError(
                f"{question_id} candidate {candidate_index} is outside score run"
            )
        generated_rank = generated_ranks[candidate_index]
        path_rank = path_ranks[candidate_index]
        cases.append(
            {
                "question_id": question_id,
                "source_candidate_index": candidate_index,
                "question": source["question"],
                "path_label": label,
                "label_class": (
                    "valid" if label in valid_labels else "spurious"
                ),
                "generator_faithfulness": label_row.get(
                    "generator_faithfulness"
                ),
                "generated_question_rank": generated_rank,
                "path_rank": path_rank,
                "path_over_generated_rank_advantage": (
                    generated_rank - path_rank
                ),
                "path_top5_generated_below_top5": (
                    path_rank <= 5 < generated_rank
                ),
                "generated_top5_path_below_top5": (
                    generated_rank <= 5 < path_rank
                ),
            }
        )

    write_jsonl(cases_path, cases)
    by_class = {
        label_class: [
            row for row in cases if row["label_class"] == label_class
        ]
        for label_class in ("valid", "spurious")
    }
    spurious_advantages = [
        row["path_over_generated_rank_advantage"]
        for row in by_class["spurious"]
    ]
    valid_advantages = [
        row["path_over_generated_rank_advantage"]
        for row in by_class["valid"]
    ]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "candidates": len(rows),
            "mean_path_over_generated_rank_advantage": (
                mean(
                    row["path_over_generated_rank_advantage"]
                    for row in rows
                )
                if rows
                else None
            ),
            "path_top5_generated_below_top5": sum(
                row["path_top5_generated_below_top5"] for row in rows
            ),
            "generated_top5_path_below_top5": sum(
                row["generated_top5_path_below_top5"] for row in rows
            ),
        }

    return {
        "labels": str(labels_path),
        "generated_run": generated["name"],
        "path_run": path["name"],
        "audited_candidates": len(cases),
        "skipped": skipped,
        "by_class": {
            label_class: summarize(rows)
            for label_class, rows in by_class.items()
        },
        "auc_path_rank_advantage_predicts_spurious": _pairwise_auc(
            spurious_advantages, valid_advantages
        ),
        "directional_hypothesis": (
            "Spurious paths should receive a larger within-question rank "
            "advantage from serialized-path scoring than from generated-question "
            "scoring. AUC > 0.5 supports this direction; AUC near 0.5 does not."
        ),
    }


def _format(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_view_report(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Representation Disagreement Audit",
        "",
        "This audit compares cross-view disagreement against ordinary same-view "
        "model disagreement on the identical candidate pool.",
        "",
        "| Pair type | Pairs | Oracle gain | Same top candidate | Rank correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, values in metrics["summary_by_pair_kind"].items():
        lines.append(
            f"| {kind} | {values['pairs']} | "
            f"{_format(values['mean_oracle_gain_over_better_arm'])} | "
            f"{_format(values['mean_same_candidate_rate'])} | "
            f"{_format(values['mean_candidate_rank_correlation'])} |"
        )
    comparison = metrics["comparison"]
    target = comparison["matched_cross_view_pair"]
    lines.extend(
        [
            "",
            "## Matched Cross-View Pair",
            "",
            f"- Runs: `{target['left_run']}` vs `{target['right_run']}`",
            f"- Exact match: {target['left_exact_match']:.3f} vs "
            f"{target['right_exact_match']:.3f}",
            f"- Oracle exact match: {target['oracle_exact_match']:.3f}",
            f"- Oracle gain over better arm: "
            f"{target['oracle_gain_over_better_arm']:.3f}",
            f"- Same-candidate rate: {target['same_candidate_rate']:.3f}",
            f"- Agreement exact match: {_format(target['agreement_exact_match'])}",
            f"- Disagreement oracle exact match: "
            f"{_format(target['disagreement_oracle_exact_match'])}",
        ]
    )
    control = comparison["closest_same_view_control"]
    if control:
        lines.extend(
            [
                "",
                "## Closest Same-View Control",
                "",
                f"- Runs: `{control['left_run']}` vs `{control['right_run']}`",
                f"- Exact match: {control['left_exact_match']:.3f} vs "
                f"{control['right_exact_match']:.3f}",
                f"- Oracle gain over better arm: "
                f"{control['oracle_gain_over_better_arm']:.3f}",
                f"- Cross-view excess oracle gain: "
                f"{comparison['cross_view_excess_oracle_gain']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            comparison["interpretation_gate"],
            "",
            "The current generated-description labels are not raw-path labels. "
            "They must not be used to claim that disagreement detects spurious "
            "reasoning until the corresponding paths are independently audited.",
        ]
    )
    spurious = metrics.get("spurious_path_test")
    if spurious:
        valid = spurious["by_class"]["valid"]
        invalid = spurious["by_class"]["spurious"]
        lines.extend(
            [
                "",
                "## Independently Audited Path Signal",
                "",
                f"- Audited candidates: {spurious['audited_candidates']}",
                "- Mean path-over-generated rank advantage, valid paths: "
                f"{_format(valid['mean_path_over_generated_rank_advantage'])}",
                "- Mean path-over-generated rank advantage, spurious paths: "
                f"{_format(invalid['mean_path_over_generated_rank_advantage'])}",
                "- AUC that path-over-generated rank advantage predicts "
                "spuriousness: "
                f"{_format(spurious['auc_path_rank_advantage_predicts_spurious'])}",
                "",
                spurious["directional_hypothesis"],
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
