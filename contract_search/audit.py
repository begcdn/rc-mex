from __future__ import annotations

import json
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any


REASONING_FAILURES = {
    "fragment_never_proposed",
    "fragment_pruned",
    "silent_incompleteness",
    "ranked_below",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _reference_fragments(row: dict[str, Any]) -> Counter[str]:
    program = row.get("program") or {}
    fragments: Counter[str] = Counter()
    if "hops" in program:
        fragments.update(item["relation"] for item in program.get("hops", ()))
        fragments.update(item["relation"] for item in program.get("constraints", ()))
        if program.get("order"):
            fragments[program["order"]["relation"]] += 1
            fragments["operator:order"] += 1
    else:
        fragments.update(item["relation"] for item in program.get("atoms", ()))
        fragments.update(
            item["relation"] for item in program.get("optional_filters", ())
        )
        if program.get("filters") or program.get("optional_filters"):
            fragments["operator:filter"] += 1
        if program.get("order"):
            fragments["operator:order"] += 1
    return fragments


def _reference_program(row: dict[str, Any]) -> dict[str, Any]:
    program = row.get("program") or {}
    if "atoms" in program:
        return program
    current = program["topic_entity"]
    atoms = []
    nodes = [current]
    for index, hop in enumerate(program.get("hops", ()), start=1):
        target = f"?v{index}"
        if hop["direction"] == "forward":
            atoms.append({"head": current, "relation": hop["relation"], "tail": target})
        else:
            atoms.append({"head": target, "relation": hop["relation"], "tail": current})
        current = target
        nodes.append(current)
    for item in program.get("constraints", ()):
        source = nodes[item["source_index"] + 1]
        atoms.append(
            {"head": source, "relation": item["relation"], "tail": item["argument"]}
        )
    order = program.get("order")
    return {
        "select": current,
        "atoms": atoms,
        "filters": [],
        "optional_filters": [],
        "exclusions": [],
        "order": (
            {
                "variable": nodes[order["source_index"] + 1],
                "descending": order["descending"],
                "limit": order["count"],
            }
            if order
            else None
        ),
    }


def _canonical_program(program: dict[str, Any] | None) -> tuple | None:
    if not program:
        return None
    variables = sorted(
        {
            term
            for atom in program.get("atoms", ())
            for term in (atom["head"], atom["tail"])
            if term.startswith("?")
        }
        | {
            item[key]
            for key in ("variable", "source")
            for item in (
                program.get("filters", ())
                if key == "variable"
                else program.get("optional_filters", ())
            )
            if key in item and item[key].startswith("?")
        }
        | (
            {program["select"]}
            if str(program.get("select", "")).startswith("?")
            else set()
        )
    )
    if len(variables) > 7:
        return None

    def encode(mapping: dict[str, str]) -> tuple:
        def term(value: str) -> str:
            return mapping.get(value, f"const:{value}")

        atoms = tuple(
            sorted(
                (
                    term(item["head"]),
                    item["relation"],
                    term(item["tail"]),
                )
                for item in program.get("atoms", ())
            )
        )
        filters = tuple(
            sorted(
                (
                    term(item["variable"]),
                    item["operator"],
                    item["argument"],
                )
                for item in program.get("filters", ())
            )
        )
        optional = tuple(
            sorted(
                (
                    term(item["source"]),
                    item["relation"],
                    item["operator"],
                    item["argument"],
                )
                for item in program.get("optional_filters", ())
            )
        )
        order = program.get("order")
        order_key = (
            (
                term(order["variable"]),
                bool(order["descending"]),
                int(order["limit"]),
            )
            if order
            else ()
        )
        return (term(program["select"]), atoms, filters, optional, order_key)

    candidates = (
        encode(dict(zip(variables, permutation, strict=True)))
        for permutation in permutations(f"?{index}" for index in range(len(variables)))
    )
    return min(candidates)


def _candidate_fragments(program: dict[str, Any] | None) -> Counter[str]:
    if not program:
        return Counter()
    fragments: Counter[str] = Counter(
        item["relation"] for item in program.get("atoms", ())
    )
    fragments.update(
        item["relation"] for item in program.get("optional_filters", ())
    )
    if program.get("filters") or program.get("optional_filters"):
        fragments["operator:filter"] += 1
    if program.get("order"):
        fragments["operator:order"] += 1
    return fragments


def _is_strict_submultiset(left: Counter[str], right: Counter[str]) -> bool:
    return left != right and all(count <= right[item] for item, count in left.items())


def _final_beam(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    trace = prediction.get("trace", ())
    return trace[-1].get("beam", ()) if trace else []


def classify_failure(
    prediction: dict[str, Any],
    ceiling: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if ceiling.get("status") != "exact":
        return "evidence_absent", {}
    if prediction.get("exact"):
        return "correct_answer", {}

    reference = _reference_fragments(ceiling)
    selected = _candidate_fragments(prediction.get("program_state"))
    proposed_relations = {
        atom["relation"]
        for round_trace in prediction.get("trace", ())
        for atom in round_trace.get("proposed_atoms", ())
    }
    proposed_operators = {
        token
        for token in ("operator:filter", "operator:order")
        if any(
            _candidate_fragments(item.get("program_state"))[token]
            for round_trace in prediction.get("trace", ())
            for item in round_trace.get("beam", ())
        )
    }
    proposed = proposed_relations | proposed_operators
    missing_never_proposed = sorted(
        fragment for fragment in reference if fragment not in proposed
    )
    reference_program = _reference_program(ceiling)
    reference_key = _canonical_program(reference_program)
    final_has_reference = any(
        _canonical_program(item.get("program_state")) == reference_key
        for item in _final_beam(prediction)
    )

    if missing_never_proposed:
        category = "fragment_never_proposed"
    elif final_has_reference:
        category = "ranked_below"
    elif (
        prediction.get("predicted_answers")
        and _is_strict_submultiset(selected, reference)
    ):
        category = "silent_incompleteness"
    elif _canonical_program(prediction.get("program_state")) == reference_key:
        category = "execution_normalization"
    else:
        category = "fragment_pruned"
    return category, {
        "reference_fragments": dict(reference),
        "selected_fragments": dict(selected),
        "missing_never_proposed": missing_never_proposed,
        "reference_program_in_final_beam": final_has_reference,
    }


def run_failure_audit(
    predictions_path: Path,
    ceiling_path: Path,
    output: Path,
) -> dict[str, Any]:
    predictions = _read_jsonl(predictions_path)
    ceiling = {row["id"]: row for row in _read_jsonl(ceiling_path)}
    rows = []
    counts: Counter[str] = Counter()
    for prediction in predictions:
        reference = ceiling.get(prediction["id"])
        if reference is None:
            raise KeyError(f"missing ceiling row for {prediction['id']}")
        category, details = classify_failure(prediction, reference)
        counts[category] += 1
        rows.append(
            {
                "id": prediction["id"],
                "question": prediction["question"],
                "category": category,
                "f1": prediction["f1"],
                "selected_program": prediction.get("program_state"),
                **details,
            }
        )

    reasoning_failures = sum(counts[name] for name in REASONING_FAILURES)
    silent_rate = (
        counts["silent_incompleteness"] / reasoning_failures
        if reasoning_failures
        else 0.0
    )
    metrics = {
        "questions": len(rows),
        "categories": dict(sorted(counts.items())),
        "reasoning_failures": reasoning_failures,
        "silent_incompleteness_rate": silent_rate,
        "gate_1_threshold": 0.20,
        "gate_1_passed": silent_rate >= 0.20,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "phase0_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Phase 0 Failure Audit",
        "",
        f"- Questions: {len(rows)}",
        f"- Reasoning-stage failures: {reasoning_failures}",
        (
            "- Silent incompleteness: "
            f"{counts['silent_incompleteness']} ({silent_rate:.1%})"
        ),
        f"- Gate 1 (at least 20%): {'PASS' if metrics['gate_1_passed'] else 'FAIL'}",
        "",
        "| category | count |",
        "|---|---:|",
        *[f"| {name} | {count} |" for name, count in sorted(counts.items())],
        "",
        "Gold reference programs were used only by this offline audit.",
    ]
    (output / "phase0_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return metrics
