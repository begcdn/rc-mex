from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .oracle.webqsp import compile_parse
from .oracle.sparql import compile_sparql
from .conjunctive import exact_f1, execute_conjunctive
from .program import answer_set_f1, execute_reference
from .substrate import GNNRAGDataset, IndexedSubgraph


def _official_questions(path: Path) -> dict[str, dict[str, Any]]:
    return {
        question["QuestionId"]: question
        for question in json.loads(path.read_text(encoding="utf-8"))["Questions"]
    }


def webqsp_ceiling(
    official_path: Path,
    substrate_folder: Path,
    output: Path,
) -> dict[str, Any]:
    official = _official_questions(official_path)
    rows = []
    counts: Counter[str] = Counter()
    f1_total = 0.0
    for graph_row in GNNRAGDataset(substrate_folder, "dev"):
        if not graph_row.gold_answers:
            counts["empty_gold"] += 1
            rows.append(
                {
                    "id": graph_row.question_id,
                    "question": graph_row.question,
                    "status": "empty_gold",
                    "gold_answers": [],
                }
            )
            continue
        question = official.get(graph_row.question_id)
        if question is None:
            counts["missing_official_question"] += 1
            continue
        graph = IndexedSubgraph(graph_row.triples)
        candidates = []
        for parse_index, parse in enumerate(question.get("Parses", [])):
            program = compile_parse(parse)
            if program is None:
                continue
            execution = execute_reference(graph, program)
            f1 = answer_set_f1(execution.answers, graph_row.gold_answers)
            candidates.append((f1, parse_index, program, execution))
        if not candidates:
            counts["uncompiled"] += 1
            rows.append(
                {
                    "id": graph_row.question_id,
                    "question": graph_row.question,
                    "status": "uncompiled",
                    "gold_answers": list(graph_row.gold_answers),
                }
            )
            continue
        f1, parse_index, program, execution = max(candidates, key=lambda item: item[0])
        if execution.traversed_identity_hop:
            counts["identity_noop"] += 1
            rows.append(
                {
                    "id": graph_row.question_id,
                    "question": graph_row.question,
                    "status": "identity_noop",
                    "parse_index": parse_index,
                    "gold_answers": list(graph_row.gold_answers),
                }
            )
            continue
        exact = set(execution.answers) == set(graph_row.gold_answers)
        counts["exact" if exact else "not_exact"] += 1
        f1_total += f1
        rows.append(
            {
                "id": graph_row.question_id,
                "question": graph_row.question,
                "status": "exact" if exact else "not_exact",
                "parse_index": parse_index,
                "gold_answers": list(graph_row.gold_answers),
                "executed_answers": sorted(execution.answers),
                "f1": f1,
                "has_constraints": bool(program.constraints),
                "has_order": program.order is not None,
                "identity_hop_in_reference": execution.traversed_identity_hop,
                "program": {
                    "topic_entity": program.topic_entity,
                    "hops": [hop.__dict__ for hop in program.hops],
                    "constraints": [item.__dict__ for item in program.constraints],
                    "order": program.order.__dict__ if program.order else None,
                },
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "ceiling_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    evaluated = counts["exact"] + counts["not_exact"]
    metrics = {
        "dataset": "webqsp",
        "split": "gnnrag_dev",
        "substrate": "gnnrag_mid_keyed",
        "questions": len(rows),
        "compiled_and_executed": evaluated,
        "exact": counts["exact"],
        "exact_rate_all": counts["exact"] / len(rows) if rows else 0.0,
        "exact_rate_compiled": counts["exact"] / evaluated if evaluated else 0.0,
        "mean_f1_all": f1_total / len(rows) if rows else 0.0,
        "uncompiled": counts["uncompiled"],
        "missing_official_question": counts["missing_official_question"],
        "empty_gold": counts["empty_gold"],
        "identity_noop_excluded": counts["identity_noop"],
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return metrics


def cwq_ceiling(
    official_path: Path,
    substrate_folder: Path,
    output: Path,
) -> dict[str, Any]:
    official_rows = json.loads(official_path.read_text(encoding="utf-8"))
    official = {row["ID"]: row for row in official_rows}
    rows = []
    counts: Counter[str] = Counter()
    f1_total = 0.0
    by_type: dict[str, Counter[str]] = {}
    for graph_row in GNNRAGDataset(substrate_folder, "dev"):
        question = official.get(graph_row.question_id)
        if question is None:
            counts["missing_official_question"] += 1
            continue
        composition = question["compositionality_type"]
        type_counts = by_type.setdefault(composition, Counter())
        program = compile_sparql(question["sparql"])
        if program is None:
            counts["uncompiled"] += 1
            type_counts["uncompiled"] += 1
            rows.append(
                {
                    "id": graph_row.question_id,
                    "question": graph_row.question,
                    "composition_type": composition,
                    "status": "uncompiled",
                }
            )
            continue
        execution = execute_conjunctive(IndexedSubgraph(graph_row.triples), program)
        exact, f1 = exact_f1(execution.answers, graph_row.gold_answers)
        status = "exact" if exact else "not_exact"
        counts[status] += 1
        type_counts[status] += 1
        counts["identity_atom_in_reference"] += int(execution.traversed_identity_atom)
        f1_total += f1
        rows.append(
            {
                "id": graph_row.question_id,
                "question": graph_row.question,
                "composition_type": composition,
                "status": status,
                "gold_answers": list(graph_row.gold_answers),
                "executed_answers": sorted(execution.answers),
                "f1": f1,
                "identity_atom_in_reference": execution.traversed_identity_atom,
                "program": {
                    "select": program.select,
                    "atoms": [atom.__dict__ for atom in program.atoms],
                    "filters": [item.__dict__ for item in program.filters],
                    "optional_filters": [
                        item.__dict__ for item in program.optional_filters
                    ],
                    "exclusions": list(program.exclusions),
                    "order": program.order.__dict__ if program.order else None,
                },
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    with (output / "ceiling_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    evaluated = counts["exact"] + counts["not_exact"]
    metrics = {
        "dataset": "cwq",
        "split": "gnnrag_dev",
        "substrate": "gnnrag_mid_keyed",
        "questions": len(rows),
        "compiled_and_executed": evaluated,
        "exact": counts["exact"],
        "exact_rate_all": counts["exact"] / len(rows) if rows else 0.0,
        "exact_rate_compiled": counts["exact"] / evaluated if evaluated else 0.0,
        "mean_f1_all": f1_total / len(rows) if rows else 0.0,
        "uncompiled": counts["uncompiled"],
        "missing_official_question": counts["missing_official_question"],
        "identity_atom_in_reference": counts["identity_atom_in_reference"],
        "by_composition_type": {
            name: dict(values) for name, values in sorted(by_type.items())
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return metrics
