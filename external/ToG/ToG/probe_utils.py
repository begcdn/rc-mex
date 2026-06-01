from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def answer_match(value: Any, gold_answers: list[str]) -> bool:
    pred = normalize_answer(value)
    if not pred:
        return False
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if pred == normalized_gold or pred in normalized_gold or normalized_gold in pred:
            return True
    return False


def gold_answers_for(data: dict[str, Any], dataset: str) -> list[str]:
    if dataset == "webqsp":
        out = []
        for parse in data.get("Parses", []) or []:
            for answer in parse.get("Answers", []) or []:
                if answer.get("EntityName"):
                    out.append(str(answer["EntityName"]))
                elif answer.get("AnswerArgument"):
                    out.append(str(answer["AnswerArgument"]))
        return sorted(set(out))
    raw = data.get("answer", data.get("answers", data.get("label", "")))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        return [str(value) for value in raw.values()]
    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("entity_name", "answer_argument", "answer", "EntityName", "AnswerArgument"):
                    if item.get(key):
                        out.append(str(item[key]))
                        break
        return sorted(set(out))
    return []


def question_id_for(data: dict[str, Any], fallback: int) -> str:
    return str(data.get("ID") or data.get("QuestionId") or data.get("id") or fallback)


def candidate_rows(
    depth: int,
    total_entities_id: list[str],
    total_relations: list[str],
    total_candidates: list[str],
    total_topic_entities: list[str],
    total_head: list[bool],
    total_scores: list[float],
) -> list[dict[str, Any]]:
    rows = []
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    for rank, (entity_id, relation, candidate, topic_entity, head, score) in enumerate(
        sorted(zipped, key=lambda row: row[5], reverse=True),
        start=1,
    ):
        rows.append(
            {
                "depth": depth,
                "rank": rank,
                "entity_id": entity_id,
                "answer": candidate,
                "score": float(score),
                "relation": relation,
                "topic_entity_id": topic_entity,
                "head": bool(head),
                "path_trace": f"{topic_entity} --{relation}--> {candidate}",
            }
        )
    return rows


def retained_rows(candidates: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    return [row for row in candidates[:width] if row["score"] != 0]


def summarize_question(row: dict[str, Any]) -> dict[str, Any]:
    generated = row["candidate_entities_generated"]
    retained = row["retained_beam_entities"]
    gold_answers = row["gold_answers"]
    row["gold_answer_generated"] = any(answer_match(c["answer"], gold_answers) or answer_match(c["entity_id"], gold_answers) for c in generated)
    row["gold_answer_retained"] = any(answer_match(c["answer"], gold_answers) or answer_match(c["entity_id"], gold_answers) for c in retained)
    row["tog_correct"] = answer_match(row.get("tog_final_answer", ""), gold_answers)
    row["gold_ranked_below_wrong_answer"] = bool(
        row["gold_answer_generated"]
        and not row["tog_correct"]
        and retained
        and not (answer_match(retained[0]["answer"], gold_answers) or answer_match(retained[0]["entity_id"], gold_answers))
    )
    row["main_failure"] = main_failure(row)
    row["paths_per_answer"] = paths_per_answer(generated)
    return row


def main_failure(row: dict[str, Any]) -> str:
    if row["tog_correct"]:
        return "tog_correct"
    if not row["gold_answer_generated"]:
        return "gold_not_generated"
    if not row["gold_answer_retained"]:
        return "gold_pruned"
    if row["gold_ranked_below_wrong_answer"]:
        return "gold_ranked_low"
    return "unknown"


def paths_per_answer(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        counts[candidate["answer"]] += 1
    return dict(counts)


def print_question_log(row: dict[str, Any], index: int, total: int) -> None:
    print("\n" + "=" * 52)
    print(f"Question {index}/{total}")
    print(f"Q: {row['question']}")
    print(f"Gold: {row['gold_answers']}")
    print(f"ToG answer: {row.get('tog_final_answer', '')}")
    print(f"Gold generated: {'yes' if row['gold_answer_generated'] else 'no'}")
    print(f"Gold retained: {'yes' if row['gold_answer_retained'] else 'no'}")
    print("Top candidates:")
    for candidate in row["retained_beam_entities"][:3]:
        print(f"{candidate['rank']}. {candidate['answer']} score={candidate['score']:.4f} path={candidate['path_trace']}")
    print(f"Main failure: {row['main_failure']}")
    print("=" * 52)


def write_probe_outputs(output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "candidate_paths.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = compute_metrics(rows)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "report.md").write_text(write_report(metrics), encoding="utf-8")


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    generated = sum(1 for row in rows if row["gold_answer_generated"])
    retained = sum(1 for row in rows if row["gold_answer_retained"])
    ranked_low = sum(1 for row in rows if row["gold_ranked_below_wrong_answer"])
    multiple_paths = sum(1 for row in rows if any(count > 1 for count in row["paths_per_answer"].values()))
    failure_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        failure_counts[row["main_failure"]] += 1
    return {
        "total_questions": total,
        "gold_generated_rate": generated / total if total else 0.0,
        "gold_retained_rate": retained / total if total else 0.0,
        "gold_generated_but_ranked_or_selected_incorrectly_rate": ranked_low / total if total else 0.0,
        "multiple_supporting_paths_rate": multiple_paths / total if total else 0.0,
        "average_generated_candidates": sum(len(row["candidate_entities_generated"]) for row in rows) / total if total else 0.0,
        "average_retained_candidates": sum(len(row["retained_beam_entities"]) for row in rows) / total if total else 0.0,
        "failure_counts": dict(failure_counts),
    }


def write_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Minimal ToG Evidence Probe",
        "",
        "This smoke test keeps ToG search/scoring unchanged and records whether answer reranking could help.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation Checklist",
        "",
        "1. How often was the gold answer generated?",
        f"   - `{metrics['gold_generated_rate']:.3f}`",
        "2. How often was the gold answer retained?",
        f"   - `{metrics['gold_retained_rate']:.3f}`",
        "3. How often was the gold generated but ranked/selected incorrectly?",
        f"   - `{metrics['gold_generated_but_ranked_or_selected_incorrectly_rate']:.3f}`",
        "4. Are there multiple supporting paths for candidate answers?",
        f"   - `{metrics['multiple_supporting_paths_rate']:.3f}`",
        "5. Is answer scoring a bottleneck?",
        "   - Only if gold generation/retention is non-trivial and ranked-low cases exist.",
        "",
    ]
    return "\n".join(lines)
