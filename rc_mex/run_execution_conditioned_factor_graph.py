"""Controlled evaluation of execution-conditioned factor-graph inference.

The runner evaluates several frozen selectors over one shared candidate pool.
It uses gold topic entities to isolate query-graph inference; gold relations and
gold programs are never exposed to proposal generation or ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.kg import normalize_text
from rc_mex.execution_conditioned_factor_graph import (
    FactorGraphModels,
    KQARuntime,
    WebSubgraphRuntime,
    build_candidate_pool,
    rank_candidate_pool,
    score_answers,
)
from rc_mex.factor_graph_ir import abstract_graph, compile_kqa_program, compile_webqsp_question, graph_key, serialize_graph


ARMS = ("learned_product", "grammar_product", "grammar_topology_product", "source_aware")


def load_selection(path: Path | None, key: str) -> list[str | int] | None:
    if path is None:
        return None
    rows = json.load(open(path, encoding="utf-8"))
    return [row[key] if isinstance(row, dict) else row for row in rows]


def top_diagnostics(ranked, limit: int = 10) -> list[dict]:
    return [
        {
            "rank": rank,
            "score": score,
            "graph": serialize_graph(candidate.graph),
            "answers": sorted(candidate.answers)[:50],
            "answer_count": len(candidate.answers),
            "binding_count": candidate.binding_count,
            "learned_topology": candidate.learned_topology,
        }
        for rank, (score, candidate) in enumerate(ranked[:limit], 1)
    ]


def evaluate_one(question_id, question, slots, runtime, gold_graph, gold, models, semantic_schema) -> dict:
    candidates, learned_topologies = build_candidate_pool(models, runtime, question, slots)
    rankings = rank_candidate_pool(models, question, slots, candidates, semantic_schema)
    result = {
        "id": question_id,
        "question": question,
        "linked_entities": slots,
        "gold": sorted(gold),
        "candidate_pool": {
            "count": len(candidates),
            "learned_count": sum(candidate.learned_topology for candidate in candidates),
            "grammar_added_count": sum(not candidate.learned_topology for candidate in candidates),
            "gold_answer_generated": any(candidate.answers & gold for candidate in candidates),
            "exact_denotation_generated": any(set(candidate.answers) == gold for candidate in candidates),
            "exact_graph_generated": any(candidate.graph == gold_graph for candidate in candidates),
            "gold_topology_learned": any(graph_key(topology) == graph_key(abstract_graph(gold_graph)) for topology in learned_topologies),
        },
        "arms": {},
    }
    for arm in ARMS:
        ranked = rankings.get(arm, [])
        predicted = set(ranked[0][1].answers) if ranked else set()
        result["arms"][arm] = {
            "predicted": sorted(predicted),
            **score_answers(predicted, gold),
            "gold_best_rank": next((rank for rank, (_, candidate) in enumerate(ranked, 1) if candidate.answers & gold), None),
            "top_candidates": top_diagnostics(ranked),
        }
    return result


def kqa_rows(args, models) -> list[dict]:
    questions = json.load(open(args.questions, encoding="utf-8"))
    selected = load_selection(args.selection, "index")
    indices = [int(index) for index in selected] if selected is not None else list(range(len(questions)))
    runtime = KQARuntime(args.kb)
    rows = []
    for index in indices:
        row = questions[index]
        compiled = compile_kqa_program(row.get("program") or [])
        if compiled is None:
            continue
        gold_graph, slots = compiled
        result = evaluate_one(index, str(row.get("question", "")), slots, runtime, gold_graph, {normalize_text(row.get("answer", ""))}, models, False)
        rows.append(result)
        print_progress(len(rows), args.limit, result)
        if len(rows) >= args.limit:
            break
    return rows


def load_web_subgraphs(path: Path, wanted: set[str]) -> dict[str, dict]:
    found = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if str(row.get("id")) in wanted:
                found[str(row["id"])] = row
                if len(found) == len(wanted):
                    break
    return found


def web_rows(args, models) -> list[dict]:
    official_rows = json.load(open(args.questions, encoding="utf-8"))["Questions"]
    official = {str(row["QuestionId"]): row for row in official_rows}
    selected = load_selection(args.selection, "id")
    if selected is None:
        selected = [str(row["QuestionId"]) for row in official_rows if compile_webqsp_question(row) is not None][: args.limit]
    selected = [str(item) for item in selected]
    subgraphs = load_web_subgraphs(args.subgraphs, set(selected))
    rows = []
    for question_id in selected:
        if question_id not in official or question_id not in subgraphs:
            continue
        official_row, subgraph = official[question_id], subgraphs[question_id]
        compiled = compile_webqsp_question(official_row)
        if compiled is None:
            continue
        gold_graph, slots = compiled
        question = str(official_row.get("ProcessedQuestion") or official_row.get("RawQuestion") or "")
        gold = {normalize_text(answer) for answer in subgraph.get("answer", [])}
        result = evaluate_one(question_id, question, slots, WebSubgraphRuntime(subgraph), gold_graph, gold, models, True)
        rows.append(result)
        print_progress(len(rows), args.limit, result)
        if len(rows) >= args.limit:
            break
    return rows


def print_progress(position: int, total: int, row: dict) -> None:
    arm_values = " ".join(f"{arm}={row['arms'][arm]['f1']:.3f}" for arm in ARMS)
    print(
        f"[{position}/{total}] {row['id']} candidates={row['candidate_pool']['count']} "
        f"gold_generated={row['candidate_pool']['gold_answer_generated']} {arm_values}",
        flush=True,
    )


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    pool = {
        "count": count,
        "gold_answer_generated": sum(row["candidate_pool"]["gold_answer_generated"] for row in rows) / count if count else 0.0,
        "exact_denotation_generated": sum(row["candidate_pool"]["exact_denotation_generated"] for row in rows) / count if count else 0.0,
        "exact_graph_generated": sum(row["candidate_pool"]["exact_graph_generated"] for row in rows) / count if count else 0.0,
        "gold_topology_learned": sum(row["candidate_pool"]["gold_topology_learned"] for row in rows) / count if count else 0.0,
        "average_candidates": sum(row["candidate_pool"]["count"] for row in rows) / count if count else 0.0,
    }
    arms = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in rows]
        arms[arm] = {
            "count": count,
            "exact_match": sum(item["exact_match"] for item in values) / count if count else 0.0,
            "hits_at_1": sum(item["hits_at_1"] for item in values) / count if count else 0.0,
            "precision": sum(item["precision"] for item in values) / count if count else 0.0,
            "recall": sum(item["recall"] for item in values) / count if count else 0.0,
            "f1": sum(item["f1"] for item in values) / count if count else 0.0,
        }
    return {"candidate_pool": pool, "arms": arms}


def report(dataset: str, metrics: dict) -> str:
    pool = metrics["candidate_pool"]
    lines = [
        "# Execution-Conditioned Factor-Graph Evaluation",
        "",
        f"Dataset: `{dataset}`",
        "",
        "This is a controlled query-inference evaluation with gold topic entities. Gold relations and gold query edges are used only for diagnostics.",
        "",
        "## Candidate Pool",
        "",
        f"- Questions: {pool['count']}",
        f"- Gold-answer generated: {pool['gold_answer_generated']:.3f}",
        f"- Exact denotation generated: {pool['exact_denotation_generated']:.3f}",
        f"- Exact graph generated: {pool['exact_graph_generated']:.3f}",
        f"- Gold topology from learned proposer: {pool['gold_topology_learned']:.3f}",
        f"- Average executable candidates: {pool['average_candidates']:.1f}",
        "",
        "## Shared-Pool Selectors",
        "",
        "| Selector | EM | Hits@1 | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = metrics["arms"][arm]
        lines.append(f"| {arm} | {item['exact_match']:.3f} | {item['hits_at_1']:.3f} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    lines.extend(("", "The selector comparison is diagnostic. It tests how grammar completion and topology evidence affect one frozen proof pool; it does not treat any fusion rule as settled."))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("kqa", "webqsp"), required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("runs/execution_conditioned_factor_graph_models"))
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--kb", type=Path, default=Path("data/kqa_pro/kb.json"))
    parser.add_argument("--subgraphs", type=Path, default=Path("data/webqsp/test.jsonl"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/execution_conditioned_factor_graph"))
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.questions is None:
        args.questions = Path("data/kqa_pro/val.json" if args.dataset == "kqa" else "data/pattern_alignment/webqsp/WebQSP.test.json")
    ensure_dir(args.output)
    print(f"Loading frozen factor models from {args.model_dir}", flush=True)
    models = FactorGraphModels(args.model_dir)
    print(f"Evaluating {args.dataset} (limit={args.limit})", flush=True)
    rows = kqa_rows(args, models) if args.dataset == "kqa" else web_rows(args, models)
    metrics = aggregate(rows)
    with open(args.output / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output / "metrics.json", metrics)
    (args.output / "report.md").write_text(report(args.dataset, metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote outputs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
