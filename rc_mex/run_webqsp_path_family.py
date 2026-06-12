"""Path-family search on RoG-preprocessed Freebase benchmarks (WebQSP / CWQ).

Breadth port of the KQA Pro pipeline: same search core, same micro-agents,
new KB. Each question ships its own Freebase subgraph (name-level triples)
from rmanluo/RoG-webqsp / RoG-cwq — the substrate used by RoG, GCR and
SubgraphRAG, so Hits@1 is comparable to published tables.

Adapter notes:
- Entities are keyed by normalized surface name; triples become forward +
  backward relation entries in the KQA kb shape. No concept inventory exists,
  so the structural concept channel is a designed no-op.
- Freebase CVT nodes (mids like "m.0k8nh0b") are traversal-only: they stay in
  the graph (2-hop answers route through them) but are removed from answer
  pools before ranking and adjudication — an opaque internal node can never
  be a named answer in these benchmarks.
- gold_hop_count is approximated by BFS distance from the topic entities to
  the nearest gold answer inside the subgraph (eval bookkeeping only; the
  search never reads it).

Methods (same keys as the KQA runner so the diag tools work unchanged):
  path_family_any_hop_concept   symbolic control (no LLM)
  path_family_any_hop_llm       + micro-agent 2 union retention
  path_family_any_hop_adjudicated  + micro-agent 3 entity-aware adjudication

Usage:
  python3 -m rc_mex.run_webqsp_path_family --data data/webqsp/test.jsonl --output runs/webqsp_test [--limit 50]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import Counter, deque

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.run_proof_state_search_smoke import (
    SmokeExample,
    apply_final_adjudication,
    build_method_regression_audit,
    f1,
    precision,
    recall,
    run_path_family_any_hop_concept,
    run_path_family_any_hop_llm,
    write_method_regression_markdown,
)

CVT_PATTERN = re.compile(r"^(m|g)\.[0-9a-z_]+$")
STORED_CANDIDATES = 25
STORED_PATHS_PER_CANDIDATE = 2
METHOD_KEYS = ["path_family_any_hop_concept", "path_family_any_hop_llm", "path_family_any_hop_adjudicated"]


def build_kb(triples: list[list[str]]) -> dict:
    entities: dict[str, dict] = {}

    def ensure(name: str) -> str:
        entity_id = normalize_text(name)
        if entity_id not in entities:
            entities[entity_id] = {"name": str(name), "instanceOf": [], "relations": [], "attributes": []}
        return entity_id

    for triple in triples:
        if len(triple) != 3:
            continue
        head, relation, tail = (str(part).strip() for part in triple)
        if not head or not relation or not tail:
            continue
        head_id, tail_id = ensure(head), ensure(tail)
        entities[head_id]["relations"].append({"predicate": relation, "object": tail_id, "direction": "forward"})
        entities[tail_id]["relations"].append({"predicate": relation, "object": head_id, "direction": "backward"})
    return {"concepts": {}, "entities": entities}


def approx_gold_hop_count(kb: dict, start_ids: set[str], gold_ids: set[str], max_depth: int = 4) -> int:
    """BFS distance (undirected) from topic entities to the nearest gold answer.

    Returns 0 when gold is unreachable within max_depth or absent from the
    subgraph. Eval bookkeeping only — the search never reads this."""
    if not start_ids or not gold_ids:
        return 0
    if start_ids & gold_ids:
        return 0
    entities = kb["entities"]
    seen = set(start_ids)
    frontier = deque((entity_id, 0) for entity_id in start_ids)
    while frontier:
        entity_id, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for relation in entities.get(entity_id, {}).get("relations", []):
            neighbor = relation["object"]
            if neighbor in seen:
                continue
            if neighbor in gold_ids:
                return depth + 1
            seen.add(neighbor)
            frontier.append((neighbor, depth + 1))
    return 0


def filter_cvt_and_rescore(result: dict, gold_ids: set[str]) -> None:
    """Drop CVT mids from the answer pool and recompute the headline fields.

    Order is preserved (the pool is already score-sorted), so this is a pure
    candidate filter, never a re-ranking."""
    candidates = [c for c in result.get("candidate_answers", []) if not CVT_PATTERN.match(str(c["answer_id"]))]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    predicted = {str(candidates[0]["answer_id"])} if candidates else set()
    pool_ids = {str(c["answer_id"]) for c in candidates}
    p = precision(gold_ids, predicted)
    r = recall(gold_ids, predicted)
    result.update(
        {
            "candidate_answers": candidates,
            "top_answer": candidates[0] if candidates else {},
            "hits_at_1": bool(predicted & gold_ids),
            "exact_match": predicted == gold_ids if gold_ids else not predicted,
            "final_answer_precision": p,
            "final_answer_recall": r,
            "final_answer_f1": f1(p, r),
            "gold_generated": bool(pool_ids & gold_ids),
        }
    )


def truncate_for_storage(result: dict) -> None:
    result.pop("audit_trace", None)
    result.pop("debug_trace", None)
    result.pop("trace", None)
    result.pop("family_records", None)
    candidates = result.get("candidate_answers", [])[:STORED_CANDIDATES]
    for candidate in candidates:
        candidate["paths"] = (candidate.get("paths") or [])[:STORED_PATHS_PER_CANDIDATE]
    result["candidate_answers"] = candidates


def empty_result(mode: str, reason: str) -> dict:
    return {
        "mode": mode,
        "skipped_reason": reason,
        "candidate_answers": [],
        "top_answer": {},
        "hits_at_1": False,
        "exact_match": False,
        "final_answer_precision": 0.0,
        "final_answer_recall": 0.0,
        "final_answer_f1": 0.0,
        "gold_generated": False,
        "llm_usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL from rc_mex.download_rog_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--relation-cap", type=int, default=30)
    parser.add_argument("--sample-entities", type=int, default=25)
    parser.add_argument("--max-branch-entities", type=int, default=40)
    parser.add_argument("--noisy-branch-threshold", type=int, default=25)
    args = parser.parse_args()

    from rc_mex.micro_agents import probe_llm_endpoint

    probe = probe_llm_endpoint()
    if probe["ok"]:
        print(f"LLM endpoint OK: {probe['url']} model={probe['model']}")
    else:
        print(f"WARNING: LLM endpoint unavailable ({probe['error']}); LLM methods fall back to symbolic.")

    source_rows = []
    with open(args.data) as fh:
        for line in fh:
            source_rows.append(json.loads(line))
    source_rows = source_rows[args.offset :]
    if args.limit:
        source_rows = source_rows[: args.limit]
    output_dir = ensure_dir(args.output)
    print(f"{len(source_rows)} questions from {args.data}")

    knobs = {
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "relation_cap": args.relation_cap,
        "sample_entities": args.sample_entities,
        "max_branch_entities": args.max_branch_entities,
        "noisy_branch_threshold": args.noisy_branch_threshold,
    }
    rows = []
    selection = Counter()
    started = time.time()
    with open(f"{output_dir}/predictions.jsonl", "w") as out:
        for index, source in enumerate(source_rows, start=1):
            kb = build_kb(source.get("graph") or [])
            graph = KnowledgeGraph(kb)
            gold_ids = {normalize_text(a) for a in source.get("answer") or [] if str(a).strip()}
            start_ids = {normalize_text(e) for e in source.get("q_entity") or [] if str(e).strip()}
            start_ids &= set(kb["entities"].keys())
            selection["total"] += 1
            selection["start_resolved"] += bool(start_ids)
            selection["gold_in_subgraph"] += bool(gold_ids & set(kb["entities"].keys()))
            hop = approx_gold_hop_count(kb, start_ids, gold_ids)

            if start_ids:
                start_name = next(str(e) for e in source.get("q_entity") if normalize_text(e) in start_ids)
                example = SmokeExample(
                    question_id=str(source.get("id", index)),
                    question=str(source.get("question", "")),
                    program_index=0,
                    start_entity_ids=start_ids,
                    start_entity_name=start_name,
                    gold_answer_ids=gold_ids,
                    gold_answer_labels=[str(a) for a in source.get("answer") or []],
                    hop_count=hop,
                )
                res_concept = run_path_family_any_hop_concept(graph=graph, example=example, **knobs)
                res_llm = run_path_family_any_hop_llm(graph=graph, example=example, **knobs)
                for res in (res_concept, res_llm):
                    res.pop("audit_trace", None)
                    filter_cvt_and_rescore(res, gold_ids)
                res_adj = copy.deepcopy(res_llm)
                apply_final_adjudication(
                    graph=graph,
                    result=res_adj,
                    question=example.question,
                    start_entity_name=example.start_entity_name,
                    gold_answer_ids=gold_ids,
                )
                adjudication = res_adj.get("final_adjudication", {})
                retention_usage = res_llm.get("llm_usage", {})
                res_adj["llm_usage"] = {
                    "llm_calls": int(retention_usage.get("llm_calls", 0)) + (1 if adjudication.get("consulted") else 0),
                    "prompt_tokens": int(retention_usage.get("prompt_tokens", 0)) + int(adjudication.get("prompt_tokens", 0)),
                    "completion_tokens": int(retention_usage.get("completion_tokens", 0))
                    + int(adjudication.get("completion_tokens", 0)),
                }
            else:
                selection["unresolved_start"] += 1
                start_name = str((source.get("q_entity") or [""])[0])
                res_concept = empty_result("path_family_any_hop_concept", "topic entity not in subgraph")
                res_llm = empty_result("path_family_any_hop_llm", "topic entity not in subgraph")
                res_adj = empty_result("path_family_any_hop_adjudicated", "topic entity not in subgraph")

            for res in (res_concept, res_llm, res_adj):
                truncate_for_storage(res)
            row = {
                "question_id": str(source.get("id", index)),
                "question": str(source.get("question", "")),
                "start_entity": {"name": start_name},
                "gold_answers": [str(a) for a in source.get("answer") or []],
                "gold_answer_ids": sorted(gold_ids),
                "gold_hop_count": hop,
                "subgraph_triples": len(source.get("graph") or []),
                "path_family_any_hop_concept": res_concept,
                "path_family_any_hop_llm": res_llm,
                "path_family_any_hop_adjudicated": res_adj,
            }
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if index % 10 == 0 or index == len(source_rows):
                elapsed = time.time() - started
                rate = elapsed / index
                print(
                    f"  ... {index}/{len(source_rows)} ({elapsed:.0f}s, ~{rate * (len(source_rows) - index):.0f}s left) "
                    f"hits@1 concept/llm/adj: "
                    f"{sum(r['path_family_any_hop_concept']['hits_at_1'] for r in rows)}/"
                    f"{sum(r['path_family_any_hop_llm']['hits_at_1'] for r in rows)}/"
                    f"{sum(r['path_family_any_hop_adjudicated']['hits_at_1'] for r in rows)}",
                    flush=True,
                )

    metrics: dict = {"selection": dict(selection), "methods": {}}
    for key in METHOD_KEYS:
        by_hop: dict[str, dict] = {}
        for row in rows:
            bucket = str(row["gold_hop_count"]) if row["gold_hop_count"] else "unreachable"
            slot = by_hop.setdefault(bucket, {"questions": 0, "hits_at_1": 0})
            slot["questions"] += 1
            slot["hits_at_1"] += bool(row[key]["hits_at_1"])
        metrics["methods"][key] = {
            "hits_at_1": sum(bool(row[key]["hits_at_1"]) for row in rows),
            "total": len(rows),
            "gold_generated": sum(bool(row[key].get("gold_generated")) for row in rows),
            "avg_f1": sum(float(row[key].get("final_answer_f1", 0.0)) for row in rows) / max(1, len(rows)),
            "by_approx_hop": by_hop,
            "llm_cost_per_question": {
                "avg_llm_calls": sum(row[key].get("llm_usage", {}).get("llm_calls", 0) for row in rows) / max(1, len(rows)),
                "avg_prompt_tokens": sum(row[key].get("llm_usage", {}).get("prompt_tokens", 0) for row in rows)
                / max(1, len(rows)),
                "avg_completion_tokens": sum(row[key].get("llm_usage", {}).get("completion_tokens", 0) for row in rows)
                / max(1, len(rows)),
            },
        }
    write_json(f"{output_dir}/metrics.json", {"args": vars(args), "metrics": metrics})

    audits = [
        (
            "llm_value_audit",
            "path_family_any_hop_concept",
            "path_family_any_hop_llm",
            "WebQSP/CWQ: LLM union retention vs symbolic control",
            "Isolates micro-agent 2 on the new KB: same search, retention union adds LLM-selected families.",
        ),
        (
            "adjudicator_value_audit",
            "path_family_any_hop_llm",
            "path_family_any_hop_adjudicated",
            "WebQSP/CWQ: Final adjudication vs any-hop LLM",
            "Isolates micro-agent 3 on the new KB: one entity-aware listwise call over the top final candidates.",
        ),
    ]
    for name, baseline_key, candidate_key, title, note in audits:
        audit = build_method_regression_audit(rows, baseline_key, candidate_key, title, note)
        write_json(f"{output_dir}/{name}.json", audit)
        with open(f"{output_dir}/{name}.md", "w") as fh:
            fh.write(write_method_regression_markdown(audit))
        print(f"{name}: {json.dumps(audit['summary'])}")

    print(json.dumps({k: v["hits_at_1"] for k, v in metrics["methods"].items()}, indent=2))
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
