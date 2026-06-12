"""Offline eval of micro-agent 4: frontier relation proposal (WebQSP/CWQ).

Both 8B and 32B retention stall at the same pool-recall ceiling, so recall is
provably proposal-bound, not model-bound: the symbolic hybrid ranker's top-10
misses the gold edge on ~19% of 1-hop questions. This eval asks: if an LLM
picked up to 10 relations from the symbolic top-M shortlist and we kept the
UNION with the symbolic top-10, how much gold-edge recall comes back?

Gold edges are derived from the data (edges from a topic entity to a gold
answer in the compressed subgraph) and used only to score, never in prompts.

Usage (train split only — this is a tuning measurement):
  python3 -m rc_mex.diag_relation_proposal_eval --data data/webqsp/train.jsonl \
      [--limit 500] [--shortlist 40] [--model qwen3:8b]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.micro_agents import DEFAULT_MODEL, propose_relations
from rc_mex.run_proof_state_search_smoke import RELATION_PROPOSAL_K, rank_relations_hybrid
from rc_mex.run_webqsp_path_family import build_kb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--shortlist", type=int, default=40)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-cases", type=int, default=0, help="Print this many recovered/lost cases.")
    args = parser.parse_args()

    rows = []
    with open(args.data) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    counts: Counter[str] = Counter()
    shown = 0
    started = time.time()
    for index, row in enumerate(rows, start=1):
        kb = build_kb(row.get("graph") or [])
        graph = KnowledgeGraph(kb)
        start_ids = {normalize_text(e) for e in row.get("q_entity") or [] if str(e).strip()} & set(kb["entities"])
        gold_ids = {normalize_text(a) for a in row.get("answer") or [] if str(a).strip()}
        if not start_ids or not gold_ids:
            continue
        start_id = sorted(start_ids)[0]
        gold_edges = {
            (rel["predicate"], rel["direction"])
            for sid in start_ids
            for rel in kb["entities"][sid]["relations"]
            if rel["object"] in gold_ids
        }
        if not gold_edges:
            counts["no_direct_gold_edge"] += 1
            continue
        counts["total"] += 1
        frontier = graph.candidate_relations(sorted(start_ids), cap=100000, sample_entities=25)
        ranked = rank_relations_hybrid(row["question"], frontier)
        keys = [(c["relation_id"], c["direction"]) for c in ranked]
        symbolic_top = set(keys[:RELATION_PROPOSAL_K])
        shortlist_keys = keys[: args.shortlist]
        symbolic_hit = bool(gold_edges & symbolic_top)
        oracle_hit = bool(gold_edges & set(shortlist_keys))
        counts["symbolic_top10"] += symbolic_hit
        counts[f"oracle_top{args.shortlist}"] += oracle_hit

        labels = [f"{rid.replace('_', ' ')} [{direction}]" for rid, direction in shortlist_keys]
        start_name = graph.entity_name(start_id)
        response = propose_relations(
            question=row["question"],
            start_entity_name=start_name,
            relation_labels=labels,
            model=args.model,
            top_k=RELATION_PROPOSAL_K,
        )
        if response["error"]:
            counts["llm_server_error"] += 1
            counts["union_top"] += symbolic_hit
            continue
        llm_keys = {shortlist_keys[i] for i in response["picks"] if 0 <= i < len(shortlist_keys)}
        union_hit = bool(gold_edges & (symbolic_top | llm_keys))
        counts["union_top"] += union_hit
        counts["llm_alone"] += bool(gold_edges & llm_keys)
        counts["recovered"] += union_hit and not symbolic_hit
        counts["llm_calls"] += 1
        if union_hit != symbolic_hit and shown < args.show_cases:
            shown += 1
            print(f"[recovered] {row['question'][:70]!r} gold={sorted(gold_edges)[:2]}")
        if index % 50 == 0:
            print(f"  ... {index}/{len(rows)} ({time.time()-started:.0f}s)", flush=True)

    print(json.dumps(dict(counts), indent=2))
    total = max(1, counts["total"])
    print(f"symbolic top-{RELATION_PROPOSAL_K}: {counts['symbolic_top10']}/{total} = {counts['symbolic_top10']/total:.1%}")
    print(f"oracle  top-{args.shortlist}: {counts[f'oracle_top{args.shortlist}']}/{total} = {counts[f'oracle_top{args.shortlist}']/total:.1%} (ceiling)")
    print(f"union (symbolic ∪ LLM): {counts['union_top']}/{total} = {counts['union_top']/total:.1%}  (recovered {counts['recovered']})")


if __name__ == "__main__":
    main()
