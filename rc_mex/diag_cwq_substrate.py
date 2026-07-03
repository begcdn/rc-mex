"""CWQ substrate measurement — the design-deciding numbers for multi-hop v2.

Same methodology that motivated the query-selection pivot on WebQSP: before
designing chaining, measure what the KB shape actually demands. Per question
(post-CVT compression, gold used only to locate answers and score):

  depth        BFS hops (undirected) from topic entities to nearest gold
  1-hop set    F1 of the best single (predicate, direction) full target set
  2-chain set  F1 of the best (p1,d1)->(p2,d2) chain's full target set,
               where hop-2 executes from ALL hop-1 targets (query semantics,
               not path-to-one-entity semantics)
  operators    fraction of questions whose surface form triggers our
               existing zero-LLM operators (scope words in question via
               qualifier values; ordinal first/last lexicon)

Usage:
  python3 -m rc_mex.diag_cwq_substrate --data data/cwq/train.jsonl --limit 300 [--offset 0]
"""

from __future__ import annotations

import argparse
import json
from collections import deque

from cigr_d_mvp1.kg import normalize_text
from rc_mex.run_webqsp_path_family import build_kb
from rc_mex.run_query_selection import ORDINAL_MAX, ORDINAL_MIN, is_junk_predicate


def bfs_depth(kb: dict, starts: set[str], golds: set[str], maxd: int = 4) -> int | None:
    seen = set(starts)
    queue = deque((s, 0) for s in starts)
    while queue:
        node, d = queue.popleft()
        if node in golds:
            return d
        if d >= maxd:
            continue
        for rel in kb["entities"].get(node, {}).get("relations", []):
            if rel["object"] not in seen:
                seen.add(rel["object"])
                queue.append((rel["object"], d + 1))
    return None


def hop_targets(kb: dict, sources: set[str], predicate: str, direction: str) -> set[str]:
    out = set()
    for sid in sources:
        for rel in kb["entities"].get(sid, {}).get("relations", []):
            if rel["predicate"] == predicate and rel["direction"] == direction:
                out.add(rel["object"])
    return out


def grouped_relations(kb: dict, sources: set[str]) -> dict[tuple[str, str], set[str]]:
    pairs: dict[tuple[str, str], set[str]] = {}
    for sid in sources:
        for rel in kb["entities"].get(sid, {}).get("relations", []):
            if not is_junk_predicate(rel["predicate"]):
                pairs.setdefault((rel["predicate"], rel["direction"]), set()).add(rel["object"])
    return pairs


def set_f1(targets: set[str], golds: set[str]) -> float:
    inter = len(targets & golds)
    if not inter:
        return 0.0
    p, r = inter / len(targets), inter / len(golds)
    return 2 * p * r / (p + r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--fanout-cap", type=int, default=2000, help="Skip hop-2 expansion from absurd hop-1 fanouts.")
    args = parser.parse_args()

    rows = []
    with open(args.data) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    n = 0
    depth_hist: dict = {}
    hop1_exact = hop1_any = 0
    chain_exact = chain_any = 0
    hop1_f1_sum = chain_f1_sum = 0.0
    ordinal_q = scope_shape = 0
    for src in rows:
        kb = build_kb(src.get("graph") or [])
        golds = {normalize_text(a) for a in src.get("answer") or [] if str(a).strip()} & set(kb["entities"])
        starts = {normalize_text(e) for e in src.get("q_entity") or [] if str(e).strip()} & set(kb["entities"])
        if not golds or not starts:
            depth_hist["unanswerable"] = depth_hist.get("unanswerable", 0) + 1
            n += 1
            continue
        n += 1
        d = bfs_depth(kb, starts, golds)
        depth_hist[d if d is not None else ">4"] = depth_hist.get(d if d is not None else ">4", 0) + 1

        question = str(src.get("question", ""))
        if ORDINAL_MAX.search(question) or ORDINAL_MIN.search(question):
            ordinal_q += 1

        hop1 = grouped_relations(kb, starts)
        best1 = 0.0
        has_qualed_gold_path = False
        for (prd, drn), targets in hop1.items():
            f = set_f1(targets, golds)
            best1 = max(best1, f)
        hop1_f1_sum += best1
        hop1_exact += best1 == 1.0
        hop1_any += best1 > 0

        # best 2-chain: hop-2 executes from the full hop-1 target set
        best2 = best1
        for (p1, d1), mid in hop1.items():
            if not mid or len(mid) > args.fanout_cap:
                continue
            for (p2, d2), targets in grouped_relations(kb, mid).items():
                targets = targets - starts
                if targets:
                    best2 = max(best2, set_f1(targets, golds))
        chain_f1_sum += best2
        chain_exact += best2 == 1.0
        chain_any += best2 > 0

    print(f"{n} questions | depth histogram: {dict(sorted(depth_hist.items(), key=lambda kv: str(kv[0])))}")
    print(f"1-hop  : any-overlap {hop1_any}/{n} = {hop1_any/n:.1%} | exact-set {hop1_exact}/{n} = {hop1_exact/n:.1%} | F1 ceiling {hop1_f1_sum/n:.3f}")
    print(f"2-chain: any-overlap {chain_any}/{n} = {chain_any/n:.1%} | exact-set {chain_exact}/{n} = {chain_exact/n:.1%} | F1 ceiling {chain_f1_sum/n:.3f}")
    print(f"ordinal-lexicon questions: {ordinal_q}/{n} = {ordinal_q/n:.1%}")


if __name__ == "__main__":
    main()
