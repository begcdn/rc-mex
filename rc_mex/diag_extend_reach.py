"""Design probe for the EXTEND move (selection round 2).

Arm F showed the abstain rows are not selection failures: the forced pick
lands on the correct intermediate entity (anthem->country, school->student)
but the menu never contained the second hop, because chain grounding depends
on the intent call's second property name. This measures the ceiling of the
architectural fix — letting selection answer "option k, then continue" and
executing a fresh hop from option k's result set:

  per abstained (or all) row:
    gold in KB / gold on rebuilt mixed menu (context)
    EXTEND reach: best set-F1 over all (predicate, direction) groups executed
    from the stored pick's full target set; hit if some group contains gold.

Usage:
  python3 -m rc_mex.diag_extend_reach --data data/cwq/train.jsonl \
      --predictions ~/Documents/cwq_dev300f/predictions.jsonl --limit 300 \
      [--class abstained|all|miss]
"""

from __future__ import annotations

import argparse
import json

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.run_webqsp_path_family import build_kb
from rc_mex.run_query_selection import (
    build_candidate_paths,
    build_chain_candidates,
    build_intersection_candidates,
    display_relation,
    is_junk_predicate,
    merge_mixed_menu,
)


def set_f1(t: set, g: set) -> float:
    inter = len(t & g)
    if not inter:
        return 0.0
    p, r = inter / len(t), inter / len(g)
    return 2 * p * r / (p + r)


def grouped(kb, sources: set[str], exclude: set[str]) -> dict:
    out: dict[tuple[str, str], set[str]] = {}
    for sid in sources:
        for rel in kb["entities"].get(sid, {}).get("relations", []):
            if not is_junk_predicate(rel["predicate"]):
                out.setdefault((rel["predicate"], rel["direction"]), set()).add(rel["object"])
    return {k: v - exclude for k, v in out.items() if v - exclude}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--row-class", default="abstained", choices=["abstained", "all", "miss"])
    args = parser.parse_args()

    data = []
    with open(args.data) as fh:
        for i, line in enumerate(fh):
            if i >= args.offset + args.limit:
                break
            data.append(json.loads(line))
    data = data[args.offset:]
    by_id = {str(d["id"]): d for d in data}
    rows = [json.loads(l) for l in open(args.predictions)]

    def is_hit(r):
        golds = {normalize_text(g) for g in r["gold_answers"]}
        pred = r.get("predicted") or []
        return bool(pred) and normalize_text(pred[0]) in golds

    if args.row_class == "abstained":
        rows = [r for r in rows if r.get("abstained")]
    elif args.row_class == "miss":
        rows = [r for r in rows if not is_hit(r)]

    n = kb_ok = menu_gold = pick_found = 0
    reach_any = reach_strong = 0
    f1_menu_sum = f1_extend_sum = 0.0
    details = []
    for r in rows:
        datum = by_id.get(str(r["question_id"]))
        if datum is None:
            continue
        n += 1
        kb = build_kb(datum.get("graph") or [])
        graph = KnowledgeGraph(kb)
        golds = {normalize_text(a) for a in r["gold_answers"]} & set(kb["entities"])
        starts = {normalize_text(e) for e in (datum.get("q_entity") or []) if str(e).strip()} & set(kb["entities"])
        if not golds:
            continue
        kb_ok += 1
        hop1 = build_candidate_paths(kb, graph, starts, r["question"], r.get("described_names") or [])
        chains = build_chain_candidates(kb, graph, starts, hop1, r.get("described_names_2") or [])
        menu = merge_mixed_menu(hop1, chains)
        menu = menu + build_intersection_candidates(menu)
        best_menu = max((set_f1(set(p["targets"]), golds) for p in menu), default=0.0)
        f1_menu_sum += best_menu
        menu_gold += best_menu > 0

        sel = r.get("selected") or {}
        pick = next(
            (p for p in menu if p["predicate"] == sel.get("predicate") and p["direction"] == sel.get("direction")),
            None,
        )
        if pick is None:
            continue
        pick_found += 1
        frontier = set(pick["targets"])
        best_ext = 0.0
        best_lab = None
        for (prd, drn), targets in grouped(kb, frontier, starts | frontier).items():
            f = set_f1(targets, golds)
            if f > best_ext:
                best_ext, best_lab = f, f"{display_relation(prd)} ({drn}) n={len(targets)}"
        f1_extend_sum += best_ext
        reach_any += best_ext > 0
        reach_strong += best_ext >= 0.5
        details.append((r["question"][:70], round(best_menu, 2), round(best_ext, 2), best_lab))

    print(f"{n} rows | gold-in-KB {kb_ok} | pick matched on rebuilt menu {pick_found}")
    print(f"gold on menu already: {menu_gold}/{kb_ok} (best-menu F1 avg {f1_menu_sum/max(kb_ok,1):.3f})")
    print(f"EXTEND from stored pick: any-overlap {reach_any}/{pick_found}, F1>=0.5 {reach_strong}/{pick_found}, avg best F1 {f1_extend_sum/max(pick_found,1):.3f}")
    for q, bm, be, lab in details:
        print(f"  menuF1={bm:<5} extendF1={be:<5} {q!r} via {lab}")


if __name__ == "__main__":
    main()
