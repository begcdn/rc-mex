"""Offline replay harness for the query-selection runner (method v2).

Rebuilds each question's menu locally from the stored intent names
(described_names in predictions.jsonl), verifies it matches the server run
byte-for-byte, then replays the stored LLM picks through the (possibly
modified) deterministic stages. LLM seats are never simulated: selection and
member picks come from the stored run; everything downstream — scope,
ordinal, member ordering — is recomputed live. This gives exact end-to-end
deltas for algorithmic changes without a GPU.

Also reports the stage-ceiling decomposition:
  KB ceiling      — gold present in subgraph at all
  menu ceiling    — some option's target set intersects gold (grounding)
  oracle select   — Hits@1 if the selector always picked a gold option
  oracle F1       — best achievable set-F1 over the menu

Usage:
  python3 -m rc_mex.diag_replay_qsel \
      --data data/webqsp/train.jsonl --predictions ~/Documents/<run>/predictions.jsonl \
      [--offset 150] [--limit 250] [--show-misses 0] [--inspect <question substring>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.run_webqsp_path_family import build_kb
from rc_mex.run_query_selection import (
    build_candidate_paths,
    display_relation,
    question_type_evidence,
    refine_members,
)


def load_rows(pred_path: str, data_path: str, offset: int, limit: int):
    data = [json.loads(l) for l in open(data_path)]
    if offset or limit:
        data = data[offset:]
        if limit:
            data = data[:limit]
    by_id = {str(d["id"]): d for d in data}
    rows = [json.loads(l) for l in open(pred_path)]
    return [(r, by_id[str(r["question_id"])]) for r in rows if str(r["question_id"]) in by_id]


def rebuild(row, datum):
    kb = build_kb(datum.get("graph") or [])
    graph = KnowledgeGraph(kb)
    golds = {normalize_text(a) for a in row["gold_answers"]}
    starts = {normalize_text(e) for e in (datum.get("q_entity") or []) if str(e).strip()} & set(kb["entities"])
    question_types = row.get("question_types") or question_type_evidence(kb, row["question"])
    paths = build_candidate_paths(kb, graph, starts, row["question"], row.get("described_names") or [])
    return kb, graph, starts, golds, question_types, paths


def find_selected(paths, selected):
    if not selected:
        return None
    for p in paths:
        if p["predicate"] == selected.get("predicate") and p["direction"] == selected.get("direction"):
            return p
    return None


def replay_refinement(kb, graph, row, chosen, question_types, starts):
    """Run the SHIPPING refinement chain (refine_members) with the stored
    run's member pick replayed as the micro-call. Where the stored pick is no
    longer in the (re-scoped) set, the call abstains and we flag the row —
    the deterministic order stands as a conservative lower bound."""
    trace = {"member_unknown": False}

    def replay_member_call(members, blocks):
        if row.get("refined") and row.get("predicted"):
            picked_name = row["predicted"][0]
            for m in members:
                if graph.entity_name(m) == picked_name:
                    return m
            trace["member_unknown"] = True
            return None
        return "ALL"  # stored run kept the whole set (or made no call)

    members, flags = refine_members(
        kb,
        graph,
        row["question"],
        question_types,
        chosen,
        member_call=replay_member_call,
        topic_names=[graph.entity_name(s) for s in starts],
    )
    trace.update(flags)
    return members, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--inspect", default=None, help="Dump full menu/qualifier detail for questions containing this substring.")
    parser.add_argument("--show-misses", type=int, default=0)
    args = parser.parse_args()

    pairs = load_rows(args.predictions, args.data, args.offset, args.limit)
    print(f"{len(pairs)} rows joined")

    stats = Counter()
    f1_sum = 0.0
    replay_f1_sum = [0.0]
    changed = []
    for row, datum in pairs:
        golds = {normalize_text(a) for a in row["gold_answers"]}
        stored_pred = [normalize_text(p) for p in (row.get("predicted") or [])]
        stored_hit = bool(stored_pred) and stored_pred[0] in golds
        stats["stored_hits"] += stored_hit

        kb, graph, starts, golds, question_types, paths = rebuild(row, datum)
        rebuilt_menu = [(p["predicate"], p["direction"], len(p["targets"])) for p in paths]
        stored_menu = [(c["predicate"], c["direction"], c["size"]) for c in (row.get("candidates") or [])]
        stats["menu_match"] += rebuilt_menu == stored_menu
        stats["gold_in_kb"] += bool(golds & set(kb["entities"]))

        gold_paths = [p for p in paths if set(p["targets"]) & golds]
        stats["menu_gold"] += bool(gold_paths)
        if gold_paths:
            best_f1 = 0.0
            for p in gold_paths:
                t = set(p["targets"])
                prec = len(t & golds) / len(t)
                rec = len(t & golds) / len(golds) if golds else 0.0
                best_f1 = max(best_f1, 2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            stats["oracle_select"] += 1  # oracle also orders gold first
            f1_sum += best_f1

        chosen = find_selected(paths, row.get("selected"))
        if chosen is None:
            stats["selected_missing"] += 1
            continue
        members, trace = replay_refinement(kb, graph, row, chosen, question_types, starts)
        stats["replay_scope"] += trace["scope"]
        stats["replay_ordinal"] += trace["ordinal"]
        stats["member_unknown"] += trace["member_unknown"]
        replay_hit = bool(members) and members[0] in golds
        stats["replay_hits"] += replay_hit
        pred_ids = set(members)
        prec = len(pred_ids & golds) / len(pred_ids) if pred_ids else 0.0
        rec = len(pred_ids & golds) / len(golds) if golds else 0.0
        stats_f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        replay_f1_sum[0] += stats_f1
        if replay_hit != stored_hit:
            changed.append((row["question"][:60], stored_hit, replay_hit, graph.entity_name(members[0]) if members else None, sorted(golds)[:2], trace))

        if args.inspect and args.inspect.lower() in row["question"].lower():
            print(f"\n=== INSPECT: {row['question']!r} | golds {sorted(golds)[:4]}")
            print(f"  described: {row.get('described_names')}, types: {question_types}")
            for i, p in enumerate(paths):
                mark = " <GOLD>" if set(p["targets"]) & golds else ""
                sel = " <SELECTED>" if p is chosen else ""
                print(f"  [{i+1}] {display_relation(p['predicate'])} ({p['direction']}) n={len(p['targets'])}{mark}{sel}")
                for m in p["targets"][:6]:
                    print(f"        {graph.entity_name(m)!r} quals={p['member_quals'].get(m, {})}")
            print(f"  replay trace: {trace} -> top1 {graph.entity_name(members[0]) if members else None!r}")

    n = len(pairs)
    print(f"\nmenu reproduction: {stats['menu_match']}/{n} (selected path missing after rebuild: {stats['selected_missing']})")
    print(f"replay hits {stats['replay_hits']} (F1 {replay_f1_sum[0]/n:.3f}) vs stored {stats['stored_hits']} (member_unknown {stats['member_unknown']})")
    print("\nstage ceilings:")
    print(f"  gold in KB:            {stats['gold_in_kb']}/{n} = {stats['gold_in_kb']/n:.1%}")
    print(f"  gold on menu:          {stats['menu_gold']}/{n} = {stats['menu_gold']/n:.1%}   <- grounding ceiling")
    print(f"  oracle select Hits@1:  {stats['oracle_select']}/{n} = {stats['oracle_select']/n:.1%}")
    print(f"  oracle F1 (best set):  {f1_sum/n:.3f}")
    print(f"\nreplay operators: scope {stats['replay_scope']}, ordinal {stats['replay_ordinal']}")
    if changed:
        print(f"\nchanged vs stored ({len(changed)}):")
        for q, old, new, top1, g, tr in changed[:30]:
            arrow = "FIXED" if new else "BROKE"
            print(f"  [{arrow}] {q!r} top1={top1!r} gold~{g} {tr}")


if __name__ == "__main__":
    main()
