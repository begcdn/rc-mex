"""Selection-quality probe: does a well-posed, type-aware selection task lift
the LLM selector's accuracy on WebQSP?

Diagnosis showed the selector picks the wrong candidate ~70% of the time even
when gold is in its window, because the prompt is information-starved (name +
truncated path, no type signal). This probe re-poses selection over the SAME
stored candidate pools two ways and measures rescue rate:

  OLD:  "{name} (path: {readable[:180]})"               -> adjudicate_answer_candidates
  NEW:  "{name} [type: ...] -- via {clean relation}"    -> select_answer_typed,
        with the question's expected answer-type stated explicitly.

Both decide UNBOUNDED (the pick is authoritative — no 0.45 bonus), so this also
folds in the bounded-bonus fix. Metric: among questions where gold is in the
top-K window, how often the selector picks a gold answer. Gold is used only to
score, never in any prompt.

Run on a TRAIN predictions file (tuning), confirm on test later.
Usage:
  python3 -m rc_mex.diag_selection_quality_eval \
      --data data/webqsp/train.jsonl --predictions runs/<train_run>/predictions.jsonl \
      [--method path_family_any_hop_llm] [--pool-size 12] [--model qwen3:8b] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.micro_agents import DEFAULT_MODEL, adjudicate_answer_candidates, select_answer_typed
from rc_mex.run_proof_state_search_smoke import extract_answer_concept
from rc_mex.run_webqsp_path_family import build_kb


def clean_relation(predicate: str) -> str:
    """Freebase predicate -> readable tail (people.person.parents -> parents;
    composite 'a.b.c / d.e.f' -> 'c / f')."""
    parts = [seg.strip().split(".")[-1].replace("_", " ") for seg in str(predicate).split("/")]
    return " / ".join(p for p in parts if p)


def candidate_clean_path(candidate: dict) -> str:
    evidence = (candidate.get("paths") or [{}])[0].get("evidence", []) or []
    rels = [clean_relation(step.get("predicate", "")) for step in evidence]
    rels = [r for r in rels if r]
    return " then ".join(rels) if rels else "(direct)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Original RoG JSONL (for the subgraph / types).")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", default="path_family_any_hop_llm", help="Pool to select over (pre-adjudication).")
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show-cases", type=int, default=0)
    args = parser.parse_args()

    data_by_id = {}
    with open(args.data) as fh:
        for line in fh:
            row = json.loads(line)
            data_by_id[str(row.get("id", ""))] = row

    preds = []
    with open(args.predictions) as fh:
        for line in fh:
            preds.append(json.loads(line))
    if args.limit:
        preds = preds[: args.limit]

    counts: Counter[str] = Counter()
    shown = 0
    started = time.time()
    for index, pred in enumerate(preds, start=1):
        qid = str(pred.get("question_id", ""))
        data_row = data_by_id.get(qid)
        if data_row is None:
            counts["no_data_match"] += 1
            continue
        res = pred.get(args.method, {})
        gold = set(pred.get("gold_answer_ids", []))
        pool = (res.get("candidate_answers") or [])[: args.pool_size]
        if not pool or not gold:
            continue
        gold_in_window = {str(c["answer_id"]) for c in pool} & gold
        if not gold_in_window:
            continue  # selector cannot rescue what it is not shown
        counts["gold_in_window"] += 1
        counts["baseline_top1_correct"] += str(pool[0]["answer_id"]) in gold

        graph = KnowledgeGraph(build_kb(data_row.get("graph") or []))
        start_name = pred.get("start_entity", {}).get("name", "")
        extracted = extract_answer_concept(graph, pred["question"], [start_name])
        answer_type = extracted["concept_name"] if extracted else ""

        gold_labels = {str(c["answer_label"]) for c in pool if str(c["answer_id"]) in gold}

        old_blocks = []
        new_blocks = []
        for c in pool:
            readable = (c.get("paths") or [{}])[0].get("readable", "")[:180]
            old_blocks.append(f"{c['answer_label']} (path: {readable})")
            types = graph.entity_type_names(str(c["answer_id"]), limit=4)
            type_str = ", ".join(types) if types else "unknown"
            new_blocks.append(f"{c['answer_label']} [type: {type_str}] -- via {candidate_clean_path(c)}")

        old = adjudicate_answer_candidates(pred["question"], start_name, old_blocks, model=args.model)
        new = select_answer_typed(pred["question"], start_name, answer_type, new_blocks, model=args.model)
        if old["error"] or new["error"]:
            counts["llm_error"] += 1
            continue

        old_hit = old["pick"] is not None and str(pool[old["pick"]]["answer_label"]) in gold_labels
        new_hit = new["pick"] is not None and str(pool[new["pick"]]["answer_label"]) in gold_labels
        counts["OLD_rescue"] += old_hit
        counts["NEW_rescue"] += new_hit
        counts["NEW_fixes_OLD_miss"] += new_hit and not old_hit
        counts["NEW_breaks_OLD_hit"] += old_hit and not new_hit

        if args.show_cases and shown < args.show_cases and new_hit and not old_hit:
            shown += 1
            print(f"[NEW fixes] q={pred['question'][:60]!r} type={answer_type!r}")
            print(f"   OLD picked: {pool[old['pick']]['answer_label'] if old['pick'] is not None else None!r}")
            print(f"   NEW picked: {pool[new['pick']]['answer_label']!r}  (gold)")
        if index % 50 == 0:
            print(f"  ... {index}/{len(preds)} ({time.time()-started:.0f}s)", flush=True)

    w = max(1, counts["gold_in_window"])
    print(json.dumps(dict(counts), indent=2))
    print()
    print(f"gold-in-window questions: {counts['gold_in_window']}")
    print(f"  baseline heuristic top-1 correct: {counts['baseline_top1_correct']}/{w} = {counts['baseline_top1_correct']/w:.0%}")
    print(f"  OLD selector rescue:  {counts['OLD_rescue']}/{w} = {counts['OLD_rescue']/w:.0%}")
    print(f"  NEW selector rescue:  {counts['NEW_rescue']}/{w} = {counts['NEW_rescue']/w:.0%}")
    print(f"  NEW fixes / breaks vs OLD: +{counts['NEW_fixes_OLD_miss']} / -{counts['NEW_breaks_OLD_hit']}")


if __name__ == "__main__":
    main()
