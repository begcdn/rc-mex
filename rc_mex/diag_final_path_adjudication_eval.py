"""Offline eval of micro-agent 3: cross-depth final-path adjudication.

After depth-agnostic retention, final answer pools mix depth-1 and depth-2
candidates. The dominant remaining failure is a depth-1 intermediate that
passes the type filter beating the depth-2 gold (42 of 65 two-hop ranking
failures on post-fix train): the type check cannot reject it, but the paths
differ — one explains the whole question, the other a prefix. This eval asks:
if an LLM adjudicated the final retained path families listwise (one call per
question, mixed depths), and candidates from its top pick received a bonus,
how many failures convert and how many successes break?

Pure subtask measurement from an existing run's predictions.jsonl.
Gold is used only to score, never in any prompt.

Usage:
  python3 -m rc_mex.diag_final_path_adjudication_eval \
      --predictions runs/<run>/predictions.jsonl [--method path_family_any_hop_llm] \
      [--model qwen3:8b] [--bonus 0.45] [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from rc_mex.micro_agents import DEFAULT_MODEL, adjudicate_answer_candidates, rank_relation_paths, relation_path_label


def candidate_best_label(candidate: dict) -> str:
    paths = candidate.get("paths") or []
    if not paths:
        return ""
    return relation_path_label(paths[0].get("evidence", []) or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", default="path_family_any_hop_llm")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--bonus", type=float, default=0.45)
    parser.add_argument("--entity-aware", action="store_true", help="Show answer names + evidence paths (world knowledge) instead of relation labels only.")
    parser.add_argument("--prompt-version", default="final_adjudicator_v1", help="Adjudicator prompt variant (entity-aware mode only), e.g. final_adjudicator_v2.")
    parser.add_argument("--pool-size", type=int, default=12, help="How many top candidates the adjudicator sees (stored predictions keep 25).")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.predictions) as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    counts: Counter[str] = Counter()
    started = time.time()
    for index, row in enumerate(rows, start=1):
        res = row[args.method]
        gold_ids = set(row["gold_answer_ids"])
        hops = row["gold_hop_count"]
        cands = res.get("candidate_answers", [])
        counts["total"] += 1
        if not cands:
            continue
        was_hit = bool(res["hits_at_1"])
        counts[f"{hops}hop_before"] += was_hit

        if args.entity_aware:
            top = cands[: args.pool_size]
            blocks = []
            for c in top:
                readable = (c.get("paths") or [{}])[0].get("readable", "")[:180]
                blocks.append(f"{c['answer_label']} (path: {readable})")
            if len(blocks) < 2:
                counts[f"{hops}hop_after"] += was_hit
                counts["single_label_skip"] += 1
                continue
            response = adjudicate_answer_candidates(
                question=row["question"],
                start_entity_name=row["start_entity"]["name"],
                candidate_blocks=blocks,
                model=args.model,
                prompt_version=args.prompt_version,
            )
            if response["error"]:
                counts["llm_server_error"] += 1
                counts[f"{hops}hop_after"] += was_hit
                continue
            if response["pick"] is None:
                counts["llm_parse_failure"] += 1
                counts[f"{hops}hop_after"] += was_hit
                continue
            chosen_id = str(top[response["pick"]]["answer_id"])
            rescored = sorted(
                (
                    -(float(c["score"]) + (args.bonus if str(c["answer_id"]) == chosen_id else 0.0)),
                    str(c["answer_label"]),
                    str(c["answer_id"]),
                )
                for c in cands
            )
        else:
            labels = []
            for c in cands:
                label = candidate_best_label(c)
                if label and label not in labels:
                    labels.append(label)
            if len(labels) < 2:
                counts[f"{hops}hop_after"] += was_hit
                counts["single_label_skip"] += 1
                continue
            response = rank_relation_paths(
                question=row["question"],
                start_entity_name=row["start_entity"]["name"],
                path_labels=labels,
                model=args.model,
                top_k=1,
            )
            if response["error"]:
                counts["llm_server_error"] += 1
                counts[f"{hops}hop_after"] += was_hit
                continue
            if not response["picks"]:
                counts["llm_parse_failure"] += 1
                counts[f"{hops}hop_after"] += was_hit
                continue
            chosen_label = labels[response["picks"][0]]
            rescored = sorted(
                (
                    -(float(c["score"]) + (args.bonus if candidate_best_label(c) == chosen_label else 0.0)),
                    str(c["answer_label"]),
                    str(c["answer_id"]),
                )
                for c in cands
            )
        now_hit = rescored[0][2] in gold_ids
        counts[f"{hops}hop_after"] += now_hit
        counts["fixed"] += now_hit and not was_hit
        counts["broken"] += was_hit and not now_hit
        if index % 50 == 0:
            print(f"  ... {index}/{len(rows)} ({time.time()-started:.0f}s)", flush=True)

    print(json.dumps(dict(counts), indent=2))
    for hops in [1, 2, 3]:
        if counts[f"{hops}hop_before"] or counts[f"{hops}hop_after"]:
            print(f"{hops}-hop: {counts[f'{hops}hop_before']} -> {counts[f'{hops}hop_after']}")
    print(f"fixed {counts['fixed']}  broken {counts['broken']}")


if __name__ == "__main__":
    main()
