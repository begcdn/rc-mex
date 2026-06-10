"""Offline eval of micro-agent 2: relation-path plausibility for family retention.

The hop-2 family beam keeps only the top-3 path families by symbolic retention
score; on held-out data a gold-containing family exists for most questions but
survives the beam less than 60% of the time. This eval asks: if an LLM ranked
the candidate relation paths listwise (one call per question), would the
gold-containing family make the top-3 more often?

Pure subtask measurement from an existing run's predictions.jsonl — no search
or pipeline changes. Gold is used only to score, never in any prompt.

Usage:
  python3 -m rc_mex.diag_path_plausibility_eval \
      --predictions runs/heldout_train_320/predictions.jsonl \
      [--model qwen3:8b] [--max-paths 60] [--limit 0] [--show-cases]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from typing import Any

from rc_mex.micro_agents import DEFAULT_MODEL, call_local_llm

PATH_PLAUSIBILITY_PROMPT_VERSION = "path_plausibility_v1"
PATH_PLAUSIBILITY_PROMPT_TEMPLATE = """A question is answered by following relations in a knowledge base, starting from the starting entity. [forward] follows the relation, [backward] follows it in reverse.

Starting entity: "{start}"
Question: "{question}"

Candidate relation paths (hop 1 -> hop 2):
{paths}

Pick the 3 paths most likely to reach the answer to the question. Reply with only the 3 numbers separated by commas, best first."""


def family_path_label(summary: dict[str, Any]) -> str:
    by_hop: dict[int, list[str]] = defaultdict(list)
    for step in summary.get("evidence", []) or []:
        hop = int(step.get("hop", 0))
        label = f"{str(step.get('relation_id', '')).replace('_', ' ')} [{step.get('direction', '')}]"
        if label not in by_hop[hop]:
            by_hop[hop].append(label)
    return " -> ".join(" & ".join(by_hop[hop]) for hop in sorted(by_hop))


def parse_pick_numbers(raw: str, count: int) -> list[int]:
    picks: list[int] = []
    for token in re.findall(r"\d+", raw):
        value = int(token)
        if 1 <= value <= count and value not in picks:
            picks.append(value)
        if len(picks) == 3:
            break
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", default="path_family_concept_verifier")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-paths", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions (0 = all).")
    parser.add_argument("--show-cases", action="store_true")
    args = parser.parse_args()

    rows = []
    with open(args.predictions) as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    counts: Counter[str] = Counter()
    cases = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        result = row[args.method]
        gold_ids = set(row["gold_answer_ids"])
        hop2 = next((entry for entry in result.get("audit_trace", []) if entry.get("hop") == 2), None)
        counts["total"] += 1
        if not hop2:
            counts["no_hop2_trace"] += 1
            continue
        families = list(hop2.get("all_path_families", []))[: args.max_paths]
        dropped_gold_beyond_cap = any(
            set(f.get("pool_target_entities", [])) & gold_ids
            for f in list(hop2.get("all_path_families", []))[args.max_paths:]
        )
        if dropped_gold_beyond_cap:
            counts["gold_family_beyond_path_cap"] += 1

        labels: list[str] = []
        label_is_gold: dict[str, bool] = {}
        label_best_rank: dict[str, int] = {}
        for family in families:
            label = family_path_label(family)
            if not label:
                continue
            is_gold = bool(set(family.get("pool_target_entities", [])) & gold_ids)
            if label not in label_best_rank:
                labels.append(label)
                label_best_rank[label] = int(family.get("rank", 999))
                label_is_gold[label] = is_gold
            else:
                label_is_gold[label] = label_is_gold[label] or is_gold
                label_best_rank[label] = min(label_best_rank[label], int(family.get("rank", 999)))

        gold_labels = {label for label, ok in label_is_gold.items() if ok}
        if not gold_labels:
            counts["no_gold_family_at_hop2"] += 1
            continue
        counts["gold_family_exists"] += 1

        symbolic_top3_labels = sorted(labels, key=lambda l: label_best_rank[l])[:3]
        symbolic_hit = bool(gold_labels & set(symbolic_top3_labels))
        counts["symbolic_top3_hit"] += symbolic_hit

        numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
        prompt = PATH_PLAUSIBILITY_PROMPT_TEMPLATE.format(
            start=row["start_entity"]["name"], question=row["question"], paths=numbered
        )
        response = call_local_llm(prompt, PATH_PLAUSIBILITY_PROMPT_VERSION, model=args.model, timeout=180.0)
        if response["error"]:
            counts["llm_server_error"] += 1
            if counts["llm_server_error"] == 1:
                print(f"first server error: {response['error']}")
            continue
        picks = parse_pick_numbers(response["text"], len(labels))
        if not picks:
            counts["llm_parse_failure"] += 1
        llm_top3_labels = [labels[i - 1] for i in picks]
        llm_hit = bool(gold_labels & set(llm_top3_labels))
        counts["llm_top3_hit"] += llm_hit
        counts["both_hit"] += symbolic_hit and llm_hit
        counts["llm_only_hit"] += llm_hit and not symbolic_hit
        counts["symbolic_only_hit"] += symbolic_hit and not llm_hit
        counts["neither_hit"] += not symbolic_hit and not llm_hit
        if args.show_cases and llm_hit != symbolic_hit:
            cases.append(
                {
                    "question": row["question"][:90],
                    "who_won": "llm" if llm_hit else "symbolic",
                    "gold_label": sorted(gold_labels)[0],
                    "symbolic_top3": symbolic_top3_labels,
                    "llm_top3": llm_top3_labels,
                    "llm_raw": response["text"][:80],
                }
            )
        if index % 25 == 0:
            print(f"  ... {index}/{len(rows)} ({time.time()-started:.0f}s)", flush=True)

    print(json.dumps(dict(counts), indent=2))
    exists = counts["gold_family_exists"]
    print(f"gold family in top-3: symbolic {counts['symbolic_top3_hit']}/{exists}  llm {counts['llm_top3_hit']}/{exists}")
    print(f"llm recovers (llm yes, symbolic no): {counts['llm_only_hit']}   llm loses (symbolic yes, llm no): {counts['symbolic_only_hit']}")
    if args.show_cases:
        for case in cases:
            print(f"[{case['who_won']:8}] q={case['question']}")
            print(f"           gold path: {case['gold_label']}")
            print(f"           llm picked: {case['llm_top3']}  raw={case['llm_raw']!r}")


if __name__ == "__main__":
    main()
