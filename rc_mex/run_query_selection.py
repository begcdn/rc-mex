"""Query-selection KGQA (method v2) — WebQSP/CWQ RoG subgraphs.

The answer to a KGQA question is the RESULT SET of a small query, not a
ranked entity. Measured basis (WebQSP test substrate): post-CVT, 99% of
answerable questions are one property from the topic entity, and the correct
property's full target set has F1 ceiling 0.897 (78% exact set).

Per question (2 LLM calls, short outputs):
  1. intent    — LLM names 2-3 candidate property names, no schema shown
  2. grounding — embed those names against the relations actually present
                 (per-segment max for composite CVT relations), UNION with
                 the question-embedding channel; junk predicates dropped
  3. execute   — algorithm walks each grounded (predicate, direction),
                 collecting full target sets + type evidence (zero-LLM:
                 question embedding vs subgraph type inventory)
  4. select    — LLM picks ONE property from ~a dozen options, each shown
                 with its answers; explicit abstain option; on abstain or
                 error, fall back to the question-embedding top candidate
  5. answer    — the selected property's full target set (members ordered
                 by type-match, then name)

Hard-fails at startup if MiniLM or the LLM endpoint is missing (no silent
degradation), and counts empty LLM completions as a run canary.

Usage:
  python3 -m rc_mex.run_query_selection --data data/webqsp/train150.jsonl --output runs/qsel_train150 [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

import numpy as np

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.diag_relation_description_eval import relation_similarity
from rc_mex.diag_selection_quality_eval import clean_relation
from rc_mex.micro_agents import (
    DEFAULT_MODEL,
    describe_target_relation,
    probe_llm_endpoint,
    select_query_path,
    select_set_member,
)
from rc_mex.run_proof_state_search_smoke import (
    rank_relations_hybrid,
    semantic_embedding,
    semantic_relation_model_available,
)
from rc_mex.run_webqsp_path_family import build_kb

QUESTION_CHANNEL_K = 10
DESCRIPTION_CHANNEL_K = 8
RAW_UNION_CAP = 18
DISTINCT_OPTIONS = 10
EXAMPLES_PER_OPTION = 3
REFINE_MIN_SET = 2
REFINE_MAX_SET = 12
JUNK_PREDICATE_MARKERS = (
    "freebase.valuenotation",
    "common.image",
    "appears_in_topic_gallery",
    "common.topic.webpage",
    "common.topic.article",
    "type.object",
    "dataworld.",
)


def is_junk_predicate(predicate: str) -> bool:
    return any(marker in predicate for marker in JUNK_PREDICATE_MARKERS)


def cosine(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0


def question_type_evidence(kb: dict, question: str, top_n: int = 3) -> list[str]:
    """Zero-LLM type channel (measured 68% top-3): embed the question against
    the subgraph's own type inventory."""
    inventory = sorted({c["name"] for cid, c in kb["concepts"].items() if cid.startswith("type:")})
    if not inventory:
        return []
    qv = semantic_embedding(question)
    ranked = sorted(((cosine(qv, semantic_embedding(t)), t) for t in inventory), reverse=True)
    return [t for _, t in ranked[:top_n]]


def entity_type_names(kb: dict, entity_id: str) -> list[str]:
    return [
        kb["concepts"][cid]["name"]
        for cid in kb["entities"].get(entity_id, {}).get("instanceOf", [])
        if cid.startswith("type:")
    ]


def build_candidate_paths(kb, graph, starts, question, described_names):
    """Union of the question-embedding and description-embedding channels,
    each candidate a (predicate, direction) with its full target set."""
    frontier = graph.candidate_relations(sorted(starts), cap=100000, sample_entities=25)
    ranked = rank_relations_hybrid(question, frontier)
    directions = {}
    for c in ranked:
        directions.setdefault(c["relation_id"], []).append(c["direction"])
    channel_a = [
        (c["relation_id"], c["direction"])
        for c in ranked
        if not is_junk_predicate(c["relation_id"])
    ][:QUESTION_CHANNEL_K]

    channel_b = []
    if described_names:
        name_vecs = [semantic_embedding(n) for n in described_names]
        predicates = [p for p in dict.fromkeys(c["relation_id"] for c in ranked) if not is_junk_predicate(p)]
        scored = sorted(((relation_similarity(name_vecs, p), p) for p in predicates), reverse=True)
        for _, predicate in scored[:DESCRIPTION_CHANNEL_K]:
            for direction in dict.fromkeys(directions.get(predicate, [])):
                channel_b.append((predicate, direction))

    # interleave (description channel first: it carries the semantic intent),
    # dedupe by (predicate, direction), cap the raw union
    union, seen = [], set()
    for pair_list in zip(*[iter_pad(channel_b, RAW_UNION_CAP), iter_pad(channel_a, RAW_UNION_CAP)]):
        for pair in pair_list:
            if pair and pair not in seen:
                seen.add(pair)
                union.append(pair)
    union = union[:RAW_UNION_CAP]

    # execute, then merge options whose ANSWER SETS are identical (3.4 of ~13
    # options per question were duplicate sets — pure noise for the selector).
    # Merging never drops a distinct set, so union recall is untouched.
    paths = []
    by_targets: dict[frozenset, dict] = {}
    for predicate, direction in union:
        targets = set()
        for sid in starts:
            for rel in kb["entities"][sid]["relations"]:
                if rel["predicate"] == predicate and rel["direction"] == direction:
                    targets.add(rel["object"])
        if not targets:
            continue
        key = frozenset(targets)
        if key in by_targets:
            by_targets[key]["also"].append(display_relation(predicate))
            continue
        path = {"predicate": predicate, "direction": direction, "targets": sorted(targets), "also": []}
        by_targets[key] = path
        paths.append(path)
        if len(paths) >= DISTINCT_OPTIONS:
            break
    return paths


def iter_pad(items, n):
    return items + [None] * (n - len(items))


def display_relation(predicate: str) -> str:
    """Selector-facing gloss: clean_relation plus collapsing repeated
    composite segments ('venue / venue' -> 'venue'). Kept separate from
    clean_relation so the offline probes' measurements stay comparable."""
    segments = [s.strip() for s in clean_relation(predicate).split("/")]
    deduped = list(dict.fromkeys(s for s in segments if s))
    return " / ".join(deduped)


def order_members(kb, targets: list[str], question_types: list[str]) -> list[str]:
    qtypes = set(question_types)
    return sorted(targets, key=lambda t: (not (set(entity_type_names(kb, t)) & qtypes), t))


def path_block(kb, graph, path, question_types) -> str:
    members = order_members(kb, path["targets"], question_types)
    examples = ", ".join(graph.entity_name(m) for m in members[:EXAMPLES_PER_OPTION])
    if len(members) > EXAMPLES_PER_OPTION:
        examples += ", ..."
    type_counts = Counter(t for m in members for t in entity_type_names(kb, m))
    type_str = f" [type: {', '.join(t for t, _ in type_counts.most_common(2))}]" if type_counts else ""
    reversed_str = " (reversed)" if path["direction"] == "backward" else ""
    return f"{display_relation(path['predicate'])}{reversed_str} — {len(members)} answer(s): {examples}{type_str}"


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--intent-model", default=None, help="Model for call 1 (defaults to --model).")
    parser.add_argument("--selector-model", default=None, help="Model for call 2 (defaults to --model).")
    parser.add_argument("--refiner-model", default=None, help="Model for the member-refinement call (defaults to --model).")
    args = parser.parse_args()
    intent_model = args.intent_model or args.model
    selector_model = args.selector_model or args.model
    refiner_model = args.refiner_model or args.model

    if not semantic_relation_model_available():
        sys.exit("FATAL: sentence-transformers/MiniLM unavailable — refusing to run degraded (check PYTHONPATH/HF_HOME/HF_HUB_OFFLINE).")
    probe = probe_llm_endpoint(model=selector_model)
    if not probe["ok"]:
        sys.exit(f"FATAL: LLM endpoint unavailable ({probe['error']}) — v2 requires 2 calls/question.")
    print(f"LLM endpoint OK: {probe['url']} model={probe['model']}")

    rows_in = []
    with open(args.data) as fh:
        for line in fh:
            rows_in.append(json.loads(line))
    rows_in = rows_in[args.offset:]
    if args.limit:
        rows_in = rows_in[: args.limit]
    output_dir = ensure_dir(args.output)
    print(f"{len(rows_in)} questions from {args.data}")

    stats = Counter()
    f1_sum = 0.0
    usage = Counter()
    out_rows = []
    started = time.time()
    for index, src in enumerate(rows_in, start=1):
        kb = build_kb(src.get("graph") or [])
        graph = KnowledgeGraph(kb)
        golds = {normalize_text(a) for a in src.get("answer") or [] if str(a).strip()}
        starts = {normalize_text(e) for e in src.get("q_entity") or [] if str(e).strip()} & set(kb["entities"])
        question = str(src.get("question", ""))
        start_name = str((src.get("q_entity") or [""])[0])
        stats["total"] += 1
        stats["gold_in_subgraph"] += bool(golds & set(kb["entities"]))
        row = {
            "question_id": str(src.get("id", index)),
            "question": question,
            "start_entity": start_name,
            "gold_answers": sorted(golds),
            "selected": None,
            "abstained": False,
            "fallback": False,
            "predicted": [],
        }
        if starts:
            described = describe_target_relation(question, start_name, model=intent_model)
            usage["calls"] += 1
            usage["prompt_tokens"] += described["prompt_tokens"]
            usage["completion_tokens"] += described["completion_tokens"]
            if not described["names"] and not described["error"]:
                stats["empty_completion"] += 1
            question_types = question_type_evidence(kb, question)
            paths = build_candidate_paths(kb, graph, starts, question, described["names"])
            row["described_names"] = described["names"]
            row["question_types"] = question_types
            row["candidates"] = [
                {"predicate": p["predicate"], "direction": p["direction"], "size": len(p["targets"])}
                for p in paths
            ]
            if paths:
                blocks = [path_block(kb, graph, p, question_types) for p in paths]
                selection = select_query_path(question, start_name, blocks, model=selector_model)
                usage["calls"] += 1
                usage["prompt_tokens"] += selection["prompt_tokens"]
                usage["completion_tokens"] += selection["completion_tokens"]
                if not selection["raw_response"] and not selection["error"]:
                    stats["empty_completion"] += 1
                chosen = None
                if selection["pick"] is not None:
                    chosen = paths[selection["pick"]]
                else:
                    row["abstained"] = selection["abstain"]
                    stats["abstained"] += selection["abstain"]
                    stats["selection_error"] += bool(selection["error"])
                    if selection["error"]:
                        row["selection_error"] = selection["error"][:160]
                    chosen = paths[0] if paths else None  # channel floor
                    row["fallback"] = True
                if chosen is not None:
                    members = order_members(kb, chosen["targets"], question_types)
                    # Refinement (top-1 only): when the set has several members
                    # and may answer a singular question, one micro-call picks
                    # THE member. It reorders top-1 and never mutates the set,
                    # so plural questions and answer-F1 cannot regress.
                    if REFINE_MIN_SET <= len(members) <= REFINE_MAX_SET:
                        member_blocks = []
                        for m in members:
                            types = entity_type_names(kb, m)
                            member_blocks.append(
                                graph.entity_name(m) + (f" [{', '.join(types[:2])}]" if types else "")
                            )
                        refinement = select_set_member(
                            question=question,
                            start_entity_name=start_name,
                            property_name=display_relation(chosen["predicate"]),
                            member_blocks=member_blocks,
                            model=refiner_model,
                        )
                        usage["calls"] += 1
                        usage["prompt_tokens"] += refinement["prompt_tokens"]
                        usage["completion_tokens"] += refinement["completion_tokens"]
                        if not refinement["raw_response"] and not refinement["error"]:
                            stats["empty_completion"] += 1
                        if refinement["pick"] is not None:
                            picked_member = members[refinement["pick"]]
                            members = [picked_member] + [m for m in members if m != picked_member]
                            stats["refined_top1"] += 1
                        elif refinement["whole_set"]:
                            stats["refine_whole_set"] += 1
                        row["refined"] = refinement["pick"] is not None
                    row["selected"] = {
                        "predicate": chosen["predicate"],
                        "direction": chosen["direction"],
                        "readable": display_relation(chosen["predicate"]),
                        "also": chosen.get("also", []),
                    }
                    row["predicted"] = [graph.entity_name(m) for m in members]
                    predicted_ids = set(chosen["targets"])
                    top1 = members[0]
                    stats["hits_at_1"] += top1 in golds
                    p = len(predicted_ids & golds) / len(predicted_ids) if predicted_ids else 0.0
                    r = len(predicted_ids & golds) / len(golds) if golds else 0.0
                    f1_sum += f1(p, r)
            else:
                stats["no_candidates"] += 1
        else:
            stats["unresolved_start"] += 1
        out_rows.append(row)
        if index % 10 == 0 or index == len(rows_in):
            elapsed = time.time() - started
            print(
                f"  ... {index}/{len(rows_in)} ({elapsed:.0f}s) hits@1 {stats['hits_at_1']} | "
                f"F1 {f1_sum/max(1,index):.3f} | abstain {stats['abstained']} | empty {stats['empty_completion']}",
                flush=True,
            )

    n = max(1, stats["total"])
    metrics = {
        "hits_at_1": stats["hits_at_1"],
        "hits_at_1_rate": stats["hits_at_1"] / n,
        "mean_answer_f1": f1_sum / n,
        "total": stats["total"],
        "gold_in_subgraph": stats["gold_in_subgraph"],
        "refined_top1": stats["refined_top1"],
        "refine_whole_set": stats["refine_whole_set"],
        "abstained": stats["abstained"],
        "fallbacks": stats["abstained"] + stats["selection_error"],
        "selection_errors": stats["selection_error"],
        "no_candidates": stats["no_candidates"],
        "unresolved_start": stats["unresolved_start"],
        "empty_completions": stats["empty_completion"],
        "llm_cost_per_question": {
            "avg_llm_calls": usage["calls"] / n,
            "avg_prompt_tokens": usage["prompt_tokens"] / n,
            "avg_completion_tokens": usage["completion_tokens"] / n,
        },
    }
    with open(f"{output_dir}/predictions.jsonl", "w") as out:
        for row in out_rows:
            out.write(json.dumps(row) + "\n")
    write_json(f"{output_dir}/metrics.json", {"args": vars(args), "metrics": metrics})
    print(json.dumps(metrics, indent=2))
    if stats["empty_completion"] > 0.02 * n:
        print(f"CANARY: {stats['empty_completion']} empty completions (> 2%) — check the serving backend before trusting this run.")
    print(f"Wrote {len(out_rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
