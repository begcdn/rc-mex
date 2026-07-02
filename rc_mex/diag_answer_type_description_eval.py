"""Answer-type channel probe for the query-selection method (WebQSP/CWQ).

In the query-selection formulation the selector chooses among a handful of
candidate PATHS, and the main evidence besides the property name is the type
of the target set. This measures how well the expected answer type can be
identified, HyDE-style: the LLM names the answer type WITHOUT seeing the KB's
type inventory; we embed that and match it against the types actually present
in the subgraph, then check whether the gold answer's real type is ranked
top-1/top-3.

Baselines measured alongside, same metric:
  - question-embedding: embed the raw question against the type inventory
    (no LLM), isolating the HyDE contribution;
  - the old KQA wh-rule extractor (string match), which we know fires rarely.

Gold is used only to score. Usage:
  python3 -m rc_mex.diag_answer_type_description_eval \
      --data data/webqsp/test.jsonl --predictions runs/webqsp_test_frozen/predictions.jsonl \
      [--model qwen3:8b] [--limit 300] [--show 10]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.micro_agents import DEFAULT_MODEL, describe_answer_type
from rc_mex.run_proof_state_search_smoke import (
    extract_answer_concept,
    get_semantic_relation_model,
    semantic_embedding,
)
from rc_mex.run_webqsp_path_family import build_kb


def cosine(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0


def rank_types(query_text: str, inventory: list[str]) -> list[str]:
    qv = semantic_embedding(query_text)
    return [t for _, t in sorted(((cosine(qv, semantic_embedding(t)), t) for t in inventory), reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    if get_semantic_relation_model() is None:
        print("WARNING: semantic model unavailable; this probe needs it.")
        return

    data = {str(json.loads(l)["id"]): json.loads(l) for l in open(args.data)}
    preds = [json.loads(l) for l in open(args.predictions)]

    from collections import Counter

    counts: Counter[str] = Counter()
    shown = 0
    for pred in preds:
        if args.limit and counts["scored"] >= args.limit:
            break
        d = data.get(str(pred["question_id"]))
        if not d:
            continue
        kb = build_kb(d.get("graph") or [])
        graph = KnowledgeGraph(kb)
        golds = {normalize_text(a) for a in pred["gold_answers"]} & set(kb["entities"])
        if not golds:
            continue
        # canonical type names only (skip head-noun aliases)
        inventory = sorted({c["name"] for cid, c in kb["concepts"].items() if cid.startswith("type:")})
        gold_types = set()
        for gid in golds:
            for cid in kb["entities"][gid].get("instanceOf", []):
                if cid.startswith("type:"):
                    gold_types.add(kb["concepts"][cid]["name"])
        if not inventory or not gold_types:
            counts["untyped_gold_or_empty_inventory"] += 1
            continue
        counts["scored"] += 1

        described = describe_answer_type(pred["question"], model=args.model)
        if described["error"] or not described["text"]:
            counts["llm_error"] += 1
            continue
        llm_ranked = rank_types(described["text"], inventory)
        q_ranked = rank_types(pred["question"], inventory)
        counts["llm_top1"] += llm_ranked[0] in gold_types
        counts["llm_top3"] += bool(set(llm_ranked[:3]) & gold_types)
        counts["q_top1"] += q_ranked[0] in gold_types
        counts["q_top3"] += bool(set(q_ranked[:3]) & gold_types)

        extracted = extract_answer_concept(graph, pred["question"], [pred["start_entity"]["name"]])
        if extracted and extracted.get("concept_ids"):
            counts["extractor_fired"] += 1
            names = {kb["concepts"].get(cid, {}).get("name", "") for cid in extracted["concept_ids"]}
            counts["extractor_correct"] += bool(names & gold_types)

        if shown < args.show:
            shown += 1
            mark = "HIT" if llm_ranked[0] in gold_types else ("top3" if set(llm_ranked[:3]) & gold_types else "MISS")
            print(f"[{mark}] q={pred['question'][:48]!r} | LLM said {described['text']!r} -> matched {llm_ranked[0]!r} | gold types {sorted(gold_types)[:3]}")

        if counts["scored"] % 50 == 0:
            print(f"  ... scored {counts['scored']}", flush=True)

    s = max(1, counts["scored"] - counts["llm_error"])
    print()
    print(json.dumps(dict(counts), indent=2))
    print(f"\nLLM-described type -> embed-match: top-1 {counts['llm_top1']}/{s} = {counts['llm_top1']/s:.0%} | top-3 {counts['llm_top3']}/{s} = {counts['llm_top3']/s:.0%}")
    print(f"question-embedding baseline:       top-1 {counts['q_top1']}/{s} = {counts['q_top1']/s:.0%} | top-3 {counts['q_top3']}/{s} = {counts['q_top3']/s:.0%}")
    print(f"old wh-rule extractor: fired {counts['extractor_fired']}/{s}, correct when fired {counts['extractor_correct']}")


if __name__ == "__main__":
    main()
