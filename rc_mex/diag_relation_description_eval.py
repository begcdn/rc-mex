"""Realistic test of HyDE-style schema linking (the "semantic search on LLM
output" idea), on the cases where our question-embedding proposal buries the
answer relation.

Mechanism under test: instead of embedding the QUESTION and matching it to
relation names (which fails on colloquial<->formal mismatch), have the LLM
NAME the property that answers the question (without seeing the schema), embed
THAT, and rank the real relations by similarity to it. The ceiling (matching on
the target relation's own gloss) was 100% rescue; this measures the realistic
rescue with an actual LLM description.

Scope: depth-1 gold-not-generated cases where the needed relation ranked below
the top-K question-proposal cutoff. Gold is used only to locate the needed
relation and to score, never in any prompt.

Usage:
  python3 -m rc_mex.diag_relation_description_eval \
      --data data/webqsp/test.jsonl --predictions runs/<run>/predictions.jsonl \
      [--model qwen3:8b] [--limit 0] [--show 12]
"""

from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.micro_agents import DEFAULT_MODEL, describe_target_relation
from rc_mex.run_proof_state_search_smoke import (
    RELATION_PROPOSAL_K,
    get_semantic_relation_model,
    rank_relations_hybrid,
    semantic_embedding,
)
from rc_mex.run_webqsp_path_family import build_kb
from rc_mex.diag_selection_quality_eval import clean_relation


def bfs_first_relation(kb: dict, starts: set[str], golds: set[str], maxd: int = 1):
    ents = kb["entities"]
    seen = {s: (0, None) for s in starts}
    q = deque(starts)
    while q:
        e = q.popleft()
        d, r1 = seen[e]
        if d >= maxd:
            continue
        for rel in ents.get(e, {}).get("relations", []):
            nb = rel["object"]
            if nb in seen:
                continue
            seen[nb] = (d + 1, r1 if d >= 1 else rel["predicate"])
            if nb in golds:
                return seen[nb]
            q.append(nb)
    return (None, None)


def cosine(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    if get_semantic_relation_model() is None:
        print("WARNING: semantic model unavailable; this probe needs it.")
        return

    data = {str(json.loads(l)["id"]): json.loads(l) for l in open(args.data)}
    preds = [json.loads(l) for l in open(args.predictions)]
    if args.limit:
        preds = preds[: args.limit]

    tested = rescued = llm_err = shown = 0
    old_ranks: list[int] = []
    new_ranks: list[int] = []
    for pred in preds:
        if pred["path_family_any_hop_llm"].get("gold_generated"):
            continue
        d = data.get(str(pred["question_id"]))
        if not d:
            continue
        kb = build_kb(d.get("graph") or [])
        graph = KnowledgeGraph(kb)
        golds = {normalize_text(a) for a in pred["gold_answers"]} & set(kb["entities"])
        starts = {normalize_text(e) for e in d.get("q_entity") or []} & set(kb["entities"])
        if not golds or not starts:
            continue
        depth, needed = bfs_first_relation(kb, starts, golds)
        if depth != 1 or not needed:
            continue
        start_id = sorted(starts)[0]
        frontier = graph.candidate_relations([start_id], cap=100000, sample_entities=25)
        ranked = rank_relations_hybrid(pred["question"], frontier)
        keys = [c["relation_id"] for c in ranked]
        if needed not in keys:
            continue
        old_rank = keys.index(needed)
        if old_rank < RELATION_PROPOSAL_K:
            continue  # only the buried ones
        tested += 1
        old_ranks.append(old_rank)

        described = describe_target_relation(pred["question"], pred["start_entity"]["name"], model=args.model)
        if described["error"] or not described["text"]:
            llm_err += 1
            new_ranks.append(old_rank)
            continue
        target_vec = semantic_embedding(described["text"])
        sims = sorted(
            ((cosine(target_vec, semantic_embedding(clean_relation(rid))), rid) for rid in dict.fromkeys(keys)),
            reverse=True,
        )
        new_rank = next(i for i, (_, rid) in enumerate(sims) if rid == needed)
        new_ranks.append(new_rank)
        if new_rank < RELATION_PROPOSAL_K:
            rescued += 1
        if shown < args.show:
            shown += 1
            print(f"q={pred['question'][:50]!r} | LLM said {described['text']!r} | needed {clean_relation(needed)!r} | rank {old_rank}->{new_rank}")

    import statistics as st

    print()
    print(f"buried depth-1 cases tested: {tested}  (llm errors/empties: {llm_err})")
    if tested:
        print(f"  OLD (question embedding)  median rank {st.median(old_ranks):.0f}  | in top-{RELATION_PROPOSAL_K}: 0/{tested}")
        print(f"  NEW (LLM-described target) median rank {st.median(new_ranks):.0f}  | in top-{RELATION_PROPOSAL_K}: {rescued}/{tested} = {rescued/tested:.0%}")


if __name__ == "__main__":
    main()
