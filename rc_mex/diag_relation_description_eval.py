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


def gold_relations_depth1(kb: dict, starts: set[str], golds: set[str]) -> set[str]:
    """ALL predicates that reach a gold answer in one hop from any topic entity.

    Scoring against a single BFS-arbitrary gold path penalized the LLM for
    naming a better property than the annotation-degenerate one (e.g.
    'founder' vs a backward-nationality shotgun path); success is now any
    gold-reaching relation landing in the top-K."""
    out: set[str] = set()
    for sid in starts:
        for rel in kb["entities"].get(sid, {}).get("relations", []):
            if rel["object"] in golds:
                out.add(rel["predicate"])
    return out


def cosine(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0


def relation_similarity(name_vecs: list, relation_id: str) -> float:
    """Max over LLM-suggested names x relation gloss segments. Composite CVT
    relations ('official symbols / symbol') are matched per-segment so one
    informative segment is enough."""
    glosses = [clean_relation(relation_id)]
    glosses += [seg.strip() for seg in glosses[0].split("/") if seg.strip() and seg.strip() != glosses[0]]
    best = 0.0
    for nv in name_vecs:
        for gloss in glosses:
            best = max(best, cosine(nv, semantic_embedding(gloss)))
    return best


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
        gold_rels = gold_relations_depth1(kb, starts, golds)
        if not gold_rels:
            continue
        start_id = sorted(starts)[0]
        frontier = graph.candidate_relations(sorted(starts), cap=100000, sample_entities=25)
        ranked = rank_relations_hybrid(pred["question"], frontier)
        keys = list(dict.fromkeys(c["relation_id"] for c in ranked))
        present = [r for r in gold_rels if r in keys]
        if not present:
            continue
        old_rank = min(keys.index(r) for r in present)
        if old_rank < RELATION_PROPOSAL_K:
            continue  # only the buried ones (no gold relation in the old top-K)
        tested += 1
        old_ranks.append(old_rank)

        described = describe_target_relation(pred["question"], pred["start_entity"]["name"], model=args.model)
        if described["error"] or not described["names"]:
            llm_err += 1
            new_ranks.append(old_rank)
            continue
        name_vecs = [semantic_embedding(n) for n in described["names"]]
        sims = sorted(((relation_similarity(name_vecs, rid), rid) for rid in keys), reverse=True)
        order = [rid for _, rid in sims]
        new_rank = min(order.index(r) for r in present)
        new_ranks.append(new_rank)
        if new_rank < RELATION_PROPOSAL_K:
            rescued += 1
        if shown < args.show:
            shown += 1
            best_rel = min(present, key=lambda r: order.index(r))
            print(f"q={pred['question'][:50]!r} | LLM said {described['names']!r} | best gold rel {clean_relation(best_rel)!r} | rank {old_rank}->{new_rank}")

    import statistics as st

    print()
    print(f"buried depth-1 cases tested: {tested}  (llm errors/empties: {llm_err})")
    if tested:
        print(f"  OLD (question embedding)  median rank {st.median(old_ranks):.0f}  | in top-{RELATION_PROPOSAL_K}: 0/{tested}")
        print(f"  NEW (LLM-described target) median rank {st.median(new_ranks):.0f}  | in top-{RELATION_PROPOSAL_K}: {rescued}/{tested} = {rescued/tested:.0%}")


if __name__ == "__main__":
    main()
