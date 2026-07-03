"""Grounding-embedder bake-off: menu-gold ceiling on the dev pool.

The fresh250 grounding misses traced to a vocabulary gap (MiniLM does not
place 'adjoins' near 'border'), not to depth or channel design — the intent
names were correct. The grounding embedder is a swappable component; this
measures each candidate by the only metric that matters at that stage: how
often the rebuilt menu contains a gold-reaching option (and at what menu
size). Run one process per model (embeddings are memoized globally):

  RC_MEX_EMBED_MODEL=BAAI/bge-small-en-v1.5 python3 -m rc_mex.diag_embedder_bakeoff
"""

from __future__ import annotations

import json
import os

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.run_webqsp_path_family import build_kb
from rc_mex.run_query_selection import build_candidate_paths

SLICES = [
    (os.path.expanduser("~/Documents/qsel_train150_qwen3/predictions.jsonl"), 0, 150),
    (os.path.expanduser("~/Documents/qsel_fresh250_deepseek1/predictions.jsonl"), 150, 400),
]


def main() -> None:
    model = os.environ.get("RC_MEX_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    total = menu_gold = 0
    option_count = 0
    best_f1_sum = 0.0
    for pred_path, lo, hi in SLICES:
        data = [json.loads(l) for l in open("data/webqsp/train.jsonl")][lo:hi]
        by_id = {str(d["id"]): d for d in data}
        for r in (json.loads(l) for l in open(pred_path)):
            d = by_id.get(str(r["question_id"]))
            if not d:
                continue
            kb = build_kb(d.get("graph") or [])
            graph = KnowledgeGraph(kb)
            golds = {normalize_text(a) for a in r["gold_answers"]} & set(kb["entities"])
            starts = {normalize_text(e) for e in (d.get("q_entity") or []) if str(e).strip()} & set(kb["entities"])
            if not starts or not golds:
                continue
            paths = build_candidate_paths(kb, graph, starts, r["question"], r.get("described_names") or [])
            total += 1
            option_count += len(paths)
            best = 0.0
            for p in paths:
                t = set(p["targets"])
                inter = len(t & golds)
                if inter:
                    prec, rec = inter / len(t), inter / len(golds)
                    best = max(best, 2 * prec * rec / (prec + rec))
            menu_gold += best > 0
            best_f1_sum += best
    print(f"{model}: menu-gold {menu_gold}/{total} = {menu_gold/total:.1%} | oracle F1 {best_f1_sum/total:.3f} | mean options {option_count/total:.1f}")


if __name__ == "__main__":
    main()
