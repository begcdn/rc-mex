# Codex Task: Make Generated Questions Read Naturally

Self-contained. Do not wait on other work; nothing else depends on this.

## The observation

The generator writes faithful but unnatural questions:

> *What is the entity where the mailing address of the headquarters of Samsung Group,
> an organization founder, is located?*

A person would ask *"where is Samsung based?"*. The original questions are short and
casual; the generated ones are long, typed, and clause-heavy. The comparator has to
match one against the other.

## Why this might matter, and why it might not

**For:** general paraphrase models score poorly on these pairs — QQP-large 0.51 EM,
STS-B 0.46, NLI 0.36-0.41 — and the phrasing gap is the obvious suspect.

**Against:** `BAAI/bge-reranker-v2-m3`, used off the shelf with no training, scores
**0.65 EM on exactly the same generated questions**, beating both the trained
comparator (0.52) and the no-verifier baseline (0.53), paired p = 0.043. If
verbosity were blocking comparison, that model would have failed too.

So this is a plausible secondary gain, not the main bottleneck. Size the effort
accordingly.

## Hard constraint

**The generator must never see the original question.** Not to check hop count, not
for anything. If it can see the target, it will drift toward reconstructing the
intended question regardless of what the path says, and the method stops being a
test of the path. This is the single invariant the whole hypothesis rests on.

Everything needed is already in the path: hop count, relation ids, directions,
endpoint types. Ziad's instinct to condition on hop count is right; take it from
`len(path["hops"])`, not from the question.

## What to try

1. **Compress by hop count, from the path.** One-hop paths should produce one clause.
   The current output nests appositions and relative clauses even for single hops.
2. **Drop the anchor apposition.** "Samsung Group, an organization founder" adds
   nothing — the entity is named. `retrieval.py::dominant_type` already returns
   "entity" for ambiguous cases, which suppresses some of these.
3. **Regenerate the training targets, don't just prompt differently.** The corpus
   at `runs/inverse_verifier/naturalized_dataset_3000_executable_direction_v1/`
   contains the verbose style; the model reproduces what it was trained on. Targets
   were built with GPT-4o via `inverse_verifier/openai_naturalize.py`.

## The risk that makes this non-trivial

Compression can swallow a hop. *"What is the district represented by the position
held by Anna Bligh?"* is clumsy but preserves both hops. *"What district does Anna
Bligh represent?"* is natural and drops the intermediate position node — which
means a one-hop path and a two-hop path could generate the same question, and the
comparator could no longer tell them apart.

**Guard against this with a round-trip check**, not by eye: given the generated
question and the candidate paths, can a model recover which path produced it? If
compression makes paths indistinguishable, the naturalness gain is bought with
exactly the discriminative signal the method needs.

## How to know if it worked

Do not re-run the pipeline; it takes ~70 minutes and is not needed. Rescore an
existing run offline:

```bash
python3 -m inverse_verifier rescore \
  --predictions runs/inverse_verifier/full_pipeline_clean_types_v1/predictions.jsonl \
  --model BAAI/bge-reranker-v2-m3 --kind cross_encoder
```

Baseline to beat, same run, same comparator: **gold path 0.51, answer EM 0.65,
F1 0.709.** A generator change only helps if it moves those.

Also re-check the paraphrase models (`cross-encoder/quora-roberta-large`, 0.51 EM).
If natural phrasing is the real story, those should improve the most — that is the
cleanest confirmation the hypothesis behind this task was right.

## Do not

- Let the generator see the original question, for any reason.
- Change the comparator, the selection policy, or retrieval in the same change.
- Report a gain without a paired win/loss test; n=100 has a ±0.10 interval and
  several apparent wins in this project have evaporated on more data.
