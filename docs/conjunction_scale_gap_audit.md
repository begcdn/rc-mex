# Conjunction Scale-Gap Audit

## Question

Do the conjunction cases solved by Llama 3.1 8B but missed by Llama 3.2 3B support a branch-assembly experiment, or are they mostly unrelated errors?

The audit uses the 28 CWQ conjunction questions where the reordered-evidence arm has `Hit@1=0` for 3B and `Hit@1=1` for 8B. This is a deliberately selected diagnostic population, not an estimate of all CWQ errors.

## Result

Fourteen cases are clear failures to select the entity satisfying both branches. Three more mix branch selection with output ordering or noisy relation wording. Eleven are primarily explained by another issue: first-answer evaluation, aliases or MIDs, relation direction/type confusion, answer-set completeness, or an ambiguous question/gold pair.

This supports a small branch-assembly pilot, but not the claim that branch bookkeeping explains all of the scale gap.

The answer-blind surface-root prototype can construct at least two branches for 19/28 cases. Its structural meeting list is non-empty for 16/28 and contains a gold surface form in 14/28. Gold answers were used only to compute this post-hoc diagnostic; they are not inputs to the representation.

## Taxonomy

| Primary class | Count | Typical behavior |
|---|---:|---|
| Clear intersection/branch selection | 14 | Returns one branch, a union of both branches, or a nearby member instead of their intersection |
| Mixed branch and presentation/noisy semantics | 3 | The intersection issue is present, but first-answer ordering or malformed relation wording also matters |
| Evaluation/output ordering | 3 | Gold appears later, but `Hit@1` scores the first emitted answer |
| MID or alias realization | 3 | The selected entity is opaque or differently named rather than clearly reasoned incorrectly |
| Relation direction/type/answer-variable confusion | 5 | The model follows the wrong relation role, hierarchy level, or requested variable |

Counts assign one primary class per case. Several cases plausibly fit more than one class.

## Decision

Proceed with two matched, lossless arms on the same 400 questions:

1. `branch_grouped`: partition each triple once by proximity to entity names explicitly present in the question; boundary triples are shown as connecting evidence.
2. `junction_surfaced`: use the same partition and additionally list readable entities on branch boundaries.

The primary comparison is against the existing reordered 3B arm, with paired bootstrap intervals. Report overall, conjunction, and conjunction-with-two-roots slices. The 8B original score is a secondary compute reference, not the statistical baseline.

## Limits

- Released pilot rows do not retain upstream topic-entity IDs. Exact question/surface matching is therefore a proxy and can select generic nodes such as `Currency` or `Military Conflict`.
- The 28-case audit was selected on the 3B/8B outcome gap and cannot establish prevalence.
- Junction surfacing performs a deterministic structural operation. It may introduce distracting candidates even though it never adds or removes triples.

## Reproduction

```bash
ROOT=runs/subgraph_reader_pilot/cwq_branch_400
MODEL=/data3/ziad/models/Llama-3.2-3B-Instruct
VENV=/data3/ziad/venvs/inverse-verifier
GPU=1

"$VENV/bin/python" subgraph_reader_pilot.py prepare-branch \
  --original runs/subgraph_reader_pilot/cwq_operation_400/inputs/original.jsonl \
  --output "$ROOT/inputs" \
  --tokenizer "$MODEL"

mkdir -p "$ROOT/llama32_3b/runs"
cp runs/subgraph_reader_pilot/cwq_operation_400/llama32_3b/runs/reorder.jsonl \
  "$ROOT/llama32_3b/runs/reorder.jsonl"

CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" subgraph_reader_pilot.py run-branch-suite \
  --inputs "$ROOT/inputs" \
  --output "$ROOT/llama32_3b/runs" \
  --model "$MODEL" \
  --batch-size 8 \
  --tensor-parallel-size 1

"$VENV/bin/python" subgraph_reader_pilot.py evaluate-branch \
  --runs "$ROOT/llama32_3b/runs" \
  --metadata "$ROOT/inputs/branch_metadata.jsonl" \
  --output "$ROOT/llama32_3b/evaluation"
```
