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

## Pilot Result

The 400-question Llama 3.2 3B run gives a positive but inconclusive signal for branch grouping and no support for explicitly listing junction candidates.

| Slice | Arm | Hit@1 | F1 |
|---|---|---:|---:|
| Overall (400) | Reordered | 0.750 | 0.660 |
| Overall (400) | Branch grouped | 0.765 | 0.677 |
| Overall (400) | Junction surfaced | 0.750 | 0.674 |
| Conjunction (200) | Reordered | 0.655 | 0.599 |
| Conjunction (200) | Branch grouped | 0.675 | 0.634 |
| Conjunction (200) | Junction surfaced | 0.645 | 0.623 |
| Branchable conjunction (138) | Reordered | 0.638 | 0.587 |
| Branchable conjunction (138) | Branch grouped | 0.667 | 0.640 |
| Branchable conjunction (138) | Junction surfaced | 0.623 | 0.620 |

For branchable conjunctions, branch grouping changes F1 by `+0.0535`, with paired bootstrap 95% CI `[-0.0101, +0.1179]`. The direction is encouraging, but the interval crosses zero. It does not meet the proposed scale gate: overall 3B branch-grouped F1 is `0.677`, below the matched 8B original-evidence F1 of `0.699`.

The targeted diagnostic behaves as predicted: among the 28 previously identified conjunction cases missed by reordered 3B but solved by reordered 8B, branch grouping fixes 11 at Hit@1 and junction surfacing fixes 13. Across all conjunction questions, however, branch grouping produces 15 positive and 11 negative Hit@1 flips. The mechanism fixes real failures but also destabilizes already-correct answers.

Junction surfacing should not advance in its current form. On the 108 conjunction rows where a junction header is actually added, it changes Hit@1 by `-0.0463` and F1 by `-0.0328` relative to branch grouping; both intervals cross zero, but there is no positive aggregate signal and the no-answer rate rises.

As a reproducibility control, 208 non-branchable rows have byte-identical prompts in all three arms. Three nevertheless change Hit@1 across repeated vLLM generation, establishing a small inference nondeterminism floor. The branch-grouping trend is larger than this floor but remains unconfirmed.

**Decision:** retain reordering as the established result; retain branch grouping as a candidate requiring a better-powered conjunction test or cleaner upstream topic IDs; reject junction surfacing as currently formulated. Do not claim that 3B has matched 8B, and do not launch a full-population campaign from this pilot alone.

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
