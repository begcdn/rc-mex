# Selection Experiment Plan

## Research Question

Does denotation-only path supervision teach a selector to prefer semantically
wrong paths that happen to reach the right answer, and does inverse
reconstruction expose those paths better than ordinary model disagreement?

This is narrower than claiming a new KGQA architecture. It tests one proposed
failure mechanism before another architecture change.

## Experiment 1: Representation Disagreement

Use the same 100 WebQSP questions and the same frozen candidate pool. Rescore
every candidate with:

1. the fine-tuned generated-question comparator;
2. an off-the-shelf generated-question comparator;
3. the fine-tuned serialized-path comparator.

Compare the generated-question/path pair with the generated-question/generated-
question control. All comparisons use within-question ranks; logits from
different models are never treated as calibrated probabilities.

The old 0.79 oracle is not a new result. It is the same 64/7/8/21 paired outcome
already observed. Reconstruction is interesting only if cross-view
complementarity exceeds ordinary same-view model disagreement.

Directional prediction:

> A spurious path should rank better when read as a serialized path than when
> read through its generated question.

The raw paths must be labeled independently before testing this prediction.
Labels assigned only to generated descriptions cannot establish path
spuriousness.

## Experiment 2: Fixed-Pool Supervision

Build one candidate universe for every training question that contains both an
annotated candidate and an exact-denotation candidate. Copy the identical
candidates and train/dev split into three arms:

1. `annotated`: only annotated relation sequences are positive;
2. `denotation`: every exact-answer candidate is positive;
3. `annotated_or_denotation`: either signal is positive.

Only the labels differ. Candidate order, hard negatives, random negatives,
split, seed, base model, input representation, optimizer, and test pool remain
fixed. The manifest stores candidate-pool fingerprints.

Use `question_generated_path` for all three arms. This experiment is about the
supervision signal, so it should not simultaneously choose between path and
generated-question representations.

Primary endpoint: answer exact match on the frozen 100-question test pool.
Also report answer F1, selected annotated-path accuracy, selected
gold-equivalent accuracy, and paired wins/losses.

## Raw Path Audit

Export every non-annotated candidate that returns exactly the gold answer.
Label two independent properties:

- `path_label`: whether the raw path expresses a reliable answer rule;
- `generator_faithfulness`: whether the generated question describes that path.

Path labels:

- `direct_intent`
- `reliable_alternative`
- `correlated_shortcut`
- `entity_specific_coincidence`
- `unrelated`
- `unclear`

This avoids conflating a bad path with a generator error. Counts are
path-instance weighted, not extrapolated from deduplicated texts.

## Decision Scenarios

### A. Cross-view disagreement is ordinary ensemble noise

If its oracle gain is no larger than the matched same-view control, and path
rank advantage does not predict audited spuriousness, reject disagreement as a
mechanism. Do not build a fusion module around it.

### B. Cross-view disagreement detects spurious paths

If path-over-generated rank advantage is enriched for independently labeled
shortcuts or coincidences, retain inverse reconstruction as a diagnostic signal.
The next experiment should use it as a rejection or abstention signal, not claim
that naive score fusion solves KGQA.

### C. Denotation supervision underperforms annotated supervision

This supports the claim that answer-equivalent positives introduce harmful
semantic noise. The next controlled arm should be intent-filtered denotation
supervision, compared with RAPL-like direct semantic labels and PathISE-like
latent/MIL treatment.

### D. Denotation supervision matches or beats annotated supervision

Spurious paths may exist without causing the current selector errors, or their
extra training signal may outweigh the noise. Do not frame spurious supervision
as the bottleneck without a stronger causal intervention.

### E. Hybrid wins

The two signals are complementary, but that alone does not validate inverse
reconstruction. Inspect which candidate categories changed and repeat with
multiple seeds before retaining the result.

## Statistical Gate

The 100-question result is a smoke test. The confirmatory in-KG comparison uses
all WebQSP questions supported by the current simple-chain pipeline and reports
coverage over the complete test set. Compare arms with paired bootstrap
confidence intervals for EM/F1 and an exact McNemar test with Holm correction.
Do not infer equality from a non-significant result.

The generalization test is separate. Freeze each WebQSP-trained comparator and
evaluate it without target tuning on the fixed KQA Pro candidate sets. KQA Pro
uses a different KG and relation schema. This controlled test measures semantic
path-ranking transfer, not end-to-end KQA retrieval.

Interpret the two tests independently:

- WebQSP full test: does the supervision rule change in-distribution selection?
- WebQSP to KQA Pro: does the supervision rule learn semantics that transfer to
  another KG?
- Failure of every arm on KQA Pro indicates representation or generator
  transfer failure; it does not isolate the supervision rule.
- A target-KG advantage needs multiple training seeds before it becomes a
  research result.

## Confirmatory Execution

Generate the full supported WebQSP test candidate pool once. A limit above the
supported population means the run processes every eligible question while the
run metrics retain total-test coverage:

```bash
CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" -m inverse_verifier verify \
  --questions data/webqsp_official/WebQSP/data/WebQSP.test.json \
  --graphs data/webqsp/test.jsonl \
  --model runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model \
  --retriever-model "$SRTK_MODEL" \
  --output runs/inverse_verifier/selection_full_webqsp_candidates_v1 \
  --limit 2000 \
  --comparison-mode cross_encoder \
  --comparator-model runs/inverse_verifier/deberta_comparator_question_generated_v1/model \
  --device cuda
```

Rescore that frozen pool with each supervision arm, then run the paired
comparison:

```bash
for ARM in annotated denotation annotated_or_denotation; do
  CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" -m inverse_verifier rescore \
    --predictions runs/inverse_verifier/selection_full_webqsp_candidates_v1/predictions.jsonl \
    --comparator "runs/inverse_verifier/selection_supervision_${ARM}_seed17/model" \
    --graphs data/webqsp/test.jsonl \
    --output "runs/inverse_verifier/selection_full_webqsp_${ARM}_seed17" \
    --batch-size 4 --device cuda
done

"$VENV/bin/python" -m inverse_verifier compare-selection-runs \
  --runs \
    runs/inverse_verifier/selection_full_webqsp_annotated_seed17 \
    runs/inverse_verifier/selection_full_webqsp_denotation_seed17 \
    runs/inverse_verifier/selection_full_webqsp_annotated_or_denotation_seed17 \
  --output runs/inverse_verifier/selection_full_webqsp_comparison_seed17
```

For frozen cross-KG transfer, first materialize the same KQA Pro candidate sets
with the existing inverse generator. The generator is shared and has seen both
schemas, so this isolates transfer of the WebQSP-trained comparator rather than
claiming zero-shot transfer of the complete architecture:

```bash
CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" -m inverse_verifier evaluate \
  --data runs/inverse_verifier/data \
  --output runs/inverse_verifier/selection_kqa_candidates_v1 \
  --trained-model runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model \
  --splits test_kqa_val \
  --batch-size 16 --generation-examples 0 \
  --skip-pretrained --device cuda

"$VENV/bin/python" -m inverse_verifier prepare-comparator \
  --data runs/inverse_verifier/selection_kqa_candidates_v1/predictions.jsonl \
  --generator runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model \
  --output runs/inverse_verifier/selection_kqa_fixed_candidates_v1

for ARM in annotated denotation annotated_or_denotation; do
  CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" -m inverse_verifier evaluate-comparator \
    --data runs/inverse_verifier/selection_kqa_fixed_candidates_v1 \
    --model "runs/inverse_verifier/selection_supervision_${ARM}_seed17/model" \
    --output "runs/inverse_verifier/selection_kqa_${ARM}_seed17" \
    --split test_kqa_val --batch-size 4 --device cuda
done

"$VENV/bin/python" -m inverse_verifier compare-selection-runs \
  --runs \
    runs/inverse_verifier/selection_kqa_annotated_seed17 \
    runs/inverse_verifier/selection_kqa_denotation_seed17 \
    runs/inverse_verifier/selection_kqa_annotated_or_denotation_seed17 \
  --output runs/inverse_verifier/selection_kqa_comparison_seed17
```
