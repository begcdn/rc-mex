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

The current test has 100 questions. Report paired outcomes and uncertainty; do
not interpret a small marginal difference as established. If a supervision arm
looks promising, repeat seeds and evaluate a larger frozen question sample
before changing the architecture.
