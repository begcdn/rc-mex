# Inverse Path-to-Question Generator: Frozen Baseline

## Purpose

The component tests an inverse semantic mapping for KGQA:

```text
executable KG path -> natural-language question intent
```

The intended later use is verification: a candidate path is credible when the
question generated from that path has the same meaning as the user's question.
This report freezes the first working generator before path selection is added.

## Model and input

- Backbone: `google/flan-t5-small`.
- Retained checkpoint: `runs/inverse_verifier/joint_ranker_multi_kg_b4/model`.
- Path input contains ordered relation descriptions, traversal directions,
  source/target types, and the requested answer type.
- The topic entity is replaced by `[ENTITY]` during generation training.
- Answer entities and intermediate entity names are never supplied.
- Raw Freebase IDs are rendered as readable relation words; the generator is
  not trained to memorize unexplained relation identifiers.

The frozen checkpoint was trained jointly with:

```text
generation loss: path -> delexicalized question
ranking loss: correct question/path pair above a hard negative path
```

The generator-only evaluation ignores the ranking head completely.

This checkpoint is retained as the pre-type-aware baseline. The active follow-up
removes the ranking head and trains a pure generator with a separate auxiliary
question/type compatibility task. Its purpose is to test whether endpoint type
consistency can reject a path whose generated question resembles the original
while its answer role has changed.

## Training data and firewall

Training combines KQA Pro and official WebQSP training questions. Evaluation
questions are disjoint from training. KQA Pro additionally contains:

- 151 examples whose relation was absent from training;
- 25 two-hop relation-direction compositions absent from training;
- 223 ordinary validation examples.

WebQSP evaluation uses official held-out questions. Because WebQSP training was
seen, this is unseen-question transfer within a known Freebase schema, not a
leave-one-KG-out result. True unseen-KG transfer remains unproven.

## Gold-path generation experiment

For each held-out example:

1. Supply its annotated correct path only.
2. Generate one question.
3. Replace the topic entity in both generated and reference questions with
   `[ENTITY]`.
4. Measure semantic similarity with two frozen generic encoders: BGE-small and
   MiniLM.

There is no candidate search, negative path, path ranking, or answer execution
in this experiment.

| Split | N | BGE mean | BGE median | BGE >= 0.8 | BGE >= 0.9 |
|---|---:|---:|---:|---:|---:|
| Development | 158 | 0.951 | 0.962 | 1.000 | 0.861 |
| Unseen relation | 151 | 0.915 | 0.926 | 0.947 | 0.649 |
| Unseen two-hop composition | 25 | 0.933 | 0.938 | 1.000 | 0.800 |
| KQA Pro validation | 223 | 0.943 | 0.957 | 0.951 | 0.857 |
| Held-out WebQSP | 1,156 | 0.861 | 0.889 | 0.766 | 0.458 |
| Executable WebQSP | 986 | 0.862 | 0.887 | 0.773 | 0.449 |

On executable WebQSP, one-hop paths average 0.867 BGE similarity and two-hop
paths average 0.826. The corresponding rates above 0.8 are 78.6% and 68.5%.

## What works

- The model usually reconstructs the intended question from a correct path.
- It transfers strongly to held-out relation-direction compositions.
- Entity masking avoids relying on famous topic names as the semantic signal.
- Generated questions are readable and provide an inspectable explanation of
  what the model thinks a path means.

## Known weaknesses

1. **Hop omission.** Some two-hop paths are verbalized using only the first
   relation. For `person -> education -> degree`, the model may ask what the
   person attended rather than which degree they earned.
2. **Direction ambiguity.** Paraphrases can obscure which endpoint is the
   answer, especially for inverse relations.
3. **Schema-specific language.** Freebase relations are noisier and more
   compositional than KQA Pro labels, producing weaker generation.
4. **Semantic similarity is not logical equivalence.** A high embedding score
   can hide a missing modifier, wrong direction, or omitted hop.
5. **No unseen-KG proof yet.** The multi-KG checkpoint saw both KQA Pro and
   WebQSP training schemas.
6. **Restricted query language.** Current supervision covers simple one- to
   three-relation chains and excludes conjunction, aggregation, ordering,
   temporal clauses, and qualifiers.
7. **No calibrated acceptance threshold.** Values such as 0.8 or 0.85 are
   operational starting points, not learned probabilities.

## Validated synthetic corpus (July 2026)

The later corpus builder addresses the original generator's hop-omission failure before retraining:

1. Compile each executable path into explicit variables and complete logical facts.
2. Ground raw relation IDs with observed KG type pairs and example facts.
3. Use GPT-4o to write a natural question that expresses every fact and the return variable.
4. Randomize the intended query among executable hard-negative queries.
5. Use GPT-4o mini to select the exact represented query and verify endpoint type and fact coverage.
6. Deterministically reject internal notation, raw schemas, parentheses, vague relation wording,
   unusable relations, and rows without a validated hard negative.

The completed portion of `runs/inverse_verifier/naturalized_dataset_3000_v8` contains:

| Property | Value |
|---|---:|
| Selected source paths | 3,000 |
| Accepted rows | 1,979 |
| Train / dev | 1,781 / 198 |
| KQA Pro / WebQSP | 1,115 / 864 |
| One / two / three hop | 765 / 643 / 571 |
| Positive and negative questions | 6,531 |
| Unique positive relation sequences | 1,458 |
| Formatting-policy violations | 0 |

The API billing limit prevented verifier batch 5/5 from being submitted. A local Qwen 3 8B
verifier completed its 76 requests after calibration on the same 40-row pilot: Qwen and GPT-4o
each accepted 24 rows with 23 in common, and a compressed-prompt check reproduced all seven prior
candidate decisions. The weaker Llama 3.2 3B alternative was rejected after only 58% path-selection
agreement on its first 12 pilot items. The final manifest records this mixed verifier provenance.

The principal remaining data weakness is WebQSP's generic typing: 324 accepted rows have generic
`entity` answer types and 451 contain at least one generically typed path node. KQA Pro contributes
none of these cases. Such rows can produce semantically faithful but awkward questions because the
source graph does not identify a specific endpoint role. They remain marked in the path structures
and should be measured as a separate slice during training evaluation rather than silently removed.

### Exact data-generation procedure

The retained training corpus is
`runs/inverse_verifier/naturalized_dataset_3000_v8`. It was produced as follows.

1. **Select executable source paths.** Sample 3,000 one-, two-, and three-hop paths from the
   prepared KQA Pro and WebQSP pool. Preserve relation order, traversal direction, node types,
   endpoint answer type, KG source, and topic-entity masking.
2. **Compile each path into a query.** Convert the path into an explicit variable chain so the
   intended return variable and every hop are mechanically visible. Candidate questions are not
   generated directly from opaque relation identifiers.
3. **Ground relation semantics.** For 1,515 unique relations, collect observed source/target type
   pairs and KG facts. GPT-4o (`gpt-4o-2024-11-20`) uses this evidence to classify each relation as
   semantic, metadata, or opaque and to produce a readable glossary entry. All 1,515 entries were
   completed without API errors.
4. **Generate questions contrastively.** Construct the gold query and executable hard-negative
   queries that differ in relation, direction, composition, or return role. GPT-4o writes natural
   questions for these explicit queries. The prompt requires every represented fact, correct
   direction, and the correct answer variable while forbidding raw schema notation.
5. **Verify exact query identity.** Randomize the candidate-query order, then ask a separate
   verifier to identify exactly which query each generated question expresses. Verification also
   checks endpoint answer type, complete fact coverage, naturalness, and absence of unsupported
   facts. GPT-4o-mini (`gpt-4o-mini-2024-07-18`) handled the main verification batches.
6. **Complete the interrupted verifier batch.** The OpenAI billing limit prevented the last 76
   requests from being submitted. Qwen 3 8B completed only that final chunk after a 40-row
   calibration: GPT-4o and Qwen each accepted 24 rows, with 23 accepted by both. A compressed-prompt
   test reproduced all seven previously available decisions. Llama 3.2 3B was rejected after only
   58% agreement on its first 12 pilot items. This fallback is recorded in `manifest.json`.
7. **Apply deterministic quality gates.** Reject malformed questions, internal variables, raw KG
   syntax, parenthesized implementation text, vague relation wording, unusable/opaque relations,
   answer-role mismatches, omitted facts, unsupported facts, and rows lacking at least one validated
   hard negative. Rejection is explicit; no failed row silently enters training.
8. **Create the final split.** Write 1,781 accepted rows to `train_faithful.jsonl` and 198 to
   `dev_faithful.jsonl`. The other 1,021 selected paths remain rejected. All 3,000 selected paths are
   therefore accounted for.

The final corpus contains 10,998 contrastive-eligible generated candidate questions, all of which
received a verifier decision. Another 84 generated candidates were ineligible because their source
row did not contain at least two usable query alternatives; they were not treated as verified
training material. Every accepted row has at least one validated negative. The corpus has 6,215
unique question strings, 1,458 unique positive relation sequences, and no detected formatting-policy
violations.

### Training recommendation and interpretation

The first training run should use the `faithful_inverse` objective from a clean
`google/flan-t5-small` checkpoint. This objective combines path-to-question likelihood with a
contrastive sequence-likelihood loss that prefers the gold question/path pairing over validated hard
negatives. Starting from the older inverse-verifier checkpoint would confound the value of the new
corpus with inherited hop-omission and degenerate-generation behavior.

This dataset is suitable for the first complete training experiment, but it is not yet publication
evidence by itself. Evaluation must retain relation-disjoint, composition-disjoint, held-out-question,
and generic-WebQSP-type slices. The mixed verifier provenance and the 451 generically typed WebQSP
rows must be reported rather than hidden.

Recommended first training command:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m inverse_verifier train \
  --data runs/inverse_verifier/naturalized_dataset_3000_v8 \
  --base-model google/flan-t5-small \
  --regime faithful_synthetic \
  --objective faithful_inverse \
  --output runs/inverse_verifier/faithful_inverse_3000_v8 \
  --epochs 4 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --rank-weight 1.0 \
  --device cuda
```

The current trainer is intentionally single-device. On an L20, FLAN-T5-small and this 1,781-row
training split do not justify distributed-training complexity; exposing a second GPU would not make
the existing code use it. A second device should instead be reserved for a later controlled ablation
or larger-backbone experiment.

### Independent evaluation and direction repair

The first model trained on the validated corpus reached 1.000 development pair accuracy, but a
training-relative held-out evaluation exposed a specific failure that the random development split
hid. Gold-path question similarity remained high (0.932 on KQA Pro and 0.808 on executable WebQSP),
yet wrong-direction discrimination was only 0.126 on KQA Pro. In 147 of 206 KQA direction pairs,
the model generated exactly the same question for the forward and backward paths. The original
training corpus contained only 19 wrong-direction negatives versus 1,729 sibling-relation, 1,339
added-hop, and 1,001 missing-hop negatives.

`runs/inverse_verifier/naturalized_dataset_3000_direction_v3` repairs the supervision rather than
adding epochs to the same data:

- retain all 1,979 independently validated natural-question rows;
- add 3,211 contrast-only wrong-direction paths by flipping one asymmetric traversal at a time;
- never use these counterfactuals as natural-question generation targets;
- exclude symmetric predicates such as spouse, sibling, partner, and border sharing;
- make subject/object roles explicit in the compact path input;
- alternate direction contrasts with ordinary hard negatives across training epochs;
- preserve and flag 465 rows with generic type evidence rather than silently discarding them.

The repaired corpus has 1,781 train and 198 development rows, 4,552 naturalized negatives, no
train/development ID, question, or anchored-path overlap, and no structural direction-counterfactual
audit errors. A local 64-row training smoke test completed successfully. The next full run must use
the same frozen generalization evaluation; success requires a large increase in wrong-direction
accuracy without degrading missing-hop or wrong-relation discrimination.

## Reproduction

```bash
python3 -m inverse_verifier generate \
  --model runs/inverse_verifier/joint_ranker_multi_kg_b4/model \
  --output runs/inverse_verifier/gold_generation_full
```

Primary artifacts:

- `runs/inverse_verifier/gold_generation_full/metrics.json`
- `runs/inverse_verifier/gold_generation_full/report.md`
- `runs/inverse_verifier/gold_generation_full/gold_path_generations.jsonl`

## Frozen conclusion

The inverse generator is promising enough to serve as a semantic verifier, but
it is not yet a reliable exact path judge. Path selection should therefore be
evaluated separately for recall, and generator-based early stopping must report
false early accepts and missed gold paths rather than treating similarity as a
calibrated correctness probability.

## Type-aware follow-up

The follow-up uses the same path-to-question targets and adds training examples
of this form:

```text
Question: Which country was the author of [ENTITY] born in?
Candidate answer type: country
Compatible answer type: yes
```

```text
Question: Which country was the author of [ENTITY] born in?
Candidate answer type: person
Compatible answer type: no
```

The model still emits only a question when given a path. During candidate
verification, the auxiliary task separately checks the original question and
the generated question against the candidate endpoint type. Evaluation reports
generation similarity, positive type agreement, and wrong-type rejection; high
positive agreement without wrong-type rejection is treated as failure.

The first test does not add LLM-written synthetic questions. Gold benchmark
questions remain the path-to-question targets, while only the incompatible type
pairs are synthesized. This isolates the type-consistency hypothesis before
introducing possible teacher-model noise.
