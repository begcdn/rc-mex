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
