# Inverse Path Verifier: Opus Handoff

## What This Project Is Now

Ignore the old RC-MEX relation-card work. It was abandoned and is not the current
research direction.

The current hypothesis is:

> A high-recall KG path proposer can be made more precise by reconstructing the
> natural-language question implied by each executable path, then selecting the
> reconstruction that matches the user's complete intended question.

The inverse generator is not the final answer model. It maps:

```text
executable KG path -> question/intent represented by that path
```

A separate semantic comparator maps:

```text
original question + generated question -> scalar equivalence score
```

The selected path is executed to obtain the answer entities.

## Current Controlled Architecture

### 1. Inputs

- Official WebQSP question.
- A pre-extracted local Freebase neighborhood from `data/webqsp/test.jsonl`.
- The provided/gold topic entity is used to avoid entity-linking confounds.
- Gold relation paths and answers are used only for evaluation.

### 2. Candidate proposer

Implementation: `inverse_verifier/retrieval.py`

Class: `SRTKPathRetriever`

- Uses the released `drt/srtk-scorer`.
- Expands executable paths from the topic entity.
- Every stored KG edge is traversable forward or backward.
- Backward traversal is not inherently wrong: Freebase may store a relation in
  the opposite orientation from the question's starting entity.
- Maximum path depth: 2.
- SRTK beam/path cap: 200.
- Returns relation-path families, endpoint answer pools, supporting triples, and
  a retrieved subgraph.

The paths are not random. SRTK question-conditionally scores them, but the pool
is intentionally wide and therefore noisy.

### 3. Inverse path-to-question generator

Main implementation: `inverse_verifier/model.py`

Serialization/grounding: `inverse_verifier/data.py`

Current server checkpoint:

```text
runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model
```

This is a fine-tuned Qwen2.5 3B model with LoRA. It receives a grounded path
serialization containing:

- anchor entity and type;
- every relation;
- direction and subject/object roles;
- intermediate and endpoint types;
- relation glossary descriptions.

It emits only the question represented by that path.

Desired behavior for a wrong reversed path is to generate the corresponding
wrong/reversed question faithfully. The generator must not silently repair the
path into the user's intended question.

### 4. Semantic question comparator

Implementation: `inverse_verifier/comparator.py`

Current server checkpoint:

```text
runs/inverse_verifier/deberta_comparator_question_generated_v1/model
```

- DeBERTa-v3-base scalar cross-encoder.
- Current input mode: `question_generated`.
- Input contains only the original question and one generated question.
- It does not see the path in this mode.
- It independently scores every candidate pair.
- It can rank any number of candidates; it is not a two-question classifier.
- Training uses listwise multi-positive softmax over complete candidate sets.
- At inference, all verified candidate scores are sorted globally.

Other ablations remain available:

- `question_path`
- `question_generated_path`
- cosine similarity using BGE-small

### 5. Full pipeline

Implementation: `inverse_verifier/selector.py`

Function: `run_verifier_pipeline`

- SRTK proposes up to 200 paths.
- The first 100 are verbalized by the generator in batches of five.
- Cross-encoder mode has no early similarity threshold.
- The comparator scores all generated questions.
- The highest-scoring path is executed; its endpoint pool is the prediction.
- Current `predictions.jsonl` stores the selected candidate and top 10 scored
  candidates, but not all 100. This limits complete generator auditing.

## Evaluation Firewall

Gold is allowed only for:

- selecting supported examples;
- checking whether the annotated path exists in the supplied graph;
- proposal/path recall diagnostics;
- final answer metrics.

Gold paths, gold hop counts, and gold answers are not passed to SRTK, the
generator, or comparator.

The current setup still uses gold topic entities and pre-extracted question
neighborhoods, so it is controlled KGQA rather than fully open end-to-end QA.

## Semantic Comparator Benchmark

The original held-out path benchmark was unsuitable for measuring the comparator:
a nominally wrong path can generate a question semantically equivalent to the
original, while a nominally gold path can be verbalized incorrectly.

We therefore created a text-only semantic benchmark:

```text
/Users/ziad/Documents/inverse_verifier/comparator_kqa_semantic_adjudicated_v1
```

The judge saw only:

- original question;
- candidate generated question.

It did not see paths, path labels, endpoints, or negative categories. GPT-4o
first labeled exact answer-set equivalence. A second difference-first pass
adjudicated:

- all 92 path-label/semantic-label disagreements;
- all 3 ambiguous cases;
- 30 randomly sampled agreements.

Disputed candidate sets were excluded rather than forced to a label.

Final clean KQA slice:

- 179 scorable candidate sets.
- 15 disputed sets excluded.
- 29 source sets had no equivalent generated question.
- Agreement-control disagreement: 1/30, so this is a useful silver benchmark,
  not human gold.

Question-only comparator result:

| Method | Recall@1 | MRR | Pair accuracy |
|---|---:|---:|---:|
| DeBERTa cross-encoder | 0.838 | 0.916 | 0.938 |
| BGE cosine | 0.704 | 0.845 | 0.883 |

McNemar comparison:

- both correct: 116;
- cross-encoder only: 34;
- cosine only: 10;
- both wrong: 19;
- exact p-value approximately 0.00039.

The cross-encoder is meaningfully better than cosine, but not yet reliable.
On this controlled benchmark, 23 of its 29 top-1 failures selected a
wrong-direction generated question.

The strict unseen-relation and unseen-composition slices contain only 7 and 17
clean examples. They are diagnostic only and cannot support a generalization
claim.

## Latest Full WebQSP Run

Local results:

```text
/Users/ziad/Documents/inverse_verifier/full_pipeline_question_comparator_v1
```

Server results:

```text
runs/inverse_verifier/full_pipeline_question_comparator_v1
```

Run size: 100 WebQSP questions.

| Metric | Result |
|---|---:|
| Gold path in SRTK candidates | 0.900 |
| Gold path recall@5 | 0.700 |
| Recall@10 | 0.760 |
| Recall@20 | 0.840 |
| Recall@50 | 0.870 |
| Recall@100 | 0.900 |
| Gold path availability in supplied graph | 0.920 |
| Recall given available path | 0.978 |
| Annotated gold path selected | 0.410 |
| Final answer exact match | 0.510 |
| Final answer F1 | 0.564 |
| Final answer contains a gold answer | 0.620 |
| Average verified paths | 98.52 |
| Average SRTK paths | 191.77 |
| Runtime | 4668 seconds |

The earlier generator + cosine pipeline had:

- selected gold path: 0.370;
- answer exact match: 0.440;
- answer F1: 0.454.

Thus the current generator/comparator improves final answering, but remains far
from strong KGQA performance.

## Full-Run Failure Decomposition

Of 100 questions:

- 10: annotated gold path was not proposed.
- 41: annotated gold path was selected.
- 49: annotated gold path was verified but another candidate won.
- 10 non-annotated selected paths still produced the exact gold answer.
- Only 1 selected error used the exact gold relation sequence with reversed
  direction.

Therefore, reversed-direction selection is prominent in the synthetic semantic
benchmark but is not the dominant full-pipeline failure.

Dominant observed behavior:

1. Nearby relation ambiguity:
   - "Where is JaMarcus Russell from?" chooses nationality over birthplace.
   - "What language does Egyptian people speak?" chooses languages-spoken over
     official-language.

2. Equivalent or duplicate Freebase routes:
   - county, capital, artwork, and cast relations often have multiple predicates
     that generate the same question and may return the same answer.
   - Exact annotated-path accuracy therefore underestimates some valid behavior.

3. Valid shortcut paths:
   - A direct participation relation can answer a question whose annotation uses
     a more indirect two-hop path.

4. Generator faithfulness failures:
   - Some annotated paths are rendered awkwardly or with the wrong role/type,
     causing a cleaner alternative to win.
   - Some incorrect paths are compressed into plausible near-intent questions.

5. Comparator failures:
   - It sometimes prefers nationality to birthplace, a broad relation to a
     specific relation, or a malformed reversed question.

## Important Conceptual Guardrails

- Do not ban backward traversal. Correct paths frequently require it.
- Do not assume the annotated relation sequence is the only semantically valid
  path.
- Do not treat every non-gold path as a negative when it produces the same answer
  for the same semantics.
- Do not mix generator faithfulness, semantic comparison, path correctness, and
  answer correctness into one metric.
- Do not change proposer, generator, and comparator simultaneously.
- Do not use gold paths to guide search or ranking.
- Do not add a small LLM cleanup call before measuring which component failed.
- Keep this a research experiment, not a growing configuration framework.

## Recommended Next Work

### First: improve observability without changing behavior

Store every verified candidate, not only the top 10, or stream a compact
candidate log containing:

- relation sequence;
- endpoint answers;
- generator output;
- retrieval score;
- comparator score;
- whether it matches an annotated path;
- whether its answer overlaps gold, evaluation-only.

This is required to attribute all 49 ranking failures reliably.

### Second: proposer-only comparison

Freeze the current generator and comparator. Compare:

1. SRTK top 200;
2. SRTK top 20;
3. official RoG relation-path planner beam;
4. optionally GNN-RAG later.

Measure:

- gold-path Recall@K;
- answer coverage;
- average candidates;
- annotated-gold rank;
- candidate direction/semantic noise;
- final answer metrics under the same frozen verifier.

RoG is the simplest direct alternative because it explicitly generates relation
paths. GNN-RAG proposes answer entities over a dense subgraph and extracts
shortest paths, so integration is larger.

Do not replace SRTK merely because its pool contains wrong paths: its recall is
already 90%, and 97.8% conditional on graph availability. A replacement must
preserve that recall while reducing candidate noise.

### Third: component-level audits

For every gold-present but wrong-selection case, independently determine:

- Did the gold path generate a question equivalent to the original?
- Did the selected path generate a question equivalent to the original?
- Do selected and gold paths return the same answer set?
- Is the original question genuinely ambiguous?
- Did the comparator rank two equivalent questions differently?

Only then decide whether the next intervention belongs in the proposer,
generator, or comparator.

## Code and Environment

Detailed operational history and server troubleshooting are recorded in:

```text
docs/claude_server_runbook.md
```

Local repository:

```text
/Users/ziad/Desktop/rcmex
```

GitHub:

```text
git@github.com:begcdn/rc-mex.git
```

Server repository:

```text
/data3/ziad/rcmex/rc-mex
```

Server virtual environment:

```text
/data3/ziad/venvs/inverse-verifier
```

Server environment:

```bash
export PYTHONNOUSERSITE=1
export HF_HOME=/data3/ziad/hf_cache
export HF_HUB_OFFLINE=1
```

WebQSP files on server:

```text
data/pattern_alignment/webqsp/WebQSP.test.json
data/webqsp/test.jsonl
```

Latest pushed commit at handoff:

```text
842dfba Add semantic benchmark adjudication
```

Do not stage the unrelated local file:

```text
inverse_direction_pairs_v1.tgz
```

Tests:

```bash
python3 -m pytest tests/test_inverse_verifier.py -q
```

Current result: 64 passed.

## Full Pipeline Command

```bash
cd /data3/ziad/rcmex/rc-mex

VENV=/data3/ziad/venvs/inverse-verifier
export PYTHONNOUSERSITE=1
export HF_HOME=/data3/ziad/hf_cache
export HF_HUB_OFFLINE=1

SRTK_MODEL="$(find "$HF_HOME/hub/models--drt--srtk-scorer/snapshots" \
  -mindepth 1 -maxdepth 1 -type d | head -n 1)"

CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m inverse_verifier verify \
  --questions data/pattern_alignment/webqsp/WebQSP.test.json \
  --graphs data/webqsp/test.jsonl \
  --model runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model \
  --retriever-model "$SRTK_MODEL" \
  --comparison-mode cross_encoder \
  --comparator-model runs/inverse_verifier/deberta_comparator_question_generated_v1/model \
  --output runs/inverse_verifier/full_pipeline_question_comparator_v1 \
  --limit 100 \
  --device cuda
```

## What Opus Should Not Assume

- Universality has not been demonstrated.
- Full entity linking has not been evaluated.
- Longer paths, constraints, aggregation, and complex query structures are not
  covered by this two-hop controlled run.
- The current semantic benchmark is silver, not human gold.
- Top annotated-path accuracy is not identical to semantic correctness or answer
  correctness.
- High-recall proposal is not the research contribution by itself.
- The inverse-verification hypothesis remains promising but unproven as a
  competitive KGQA method.
