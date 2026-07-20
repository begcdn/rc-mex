# Multi-Hop KGQA Research

This repository is a clean research workspace for a new method for reliable
multi-hop question answering over knowledge graphs.

The active experiment separates two jobs:

1. A recall-oriented retriever proposes many executable relation paths and
   returns their union subgraph.
2. A compact inverse generator verbalizes what each path means. Its generated
   question is compared with the user's question to verify the path.
3. The same generator is fine-tuned on an auxiliary question/type compatibility
   task. This check is run separately; the path-to-question output remains an
   ordinary question and there is no path-ranking head in the active method.

The retriever uses the released SRTK iterative relation-path implementation and
its multi-KG scorer. This is a PullNet-style retrieve-then-reason boundary, not
an assertion that we are running PullNet itself: PullNet's source and checkpoint
were not publicly released. The inverse verifier is the component under study.

## Current Contents

- `inverse_verifier/` contains the one active data, training, and evaluation pipeline.
- `inverse_verifier/retrieval.py` adapts SRTK to local question graphs and emits
  candidate paths plus a retrieved subgraph.
- `docs/inverse_verifier_generation_report.md` freezes the generator training,
  results, and known weaknesses.
- `docs/inverse_verifier_hypothesis.md` states the mechanism and falsifying experiment.
- `AGENTS.md` and `docs/research_protocol.md` define the research discipline.
- `data/` contains local datasets and is ignored by Git.
- `runs/` contains local experimental outputs and is ignored by Git.

## Experiment

The controlled data contains one- to three-relation KQA Pro and WebQSP chains.
Intermediate and answer entity names are hidden. KQA relations and relation
compositions are withheld for dedicated transfer splits, and official WebQSP
test questions remain disjoint from the multi-KG training data.

```bash
python3 -m inverse_verifier prepare

python3 -m inverse_verifier train \
  --base-model runs/inverse_verifier/joint_ranker_multi_kg_b4/model \
  --regime multi_kg \
  --objective type_aware_generator \
  --output runs/inverse_verifier/type_aware_generator_multi_kg

python3 -m inverse_verifier generate \
  --model runs/inverse_verifier/type_aware_generator_multi_kg/model \
  --output runs/inverse_verifier/type_aware_generation_eval
```

For a quick local check, add `--limit 128` to training and
`--limit-per-split 32` to evaluation. Full outputs are limited to
`metrics.json`, `predictions.jsonl`, and `report.md`.

## Retrieval And Verification

Install the maintained retrieval dependency:

```bash
pip install -e '.[retrieval]'
```

Measure candidate-path and evidence recall without running the verifier:

```bash
python3 -m inverse_verifier retrieve \
  --limit 100 \
  --output runs/inverse_verifier/path_retrieval_100
```

Run retrieval followed by inverse verification. Candidates are checked in
batches of five; verification stops when the best semantic similarity reaches
0.85, or after 100 candidates, and otherwise selects the best seen candidate.
Scores below 0.5 are logged but deliberately have no fallback yet.

```bash
python3 -m inverse_verifier verify \
  --model runs/inverse_verifier/type_aware_generator_multi_kg/model \
  --limit 100 \
  --output runs/inverse_verifier/type_aware_path_verifier_100
```

During fine-tuning, every path/question example supplies the ordinary
path-to-question target. Examples with informative endpoint types also supply a
second task:

```text
question + candidate endpoint type -> yes/no type compatibility
```

Gold endpoint types are positive examples. Synthetic wrong endpoint types are
negative examples. At verification time, a path may pass only when both the
original question and the generated question are compatible with its endpoint
type. These compatibility decisions are separate from semantic question
similarity and are recorded independently.

The current controlled WebQSP run uses the supplied topic entities and local
question neighborhoods. It therefore tests path retrieval and verification,
not standalone entity linking or full-Freebase serving.

## Research Boundary

Before implementation, record:

1. The general failure in existing KGQA methods.
2. The computational reason for that failure.
3. The proposed mechanism and why it should address the cause.
4. The strongest matched baselines.
5. The smallest experiment that could reject the hypothesis.
6. The intermediate and final-answer measurements that determine the result.

Previous experimental methods remain recoverable from Git history, but they are
not architectural dependencies or fallbacks for this experiment.
