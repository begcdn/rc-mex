# Multi-Hop KGQA Research

This repository is a clean research workspace for a new method for reliable
multi-hop question answering over knowledge graphs.

The active experiment separates two jobs:

1. A recall-oriented retriever proposes many executable relation paths and
   returns their union subgraph.
2. A compact inverse generator verbalizes what each path means. Its generated
   question is compared with the user's question to verify the path.
3. Candidate paths are ranked by semantic similarity between the reconstructed
   question and the user's question. The endpoint entities of the selected
   executable path become the predicted answers.

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
- `docs/faithful_inverse_training.md` defines the balanced executable corpus and
  the path-faithfulness objective introduced after the full-pipeline failure analysis.
- `AGENTS.md` and `docs/research_protocol.md` define the research discipline.
- `data/` contains local datasets and is ignored by Git.
- `runs/` contains local experimental outputs and is ignored by Git.

## Experiment

The controlled data contains one- to three-relation KQA Pro and WebQSP chains.
Intermediate and answer entity names are hidden. KQA relations and relation
compositions are withheld for dedicated transfer splits, and official WebQSP
test questions remain disjoint from the multi-KG training data.

The current follow-up generates a balanced executable path corpus, optionally
naturalizes its controlled questions with a local model, and optimizes both
complete intent generation and gold-vs-hard-negative sequence likelihood.

```bash
python3 -m inverse_verifier synthesize \
  --paths 30000 \
  --output runs/inverse_verifier/faithful_data

python3 -m inverse_verifier naturalize \
  --data runs/inverse_verifier/faithful_data \
  --output runs/inverse_verifier/faithful_data_natural \
  --model qwen3:8b

python3 -m inverse_verifier train \
  --data runs/inverse_verifier/faithful_data_natural \
  --base-model runs/inverse_verifier/joint_ranker_multi_kg_b4/model \
  --regime faithful_synthetic \
  --objective faithful_inverse \
  --output runs/inverse_verifier/faithful_inverse

python3 -m inverse_verifier faithfulness \
  --data runs/inverse_verifier/faithful_data_natural/dev_faithful.jsonl \
  --model runs/inverse_verifier/faithful_inverse/model \
  --output runs/inverse_verifier/faithful_inverse_eval
```

Naturalization can use one independent Ollama server per GPU. Repeat
`--ollama-host` for every endpoint; batches run concurrently while output rows
remain in source order and the run remains resumable.

```bash
python3 -m inverse_verifier naturalize \
  --data runs/inverse_verifier/faithful_data \
  --output runs/inverse_verifier/faithful_data_natural \
  --model qwen3:8b \
  --ollama-host http://127.0.0.1:11434 \
  --ollama-host http://127.0.0.1:11435
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
  --model runs/inverse_verifier/joint_ranker_multi_kg_b4/model \
  --limit 100 \
  --output runs/inverse_verifier/full_pipeline_old_generator_100
```

This run intentionally uses the frozen pre-type-aware generator. Its 100-question
controlled run reached 90% candidate-path recall but selected the gold path only
37% of the time. The faithful inverse experiment directly targets the observed
hop omission and direction failures before full retrieval is rerun.

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

# OpenAI Batch naturalization

Naturalize a bounded, balanced faithful corpus without exposing canonical questions or entity
names to the model. Start with a dry run; it creates request files and prints a conservative cost
estimate without submitting them:

```bash
python3 -m inverse_verifier naturalize-openai \
  --data runs/inverse_verifier/faithful_data \
  --output runs/inverse_verifier/faithful_data_openai_3k \
  --max-paths 3000 \
  --max-negatives 3 \
  --dry-run
```

Then export the API key in the shell and rerun without `--dry-run`. The command submits resumable
OpenAI Batch jobs sequentially, waits for completion, validates outputs, and writes accepted train
and dev JSONL files. It defaults to the fixed `gpt-4o-mini-2024-07-18` snapshot and refuses to
proceed when its estimate exceeds `$8`.

```bash
export OPENAI_API_KEY='...'
python3 -m inverse_verifier naturalize-openai \
  --data runs/inverse_verifier/faithful_data \
  --output runs/inverse_verifier/faithful_data_openai_3k \
  --max-paths 3000 \
  --max-negatives 3
```
