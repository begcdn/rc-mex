# CIGR-D MVP 1: Execution-Witness Grounding

This package implements the first MVP for CIGR-D:

> Under controlled gold-prefix conditions, execution-derived denotational witnesses improve relation/direction grounding beyond schema-label matching.

The implementation is deliberately standalone and dependency-free. The cloned research repos in this workspace are not imported.

## MVP ladder

```text
MVP 1: execution-witness grounding works
MVP 2: bottom-up synthesis improves reachable answer recall
MVP 3: denotation merging reduces search cost
MVP 4: heuristic active controller improves budgeted accuracy
MVP 5: learned controller beats heuristic controller
```

## Download KQA Pro

```bash
python3 -m cigr_d_mvp1.download_kqa_pro --output data/kqa_pro --files kb.json,val.json
```

KQA Pro provides `kb.json` plus train/val/test JSON files with questions, SPARQL, and gold programs.

## Smoke test

```bash
python3 -m unittest discover -s tests
```

## Run MVP 1

Mock judge smoke run:

```bash
python3 -m cigr_d_mvp1.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --questions data/kqa_pro/val.json \
  --output runs/mvp1_mock \
  --max-instances 50 \
  --judge-backend mock
```

Ollama/local model run:

```bash
python3 -m cigr_d_mvp1.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --questions data/kqa_pro/val.json \
  --output runs/mvp1_ollama \
  --max-instances 200 \
  --judge-backend ollama \
  --model llama3.1
```

OpenAI-compatible local server run:

```bash
python3 -m cigr_d_mvp1.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --questions data/kqa_pro/val.json \
  --output runs/mvp1_vllm \
  --max-instances 200 \
  --judge-backend openai-compatible \
  --openai-base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

## Core matrix

The runner starts with the compact MVP1 matrix:

```text
normal relation labels + real returned entity names
anonymized relation labels + real returned entity names
anonymized relation labels + anonymized returned entity names + types
```

Methods:

```text
random
embedding_schema
schema_llm
witness_llm
```

## Outputs

Each run writes:

```text
metrics.json
report.md
instances.jsonl
rankings.jsonl
```

Read candidate-pool diagnostics first. If gold relation+direction candidate recall is below 90%, fix candidate generation before judging the LLM.

## Important limitation

Gold programs are used only to create controlled relation-grounding instances:

```text
Given question q and correct current input set S, rank the next relation/direction.
```

This tests relation grounding, not full KGQA.
