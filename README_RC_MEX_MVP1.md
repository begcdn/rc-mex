# RC-MEX MVP1: Relation Cards

This is the first MVP for **RC-MEX: Relation-Card Marginalized Execution for KGQA**.

MVP1 tests:

> Contrastive relation cards can induce reusable semantic descriptions of KG primitives from executable examples and hard negatives, remaining useful when relation names are hidden or misleading.

The code is dependency-free and uses only `kb.json` for this MVP.

## Run the synthetic tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
```

## Get KQA Pro

```bash
cd "/Users/ziad/Documents/New project"
python3 -m cigr_d_mvp1.download_kqa_pro --output data/kqa_pro --files kb.json
```

## Default local-model run

The default oracle backend is Ollama with `llama3:8b-instruct`.

If your installed Ollama tag has a different name, pass it with `--model`.

```bash
ollama serve
```

Then:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_ollama_20 \
  --max-primitives 20
```

Equivalent explicit form:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_ollama_20 \
  --max-primitives 20 \
  --oracle-backend ollama \
  --model llama3:8b-instruct
```

## Mock smoke run

The mock backend is only for pipeline testing. It is not experimental evidence.

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_mock \
  --max-primitives 20 \
  --oracle-backend mock
```

## OpenAI API run

For quick API-backed testing, use the official OpenAI backend. Start tiny because MVP1 uses one generation call per card plus one validation call per held-out pair.

```bash
export OPENAI_API_KEY="your_api_key_here"

python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_openai_tiny \
  --max-primitives 2 \
  --conditions B1 \
  --card-variants contrastive_hard \
  --oracle-backend openai
```

`--oracle-backend openai` defaults to `gpt-4o-mini`. You can override it:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_openai_20 \
  --max-primitives 20 \
  --oracle-backend openai \
  --model gpt-4o-mini
```

With an OpenAI-compatible local server:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_vllm_20 \
  --max-primitives 20 \
  --oracle-backend openai-compatible \
  --openai-base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

## MVP1 conditions

Evidence conditions:

```text
A  = normal relation names + real entity labels + type labels
B1 = anonymized relation IDs + real entity labels + type labels
B2 = anonymized relation IDs + anonymized entity labels + type labels
B3 = anonymized relation IDs + anonymized entity labels + no type labels
C  = misleading relation names + real entity labels + type labels
```

Card variants:

```text
contrastive_hard = positives + hard negatives
random_negative  = positives + random negatives
name_only        = visible relation/type evidence only
```

## Outputs

Each run writes:

```text
relation_cards.jsonl
validation_predictions.jsonl
primitive_samples.jsonl
metrics.json
report.md
examples_summary.json
primitive_metrics.jsonl
debug_examples.md
debug_report.html
```

Validation is pair classification:

```text
Given a frozen card and ordered pair (h,t), does the pair satisfy the card predicate?
```

It reports positive accuracy, hard-negative rejection, random-negative rejection, swapped-direction rejection, F1, direction accuracy, opaque rate, and cost estimates.

Open `debug_report.html` after a run to inspect primitives, card descriptions, examples, false positives, false negatives, direction errors, and automatic diagnosis labels.

For more terminal detail during a run:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_debug \
  --max-primitives 2 \
  --model llama3:8b-instruct \
  --verbose \
  --debug-examples-per-primitive 3
```

To skip metadata-looking relations such as external IDs, URLs, and source/provenance relations:

```bash
python3 -m rc_mex.run_mvp1 \
  --kb data/kqa_pro/kb.json \
  --output runs/rc_mex_mvp1_no_metadata \
  --max-primitives 20 \
  --exclude-metadata-relations
```
