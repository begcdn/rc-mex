#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-/data3/ziad/rcmex/rc-mex}"

VENV="${VENV:-/data3/ziad/venvs/inverse-verifier}"
PY="$VENV/bin/python"
MODEL="${MODEL:-/data3/ziad/models/Llama-3.2-3B-Instruct}"
GPU="${GPU:-1}"
OPENAI_WORKERS="${OPENAI_WORKERS:-3}"
ROOT="${ROOT:-runs/subgraph_reader_pilot/cwq_full_scale_campaign_v1}"
RELEASE="${RELEASE:-data/subgraphrag_release}"
OFFICIAL="${OFFICIAL:-data/pattern_alignment/transfer/cwq.json}"
LLAMA_SOURCE="$RELEASE/results/KGQA/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl"
GPT_SOURCE="$RELEASE/results/KGQA/cwq/SubgraphRAG/gpt-4o-mini/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/data3/ziad/hf_cache}"

if [[ ! -f "$LLAMA_SOURCE" || ! -f "$GPT_SOURCE" ]]; then
  export HF_HUB_OFFLINE=0
  "$PY" - <<PY
from huggingface_hub import hf_hub_download
for filename in (
    "results/KGQA/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl",
    "results/KGQA/cwq/SubgraphRAG/gpt-4o-mini/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl",
):
    print(hf_hub_download("siqim311/SubgraphRAG", filename=filename, local_dir="$RELEASE"))
PY
fi

"$PY" reader_scale_campaign.py prepare \
  --source "$LLAMA_SOURCE" \
  --gpt-source "$GPT_SOURCE" \
  --official "$OFFICIAL" \
  --output "$ROOT" \
  --tokenizer "$MODEL"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"

(
  "$PY" reader_scale_campaign.py run-openai \
    --inputs "$ROOT/inputs" \
    --output "$ROOT/gpt4o_mini" \
    --workers "$OPENAI_WORKERS"
) >"$ROOT/logs/openai.log" 2>&1 &
OPENAI_PID=$!
echo "OpenAI API runner PID: $OPENAI_PID (workers=$OPENAI_WORKERS)"

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" "$PY" reader_scale_campaign.py run-local \
  --inputs "$ROOT/inputs" \
  --output "$ROOT/llama32_3b/runs" \
  --model "$MODEL" \
  --batch-size 8 \
  --tensor-parallel-size 1 \
  2>&1 | tee "$ROOT/logs/llama32_3b.log"

echo "Local model complete; waiting for OpenAI API runner"
wait "$OPENAI_PID"

"$PY" reader_scale_campaign.py evaluate \
  --local-runs "$ROOT/llama32_3b/runs" \
  --gpt-runs "$ROOT/gpt4o_mini/runs" \
  --metadata "$ROOT/inputs/metadata.jsonl" \
  --output "$ROOT/evaluation"

tar -czf /tmp/cwq_full_scale_campaign_v1_results.tgz \
  -C "$ROOT" manifest.json evaluation llama32_3b/runs gpt4o_mini/runs \
  gpt4o_mini/run_manifest.json logs

echo "Campaign complete"
echo "Results archive: /tmp/cwq_full_scale_campaign_v1_results.tgz"
