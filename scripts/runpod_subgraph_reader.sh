#!/usr/bin/env bash
set -euo pipefail

# Disposable RunPod runner for the controlled CWQ graph-interface experiment.
# Use a 48 GB GPU and a RunPod PyTorch template with at least 60 GB of disk.

MODE="${1:-all}"
case "$MODE" in
  prepare|smoke|full|all) ;;
  *) echo "usage: $0 [prepare|smoke|full|all]" >&2; exit 2 ;;
esac

ROOT="${RUNPOD_ROOT:-/workspace}"
REPO="$ROOT/rcmex"
VENV="$ROOT/venvs/subgraph-reader"
MODEL_DIR="$ROOT/models/Meta-Llama-3.1-8B-Instruct"
HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
OUT="$ROOT/runs/subgraph_reader_pilot/cwq_structure_400"
SOURCE_DIR="$ROOT/data/subgraphrag_release"
SOURCE_FILE="$SOURCE_DIR/results/KGQA/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl"
REPO_URL="${REPO_URL:-https://github.com/begcdn/rc-mex.git}"
CODE_REVISION="${CODE_REVISION:-main}"

export HF_HOME
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

prepare_runtime() {
  if [[ ! -d "$REPO/.git" ]]; then
    git clone "$REPO_URL" "$REPO"
  fi
  git -C "$REPO" fetch origin "$CODE_REVISION"
  git -C "$REPO" checkout --detach FETCH_HEAD

  if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --upgrade pip
  if ! "$VENV/bin/python" -c 'import vllm; assert vllm.__version__ == "0.8.5.post1"' 2>/dev/null; then
    "$VENV/bin/python" -m pip install "vllm==0.8.5.post1"
  fi
  "$VENV/bin/python" -m pip install "huggingface_hub>=0.25,<1" pytest
}

download_inputs() {
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    : "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with Meta-Llama-3.1-8B-Instruct access}"
    log "Downloading the exact Llama checkpoint"
    MODEL_DIR="$MODEL_DIR" "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir=os.environ["MODEL_DIR"],
    token=os.environ["HF_TOKEN"],
)
PY
  fi

  if [[ ! -f "$SOURCE_FILE" ]]; then
    log "Downloading SubgraphRAG's published CWQ predictions"
    SOURCE_DIR="$SOURCE_DIR" "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="siqim311/SubgraphRAG",
    filename="results/KGQA/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl",
    local_dir=os.environ["SOURCE_DIR"],
)
PY
  fi
}

prepare_experiment() {
  mkdir -p "$OUT"
  cd "$REPO"
  "$VENV/bin/python" -m pytest -q
  "$VENV/bin/python" subgraph_reader_pilot.py prepare-structure \
    --source "$SOURCE_FILE" \
    --output "$OUT" \
    --official-cwq data/pattern_alignment/transfer/cwq.json \
    --tokenizer "$MODEL_DIR" \
    --per-type 200 \
    --seed 17

  {
    echo "git_commit=$(git rev-parse HEAD)"
    echo "created_at=$(date --iso-8601=seconds)"
    "$VENV/bin/python" -c 'import torch, vllm; print(f"torch={torch.__version__}"); print(f"vllm={vllm.__version__}")'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  } > "$OUT/environment.txt"
}

run_smoke() {
  cd "$REPO"
  rm -rf "$OUT/smoke"
  "$VENV/bin/python" subgraph_reader_pilot.py run-all-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/smoke" \
    --model "$MODEL_DIR" \
    --batch-size 2 \
    --tensor-parallel-size 1 \
    --limit 2

  OUT="$OUT" "$VENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT"]) / "smoke"
for arm in ("original", "reorder", "structured", "adjacency_flat", "adjacency_graph"):
    rows = [json.loads(line) for line in (root / f"{arm}.jsonl").open()]
    assert len(rows) == 2, (arm, len(rows))
    assert all(row.get("prediction", "").strip() for row in rows), arm
print("Smoke test passed for all five arms")
PY
}

run_full() {
  cd "$REPO"
  "$VENV/bin/python" subgraph_reader_pilot.py run-all-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/runs" \
    --model "$MODEL_DIR" \
    --batch-size 8 \
    --tensor-parallel-size 1

  "$VENV/bin/python" subgraph_reader_pilot.py evaluate-graph \
    --runs "$OUT/runs" \
    --metadata "$OUT/inputs/graph_metadata.jsonl" \
    --output "$OUT/evaluation"
  log "Finished. Results: $OUT/evaluation/metrics.json"
}

prepare_runtime
download_inputs

case "$MODE" in
  prepare) prepare_experiment ;;
  smoke) prepare_experiment; run_smoke ;;
  full) prepare_experiment; run_full ;;
  all) prepare_experiment; run_smoke; run_full ;;
esac
