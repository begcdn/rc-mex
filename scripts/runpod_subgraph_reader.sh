#!/usr/bin/env bash
set -euo pipefail

# Two-stage RunPod runner for the controlled CWQ graph-interface experiment.
# Prepare a persistent /workspace volume on a CPU Pod, then attach the same
# volume to a 48 GB GPU Pod for inference.

MODE="${1:-gpu-all}"
case "$MODE" in
  cpu-prepare|gpu-smoke|gpu-full|gpu-all|gpu-operation|all|prepare|smoke|full) ;;
  *)
    echo "usage: $0 [cpu-prepare|gpu-smoke|gpu-full|gpu-all|gpu-operation|all]" >&2
    exit 2
    ;;
esac

ROOT="${RUNPOD_ROOT:-/workspace}"
REPO="$ROOT/rcmex"
CPU_VENV="$ROOT/venvs/subgraph-reader-prep"
GPU_VENV="$ROOT/venvs/subgraph-reader-gpu"
MODEL_DIR="$ROOT/models/Meta-Llama-3.1-8B-Instruct"
HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
TMPDIR="${TMPDIR:-$ROOT/tmp}"
OUT="$ROOT/runs/subgraph_reader_pilot/cwq_structure_400"
SOURCE_DIR="$ROOT/data/subgraphrag_release"
SOURCE_FILE="$SOURCE_DIR/results/KGQA/cwq/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl"
REPO_URL="${REPO_URL:-https://github.com/begcdn/rc-mex.git}"
CODE_REVISION="${CODE_REVISION:-main}"

export HF_HOME
export PIP_CACHE_DIR
export TMPDIR
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

checkout_code() {
  if [[ ! -d "$REPO/.git" ]]; then
    git clone "$REPO_URL" "$REPO"
  fi
  git -C "$REPO" fetch origin "$CODE_REVISION"
  git -C "$REPO" checkout --detach FETCH_HEAD
}

prepare_cpu_runtime() {
  checkout_code
  if [[ ! -x "$CPU_VENV/bin/python" ]]; then
    python3 -m venv "$CPU_VENV"
  fi
  "$CPU_VENV/bin/python" -m pip install --upgrade pip
  "$CPU_VENV/bin/python" -m pip install \
    "huggingface_hub>=0.25,<1" \
    "hf_transfer>=0.1.8" \
    "transformers==4.46.3" \
    pytest
}

select_gpu_python() {
  local candidate
  for candidate in "${GPU_PYTHON:-}" python3.11 python3.10 python3.9 python3; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
    then
      GPU_PYTHON="$candidate"
      return
    fi
  done
  echo "GPU stage requires Python 3.9 or newer; use a current RunPod PyTorch template" >&2
  exit 1
}

prepare_gpu_runtime() {
  select_gpu_python
  if [[ ! -x "$GPU_VENV/bin/python" ]]; then
    "$GPU_PYTHON" -m venv "$GPU_VENV"
  fi
  "$GPU_VENV/bin/python" -m pip install --upgrade pip
  if ! "$GPU_VENV/bin/python" -c 'from importlib.metadata import version; assert version("vllm") == "0.8.5.post1"' 2>/dev/null; then
    "$GPU_VENV/bin/python" -m pip install "vllm==0.8.5.post1"
  fi
  "$GPU_VENV/bin/python" -m pip install \
    "transformers==4.51.3" \
    "tokenizers==0.21.1"
}

download_inputs() {
  if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    : "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with Meta-Llama-3.1-8B-Instruct access}"
    log "Downloading the exact Llama checkpoint"
    MODEL_DIR="$MODEL_DIR" "$CPU_VENV/bin/python" - <<'PY'
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
    SOURCE_DIR="$SOURCE_DIR" "$CPU_VENV/bin/python" - <<'PY'
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
  "$CPU_VENV/bin/python" -m pytest \
    tests/test_subgraph_organizer.py \
    tests/test_subgraph_reader_pilot.py \
    -q
  "$CPU_VENV/bin/python" subgraph_reader_pilot.py prepare-structure \
    --source "$SOURCE_FILE" \
    --output "$OUT" \
    --official-cwq resources/complexwebquestions/cwq_test_structure.json \
    --tokenizer "$MODEL_DIR" \
    --per-type 200 \
    --seed 17

  {
    echo "git_commit=$(git rev-parse HEAD)"
    echo "created_at=$(date --iso-8601=seconds)"
    "$CPU_VENV/bin/python" - <<'PY'
from importlib.metadata import version

print(f"transformers={version('transformers')}")
print(f"huggingface_hub={version('huggingface_hub')}")
PY
  } > "$OUT/preparation_environment.txt"
  log "CPU preparation finished. Persistent data is ready at $OUT"
}

verify_gpu_stage() {
  [[ -x "$GPU_VENV/bin/python" ]] || {
    echo "Missing GPU environment at $GPU_VENV" >&2
    exit 1
  }
  [[ -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
    echo "Missing complete model at $MODEL_DIR; run cpu-prepare first" >&2
    exit 1
  }
  for arm in original reorder structured adjacency_flat adjacency_graph; do
    [[ -f "$OUT/inputs/$arm.jsonl" ]] || {
      echo "Missing prepared arm $arm; run cpu-prepare first" >&2
      exit 1
    }
  done

  cd "$REPO"
  "$GPU_VENV/bin/python" - <<'PY'
import torch
import vllm

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the prepared environment")
print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
  {
    echo "gpu_started_at=$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  } > "$OUT/gpu_environment.txt"
}

run_smoke() {
  cd "$REPO"
  rm -rf "$OUT/smoke"
  "$GPU_VENV/bin/python" subgraph_reader_pilot.py run-all-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/smoke" \
    --model "$MODEL_DIR" \
    --batch-size 2 \
    --tensor-parallel-size 1 \
    --limit 2

  OUT="$OUT" "$GPU_VENV/bin/python" - <<'PY'
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
  "$GPU_VENV/bin/python" subgraph_reader_pilot.py run-all-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/runs" \
    --model "$MODEL_DIR" \
    --batch-size 8 \
    --tensor-parallel-size 1

  "$GPU_VENV/bin/python" subgraph_reader_pilot.py evaluate-graph \
    --runs "$OUT/runs" \
    --metadata "$OUT/inputs/graph_metadata.jsonl" \
    --output "$OUT/evaluation"
  log "Finished. Results: $OUT/evaluation/metrics.json"
}

run_operation() {
  cd "$REPO"
  for arm in original reorder original_operation reorder_operation; do
    [[ -f "$OUT/inputs/$arm.jsonl" ]] || {
      echo "Missing prepared operation arm $arm; run cpu-prepare with the current code first" >&2
      exit 1
    }
  done

  rm -rf "$OUT/operation_smoke"
  "$GPU_VENV/bin/python" subgraph_reader_pilot.py run-operation-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/operation_smoke" \
    --model "$MODEL_DIR" \
    --batch-size 2 \
    --tensor-parallel-size 1 \
    --limit 2

  "$GPU_VENV/bin/python" subgraph_reader_pilot.py run-operation-suite \
    --inputs "$OUT/inputs" \
    --output "$OUT/operation_runs" \
    --model "$MODEL_DIR" \
    --batch-size 8 \
    --tensor-parallel-size 1

  "$GPU_VENV/bin/python" subgraph_reader_pilot.py evaluate-operation \
    --runs "$OUT/operation_runs" \
    --metadata "$OUT/inputs/graph_metadata.jsonl" \
    --output "$OUT/operation_evaluation"
  log "Operation experiment finished. Results: $OUT/operation_evaluation/metrics.json"
}

case "$MODE" in
  cpu-prepare|prepare)
    prepare_cpu_runtime
    download_inputs
    prepare_experiment
    ;;
  gpu-smoke|smoke)
    prepare_gpu_runtime
    verify_gpu_stage
    run_smoke
    ;;
  gpu-full|full)
    prepare_gpu_runtime
    verify_gpu_stage
    run_full
    ;;
  gpu-all)
    prepare_gpu_runtime
    verify_gpu_stage
    run_smoke
    run_full
    ;;
  gpu-operation)
    prepare_gpu_runtime
    verify_gpu_stage
    run_operation
    ;;
  all)
    prepare_cpu_runtime
    download_inputs
    prepare_experiment
    prepare_gpu_runtime
    verify_gpu_stage
    run_smoke
    run_full
    ;;
esac
