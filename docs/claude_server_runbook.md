# Claude Server Runbook

This file records the working server setup and previous operational failures.
Read it before changing dependencies, downloading models, or giving Ziad a run
command. Do not include API keys in code, shell history, commits, or reports.

## Machines and Repository

Local Mac repository:

```text
/Users/ziad/Desktop/rcmex
```

Local result collection:

```text
/Users/ziad/Documents/inverse_verifier
```

Git remote:

```text
git@github.com:begcdn/rc-mex.git
```

Server SSH:

```text
ziad@10.249.44.191
```

Server repository:

```text
/data3/ziad/rcmex/rc-mex
```

Ziad normally transfers source through GitHub:

1. Commit and push locally.
2. Run `git pull origin main` on the server.
3. Transfer large datasets, checkpoints, and result folders with `rsync`.

Do not commit model weights, generated datasets, or run outputs.

## Working Python Environment

Use this virtual environment:

```text
/data3/ziad/venvs/inverse-verifier
```

Standard shell setup:

```bash
cd /data3/ziad/rcmex/rc-mex

VENV=/data3/ziad/venvs/inverse-verifier
export PYTHONNOUSERSITE=1
export HF_HOME=/data3/ziad/hf_cache
export HF_HUB_OFFLINE=1
```

Run modules from the repository root:

```bash
"$VENV/bin/python" -m inverse_verifier --help
```

Do not use bare `python3` for model runs unless its environment has been
explicitly checked.

Do not prepend `/data3/ziad/pylibs` to `PYTHONPATH`. That directory previously
supplied:

```text
torch 2.12.0+cu130
CUDA runtime 13.0
```

The server NVIDIA driver supports CUDA 12.4, so that combination failed with:

```text
RuntimeError: The NVIDIA driver on your system is too old
```

The dedicated virtual environment was created to avoid that mismatch and to
isolate `torch`, `transformers`, `sentence-transformers`, and `protobuf`.

Before a long run, verify the interpreter and CUDA:

```bash
"$VENV/bin/python" - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Also verify core imports:

```bash
"$VENV/bin/python" - <<'PY'
import transformers
import sentence_transformers
import srtk

print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("srtk:", srtk.__file__)
PY
```

## Installation Problems Already Encountered

### Editable install

This command failed:

```bash
pip install -e .
```

The repository uses a flat layout with several top-level packages and data
directories, so setuptools automatic package discovery rejected it. The current
workflow does not require editable installation. Run `python -m
inverse_verifier` from the repository root.

### User-site contamination

The server user site previously mixed incompatible versions of PyTorch and
Transformers and produced:

```text
ImportError: cannot import name 'AuxRequest' from
torch.nn.attention.flex_attention
```

Keep:

```bash
export PYTHONNOUSERSITE=1
```

Do not combine the dedicated virtual environment with packages under
`~/.local/lib/python3.10/site-packages` or `/data3/ziad/pylibs`.

### Virtual environment initially lacked pip

The first virtual environment had neither `pip` nor `ensurepip`. That has since
been repaired. Do not recreate or overwrite the working environment casually.

### Protobuf

Transformers tokenization previously failed because `protobuf` was missing.
It is installed in the dedicated environment. If the error returns, install
into that environment only:

```bash
"$VENV/bin/python" -m pip install protobuf
```

### SRTK

The full retriever needs the `srtk` package. The earlier error was:

```text
ModuleNotFoundError: No module named 'srtk'
```

It is now installed in the dedicated environment. Verify with the import check
above rather than reinstalling preemptively.

### Hugging Face network access

The server often cannot reach Hugging Face. Typical failure:

```text
LocalEntryNotFoundError
ConnectTimeout
```

Normal runs are offline:

```bash
export HF_HUB_OFFLINE=1
```

Resolve models to an existing local snapshot or explicit model directory. If a
model is missing, downloading on the Mac and transferring it to the server is
more reliable than repeatedly retrying from the server.

## Hardware

The server has eight shared NVIDIA L20 GPUs, each with approximately 46 GB VRAM.
Observed driver/runtime:

```text
Driver 550.144.03
CUDA 12.4
```

Other users may occupy GPUs. Never kill another user's process.

Inspect GPUs:

```bash
nvidia-smi
```

Live monitoring:

```bash
watch -n 1 nvidia-smi
```

List compute processes and owners:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
  --format=csv,noheader
```

Then inspect a PID:

```bash
ps -o user,pid,etime,cmd -p PID
```

Select one free physical GPU:

```bash
CUDA_VISIBLE_DEVICES=2 "$VENV/bin/python" -m inverse_verifier ...
```

Inside the process, that selected physical GPU appears as `cuda:0`. Do not pass
`cuda:2` after setting `CUDA_VISIBLE_DEVICES=2`; use `--device cuda`.

An out-of-memory error previously occurred because another process occupied
almost all of GPU 0. Changing to a genuinely free GPU fixed it. Reducing batch
size is secondary to checking ownership and free VRAM first.

## Long Runs and tmux

Use tmux so SSH disconnection does not stop a run:

```bash
tmux new -s inverse-eval
```

Run the command inside that tmux session.

Detach from inside tmux:

```text
Ctrl-b, then d
```

Attach later:

```bash
tmux attach -t inverse-eval
```

Check sessions:

```bash
tmux ls
```

From another terminal, force-detach a connected client:

```bash
tmux detach-client -s inverse-eval
```

The message `no current client` means the session already has no attached
client; it is not an error if `tmux ls` still shows the session.

## Git Safety

Before pulling:

```bash
cd /data3/ziad/rcmex/rc-mex
git status --short
git pull origin main
```

A previous pull was blocked because untracked `inverse_verifier/*.py` and
`pyproject.toml` files would have been overwritten. Do not delete such files
blindly. Move them to a timestamped backup, inspect them, then pull.

Local Mac currently has an unrelated untracked archive:

```text
inverse_direction_pairs_v1.tgz
```

Do not stage or commit it.

## Data Paths

Correct server WebQSP paths:

```text
data/pattern_alignment/webqsp/WebQSP.test.json
data/webqsp/test.jsonl
```

The following path does not exist on the server:

```text
data/webqsp_official/WebQSP/data/WebQSP.test.json
```

Find data before assuming a path:

```bash
find data -type f \( -name 'WebQSP.test.json' -o -name 'test.jsonl' \) | sort
```

Important KQA Pro paths used previously:

```text
data/kqa_pro/kb.json
data/kqa_pro/val.json
```

## Cached and Trained Models

Resolve BGE:

```bash
BGE_MODEL="$(find "$HF_HOME/hub/models--BAAI--bge-small-en-v1.5/snapshots" \
  -mindepth 1 -maxdepth 1 -type d | head -n 1)"
```

Known BGE snapshot:

```text
/data3/ziad/hf_cache/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
```

Resolve SRTK:

```bash
SRTK_MODEL="$(find "$HF_HOME/hub/models--drt--srtk-scorer/snapshots" \
  -mindepth 1 -maxdepth 1 -type d | head -n 1)"
```

Known SRTK snapshot:

```text
/data3/ziad/hf_cache/hub/models--drt--srtk-scorer/snapshots/53f281a33d497ace4d9b0adbc1da6f711a43c3d9
```

Other explicit server model directories:

```text
/data3/ziad/models/flan-t5-small
/data3/ziad/models/deberta-v3-base
```

Current inverse generator:

```text
runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1/model
```

Current question-only comparator:

```text
runs/inverse_verifier/deberta_comparator_question_generated_v1/model
```

Do not assume a checkpoint exists merely because code references it. Check:

```bash
test -f MODEL_PATH/config.json
find MODEL_PATH -maxdepth 1 -type f -print
```

## Ollama History

Ollama is not used by the current SRTK + Qwen checkpoint + DeBERTa comparator
pipeline.

The system Ollama server was reachable at:

```text
http://127.0.0.1:11434
```

Installed models included:

```text
qwen2.5:7b
qwen3:8b
qwen3:4b
```

Additional Ollama instances were temporarily launched on ports 11435-11438.
Those ports do not persist automatically and sometimes pointed at the wrong
model directory. Always check:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

If manually launching another instance, set both `OLLAMA_MODELS` and a URL with
an HTTP scheme. A prior command used `127.0.0.1:11435` without `http://`, which
caused:

```text
urllib.error.URLError: unknown url type: 127.0.0.1
```

Do not restart or replace a shared Ollama service without checking ownership.

## File Transfer

Commands involving `/Users/ziad/...` must run on the Mac, not inside the server
SSH shell.

Mac to server:

```bash
rsync -avP \
  /Users/ziad/Documents/inverse_verifier/SOURCE_FOLDER/ \
  ziad@10.249.44.191:/data3/ziad/rcmex/rc-mex/runs/inverse_verifier/DESTINATION_FOLDER/
```

Server results to Mac:

```bash
rsync -avP \
  ziad@10.249.44.191:/data3/ziad/rcmex/rc-mex/runs/inverse_verifier/RUN_NAME/ \
  /Users/ziad/Documents/inverse_verifier/RUN_NAME/
```

Using a Mac path while logged into the Linux server previously caused
`change_dir ... No such file or directory`.

## Current Full Pipeline Command

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

Check that the chosen GPU is free before using `CUDA_VISIBLE_DEVICES=0`.

The latest 100-question run took approximately 4,668 seconds because it
generated and scored an average of 98.5 candidate questions per example.

## Current Result Locations

Mac:

```text
/Users/ziad/Documents/inverse_verifier/full_pipeline_question_comparator_v1
/Users/ziad/Documents/inverse_verifier/comparator_kqa_semantic_adjudicated_v1
/Users/ziad/Documents/inverse_verifier/semantic_adjudicated_test_kqa_val_eval
```

Server:

```text
runs/inverse_verifier/full_pipeline_question_comparator_v1
runs/inverse_verifier/faithful_inverse_qwen25_3b_lora_v1
runs/inverse_verifier/deberta_comparator_question_generated_v1
```

Research and architecture context is in:

```text
docs/opus_handoff_inverse_verifier.md
```

## Before Giving Ziad Any Run Command

1. Confirm whether the command runs on Mac or server.
2. Use the dedicated virtual-environment Python.
3. Check the actual input paths.
4. Check that every model directory exists.
5. Check that the selected GPU is free.
6. Use tmux for a run longer than a few minutes.
7. State the expected output directory.
8. Provide the matching `rsync` command to copy results back.
9. Avoid adding flags or output files that are not needed by the experiment.
10. Do not change the scientific method merely to work around an environment
    problem.
