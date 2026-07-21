from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.request
import uuid
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .synthetic import relation_words


MODEL = "gpt-4o-mini-2024-07-18"
# Conservative published rates. Actual Batch charges may be lower.
BATCH_INPUT_USD_PER_MILLION = 0.15
BATCH_OUTPUT_USD_PER_MILLION = 0.60
MAX_ESTIMATED_TOKENS_PER_BATCH = 1_250_000
ROWS_PER_REQUEST = 4
POLL_SECONDS = 60

SYSTEM_PROMPT = """You rewrite abstract knowledge-graph paths as natural English questions.

Rules:
- Use [ENTITY] exactly once as the known topic entity.
- Express every hop, in the supplied order and direction. Forward means relation(current, next); backward means relation(next, current).
- Ask for the supplied answer type. Do not invent any entity, value, date, or fact.
- Distinct paths must have distinct meanings, even when they share a prefix.
- Never mention graph traversal, hops, direction, IDs, sequences, or a resulting entity.
- Avoid procedural wording such as follow, traverse, connected to, points to, or reach.
- Do not quote raw relation labels. Turn them into ordinary English.
- If one coherent natural question cannot express the complete path, return opaque instead of dropping or changing a hop.

Return one item for every supplied id."""

OUTPUT_SCHEMA = {
    "name": "path_naturalizations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string", "enum": ["valid", "opaque"]},
                        "question": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "status", "question", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

FORBIDDEN_PHRASES = (
    "forward relation",
    "backward relation",
    "knowledge graph",
    "graph traversal",
    "first hop",
    "second hop",
    "third hop",
    "follow the",
    "traverse",
    "points to",
    "resulting entity",
    "relation sequence",
)


def sanitized_candidate(identifier: str, path: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": identifier,
        "topic": "[ENTITY]",
        "topic_type": path.get("anchor_type", "entity"),
        "hops": [
            {
                "relation": relation_words(hop["relation"]),
                "direction": hop["direction"],
                "source_type": hop.get("source_type", "entity"),
                "target_type": hop.get("target_type", "entity"),
            }
            for hop in path["hops"]
        ],
        "answer_type": path.get("answer_type", "entity"),
    }


def select_negatives(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    selected = []
    seen_types = set()
    for negative in row.get("negative_paths", []):
        category = negative.get("negative_type", "unknown")
        if category not in seen_types:
            selected.append(negative)
            seen_types.add(category)
        if len(selected) == limit:
            return selected
    for negative in row.get("negative_paths", []):
        if negative not in selected:
            selected.append(negative)
        if len(selected) == limit:
            break
    return selected


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def select_rows(source: Path, max_paths: int, max_negatives: int, seed: int = 17) -> list[dict[str, Any]]:
    buckets: dict[str, dict[tuple[int, str], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rng = random.Random(seed)
    split_sizes = {}
    for split, filename in (("train", "train_faithful.jsonl"), ("dev", "dev_faithful.jsonl")):
        source_rows = _read_rows(source / filename)
        split_sizes[split] = len(source_rows)
        for source_index, row in enumerate(source_rows):
            copied = dict(row)
            copied["negative_paths"] = select_negatives(row, max_negatives)
            copied["_naturalization"] = {"split": split, "source_index": source_index}
            path = copied["positive_path"]
            buckets[split][(len(path["hops"]), path.get("kg", "unknown"))].append(copied)
    for split_buckets in buckets.values():
        for rows in split_buckets.values():
            rng.shuffle(rows)

    total_available = sum(split_sizes.values())
    if total_available == 0:
        expected = ", ".join(
            str(source / filename)
            for filename in ("train_faithful.jsonl", "dev_faithful.jsonl")
        )
        raise FileNotFoundError(
            f"no faithful corpus rows found; expected JSONL data in: {expected}"
        )
    target = min(max_paths, total_available)
    quotas = {
        split: min(size, round(target * size / total_available))
        for split, size in split_sizes.items()
    }
    while sum(quotas.values()) < target:
        split = max(split_sizes, key=lambda name: split_sizes[name] - quotas[name])
        quotas[split] += 1
    while sum(quotas.values()) > target:
        split = max(quotas, key=quotas.get)
        quotas[split] -= 1

    ordered = []
    for split in ("train", "dev"):
        split_buckets = buckets[split]
        keys = sorted(split_buckets)
        selected = 0
        while keys and selected < quotas[split]:
            remaining = []
            for key in keys:
                if split_buckets[key]:
                    ordered.append(split_buckets[key].pop())
                    selected += 1
                    if selected == quotas[split]:
                        break
                if split_buckets[key]:
                    remaining.append(key)
            keys = remaining
    return ordered


def candidate_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        meta = row["_naturalization"]
        prefix = f"{meta['split']}:{meta['source_index']}"
        items.append(sanitized_candidate(f"{prefix}:positive", row["positive_path"]))
        for index, negative in enumerate(row["negative_paths"]):
            items.append(sanitized_candidate(f"{prefix}:negative:{index}", negative))
    return items


def request_record(custom_id: str, rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    candidates = candidate_items(rows)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Naturalize these paths:\n" + json.dumps(candidates, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_schema", "json_schema": OUTPUT_SCHEMA},
            "max_completion_tokens": 1600,
        },
    }


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def prepare_requests(
    source: Path,
    output: Path,
    model: str,
    max_paths: int,
    max_negatives: int,
) -> dict[str, Any]:
    internal = output / ".openai_batch"
    internal.mkdir(parents=True, exist_ok=True)
    rows = select_rows(source, max_paths, max_negatives)
    if not rows:
        raise ValueError(f"no faithful rows found in {source}")
    source_file = internal / "selected_rows.jsonl"
    source_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = []
    for start in range(0, len(rows), ROWS_PER_REQUEST):
        records.append(request_record(f"naturalize-{start // ROWS_PER_REQUEST:06d}", rows[start:start + ROWS_PER_REQUEST], model))

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_tokens = 0
    for record in records:
        line = json.dumps(record, ensure_ascii=False)
        tokens = estimate_tokens(line)
        if current and current_tokens + tokens > MAX_ESTIMATED_TOKENS_PER_BATCH:
            chunks.append(_write_chunk(internal, len(chunks), current, current_tokens))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += tokens
    if current:
        chunks.append(_write_chunk(internal, len(chunks), current, current_tokens))

    candidates = sum(1 + len(row["negative_paths"]) for row in rows)
    estimated_input_tokens = sum(chunk["estimated_input_tokens"] for chunk in chunks)
    estimated_output_tokens = candidates * 48
    estimated_cost = (
        estimated_input_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MILLION
        + estimated_output_tokens / 1_000_000 * BATCH_OUTPUT_USD_PER_MILLION
    )
    state = {
        "model": model,
        "source": str(source),
        "max_paths": max_paths,
        "max_negatives": max_negatives,
        "rows": len(rows),
        "candidates": candidates,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_batch_cost_usd": round(estimated_cost, 4),
        "chunks": chunks,
    }
    _write_json(internal / "state.json", state)
    return state


def _write_chunk(internal: Path, index: int, lines: list[str], tokens: int) -> dict[str, Any]:
    path = internal / f"requests_{index:03d}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "request_file": str(path),
        "estimated_input_tokens": tokens,
        "status": "prepared",
    }


class OpenAIBatchClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, data: bytes | None = None, content_type: str = "application/json") -> Any:
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        return json.loads(payload) if payload else None

    def upload(self, path: Path) -> str:
        boundary = f"----inverse-verifier-{uuid.uuid4().hex}"
        file_bytes = path.read_bytes()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: application/jsonl\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        return self._request("POST", "/files", body, f"multipart/form-data; boundary={boundary}")["id"]

    def create_batch(self, file_id: str) -> dict[str, Any]:
        body = json.dumps({
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"task": "inverse-path-naturalization"},
        }).encode()
        return self._request("POST", "/batches", body)

    def retrieve_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request("GET", f"/batches/{batch_id}")

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request("POST", f"/batches/{batch_id}/cancel", b"{}")

    def download(self, file_id: str) -> bytes:
        request = urllib.request.Request(
            self.base_url + f"/files/{file_id}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()


def run_openai_naturalization(
    source: Path,
    output: Path,
    model: str = MODEL,
    max_paths: int = 3_000,
    max_negatives: int = 3,
    max_budget_usd: float = 8.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / ".openai_batch" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else prepare_requests(
        source, output, model, max_paths, max_negatives
    )
    expected_settings = {
        "model": model,
        "source": str(source),
        "max_paths": max_paths,
        "max_negatives": max_negatives,
    }
    mismatches = {
        key: (state.get(key), expected)
        for key, expected in expected_settings.items()
        if state.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "existing OpenAI batch state uses different settings: "
            f"{mismatches}. Use a new --output directory."
        )
    print(
        f"Prepared {state['rows']} paths / {state['candidates']} candidate questions in "
        f"{len(state['chunks'])} batch(es)", flush=True
    )
    print(
        f"Conservative estimate: {state['estimated_input_tokens']:,} input + "
        f"{state['estimated_output_tokens']:,} output tokens, about "
        f"${state['estimated_batch_cost_usd']:.2f}", flush=True
    )
    if state["estimated_batch_cost_usd"] > max_budget_usd:
        raise RuntimeError(
            f"estimated cost ${state['estimated_batch_cost_usd']:.2f} exceeds "
            f"the ${max_budget_usd:.2f} limit; lower --max-paths"
        )
    if dry_run:
        print("Dry run only: no API request was submitted.", flush=True)
        return state

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAIBatchClient(api_key)
    for index, chunk in enumerate(state["chunks"], 1):
        if chunk.get("status") == "completed":
            continue
        if not chunk.get("batch_id"):
            print(f"Submitting batch {index}/{len(state['chunks'])}", flush=True)
            file_id = client.upload(Path(chunk["request_file"]))
            batch = client.create_batch(file_id)
            chunk.update({"input_file_id": file_id, "batch_id": batch["id"], "status": batch["status"]})
            _write_json(state_path, state)
        while True:
            batch = client.retrieve_batch(chunk["batch_id"])
            chunk["status"] = batch["status"]
            _write_json(state_path, state)
            counts = batch.get("request_counts") or {}
            print(
                f"Batch {index}/{len(state['chunks'])}: {batch['status']} "
                f"({counts.get('completed', 0)}/{counts.get('total', '?')} requests)",
                flush=True,
            )
            if batch["status"] == "completed":
                result_path = output / ".openai_batch" / f"results_{index - 1:03d}.jsonl"
                result_path.write_bytes(client.download(batch["output_file_id"]))
                chunk.update({"output_file_id": batch["output_file_id"], "result_file": str(result_path), "status": "completed"})
                _write_json(state_path, state)
                break
            if batch["status"] in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"OpenAI batch {batch['id']} ended with status {batch['status']}")
            time.sleep(POLL_SECONDS)
    return finalize_results(output, state)


def validate_question(question: str) -> str | None:
    normalized = question.strip()
    lowered = normalized.lower()
    if normalized.count("[ENTITY]") != 1:
        return "entity_placeholder_count"
    if not normalized.endswith("?"):
        return "not_a_question"
    if any(phrase in lowered for phrase in FORBIDDEN_PHRASES):
        return "procedural_language"
    return None


def _parse_result_file(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    predictions: dict[str, dict[str, str]] = {}
    errors = []
    for line in path.open(encoding="utf-8"):
        result = json.loads(line)
        if result.get("error"):
            errors.append(result)
            continue
        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            for item in parsed["items"]:
                predictions[item["id"]] = item
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"custom_id": result.get("custom_id"), "parse_error": str(exc)})
    return predictions, errors


def finalize_results(output: Path, state: dict[str, Any]) -> dict[str, Any]:
    predictions: dict[str, dict[str, str]] = {}
    api_errors = []
    for chunk in state["chunks"]:
        parsed, errors = _parse_result_file(Path(chunk["result_file"]))
        predictions.update(parsed)
        api_errors.extend(errors)

    selected = _read_rows(output / ".openai_batch" / "selected_rows.jsonl")
    accepted_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    accepted_questions = 0
    for row in selected:
        meta = row.pop("_naturalization")
        prefix = f"{meta['split']}:{meta['source_index']}"
        positive = predictions.get(f"{prefix}:positive")
        reason = _prediction_rejection(positive)
        if reason:
            rejected.append({"id": f"{prefix}:positive", "reason": reason, "prediction": positive})
            continue
        row["canonical_question"] = row["question"]
        row["question"] = positive["question"].strip()
        accepted_questions += 1
        natural_negatives = []
        seen_questions = {row["question"].casefold()}
        for index, negative in enumerate(row["negative_paths"]):
            identifier = f"{prefix}:negative:{index}"
            prediction = predictions.get(identifier)
            negative_reason = _prediction_rejection(prediction)
            if not negative_reason and prediction["question"].strip().casefold() in seen_questions:
                negative_reason = "semantic_collapse_exact_duplicate"
            if negative_reason:
                rejected.append({"id": identifier, "reason": negative_reason, "prediction": prediction})
                continue
            copied = dict(negative)
            copied["canonical_question"] = negative["question"]
            copied["question"] = prediction["question"].strip()
            natural_negatives.append(copied)
            seen_questions.add(copied["question"].casefold())
            accepted_questions += 1
        if natural_negatives:
            row["negative_paths"] = natural_negatives
            accepted_by_split[meta["split"]].append(row)
        else:
            rejected.append({"id": prefix, "reason": "no_valid_hard_negative"})

    for split in ("train", "dev"):
        path = output / f"{split}_faithful.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted_by_split[split]),
            encoding="utf-8",
        )
    rejected_path = output / "rejected.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected + api_errors),
        encoding="utf-8",
    )
    manifest = {
        **{key: value for key, value in state.items() if key != "chunks"},
        "accepted_rows": sum(map(len, accepted_by_split.values())),
        "accepted_questions": accepted_questions,
        "rejected_items": len(rejected),
        "api_errors": len(api_errors),
        "output_files": ["train_faithful.jsonl", "dev_faithful.jsonl", "rejected.jsonl"],
    }
    _write_json(output / "manifest.json", manifest)
    print(
        f"Accepted {manifest['accepted_rows']}/{state['rows']} rows; "
        f"rejected {manifest['rejected_items']} candidate items", flush=True
    )
    return manifest


def _prediction_rejection(prediction: dict[str, str] | None) -> str | None:
    if prediction is None:
        return "missing_output"
    if prediction.get("status") != "valid":
        return "model_marked_opaque"
    return validate_question(prediction.get("question", ""))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_batch_records(
    records: list[dict[str, Any]],
    directory: Path,
    client: OpenAIBatchClient,
    label: str,
    max_estimated_tokens: int = MAX_ESTIMATED_TOKENS_PER_BATCH,
) -> list[Path]:
    """Run arbitrary Chat Completions Batch records with resumable local state."""
    directory.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(record, ensure_ascii=False) for record in records]
    fingerprint = hashlib.sha256(
        (str(max_estimated_tokens) + "\n" + "\n".join(serialized)).encode()
    ).hexdigest()
    state_path = directory / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            completed_results = [
                Path(chunk.get("result_file", ""))
                for chunk in state.get("chunks", [])
                if chunk.get("status") == "completed"
            ]
            can_reuse = (
                state.get("requests") == len(records)
                and len(completed_results) == len(state.get("chunks", []))
                and all(path.is_file() for path in completed_results)
            )
            if can_reuse:
                print(
                    f"Reusing completed {label} batch despite a request fingerprint change; "
                    "downstream validation will mark missing relation IDs opaque.",
                    flush=True,
                )
                return completed_results
            raise RuntimeError(
                f"existing {label} state does not match current requests and is not complete"
            )
    else:
        chunks = []
        lines, tokens = [], 0
        for line in serialized:
            line_tokens = estimate_tokens(line)
            if lines and tokens + line_tokens > max_estimated_tokens:
                chunks.append(_write_chunk(directory, len(chunks), lines, tokens))
                lines, tokens = [], 0
            lines.append(line)
            tokens += line_tokens
        if lines:
            chunks.append(_write_chunk(directory, len(chunks), lines, tokens))
        state = {"label": label, "fingerprint": fingerprint, "requests": len(records), "chunks": chunks}
        _write_json(state_path, state)

    for index, chunk in enumerate(state["chunks"], 1):
        if chunk.get("status") == "completed":
            continue
        if not chunk.get("batch_id"):
            print(f"Submitting {label} batch {index}/{len(state['chunks'])}", flush=True)
            file_id = client.upload(Path(chunk["request_file"]))
            batch = client.create_batch(file_id)
            chunk.update({"input_file_id": file_id, "batch_id": batch["id"], "status": batch["status"]})
            _write_json(state_path, state)
        while True:
            batch = client.retrieve_batch(chunk["batch_id"])
            chunk["status"] = batch["status"]
            _write_json(state_path, state)
            counts = batch.get("request_counts") or {}
            print(
                f"{label} batch {index}/{len(state['chunks'])}: {batch['status']} "
                f"({counts.get('completed', 0)}/{counts.get('total', '?')})",
                flush=True,
            )
            if batch["status"] == "completed":
                result_path = directory / f"results_{index - 1:03d}.jsonl"
                result_path.write_bytes(client.download(batch["output_file_id"]))
                chunk.update({"result_file": str(result_path), "status": "completed"})
                _write_json(state_path, state)
                break
            if batch["status"] in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"OpenAI {label} batch ended with {batch['status']}")
            time.sleep(POLL_SECONDS)
    return [Path(chunk["result_file"]) for chunk in state["chunks"]]


def run_chat_records_sync(
    records: list[dict[str, Any]],
    directory: Path,
    client: OpenAIBatchClient,
    label: str,
    workers: int = 6,
    retries: int = 6,
) -> list[Path]:
    """Run Chat Completions concurrently and persist every completed response."""
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "results.jsonl"
    state_path = directory / "state.json"
    serialized = [json.dumps(record, ensure_ascii=False) for record in records]
    fingerprint = hashlib.sha256("\n".join(serialized).encode()).hexdigest()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError(f"existing synchronous {label} state does not match current requests")
    else:
        state = {"label": label, "fingerprint": fingerprint, "requests": len(records)}
        _write_json(state_path, state)

    completed = set()
    if result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["custom_id"])
    pending = [record for record in records if record["custom_id"] not in completed]
    if not pending:
        print(f"Reusing {len(completed)} completed synchronous {label} requests", flush=True)
        return [result_path]

    def execute(record: dict[str, Any]) -> dict[str, Any]:
        endpoint = record["url"]
        if endpoint.startswith("/v1/"):
            endpoint = endpoint[3:]
        for attempt in range(retries + 1):
            try:
                body = client._request("POST", endpoint, json.dumps(record["body"]).encode())
                return {
                    "id": f"sync-{uuid.uuid4().hex}",
                    "custom_id": record["custom_id"],
                    "response": {"status_code": 200, "request_id": body.get("id"), "body": body},
                    "error": None,
                }
            except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == retries:
                    raise RuntimeError(
                        f"synchronous {label} request {record['custom_id']} failed"
                    ) from exc
                time.sleep(min(2 ** attempt, 30))
        raise AssertionError("unreachable")

    done = len(completed)
    with result_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(execute, record): record for record in pending}
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                print(f"{label}: {done}/{len(records)} requests", flush=True)
    return [result_path]


def recover_or_cancel_batch_records(
    directory: Path,
    client: OpenAIBatchClient,
    label: str,
) -> list[Path] | None:
    """Reuse a completed legacy batch, or cancel it before synchronous replacement."""
    state_path = directory / "state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    result_paths = []
    all_completed = True
    for index, chunk in enumerate(state.get("chunks", [])):
        batch_id = chunk.get("batch_id")
        if not batch_id:
            all_completed = False
            continue
        batch = client.retrieve_batch(batch_id)
        status = batch["status"]
        if status == "completed":
            result_path = directory / f"results_{index:03d}.jsonl"
            if not result_path.exists():
                result_path.write_bytes(client.download(batch["output_file_id"]))
            chunk.update({"result_file": str(result_path), "status": "completed"})
            result_paths.append(result_path)
        else:
            all_completed = False
            if status not in {"failed", "expired", "cancelled", "cancelling"}:
                client.cancel_batch(batch_id)
                chunk["status"] = "cancelling"
                print(f"Cancelled queued {label} batch {batch_id}", flush=True)
    _write_json(state_path, state)
    if all_completed and len(result_paths) == len(state.get("chunks", [])):
        print(f"Reusing completed legacy {label} batch", flush=True)
        return result_paths
    return None
