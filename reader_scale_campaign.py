"""Full-CWQ small-reader versus GPT-4o-mini evidence-interface campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from inverse_verifier.openai_naturalize import OpenAIBatchClient, run_batch_records
from subgraph_organizer import branch_assembly_lines, extract_triple_lines, replace_triples
from subgraph_reader_pilot import (
    _bootstrap_delta,
    _mean,
    _presentation_metadata,
    _prompt_tokens,
    _load_tokenizer,
    _run_named_suite,
    _triple_multiset,
    _cwq_metadata,
    build_conversation,
    first_answer_rank,
    needs_follow_up,
    read_jsonl,
    score_prediction,
    transform_row,
    write_jsonl,
)


LOCAL_ARMS = ("original", "reorder", "branch_grouped")
OPENAI_ARMS = ("reorder", "branch_grouped")
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"


def _branch_row(row: dict) -> tuple[dict, dict]:
    lines, metadata = branch_assembly_lines(
        row["question"],
        extract_triple_lines(row["user_query"]),
        surface_junctions=False,
    )
    transformed = dict(row)
    transformed["user_query"] = replace_triples(row["user_query"], lines)
    transformed["all_query"] = replace_triples(row["all_query"], lines)
    return transformed, metadata


def prepare_full_campaign(
    source_path: Path,
    gpt_source_path: Path,
    official_path: Path,
    output: Path,
    tokenizer_path: Path | None,
) -> dict:
    source_rows = read_jsonl(source_path)
    gpt_rows = {row["id"]: row for row in read_jsonl(gpt_source_path)}
    official = _cwq_metadata(official_path)
    tokenizer = _load_tokenizer(tokenizer_path)
    arms: dict[str, list[dict]] = {arm: [] for arm in LOCAL_ARMS}
    gpt_original = []
    metadata_rows = []

    for source in source_rows:
        if source["id"] not in official or source["id"] not in gpt_rows:
            continue
        original = dict(source)
        original["released_llama8b_prediction"] = original.pop("prediction", None)
        kind = official[source["id"]].get("compositionality_type")
        original["pilot_bucket"] = f"cwq_{kind}"
        original["answer_evidence_rank"] = first_answer_rank(original)
        reorder = transform_row(original, structured=False)
        branch_grouped, branch_info = _branch_row(original)
        raw = _triple_multiset(original)
        if any(_triple_multiset(row) != raw for row in (reorder, branch_grouped)):
            raise AssertionError(f"triple preservation failed for {original['id']}")

        arms["original"].append(original)
        arms["reorder"].append(reorder)
        arms["branch_grouped"].append(branch_grouped)
        baseline = dict(original)
        baseline["prediction"] = gpt_rows[original["id"]]["prediction"]
        baseline["released_model"] = OPENAI_MODEL
        gpt_original.append(baseline)

        presentation = _presentation_metadata(original, official[source["id"]])
        metadata_rows.append(
            {
                "id": original["id"],
                **presentation,
                **branch_info,
                "prompt_tokens": {
                    arm: _prompt_tokens(tokenizer, row["user_query"])
                    for arm, row in (
                        ("original", original),
                        ("reorder", reorder),
                        ("branch_grouped", branch_grouped),
                    )
                },
            }
        )

    inputs = output / "inputs"
    for arm, rows in arms.items():
        write_jsonl(inputs / f"{arm}.jsonl", rows)
    write_jsonl(inputs / "metadata.jsonl", metadata_rows)
    write_jsonl(output / "gpt4o_mini" / "runs" / "original.jsonl", gpt_original)
    type_counts = Counter(row["cwq_type"] for row in metadata_rows)
    manifest = {
        "source": str(source_path),
        "gpt_source": str(gpt_source_path),
        "official": str(official_path),
        "rows": len(metadata_rows),
        "type_counts": dict(type_counts),
        "branchable_rows": sum(row["branchable"] for row in metadata_rows),
        "arms": list(LOCAL_ARMS),
        "openai_model": OPENAI_MODEL,
        "gold_usage": "evaluation and structural slicing only; no gold enters prompt transformation",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_local_campaign(
    inputs: Path,
    output: Path,
    model: Path,
    batch_size: int,
    tensor_parallel_size: int,
) -> dict:
    return _run_named_suite(
        inputs,
        output,
        model,
        batch_size,
        tensor_parallel_size,
        None,
        LOCAL_ARMS,
    )


def _batch_record(row: dict, arm: str, follow_up: bool) -> dict:
    suffix = "retry" if follow_up else "primary"
    return {
        "custom_id": _request_key(arm, row["id"], suffix),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": OPENAI_MODEL,
            "messages": build_conversation(row, follow_up=follow_up),
            "temperature": 0,
            "max_tokens": 4_000,
            "frequency_penalty": 0.16,
        },
    }


def _request_key(arm: str, question_id: str, suffix: str) -> str:
    digest = hashlib.sha256(question_id.encode()).hexdigest()[:24]
    return f"{arm[:3]}-{digest}-{suffix[:1]}"


def _batch_predictions(paths: Iterable[Path]) -> dict[str, dict]:
    predictions = {}
    for path in paths:
        for result in read_jsonl(path):
            response = result.get("response") or {}
            body = response.get("body") or {}
            if response.get("status_code") != 200 or not body.get("choices"):
                raise RuntimeError(f"OpenAI request failed: {result.get('custom_id')}")
            predictions[result["custom_id"]] = {
                "prediction": body["choices"][0]["message"]["content"],
                "usage": body.get("usage") or {},
            }
    return predictions


def run_openai_campaign(inputs: Path, output: Path) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAIBatchClient(api_key)
    summary = {"model": OPENAI_MODEL, "arms": {}}

    for arm in OPENAI_ARMS:
        rows = read_jsonl(inputs / f"{arm}.jsonl")
        primary_records = [_batch_record(row, arm, False) for row in rows]
        primary_files = run_batch_records(
            primary_records,
            output / ".batch" / arm / "primary",
            client,
            f"full CWQ {arm} primary",
        )
        primary = _batch_predictions(primary_files)
        retry_rows = [
            row
            for row in rows
            if needs_follow_up(primary[_request_key(arm, row["id"], "primary")]["prediction"])
        ]
        retry = {}
        if retry_rows:
            retry_files = run_batch_records(
                [_batch_record(row, arm, True) for row in retry_rows],
                output / ".batch" / arm / "retry",
                client,
                f"full CWQ {arm} formatting retry",
            )
            retry = _batch_predictions(retry_files)

        output_rows = []
        for row in rows:
            key = _request_key(arm, row["id"], "primary")
            chosen = primary[key]
            retry_key = _request_key(arm, row["id"], "retry")
            retried = retry_key in retry
            if retried:
                chosen = retry[retry_key]
            result = dict(row)
            result["prediction"] = chosen["prediction"]
            result["follow_up_used"] = retried
            result["total_generation_tokens"] = (
                primary[key]["usage"].get("completion_tokens", 0)
                + (retry.get(retry_key, {}).get("usage") or {}).get("completion_tokens", 0)
            )
            output_rows.append(result)
        write_jsonl(output / "runs" / f"{arm}.jsonl", output_rows)
        summary["arms"][arm] = {"rows": len(rows), "formatting_retries": len(retry_rows)}

    (output / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _score_rows(run_dir: Path, arms: Iterable[str]) -> dict[str, dict[str, dict]]:
    return {
        arm: {row["id"]: row for row in read_jsonl(run_dir / f"{arm}.jsonl")}
        for arm in arms
    }


def evaluate_full_campaign(
    local_runs: Path,
    gpt_runs: Path,
    metadata_path: Path,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    models = {
        "llama32_3b": _score_rows(local_runs, LOCAL_ARMS),
        "gpt4o_mini": _score_rows(gpt_runs, LOCAL_ARMS),
    }
    metadata = {row["id"]: row for row in read_jsonl(metadata_path)}
    ids = list(metadata)
    for model_rows in models.values():
        if any(set(rows) != set(ids) for rows in model_rows.values()):
            raise ValueError("campaign run IDs do not match metadata IDs")

    scored = []
    for question_id in ids:
        row = {"id": question_id, **metadata[question_id], "scores": {}}
        for model, model_rows in models.items():
            row["scores"][model] = {
                arm: score_prediction(model_rows[arm][question_id], model_rows[arm][question_id]["prediction"])
                for arm in LOCAL_ARMS
            }
        scored.append(row)

    slices = {
        "overall": scored,
        "answer_endpoint_present": [row for row in scored if row["answer_endpoint_present"]],
        "answer_endpoint_absent": [row for row in scored if not row["answer_endpoint_present"]],
        "branchable": [row for row in scored if row["branchable"]],
        "conjunction_branchable": [
            row for row in scored if row["cwq_type"] == "conjunction" and row["branchable"]
        ],
    }
    for kind in sorted({row["cwq_type"] for row in scored}):
        slices[f"cwq_{kind}"] = [row for row in scored if row["cwq_type"] == kind]

    metrics = {
        "questions": len(scored),
        "bootstrap_samples": bootstrap_samples,
        "primary_hypothesis": (
            "evidence organization improves Llama 3.2 3B more than GPT-4o-mini"
        ),
        "slices": {},
    }
    for slice_name, rows in slices.items():
        if not rows:
            continue
        result = {"questions": len(rows), "models": {}, "interactions": {}, "model_gaps": {}}
        for model in models:
            result["models"][model] = {}
            for arm in LOCAL_ARMS:
                result["models"][model][arm] = {
                    metric: _mean([row["scores"][model][arm][metric] for row in rows])
                    for metric in ("hit_at_1", "f1", "no_answer")
                }
            result["models"][model]["effects"] = {}
            for label, arm in (("reorder_minus_original", "reorder"), ("branch_minus_original", "branch_grouped"), ("branch_minus_reorder", "branch_grouped")):
                baseline = "reorder" if label == "branch_minus_reorder" else "original"
                result["models"][model]["effects"][label] = {
                    metric: _bootstrap_delta(
                        [row["scores"][model][arm][metric] for row in rows],
                        [row["scores"][model][baseline][metric] for row in rows],
                        seed,
                        bootstrap_samples,
                    )
                    for metric in ("hit_at_1", "f1")
                }

        for label, arm in (("reorder", "reorder"), ("branch", "branch_grouped")):
            result["interactions"][f"{label}_effect_3b_minus_gpt"] = {}
            for metric in ("hit_at_1", "f1"):
                small_effect = [
                    row["scores"]["llama32_3b"][arm][metric]
                    - row["scores"]["llama32_3b"]["original"][metric]
                    for row in rows
                ]
                gpt_effect = [
                    row["scores"]["gpt4o_mini"][arm][metric]
                    - row["scores"]["gpt4o_mini"]["original"][metric]
                    for row in rows
                ]
                result["interactions"][f"{label}_effect_3b_minus_gpt"][metric] = _bootstrap_delta(
                    small_effect, gpt_effect, seed, bootstrap_samples
                )
        for arm in LOCAL_ARMS:
            result["model_gaps"][arm] = {
                metric: _bootstrap_delta(
                    [row["scores"]["gpt4o_mini"][arm][metric] for row in rows],
                    [row["scores"]["llama32_3b"][arm][metric] for row in rows],
                    seed,
                    bootstrap_samples,
                )
                for metric in ("hit_at_1", "f1")
            }
        metrics["slices"][slice_name] = result

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_jsonl(output / "paired_diagnostics.jsonl", scored)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--gpt-source", type=Path, required=True)
    prepare.add_argument("--official", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path)
    local = commands.add_parser("run-local")
    local.add_argument("--inputs", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--model", type=Path, required=True)
    local.add_argument("--batch-size", type=int, default=8)
    local.add_argument("--tensor-parallel-size", type=int, default=1)
    openai = commands.add_parser("run-openai")
    openai.add_argument("--inputs", type=Path, required=True)
    openai.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--local-runs", type=Path, required=True)
    evaluate.add_argument("--gpt-runs", type=Path, required=True)
    evaluate.add_argument("--metadata", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_full_campaign(args.source, args.gpt_source, args.official, args.output, args.tokenizer)
    elif args.command == "run-local":
        result = run_local_campaign(args.inputs, args.output, args.model, args.batch_size, args.tensor_parallel_size)
    elif args.command == "run-openai":
        result = run_openai_campaign(args.inputs, args.output)
    else:
        result = evaluate_full_campaign(
            args.local_runs,
            args.gpt_runs,
            args.metadata,
            args.output,
            args.bootstrap_samples,
            args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
