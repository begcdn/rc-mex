"""Controlled reader pilot for losslessly organized SubgraphRAG evidence."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from subgraph_organizer import (
    _parse_triple,
    adjacency_graph_lines,
    decode_adjacency_graph,
    extract_triple_lines,
    replace_triples,
    transform_row,
)


# Copied from the published SubgraphRAG reason/prompts.py. Keeping the original
# demonstration fixed is part of the matched reader comparison.
ICL_USER_PROMPT = """Triplets:
(Lou Seal,sports.mascot.team,San Francisco Giants)
(San Francisco Giants,sports.sports_team.championships,2012 World Series)
(San Francisco Giants,sports.sports_championship_event.champion,2014 World Series)
(San Francisco Giants,time.participant.event,2014 Major League Baseball season)
(San Francisco Giants,time.participant.event,2010 World Series)
(San Francisco Giants,time.participant.event,2010 Major League Baseball season)
(San Francisco Giants,sports.sports_team.championships,2014 World Series)
(San Francisco Giants,sports.sports_team.team_mascot,Crazy Crab)
(San Francisco Giants,sports.sports_team.championships,2010 World Series)
(San Francisco Giants,sports.professional_sports_team.owner_s,Bill Neukom)
(San Francisco Giants,time.participant.event,2012 World Series)
(San Francisco,sports.sports_team_location.teams,San Francisco Giants)
(San Francisco Giants,sports.sports_team.arena_stadium,AT&T Park)
(AT&T Park,location.location.events,2012 World Series)
(m.011zsc4_,organization.leadership.organization,San Francisco Giants)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Giants)
(AT&T Park,location.location.events,2010 World Series)
(Crazy Crab,sports.mascot.team,San Francisco Giants)
(New York Giants,baseball.baseball_team.league,National League)
(San Francisco Giants,sports.sports_team.colors,Black)
(San Francisco Giants,sports.sports_team.previously_known_as,New York Gothams)
(m.0k079qm,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(m.0k079ry,base.schemastaging.team_training_ground_relationship.team,San Francisco Giants)
(2010 World Series,time.event.locations,AT&T Park)
(San Francisco Giants,time.participant.event,2012 Major League Baseball season)
(San Francisco Giants,baseball.baseball_team.league,National League)
(m.0crtd80,sports.sports_league_participation.league,National League West)
(San Francisco Giants,sports.sports_team.location,San Francisco)
(San Francisco Giants,sports.sports_team.sport,Baseball)
(m.05n6dtn,baseball.baseball_team_stats.team,San Francisco Giants)


Question:
What year did the team with mascot named Lou Seal win the World Series?"""

ICL_ASSISTANT_PROMPT = """To find the year the team with mascot named Lou Seal won the World Series, we need to find the team with mascot named Lou Seal and then find the year they won the World Series.

From the triplets, we can see that Lou Seal is the mascot of the San Francisco Giants.

Now, we need to find the year the San Francisco Giants won the World Series.

From the triplets, we can see that San Francisco Giants won the 2010 World Series and 2012 World Series and 2014 World Series.

So, the team with mascot named Lou Seal (San Francisco Giants) won the World Series in 2010, 2012, and 2014.

Therefore, the formatted answers are:

ans: 2014 (2014 World Series)
ans: 2012 (2012 World Series)
ans: 2010 (2010 World Series)"""

ARM_NAMES = ("original", "reorder", "structured")
GRAPH_ARM_NAMES = ("adjacency_flat", "adjacency_graph")
ALL_ARM_NAMES = ARM_NAMES + GRAPH_ARM_NAMES


def _prompt_evidence_lines(prompt: str) -> list[str]:
    body = prompt.split("Triplets:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    return body.splitlines()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def first_answer_rank(row: dict) -> int | None:
    """Rank of the first triple whose endpoint exactly names a gold entity."""
    answers = row.get("a_entity") or row.get("ground_truth") or []
    normalized_answers = {normalize(str(answer)) for answer in answers if str(answer).strip()}
    for rank, line in enumerate(extract_triple_lines(row["user_query"]), start=1):
        triple = _parse_triple(line, rank - 1)
        if normalize(triple.head) in normalized_answers or normalize(triple.tail) in normalized_answers:
            return rank
    return None


def select_pilot_rows(rows: Sequence[dict], per_slice: int, seed: int) -> list[dict]:
    ranked = [(index, row, first_answer_rank(row)) for index, row in enumerate(rows)]
    shallow = [item for item in ranked if item[2] is not None and 1 <= item[2] <= 10]
    deep = [item for item in ranked if item[2] is not None and 51 <= item[2] <= 100]
    count = min(per_slice, len(shallow), len(deep))
    if count == 0:
        raise ValueError("no examples available in one or both evidence-depth slices")

    rng = random.Random(seed)
    selected = rng.sample(shallow, count) + rng.sample(deep, count)
    selected.sort(key=lambda item: item[0])
    output = []
    for _, source, rank in selected:
        row = dict(source)
        row["released_prediction"] = row.pop("prediction", None)
        row["pilot_bucket"] = "shallow_1_10" if rank <= 10 else "deep_51_100"
        row["answer_evidence_rank"] = rank
        output.append(row)
    return output


def _presented_triple_lines(prompt: str) -> list[str]:
    body = prompt.split("Triplets:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    lines = body.splitlines()
    invalid = [
        line
        for line in lines
        if not (line.startswith("(") and line.endswith(")"))
        and not (line.startswith("[") and line.endswith("]"))
    ]
    if invalid:
        raise ValueError(f"unexpected non-triple presentation lines: {invalid[:3]}")
    return [line for line in lines if line.startswith("(") and line.endswith(")")]


def _triple_multiset(row: dict) -> Counter[str]:
    return Counter(_presented_triple_lines(row["user_query"]))


def prepare_pilot(source: Path, output: Path, per_slice: int, seed: int) -> dict:
    selected = select_pilot_rows(read_jsonl(source), per_slice, seed)
    arms: dict[str, list[dict]] = {name: [] for name in ARM_NAMES}
    for original in selected:
        reorder = transform_row(original, structured=False)
        structured = transform_row(original, structured=True)
        raw = _triple_multiset(original)
        if _triple_multiset(reorder) != raw or _triple_multiset(structured) != raw:
            raise AssertionError(f"triple preservation failed for {original['id']}")
        reorder_flat = extract_triple_lines(reorder["user_query"])
        structured_flat = _presented_triple_lines(structured["user_query"])
        if reorder_flat != structured_flat:
            raise AssertionError(f"the two organized arms disagree for {original['id']}")
        arms["original"].append(original)
        arms["reorder"].append(reorder)
        arms["structured"].append(structured)

    for arm, arm_rows in arms.items():
        write_jsonl(output / "inputs" / f"{arm}.jsonl", arm_rows)
    counts = Counter(row["pilot_bucket"] for row in selected)
    manifest = {
        "source": str(source),
        "seed": seed,
        "requested_per_slice": per_slice,
        "rows": len(selected),
        "slice_counts": dict(counts),
        "arms": list(ARM_NAMES),
        "gold_usage": "offline sample selection and evaluation only",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _sparql_relations(sparql: str) -> list[str]:
    pattern = (
        r"(?:\?\w+|ns:[A-Za-z0-9_#.-]+)\s+"
        r"ns:([A-Za-z0-9_#.-]+)\s+"
        r"(?:\?\w+|ns:[A-Za-z0-9_#.-]+|['\"])")
    return [match.group(1) for match in re.finditer(pattern, sparql)]


def _cwq_metadata(official_path: Path) -> dict[str, dict]:
    rows = json.loads(official_path.read_text(encoding="utf-8"))
    return {row["ID"]: row for row in rows}


def _presentation_metadata(row: dict, official: dict | None) -> dict:
    triples = [
        _parse_triple(line, index)
        for index, line in enumerate(extract_triple_lines(row["user_query"]))
    ]
    retrieved_relations = {triple.relation for triple in triples}
    gold_relations = _sparql_relations(official["sparql"]) if official else []
    relation_complete = bool(gold_relations) and set(gold_relations).issubset(retrieved_relations)
    answer_present = first_answer_rank(row) is not None
    return {
        "cwq_type": official.get("compositionality_type") if official else None,
        "gold_relation_count": len(gold_relations),
        "gold_relations": gold_relations,
        "gold_relations_present": relation_complete,
        "answer_endpoint_present": answer_present,
        # This is only a diagnostic proxy. Matching relation names and an
        # answer endpoint do not prove that the required connected proof exists.
        "evidence_proxy_complete": relation_complete and answer_present,
    }


def _load_tokenizer(path: Path | None):
    if path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _prompt_tokens(tokenizer, prompt: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def _adjacency_row(row: dict, *, organize_groups: bool) -> dict:
    raw_lines = extract_triple_lines(row["user_query"])
    adjacency_lines = adjacency_graph_lines(
        row["question"], raw_lines, organize_groups=organize_groups
    )
    transformed = dict(row)
    transformed["user_query"] = replace_triples(row["user_query"], adjacency_lines)
    transformed["all_query"] = replace_triples(row["all_query"], adjacency_lines)
    return transformed


def _graph_variants(source: dict) -> tuple[dict, dict]:
    raw = [
        (triple.head, triple.relation, triple.tail)
        for index, line in enumerate(extract_triple_lines(source["user_query"]))
        for triple in [_parse_triple(line, index)]
    ]
    adjacency_flat = _adjacency_row(source, organize_groups=False)
    adjacency_graph = _adjacency_row(source, organize_groups=True)
    flat_decoded = decode_adjacency_graph(
        _prompt_evidence_lines(adjacency_flat["user_query"])
    )
    graph_decoded = decode_adjacency_graph(
        _prompt_evidence_lines(adjacency_graph["user_query"])
    )
    if Counter(flat_decoded) != Counter(raw) or Counter(graph_decoded) != Counter(raw):
        raise AssertionError(f"adjacency graph lost evidence for {source['id']}")
    if Counter(_prompt_evidence_lines(adjacency_flat["user_query"])[1:]) != Counter(
        _prompt_evidence_lines(adjacency_graph["user_query"])[1:]
    ):
        raise AssertionError(f"adjacency arms differ by more than order for {source['id']}")
    return adjacency_flat, adjacency_graph


def prepare_graph_arms(
    original_path: Path,
    output: Path,
    official_cwq: Path,
    tokenizer_path: Path | None,
) -> dict:
    """Add matched flat and graph-ordered adjacency arms to an existing pilot."""
    original_rows = read_jsonl(original_path)
    official = _cwq_metadata(official_cwq)
    tokenizer = _load_tokenizer(tokenizer_path)
    arm_rows: dict[str, list[dict]] = {name: [] for name in GRAPH_ARM_NAMES}
    metadata_rows = []

    for source in original_rows:
        adjacency_flat, adjacency_graph = _graph_variants(source)
        arm_rows["adjacency_flat"].append(adjacency_flat)
        arm_rows["adjacency_graph"].append(adjacency_graph)
        row_metadata = {"id": source["id"], **_presentation_metadata(source, official.get(source["id"]))}
        row_metadata["prompt_tokens"] = {
            "original": _prompt_tokens(tokenizer, source["user_query"]),
            "adjacency_flat": _prompt_tokens(tokenizer, adjacency_flat["user_query"]),
            "adjacency_graph": _prompt_tokens(tokenizer, adjacency_graph["user_query"]),
        }
        metadata_rows.append(row_metadata)

    output.mkdir(parents=True, exist_ok=True)
    for arm, rows in arm_rows.items():
        write_jsonl(output / f"{arm}.jsonl", rows)
    write_jsonl(output / "graph_metadata.jsonl", metadata_rows)
    type_counts = Counter(row["cwq_type"] for row in metadata_rows)
    proxy_counts = Counter(row["evidence_proxy_complete"] for row in metadata_rows)
    manifest = {
        "source": str(original_path),
        "official_cwq": str(official_cwq),
        "tokenizer": str(tokenizer_path) if tokenizer_path else None,
        "rows": len(original_rows),
        "arms": list(GRAPH_ARM_NAMES),
        "cwq_type_counts": dict(type_counts),
        "evidence_proxy_counts": {str(key).lower(): value for key, value in proxy_counts.items()},
        "evidence_proxy_warning": "relation coverage plus answer endpoint; not proof of a connected gold derivation",
    }
    (output / "graph_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_structure_pilot(
    source_path: Path,
    output: Path,
    official_cwq: Path,
    tokenizer_path: Path | None,
    per_type: int,
    seed: int,
) -> dict:
    """Build a balanced composition/conjunction reader gate from released outputs."""
    official = _cwq_metadata(official_cwq)
    tokenizer = _load_tokenizer(tokenizer_path)
    eligible: dict[str, list[tuple[dict, dict]]] = {
        "composition": [],
        "conjunction": [],
    }
    for source in read_jsonl(source_path):
        official_row = official.get(source["id"])
        if not official_row:
            continue
        info = _presentation_metadata(source, official_row)
        kind = info["cwq_type"]
        if kind in eligible and info["evidence_proxy_complete"]:
            eligible[kind].append((source, info))
    if any(len(rows) < per_type for rows in eligible.values()):
        raise ValueError(
            f"requested {per_type} per type but found "
            + ", ".join(f"{kind}={len(rows)}" for kind, rows in eligible.items())
        )

    rng = random.Random(seed)
    selected = []
    for kind, rows in eligible.items():
        selected.extend(rng.sample(rows, per_type))
    selected.sort(key=lambda item: item[0]["id"])

    arms: dict[str, list[dict]] = {arm: [] for arm in ALL_ARM_NAMES}
    metadata_rows = []
    for source, info in selected:
        original = dict(source)
        original["released_prediction"] = original.pop("prediction", None)
        original["pilot_bucket"] = f"cwq_{info['cwq_type']}"
        original["answer_evidence_rank"] = first_answer_rank(original)
        reorder = transform_row(original, structured=False)
        structured = transform_row(original, structured=True)
        adjacency_flat, adjacency_graph = _graph_variants(original)

        raw = _triple_multiset(original)
        if _triple_multiset(reorder) != raw or _triple_multiset(structured) != raw:
            raise AssertionError(f"triple preservation failed for {original['id']}")
        arms["original"].append(original)
        arms["reorder"].append(reorder)
        arms["structured"].append(structured)
        arms["adjacency_flat"].append(adjacency_flat)
        arms["adjacency_graph"].append(adjacency_graph)

        metadata_rows.append(
            {
                "id": original["id"],
                **info,
                "prompt_tokens": {
                    "original": _prompt_tokens(tokenizer, original["user_query"]),
                    "adjacency_flat": _prompt_tokens(tokenizer, adjacency_flat["user_query"]),
                    "adjacency_graph": _prompt_tokens(tokenizer, adjacency_graph["user_query"]),
                },
            }
        )

    inputs = output / "inputs"
    for arm, rows in arms.items():
        write_jsonl(inputs / f"{arm}.jsonl", rows)
    write_jsonl(inputs / "graph_metadata.jsonl", metadata_rows)
    manifest = {
        "source": str(source_path),
        "official_cwq": str(official_cwq),
        "tokenizer": str(tokenizer_path) if tokenizer_path else None,
        "seed": seed,
        "requested_per_type": per_type,
        "rows": len(selected),
        "arms": list(ALL_ARM_NAMES),
        "available_evidence_proxy_complete": {
            kind: len(rows) for kind, rows in eligible.items()
        },
        "selected_type_counts": dict(Counter(info["cwq_type"] for _, info in selected)),
        "gold_usage": "offline sample selection and structural evaluation only",
        "evidence_proxy_warning": "relation coverage plus answer endpoint; not proof of a connected gold derivation",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_conversation(row: dict, *, follow_up: bool = False) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": row["sys_query"]},
        {"role": "user", "content": ICL_USER_PROMPT},
        {"role": "assistant", "content": ICL_ASSISTANT_PROMPT},
        {"role": "user", "content": row["user_query"]},
    ]
    if follow_up:
        # This intentionally mirrors SubgraphRAG's dc branch: the formatting
        # request is appended without inserting the first model response.
        conversation.append({"role": "user", "content": row["cot_query"]})
    return conversation


def needs_follow_up(prediction: str) -> bool:
    lowered = prediction.lower()
    return (
        "ans:" not in lowered
        or "ans: not available" in lowered
        or "ans: no information available" in lowered
    )


def _load_vllm(model_path: Path, tensor_parallel_size: int):
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required to reproduce SubgraphRAG inference") from exc

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        max_seq_len_to_capture=16_384,
    )
    params = SamplingParams(temperature=0, max_tokens=4_000, frequency_penalty=0.16)
    return llm, params, vllm.__version__


def _run_with_engine(
    input_path: Path,
    output_path: Path,
    llm,
    params,
    batch_size: int,
    limit: int | None,
) -> dict:
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    input_ids = [row["id"] for row in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("input contains duplicate question ids")
    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    completed = {row["id"] for row in completed_rows}
    if len(completed_rows) != len(completed):
        raise ValueError("output checkpoint contains duplicate question ids")
    if not completed.issubset(input_ids):
        raise ValueError("output checkpoint contains ids absent from the input")
    pending = [row for row in rows if row["id"] not in completed]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            conversations = [build_conversation(row) for row in batch]
            results = llm.chat(messages=conversations, sampling_params=params, use_tqdm=False)
            predictions = [result.outputs[0].text for result in results]

            retry_indices = [index for index, text in enumerate(predictions) if needs_follow_up(text)]
            if retry_indices:
                retry_conversations = [
                    build_conversation(batch[index], follow_up=True) for index in retry_indices
                ]
                retries = llm.chat(
                    messages=retry_conversations,
                    sampling_params=params,
                    use_tqdm=False,
                )
                for index, result in zip(retry_indices, retries):
                    predictions[index] = result.outputs[0].text

            for row, prediction in zip(batch, predictions):
                result_row = dict(row)
                result_row["prediction"] = prediction
                output.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            output.flush()
            done = len(completed) + min(start + len(batch), len(pending))
            print(f"{input_path.stem}: {done}/{len(rows)}", flush=True)

    return {"input": str(input_path), "output": str(output_path), "rows": len(rows)}


def run_inference(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
) -> dict:
    llm, params, version = _load_vllm(model_path, tensor_parallel_size)
    result = _run_with_engine(input_path, output_path, llm, params, batch_size, limit)
    result["model"] = str(model_path)
    result["vllm"] = version
    return result


def _run_named_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
    arms: Sequence[str],
) -> dict:
    """Run every presentation arm while retaining one GPU/model allocation."""
    llm, params, version = _load_vllm(model_path, tensor_parallel_size)
    results = {}
    for arm in arms:
        results[arm] = _run_with_engine(
            inputs / f"{arm}.jsonl",
            output / f"{arm}.jsonl",
            llm,
            params,
            batch_size,
            limit,
        )
    manifest = {
        "model": str(model_path),
        "vllm": version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sampling": {"temperature": 0, "max_tokens": 4_000, "frequency_penalty": 0.16},
        "tensor_parallel_size": tensor_parallel_size,
        "batch_size": batch_size,
        "limit": limit,
        "arms": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
) -> dict:
    return _run_named_suite(
        inputs,
        output,
        model_path,
        batch_size,
        tensor_parallel_size,
        limit,
        ARM_NAMES,
    )


def run_graph_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
) -> dict:
    return _run_named_suite(
        inputs,
        output,
        model_path,
        batch_size,
        tensor_parallel_size,
        limit,
        GRAPH_ARM_NAMES,
    )


def run_all_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
) -> dict:
    return _run_named_suite(
        inputs,
        output,
        model_path,
        batch_size,
        tensor_parallel_size,
        limit,
        ALL_ARM_NAMES,
    )


def select_idle_gpu(query_output: str, max_used_mib: int) -> int | None:
    candidates = []
    for line in query_output.splitlines():
        if not line.strip():
            continue
        index_text, used_text = (field.strip() for field in line.split(",", 1))
        index, used = int(index_text), int(used_text)
        if used <= max_used_mib:
            candidates.append((used, index))
    return min(candidates)[1] if candidates else None


def wait_for_idle_gpu(max_used_mib: int, poll_seconds: int) -> int:
    while True:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpu = select_idle_gpu(query, max_used_mib)
        if gpu is not None:
            print(f"Claiming physical GPU {gpu}", flush=True)
            return gpu
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "waiting for an idle GPU", flush=True)
        time.sleep(poll_seconds)


def wait_and_run_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
    max_used_mib: int,
    poll_seconds: int,
) -> dict:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES so the queue can inspect every GPU")
    gpu = wait_for_idle_gpu(max_used_mib, poll_seconds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_suite(inputs, output, model_path, batch_size, tensor_parallel_size, limit)


def wait_and_run_graph_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
    max_used_mib: int,
    poll_seconds: int,
) -> dict:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES so the queue can inspect every GPU")
    gpu = wait_for_idle_gpu(max_used_mib, poll_seconds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_graph_suite(inputs, output, model_path, batch_size, tensor_parallel_size, limit)


def wait_and_run_all_suite(
    inputs: Path,
    output: Path,
    model_path: Path,
    batch_size: int,
    tensor_parallel_size: int,
    limit: int | None,
    max_used_mib: int,
    poll_seconds: int,
) -> dict:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES so the queue can inspect every GPU")
    gpu = wait_for_idle_gpu(max_used_mib, poll_seconds)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return run_all_suite(inputs, output, model_path, batch_size, tensor_parallel_size, limit)


def _remove_duplicates(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def extract_predictions(prediction: str) -> list[str]:
    lines = [line for line in prediction.split("\n") if "ans:" in line and "none" not in line.lower()]
    return _remove_duplicates(
        line
        for line in lines
        if "ans: not available" not in line.lower()
        and "ans: no information available" not in line.lower()
    )


def _matches(container: str, answer: str) -> bool:
    return normalize(answer) in normalize(container)


def score_prediction(row: dict, prediction: str) -> dict[str, float]:
    emitted = extract_predictions(prediction)
    predicted = sorted(emitted, key=len, reverse=True)
    answers = sorted(_remove_duplicates(row["ground_truth"]), key=len, reverse=True)
    if "when" in row["question"].lower() or "what year" in row["question"].lower():
        answers = [
            answer.split("-", 1)[0]
            if "-" in answer and answer.split("-", 1)[0].isdigit()
            else answer
            for answer in answers
        ]
    double_check = any(
        keyword in row["question"].lower()
        for keyword in (
            "when",
            "what year",
            "which year",
            "where",
            "sport",
            "what countr",
            "language",
            "nba finals",
            "world series",
        )
    )

    unmatched = list(predicted)
    matched = 0
    for answer in answers:
        for candidate in unmatched:
            candidate_answer = candidate.split("ans:")[-1].strip()
            if _matches(candidate, answer) or (
                double_check
                and (_matches(answer, candidate_answer) or _matches(answer, candidate))
            ):
                matched += 1
                unmatched.remove(candidate)
                break
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(answers) if answers else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    hit = 0.0
    if emitted:
        first = emitted[0]
        first_answer = first.split("ans:")[-1].strip()
        hit = float(
            any(
                _matches(first, answer)
                or (double_check and _matches(answer, first_answer))
                for answer in answers
            )
        )
    return {"hit_at_1": hit, "f1": f1, "no_answer": float(not predicted)}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_delta(
    left: Sequence[float], right: Sequence[float], seed: int, samples: int
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs must have the same nonzero length")
    deltas = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    boot = sorted(
        _mean([deltas[rng.randrange(len(deltas))] for _ in deltas])
        for _ in range(samples)
    )
    lower = boot[int(0.025 * samples)]
    upper = boot[min(samples - 1, int(0.975 * samples))]
    return {"delta": _mean(deltas), "ci95_low": lower, "ci95_high": upper}


def evaluate_runs(run_dir: Path, output: Path, bootstrap_samples: int, seed: int) -> dict:
    arm_rows = {
        arm: {row["id"]: row for row in read_jsonl(run_dir / f"{arm}.jsonl")}
        for arm in ARM_NAMES
    }
    ids = list(arm_rows["original"])
    if any(set(rows) != set(ids) for rows in arm_rows.values()):
        raise ValueError("the three inference arms do not contain identical question ids")

    scored_rows = []
    for question_id in ids:
        reference = arm_rows["original"][question_id]
        scores = {
            arm: score_prediction(arm_rows[arm][question_id], arm_rows[arm][question_id]["prediction"])
            for arm in ARM_NAMES
        }
        released = reference.get("released_prediction")
        scored_rows.append(
            {
                "id": question_id,
                "bucket": reference["pilot_bucket"],
                "answer_evidence_rank": reference["answer_evidence_rank"],
                "scores": scores,
                "released_scores": score_prediction(reference, released) if released else None,
            }
        )

    slices = {
        "overall": scored_rows,
        "shallow_1_10": [row for row in scored_rows if row["bucket"] == "shallow_1_10"],
        "deep_51_100": [row for row in scored_rows if row["bucket"] == "deep_51_100"],
    }
    metrics: dict = {
        "questions": len(scored_rows),
        "bootstrap_samples": bootstrap_samples,
        "slices": {},
    }
    comparisons = (
        ("reorder_minus_original", "reorder", "original"),
        ("structured_minus_original", "structured", "original"),
        ("structured_minus_reorder", "structured", "reorder"),
    )
    for slice_name, rows in slices.items():
        slice_result: dict = {"questions": len(rows), "arms": {}, "paired_differences": {}}
        for arm in ARM_NAMES:
            slice_result["arms"][arm] = {
                metric: _mean([row["scores"][arm][metric] for row in rows])
                for metric in ("hit_at_1", "f1", "no_answer")
            }
        for label, left_arm, right_arm in comparisons:
            slice_result["paired_differences"][label] = {
                metric: _bootstrap_delta(
                    [row["scores"][left_arm][metric] for row in rows],
                    [row["scores"][right_arm][metric] for row in rows],
                    seed,
                    bootstrap_samples,
                )
                for metric in ("hit_at_1", "f1")
            }
        metrics["slices"][slice_name] = slice_result

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_jsonl(output / "paired_diagnostics.jsonl", scored_rows)
    return metrics


def evaluate_graph_runs(
    run_dir: Path,
    metadata_path: Path,
    output: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Evaluate the graph interface with structural and evidence diagnostics."""
    arm_rows = {
        arm: {row["id"]: row for row in read_jsonl(run_dir / f"{arm}.jsonl")}
        for arm in ALL_ARM_NAMES
    }
    ids = list(arm_rows["original"])
    if any(set(rows) != set(ids) for rows in arm_rows.values()):
        raise ValueError("the inference arms do not contain identical question ids")
    metadata = {row["id"]: row for row in read_jsonl(metadata_path)}
    if set(metadata) != set(ids):
        raise ValueError("graph metadata and inference outputs contain different ids")

    scored_rows = []
    for question_id in ids:
        info = metadata[question_id]
        scores = {
            arm: score_prediction(
                arm_rows[arm][question_id], arm_rows[arm][question_id]["prediction"]
            )
            for arm in ALL_ARM_NAMES
        }
        scored_rows.append(
            {
                "id": question_id,
                "bucket": arm_rows["original"][question_id]["pilot_bucket"],
                "answer_evidence_rank": arm_rows["original"][question_id]["answer_evidence_rank"],
                **info,
                "scores": scores,
            }
        )

    slices: dict[str, list[dict]] = {
        "overall": scored_rows,
        "evidence_proxy_complete": [row for row in scored_rows if row["evidence_proxy_complete"]],
        "evidence_proxy_incomplete": [row for row in scored_rows if not row["evidence_proxy_complete"]],
        "gold_relations_1_2": [row for row in scored_rows if row["gold_relation_count"] <= 2],
        "gold_relations_3_plus": [row for row in scored_rows if row["gold_relation_count"] >= 3],
    }
    for composition_type in sorted({row["cwq_type"] for row in scored_rows if row["cwq_type"]}):
        slices[f"cwq_{composition_type}"] = [
            row for row in scored_rows if row["cwq_type"] == composition_type
        ]

    comparisons = (
        ("adjacency_graph_minus_original", "adjacency_graph", "original"),
        ("adjacency_graph_minus_structured", "adjacency_graph", "structured"),
        ("adjacency_graph_minus_adjacency_flat", "adjacency_graph", "adjacency_flat"),
        ("adjacency_flat_minus_original", "adjacency_flat", "original"),
    )
    metrics: dict = {
        "questions": len(scored_rows),
        "bootstrap_samples": bootstrap_samples,
        "evidence_proxy_warning": "relation coverage plus answer endpoint; not proof of a connected gold derivation",
        "slices": {},
    }
    for slice_name, rows in slices.items():
        if not rows:
            continue
        result: dict = {"questions": len(rows), "arms": {}, "paired_differences": {}}
        for arm in ALL_ARM_NAMES:
            result["arms"][arm] = {
                metric: _mean([row["scores"][arm][metric] for row in rows])
                for metric in ("hit_at_1", "f1", "no_answer")
            }
        for label, left, right in comparisons:
            result["paired_differences"][label] = {
                metric: _bootstrap_delta(
                    [row["scores"][left][metric] for row in rows],
                    [row["scores"][right][metric] for row in rows],
                    seed,
                    bootstrap_samples,
                )
                for metric in ("hit_at_1", "f1")
            }
        token_rows = [row["prompt_tokens"] for row in rows if row["prompt_tokens"]["original"]]
        if token_rows:
            result["mean_prompt_tokens"] = {
                arm: _mean([tokens[arm] for tokens in token_rows])
                for arm in ("original", "adjacency_flat", "adjacency_graph")
            }
        metrics["slices"][slice_name] = result

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_jsonl(output / "paired_diagnostics.jsonl", scored_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--per-slice", type=int, default=129)
    prepare.add_argument("--seed", type=int, default=17)

    prepare_graph = commands.add_parser("prepare-graph")
    prepare_graph.add_argument("--original", type=Path, required=True)
    prepare_graph.add_argument("--output", type=Path, required=True)
    prepare_graph.add_argument("--official-cwq", type=Path, required=True)
    prepare_graph.add_argument("--tokenizer", type=Path)

    prepare_structure = commands.add_parser("prepare-structure")
    prepare_structure.add_argument("--source", type=Path, required=True)
    prepare_structure.add_argument("--output", type=Path, required=True)
    prepare_structure.add_argument("--official-cwq", type=Path, required=True)
    prepare_structure.add_argument("--tokenizer", type=Path)
    prepare_structure.add_argument("--per-type", type=int, default=200)
    prepare_structure.add_argument("--seed", type=int, default=17)

    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument("--limit", type=int)

    suite = commands.add_parser("run-suite")
    suite.add_argument("--inputs", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--model", type=Path, required=True)
    suite.add_argument("--batch-size", type=int, default=8)
    suite.add_argument("--tensor-parallel-size", type=int, default=1)
    suite.add_argument("--limit", type=int)

    graph_suite = commands.add_parser("run-graph-suite")
    graph_suite.add_argument("--inputs", type=Path, required=True)
    graph_suite.add_argument("--output", type=Path, required=True)
    graph_suite.add_argument("--model", type=Path, required=True)
    graph_suite.add_argument("--batch-size", type=int, default=8)
    graph_suite.add_argument("--tensor-parallel-size", type=int, default=1)
    graph_suite.add_argument("--limit", type=int)

    all_suite = commands.add_parser("run-all-suite")
    all_suite.add_argument("--inputs", type=Path, required=True)
    all_suite.add_argument("--output", type=Path, required=True)
    all_suite.add_argument("--model", type=Path, required=True)
    all_suite.add_argument("--batch-size", type=int, default=8)
    all_suite.add_argument("--tensor-parallel-size", type=int, default=1)
    all_suite.add_argument("--limit", type=int)

    queued = commands.add_parser("wait-suite")
    queued.add_argument("--inputs", type=Path, required=True)
    queued.add_argument("--output", type=Path, required=True)
    queued.add_argument("--model", type=Path, required=True)
    queued.add_argument("--batch-size", type=int, default=8)
    queued.add_argument("--tensor-parallel-size", type=int, default=1)
    queued.add_argument("--limit", type=int)
    queued.add_argument("--max-used-mib", type=int, default=500)
    queued.add_argument("--poll-seconds", type=int, default=60)

    queued_graph = commands.add_parser("wait-graph-suite")
    queued_graph.add_argument("--inputs", type=Path, required=True)
    queued_graph.add_argument("--output", type=Path, required=True)
    queued_graph.add_argument("--model", type=Path, required=True)
    queued_graph.add_argument("--batch-size", type=int, default=8)
    queued_graph.add_argument("--tensor-parallel-size", type=int, default=1)
    queued_graph.add_argument("--limit", type=int)
    queued_graph.add_argument("--max-used-mib", type=int, default=500)
    queued_graph.add_argument("--poll-seconds", type=int, default=60)

    queued_all = commands.add_parser("wait-all-suite")
    queued_all.add_argument("--inputs", type=Path, required=True)
    queued_all.add_argument("--output", type=Path, required=True)
    queued_all.add_argument("--model", type=Path, required=True)
    queued_all.add_argument("--batch-size", type=int, default=8)
    queued_all.add_argument("--tensor-parallel-size", type=int, default=1)
    queued_all.add_argument("--limit", type=int)
    queued_all.add_argument("--max-used-mib", type=int, default=500)
    queued_all.add_argument("--poll-seconds", type=int, default=60)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--runs", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate.add_argument("--seed", type=int, default=17)

    evaluate_graph = commands.add_parser("evaluate-graph")
    evaluate_graph.add_argument("--runs", type=Path, required=True)
    evaluate_graph.add_argument("--metadata", type=Path, required=True)
    evaluate_graph.add_argument("--output", type=Path, required=True)
    evaluate_graph.add_argument("--bootstrap-samples", type=int, default=10_000)
    evaluate_graph.add_argument("--seed", type=int, default=17)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_pilot(args.source, args.output, args.per_slice, args.seed)
    elif args.command == "prepare-graph":
        result = prepare_graph_arms(
            args.original,
            args.output,
            args.official_cwq,
            args.tokenizer,
        )
    elif args.command == "prepare-structure":
        result = prepare_structure_pilot(
            args.source,
            args.output,
            args.official_cwq,
            args.tokenizer,
            args.per_type,
            args.seed,
        )
    elif args.command == "run":
        result = run_inference(
            args.input,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
        )
    elif args.command == "run-suite":
        result = run_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
        )
    elif args.command == "run-graph-suite":
        result = run_graph_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
        )
    elif args.command == "run-all-suite":
        result = run_all_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
        )
    elif args.command == "wait-suite":
        result = wait_and_run_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
            args.max_used_mib,
            args.poll_seconds,
        )
    elif args.command == "wait-graph-suite":
        result = wait_and_run_graph_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
            args.max_used_mib,
            args.poll_seconds,
        )
    elif args.command == "wait-all-suite":
        result = wait_and_run_all_suite(
            args.inputs,
            args.output,
            args.model,
            args.batch_size,
            args.tensor_parallel_size,
            args.limit,
            args.max_used_mib,
            args.poll_seconds,
        )
    elif args.command == "evaluate":
        result = evaluate_runs(args.runs, args.output, args.bootstrap_samples, args.seed)
    else:
        result = evaluate_graph_runs(
            args.runs,
            args.metadata,
            args.output,
            args.bootstrap_samples,
            args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
