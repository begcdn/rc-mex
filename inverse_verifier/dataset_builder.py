from __future__ import annotations

import json
import hashlib
import os
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from .openai_naturalize import (
    OpenAIBatchClient,
    _parse_result_file,
    run_chat_records_sync,
    select_rows,
)
from .query_representation import classify_metadata_relation, represent_query
from .synthetic import load_kqa_graph


GLOSSARY_MODEL = "gpt-4o-2024-11-20"
VALIDATION_MODEL = "gpt-4o-2024-11-20"
GLOSSARY_GROUP_SIZE = 6
QWEN_GROUP_SIZE = 4

GLOSSARY_SCHEMA = {
    "name": "relation_glossary",
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
                        "status": {"type": "string", "enum": ["semantic", "metadata", "opaque"]},
                        "description": {"type": "string"},
                        "subject_role": {"type": "string"},
                        "object_role": {"type": "string"},
                        "fact_template": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id", "status", "description", "subject_role", "object_role",
                        "fact_template", "confidence", "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}

GENERATION_SCHEMA = {
    "name": "faithful_question_generation",
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
                        "status": {"type": "string", "enum": ["generated", "reject"]},
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

CONTRASTIVE_SCHEMA = {
    "name": "contrastive_path_selection",
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
                        "selected_option": {"type": "string"},
                        "answer_type_matches": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id", "selected_option", "answer_type_matches", "confidence", "reason"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
}


def relation_key(kg: str, relation: str) -> str:
    return f"{kg}::{relation}"


def collect_relation_evidence(
    rows: list[dict[str, Any]],
    kqa_kb: Path,
    webqsp_graphs: Path,
    example_cap: int = 4,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for row in rows:
        for path in [row["positive_path"], *row["negative_paths"]]:
            kg = path.get("kg", row.get("kg", "unknown"))
            for hop in path["hops"]:
                key = relation_key(kg, hop["relation"])
                item = evidence.setdefault(key, {
                    "id": key,
                    "kg": kg,
                    "relation_id": hop["relation"],
                    "observed_type_pairs": [],
                    "example_facts": [],
                    "metadata_hint": classify_metadata_relation(hop["relation"]),
                })
                pair = [hop.get("source_type", "entity"), hop.get("target_type", "entity"), hop["direction"]]
                if pair not in item["observed_type_pairs"]:
                    item["observed_type_pairs"].append(pair)

    wanted_kqa = {item["relation_id"] for item in evidence.values() if item["kg"] == "kqa_pro"}
    if wanted_kqa and kqa_kb.exists():
        graph = load_kqa_graph(kqa_kb)
        for head in sorted(graph.adjacency):
            for edge in sorted(
                graph.adjacency[head],
                key=lambda item: (item.relation, item.direction, item.target),
            ):
                if edge.direction != "forward" or edge.relation not in wanted_kqa:
                    continue
                item = evidence[relation_key("kqa_pro", edge.relation)]
                fact = [graph.label_of(head), edge.relation, graph.label_of(edge.target)]
                if fact not in item["example_facts"] and len(item["example_facts"]) < example_cap:
                    item["example_facts"].append(fact)

    wanted_web = {item["relation_id"] for item in evidence.values() if item["kg"] == "webqsp"}
    if wanted_web and webqsp_graphs.exists():
        with webqsp_graphs.open(encoding="utf-8") as handle:
            for line in handle:
                for head, relation, tail in json.loads(line).get("graph", []):
                    if relation not in wanted_web:
                        continue
                    item = evidence[relation_key("webqsp", relation)]
                    fact = [head, relation, tail]
                    if fact not in item["example_facts"] and len(item["example_facts"]) < example_cap:
                        item["example_facts"].append(fact)
                if all(len(evidence[relation_key("webqsp", rel)]["example_facts"]) >= example_cap for rel in wanted_web):
                    break
    return evidence


def _chat_record(custom_id: str, model: str, system: str, payload: Any, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "max_completion_tokens": 2400,
        },
    }


def glossary_records(evidence: dict[str, dict[str, Any]], model: str) -> list[dict[str, Any]]:
    system = (
        "Create a conservative KG relation glossary from schema origin, raw relation ID, "
        "observed argument types, and real KG facts. Describe the raw fact subject-to-object, "
        "not traversal direction. fact_template must contain {subject} and {object}. Mark internal "
        "storage/provenance relations metadata. Mark opaque when evidence is insufficient or "
        "contradictory; never use vague related/associated/linked wording to hide uncertainty. "
        "Return one item for every id."
    )
    items = list(evidence.values())
    return [
        _chat_record(f"glossary-{start // GLOSSARY_GROUP_SIZE:05d}", model, system, {"relations": items[start:start + GLOSSARY_GROUP_SIZE]}, GLOSSARY_SCHEMA)
        for start in range(0, len(items), GLOSSARY_GROUP_SIZE)
    ]


def parse_batch_items(files: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    parsed, errors = {}, []
    for path in files:
        items, file_errors = _parse_result_file(path)
        parsed.update(items)
        errors.extend(file_errors)
    return parsed, errors


def validate_glossary(
    evidence: dict[str, dict[str, Any]],
    generated: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    glossary = {}
    for identifier, source in evidence.items():
        item = dict(generated.get(identifier) or {
            "id": identifier,
            "status": "opaque",
            "description": "",
            "subject_role": "",
            "object_role": "",
            "fact_template": "",
            "confidence": 0.0,
            "reason": "missing glossary output",
        })
        template = item.get("fact_template", "")
        if source["metadata_hint"]["is_metadata"]:
            item["status"] = "metadata"
            item["reason"] = "deterministic metadata pattern: " + ", ".join(
                source["metadata_hint"]["matched_patterns"]
            )
        elif (
            item.get("status") == "semantic"
            and (
                "{subject}" not in template
                or "{object}" not in template
                or float(item.get("confidence", 0.0)) < 0.6
            )
        ):
            item["status"] = "opaque"
            item["reason"] = "semantic output failed template or confidence validation"
        glossary[identifier] = item
    return glossary


def compact_query(path: dict[str, Any], glossary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    query = represent_query(path)
    variables = {item["id"]: item for item in query["variables"]}
    facts = []
    unusable = []
    for triple in query["triples"]:
        entry = glossary.get(relation_key(path["kg"], triple["predicate"]))
        if not entry or entry.get("status") != "semantic":
            unusable.append(triple["predicate"])
        facts.append({
            "subject": triple["subject"],
            "subject_type": triple["subject_type"],
            "relation_id": triple["predicate"],
            "meaning": entry.get("description", "unknown") if entry else "unknown",
            "fact_template": entry.get("fact_template", "") if entry else "",
            "object": triple["object"],
            "object_type": triple["object_type"],
        })
    return {
        "kg": query["schema_origin"],
        "anchor": {"variable": "v0", "surface": "[ENTITY]", "type": variables["v0"]["type"]},
        "variables": [{"id": item["id"], "type": item["type"]} for item in query["variables"]],
        "facts": facts,
        "return": {"variable": query["answer_variable"], "type": query["answer_type"]},
        "unusable_relations": unusable,
    }


def qwen_naturalize(items: list[dict[str, Any]], model: str, host: str) -> dict[str, dict[str, Any]]:
    prompt = (
        "/no_think\nWrite one natural English question for every explicit logical query. "
        "Raw facts and variable identity are authoritative. [ENTITY] must appear exactly once. "
        "Every fact must contribute and different variable IDs are different entities. Ask for "
        "the return variable and exact answer type. Do not mention graphs, variables, facts, "
        "forward/backward, associated, related, or linked. If unusable_relations is nonempty, "
        "return status opaque. Return JSON only: "
        '{"items":[{"id":"...","status":"valid|opaque","question":"...","reason":"..."}]}.\n'
        + json.dumps({"queries": items}, ensure_ascii=False)
    )
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 2200},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        content = json.load(response)["message"]["content"]
    return {item["id"]: item for item in json.loads(content)["items"]}


def qwen_naturalize_resilient(
    items: list[dict[str, Any]], model: str, host: str
) -> dict[str, dict[str, Any]]:
    try:
        return qwen_naturalize(items, model, host)
    except Exception as exc:
        if len(items) == 1:
            item = items[0]
            return {
                item["id"]: {
                    "id": item["id"],
                    "status": "opaque",
                    "question": "",
                    "reason": f"Qwen request failed: {type(exc).__name__}",
                }
            }
        midpoint = len(items) // 2
        return qwen_naturalize_resilient(items[:midpoint], model, host) | qwen_naturalize_resilient(
            items[midpoint:], model, host
        )


def naturalize_with_qwen(rows: list[dict[str, Any]], glossary: dict[str, dict[str, Any]], model: str, host: str, output: Path) -> dict[str, dict[str, Any]]:
    destination = output / ".dataset_builder" / "qwen_predictions.jsonl"
    predictions = {}
    if destination.exists():
        predictions = {row["id"]: row for row in (json.loads(line) for line in destination.open())}
    with destination.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rows), QWEN_GROUP_SIZE):
            pending = []
            for row in rows[start:start + QWEN_GROUP_SIZE]:
                meta = row["_naturalization"]
                prefix = f"{meta['split']}:{meta['source_index']}"
                candidates = [(f"{prefix}:positive", row["positive_path"])] + [
                    (f"{prefix}:negative:{index}", path) for index, path in enumerate(row["negative_paths"])
                ]
                for identifier, path in candidates:
                    if identifier not in predictions:
                        pending.append({"id": identifier, "query": compact_query(path, glossary)})
            if not pending:
                continue
            generated = qwen_naturalize_resilient(pending, model, host)
            for item in pending:
                prediction = generated.get(item["id"], {"id": item["id"], "status": "opaque", "question": "", "reason": "missing Qwen output"})
                predictions[item["id"]] = prediction
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Qwen naturalization: {min(start + QWEN_GROUP_SIZE, len(rows))}/{len(rows)} rows", flush=True)
    return predictions


def row_candidates(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    meta = row["_naturalization"]
    prefix = f"{meta['split']}:{meta['source_index']}"
    return [(f"{prefix}:positive", row["positive_path"])] + [
        (f"{prefix}:negative:{index}", path) for index, path in enumerate(row["negative_paths"])
    ]


def generation_records(
    rows: list[dict[str, Any]],
    glossary: dict[str, dict[str, Any]],
    qwen: dict[str, dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    system = (
        "Generate one natural English question from each explicit logical query. Raw facts and "
        "variable identity are authoritative. Express every predicate with its correct subject and "
        "object, preserve shared intermediate variables, and ask for the declared return variable "
        "and answer type. Never transfer a person's property to an institution or place, omit a "
        "fact, reverse a relation, or replace a predicate with a nearby meaning. Use [ENTITY] "
        "exactly once. The draft is optional assistance, not evidence; repair it freely. Reject only "
        "when the explicit query cannot form one coherent natural question. Avoid vague related, "
        "associated, linked, or connected wording when a specific predicate is available."
    )
    items = []
    for row in rows:
        for identifier, path in row_candidates(row):
            query = compact_query(path, glossary)
            if not query["unusable_relations"]:
                items.append({"id": identifier, "query": query, "draft": qwen.get(identifier)})
    return [
        _chat_record(
            f"generate-{start // QWEN_GROUP_SIZE:06d}", model, system,
            {"items": items[start:start + QWEN_GROUP_SIZE]}, GENERATION_SCHEMA,
        )
        for start in range(0, len(items), QWEN_GROUP_SIZE)
    ]


def contrastive_records(
    rows: list[dict[str, Any]],
    glossary: dict[str, dict[str, Any]],
    generated: dict[str, dict[str, Any]],
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    system = (
        "Infer which explicit logical query is fully expressed by each natural-language question. "
        "Compare exact predicate meanings, subject/object roles, shared variables, number of facts, "
        "and requested answer type. Select NONE if the question omits a fact, reverses a relation, "
        "collapses an intermediate entity, uses only vague related/associated wording, matches more "
        "than one option, or matches no option. Option order is random and gives no correctness hint."
    )
    items = []
    intended_options: dict[str, str] = {}
    for row in rows:
        usable = []
        for identifier, path in row_candidates(row):
            query = compact_query(path, glossary)
            if not query["unusable_relations"]:
                usable.append((identifier, path, query))
        for identifier, _path, _query in usable:
            generation = generated.get(identifier)
            if not generation or generation.get("status") != "generated":
                continue
            choices = [(other_id, other_query) for other_id, _other_path, other_query in usable]
            if len(choices) < 2:
                continue
            choices.sort(
                key=lambda choice: hashlib.sha256(
                    f"{identifier}|{choice[0]}".encode()
                ).hexdigest()
            )
            options = []
            for index, (choice_id, query) in enumerate(choices):
                option_id = chr(ord("A") + index)
                options.append({"option_id": option_id, "query": query})
                if choice_id == identifier:
                    intended_options[identifier] = option_id
            options.append({"option_id": "NONE", "query": None})
            items.append({
                "id": identifier,
                "question": generation["question"],
                "options": options,
            })
    records = [
        _chat_record(
            f"contrast-{start // QWEN_GROUP_SIZE:06d}", model, system,
            {"items": items[start:start + QWEN_GROUP_SIZE]}, CONTRASTIVE_SCHEMA,
        )
        for start in range(0, len(items), QWEN_GROUP_SIZE)
    ]
    return records, intended_options


def combine_contrastive_results(
    generated: dict[str, dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    intended_options: dict[str, str],
) -> dict[str, dict[str, Any]]:
    combined = {}
    for identifier, generation in generated.items():
        judgment = judgments.get(identifier)
        accepted = bool(
            generation.get("status") == "generated"
            and judgment
            and judgment.get("selected_option") == intended_options.get(identifier)
            and judgment.get("answer_type_matches") is True
            and float(judgment.get("confidence", 0.0)) >= 0.8
        )
        combined[identifier] = {
            "id": identifier,
            "status": "valid" if accepted else "reject",
            "question": generation.get("question", ""),
            "selected_option": (judgment or {}).get("selected_option", ""),
            "intended_option": intended_options.get(identifier, ""),
            "answer_type_matches": (judgment or {}).get("answer_type_matches", False),
            "confidence": float((judgment or {}).get("confidence", 0.0)),
            "reason": (judgment or {}).get("reason", generation.get("reason", "")),
        }
    return combined


def validation_rejection(
    path: dict[str, Any],
    prediction: dict[str, Any] | None,
    glossary: dict[str, dict[str, Any]],
) -> str | None:
    query = compact_query(path, glossary)
    if query["unusable_relations"]:
        return "unusable_relation"
    if not prediction or prediction.get("status") == "reject":
        return "contrastive_verifier_rejected"
    if prediction.get("question", "").count("[ENTITY]") != 1:
        return "invalid_entity_placeholder"
    question = prediction["question"].strip()
    if not question.endswith("?"):
        return "not_a_question"
    if prediction.get("answer_type_matches") is not True:
        return "answer_type_mismatch"
    if float(prediction.get("confidence", 0.0)) < 0.8:
        return "low_contrastive_confidence"
    if prediction.get("selected_option") != prediction.get("intended_option"):
        return "wrong_contrastive_path"
    return None


def finalize_dataset(
    rows: list[dict[str, Any]],
    validated: dict[str, dict[str, Any]],
    glossary: dict[str, dict[str, Any]],
    output: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    accepted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for original in rows:
        row = dict(original)
        meta = row.pop("_naturalization")
        prefix = f"{meta['split']}:{meta['source_index']}"
        positive = validated.get(f"{prefix}:positive")
        positive_rejection = validation_rejection(row["positive_path"], positive, glossary)
        if positive_rejection:
            rejected.append({"id": prefix, "reason": positive_rejection, "prediction": positive})
            continue
        row["canonical_question"] = row["question"]
        row["question"] = positive["question"].strip()
        negatives = []
        seen = {row["question"].casefold()}
        for index, negative in enumerate(row["negative_paths"]):
            prediction = validated.get(f"{prefix}:negative:{index}")
            question = (prediction or {}).get("question", "").strip()
            if validation_rejection(negative, prediction, glossary) or question.casefold() in seen:
                continue
            item = dict(negative)
            item["canonical_question"] = item["question"]
            item["question"] = question
            negatives.append(item)
            seen.add(question.casefold())
        if not negatives:
            rejected.append({"id": prefix, "reason": "no_valid_negative"})
            continue
        row["negative_paths"] = negatives
        accepted[meta["split"]].append(row)
    for split in ("train", "dev"):
        (output / f"{split}_faithful.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted[split]), encoding="utf-8"
        )
    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected), encoding="utf-8"
    )
    manifest.update({"accepted_train": len(accepted["train"]), "accepted_dev": len(accepted["dev"]), "rejected_rows": len(rejected)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_naturalized_dataset(
    source: Path,
    output: Path,
    kqa_kb: Path,
    webqsp_graphs: Path,
    max_paths: int = 3_000,
    max_negatives: int = 3,
    qwen_model: str = "qwen3:8b",
    ollama_host: str = "http://127.0.0.1:11434",
    glossary_model: str = GLOSSARY_MODEL,
    validation_model: str = VALIDATION_MODEL,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    output.mkdir(parents=True, exist_ok=True)
    internal = output / ".dataset_builder"
    internal.mkdir(exist_ok=True)
    rows = select_rows(source, max_paths, max_negatives)
    client = OpenAIBatchClient(api_key)
    print(f"[1/5] Selected {len(rows)} paths from {source}", flush=True)

    evidence = collect_relation_evidence(rows, kqa_kb, webqsp_graphs)
    print(
        f"[2/5] Grounded {len(evidence)} unique relations with KG examples "
        f"({sum(item['metadata_hint']['is_metadata'] for item in evidence.values())} metadata)",
        flush=True,
    )
    (internal / "relation_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    glossary_path = internal / "relation_glossary.json"
    if glossary_path.exists():
        glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
        glossary_errors = []
        print("[3/5] Reusing completed relation glossary", flush=True)
    else:
        glossary_files = run_chat_records_sync(
            glossary_records(evidence, glossary_model),
            internal / "glossary_sync",
            client,
            "relation glossary",
        )
        raw_glossary, glossary_errors = parse_batch_items(glossary_files)
        glossary = validate_glossary(evidence, raw_glossary)
    statuses = defaultdict(int)
    for item in glossary.values():
        statuses[item["status"]] += 1
    print(f"[3/5] Relation glossary: {dict(statuses)}", flush=True)
    glossary_path.write_text(json.dumps(glossary, indent=2, ensure_ascii=False), encoding="utf-8")

    qwen = naturalize_with_qwen(rows, glossary, qwen_model, ollama_host, output)
    print(f"[4/5] Qwen produced {len(qwen)} candidate outputs", flush=True)
    generation_files = run_chat_records_sync(
        generation_records(rows, glossary, qwen, validation_model),
        internal / "generation_sync_v4",
        client,
        "faithful question generation",
    )
    generated, generation_errors = parse_batch_items(generation_files)
    contrastive_requests, intended_options = contrastive_records(
        rows, glossary, generated, validation_model
    )
    contrastive_files = run_chat_records_sync(
        contrastive_requests,
        internal / "contrastive_sync_v4",
        client,
        "contrastive question verification",
    )
    judgments, contrastive_errors = parse_batch_items(contrastive_files)
    validated = combine_contrastive_results(generated, judgments, intended_options)
    validation_errors = [*generation_errors, *contrastive_errors]
    manifest = finalize_dataset(rows, validated, glossary, output, {
        "source": str(source), "requested_paths": max_paths, "selected_paths": len(rows),
        "relations": len(evidence), "glossary_entries": len(glossary),
        "glossary_errors": len(glossary_errors), "qwen_model": qwen_model,
        "glossary_model": glossary_model, "validation_model": validation_model,
        "generation_errors": len(generation_errors),
        "contrastive_errors": len(contrastive_errors),
        "validation_errors": len(validation_errors),
    })
    print(
        f"[5/5] Final dataset: {manifest['accepted_train']} train, "
        f"{manifest['accepted_dev']} dev, {manifest['rejected_rows']} rejected",
        flush=True,
    )
    return manifest
