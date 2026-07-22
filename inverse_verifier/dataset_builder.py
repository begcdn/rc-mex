from __future__ import annotations

import json
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .openai_naturalize import (
    OpenAIBatchClient,
    _parse_result_file,
    run_batch_records,
    run_chat_records_sync,
    select_rows,
)
from .query_representation import classify_metadata_relation, represent_query
from .synthetic import load_kqa_graph


GLOSSARY_MODEL = "gpt-4o-2024-11-20"
VALIDATION_MODEL = "gpt-4o-2024-11-20"
VERIFIER_MODEL = "gpt-4o-mini-2024-07-18"
GLOSSARY_GROUP_SIZE = 6
QWEN_GROUP_SIZE = 4
VAGUE_QUESTION_PHRASES = ("associated with", "related to", "linked to", "connected to")

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
                        "status": {"type": "string", "enum": ["generated"]},
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
                        "all_facts_expressed": {"type": "boolean"},
                        "uses_only_supported_facts": {"type": "boolean"},
                        "is_natural_language_question": {"type": "boolean"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id", "selected_option", "answer_type_matches", "all_facts_expressed",
                        "uses_only_supported_facts", "is_natural_language_question", "confidence", "reason"
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
            "temperature": 0,
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


def render_compact_query(query: dict[str, Any]) -> str:
    lines = [
        f"Anchor: v0 = [ENTITY] (type: {query['anchor']['type']})",
        "Required facts:",
    ]
    for index, fact in enumerate(query["facts"]):
        subject = f"{fact['subject']} (type: {fact['subject_type']})"
        object_ = f"{fact['object']} (type: {fact['object_type']})"
        template = fact.get("fact_template", "")
        try:
            statement = template.format(subject=subject, object=object_)
        except (KeyError, ValueError):
            statement = f"{subject} -- {fact['meaning']} --> {object_}"
        lines.append(f"F{index}: {statement}")
    returned = query["return"]
    lines.append(f"Return: {returned['variable']} (type: {returned['type']})")
    return "\n".join(lines)


def row_candidates(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    meta = row["_naturalization"]
    prefix = f"{meta['split']}:{meta['source_index']}"
    return [(f"{prefix}:positive", row["positive_path"])] + [
        (f"{prefix}:negative:{index}", path) for index, path in enumerate(row["negative_paths"])
    ]


def generation_records(
    rows: list[dict[str, Any]],
    glossary: dict[str, dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    system = (
        "Generate one natural English question from each explicit logical query. Raw facts and "
        "variable identity are authoritative. Express every predicate with its correct subject and "
        "object, preserve shared intermediate variables, and ask for the declared return variable "
        "and answer type. Never transfer a person's property to an institution or place, omit a "
        "fact, reverse a relation, or replace a predicate with a nearby meaning. Use [ENTITY] "
        "exactly once. Generate from the logical query itself; there is no draft to preserve. Every "
        "input query is an executable connected graph and must receive a question; do not judge factual "
        "plausibility and do not refuse complex or unusual paths. Avoid vague related, "
        "associated, linked, or connected wording when a specific predicate is available. Output "
        "ordinary natural language only: never mention variable IDs such as v0/v1, raw schemas, "
        "parentheses, type annotations, facts, or graph terminology. Type words may be used "
        "naturally as nouns, for example 'film' or 'country'."
    )
    items = []
    for row in rows:
        for identifier, path in row_candidates(row):
            query = compact_query(path, glossary)
            if not query["unusable_relations"]:
                items.append({"id": identifier, "query": render_compact_query(query)})
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
        "than one option, or matches no option. Set all_facts_expressed true only when every F-numbered "
        "fact in the selected option is stated by the question. Set uses_only_supported_facts true only "
        "when the question invents no relation absent from that option. Option order is random and gives "
        "no correctness hint. Set is_natural_language_question true only if the question exposes no "
        "variable IDs, type annotations, schemas, F-numbered facts, or graph terminology. Do not select the closest option "
        "when NONE is more accurate."
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
                options.append({"option_id": option_id, "query": render_compact_query(query)})
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
            and judgment.get("all_facts_expressed") is True
            and judgment.get("uses_only_supported_facts") is True
            and judgment.get("is_natural_language_question") is True
            and float(judgment.get("confidence", 0.0)) >= 0.8
        )
        combined[identifier] = {
            "id": identifier,
            "status": "valid" if accepted else "reject",
            "question": generation.get("question", ""),
            "selected_option": (judgment or {}).get("selected_option", ""),
            "intended_option": intended_options.get(identifier, ""),
            "answer_type_matches": (judgment or {}).get("answer_type_matches", False),
            "all_facts_expressed": (judgment or {}).get("all_facts_expressed", False),
            "uses_only_supported_facts": (judgment or {}).get("uses_only_supported_facts", False),
            "is_natural_language_question": (judgment or {}).get("is_natural_language_question", False),
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
    if (
        re.search(r"\bv\d+\b", question, flags=re.IGNORECASE)
        or "type:" in question.casefold()
        or "(" in question
        or ")" in question
    ):
        return "internal_notation_in_question"
    if any(phrase in question.casefold() for phrase in VAGUE_QUESTION_PHRASES):
        return "vague_question_wording"
    if prediction.get("answer_type_matches") is not True:
        return "answer_type_mismatch"
    if prediction.get("all_facts_expressed") is not True:
        return "missing_path_fact"
    if prediction.get("uses_only_supported_facts") is not True:
        return "unsupported_question_relation"
    if prediction.get("is_natural_language_question") is not True:
        return "non_natural_question"
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
    glossary_model: str = GLOSSARY_MODEL,
    generation_model: str = VALIDATION_MODEL,
    verifier_model: str = VERIFIER_MODEL,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    output.mkdir(parents=True, exist_ok=True)
    internal = output / ".dataset_builder"
    internal.mkdir(exist_ok=True)
    rows = select_rows(source, max_paths, max_negatives)
    use_batch = len(rows) > 100
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
        glossary_requests = glossary_records(evidence, glossary_model)
        glossary_files = (
            run_batch_records(
                glossary_requests, internal / "glossary_batch_v8", client, "relation glossary",
                max_estimated_tokens=60_000,
            )
            if use_batch
            else run_chat_records_sync(
                glossary_requests, internal / "glossary_sync_v8", client, "relation glossary",
                workers=2,
            )
        )
        raw_glossary, glossary_errors = parse_batch_items(glossary_files)
        glossary = validate_glossary(evidence, raw_glossary)
    statuses = defaultdict(int)
    for item in glossary.values():
        statuses[item["status"]] += 1
    print(f"[3/5] Relation glossary: {dict(statuses)}", flush=True)
    glossary_path.write_text(json.dumps(glossary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[4/5] Generating and independently verifying natural questions", flush=True)
    generation_requests = generation_records(rows, glossary, generation_model)
    generation_files = (
        run_batch_records(
            generation_requests, internal / "generation_batch_v8", client,
            "faithful question generation", max_estimated_tokens=60_000,
        )
        if use_batch
        else run_chat_records_sync(
            generation_requests, internal / "generation_sync_v8", client,
            "faithful question generation", workers=2,
        )
    )
    generated, generation_errors = parse_batch_items(generation_files)
    contrastive_requests, intended_options = contrastive_records(
        rows, glossary, generated, verifier_model
    )
    contrastive_files = (
        run_batch_records(
            contrastive_requests, internal / "contrastive_batch_v8", client,
            "contrastive question verification",
        )
        if use_batch
        else run_chat_records_sync(
            contrastive_requests, internal / "contrastive_sync_v8", client,
            "contrastive question verification", workers=2,
        )
    )
    judgments, contrastive_errors = parse_batch_items(contrastive_files)
    validated = combine_contrastive_results(generated, judgments, intended_options)
    validation_errors = [*generation_errors, *contrastive_errors]
    fallback_path = internal / "contrastive_batch_v8" / "fallback_manifest.json"
    verifier_fallback = (
        json.loads(fallback_path.read_text(encoding="utf-8"))
        if fallback_path.is_file()
        else None
    )
    manifest = finalize_dataset(rows, validated, glossary, output, {
        "source": str(source), "requested_paths": max_paths, "selected_paths": len(rows),
        "relations": len(evidence), "glossary_entries": len(glossary),
        "glossary_errors": len(glossary_errors),
        "glossary_model": glossary_model, "generation_model": generation_model,
        "verifier_model": verifier_model, "expensive_stages_via_batch": use_batch,
        "generation_errors": len(generation_errors),
        "contrastive_errors": len(contrastive_errors),
        "validation_errors": len(validation_errors),
        "verifier_fallback": verifier_fallback,
    })
    print(
        f"[5/5] Final dataset: {manifest['accepted_train']} train, "
        f"{manifest['accepted_dev']} dev, {manifest['rejected_rows']} rejected",
        flush=True,
    )
    return manifest
