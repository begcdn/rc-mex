from __future__ import annotations

from typing import Any

from .data import ENTITY_PLACEHOLDER, PathSpec, path_to_dict
from .synthetic import METADATA_RELATION_PARTS


REPRESENTATION_VERSION = "explicit-query-v1"
FREEBASE_KGS = {"freebase", "webqsp"}
KQA_PRO_KGS = {"kqa", "kqa_pro", "kqa pro"}


def schema_origin(kg: str) -> str:
    normalized = kg.strip().casefold()
    if normalized in FREEBASE_KGS:
        return "freebase"
    if normalized in KQA_PRO_KGS:
        return "kqa_pro"
    return "unknown"


def parse_relation_schema(relation_id: str, kg: str) -> dict[str, Any]:
    """Parse identifier structure without treating its tokens as a gloss."""
    origin = schema_origin(kg)
    if origin == "freebase":
        identifier = relation_id.removeprefix("ns:").strip("/")
        separator = "/" if "/" in identifier else "."
        parts = [part for part in identifier.split(separator) if part]
        components = {
            "domain": parts[0] if len(parts) >= 3 else None,
            "type": parts[1] if len(parts) >= 3 else None,
            "property": separator.join(parts[2:]) if len(parts) >= 3 else None,
        }
        return {
            "origin": origin,
            "raw_relation_id": relation_id,
            "components": components,
            "identifier_parts": parts,
            "parse_complete": len(parts) >= 3,
            "semantic_interpretation": "not_inferred_from_identifier_tokens",
        }
    if origin == "kqa_pro":
        return {
            "origin": origin,
            "raw_relation_id": relation_id,
            "components": {"predicate_label": relation_id},
            "identifier_parts": [relation_id],
            "parse_complete": True,
            "semantic_interpretation": "dataset_predicate_label",
        }
    return {
        "origin": origin,
        "raw_relation_id": relation_id,
        "components": {},
        "identifier_parts": [relation_id],
        "parse_complete": False,
        "semantic_interpretation": "unknown_schema",
    }


def classify_metadata_relation(relation_id: str) -> dict[str, Any]:
    lowered = relation_id.casefold()
    matched = [pattern for pattern in METADATA_RELATION_PARTS if pattern in lowered]
    return {
        "is_metadata": bool(matched),
        "matched_patterns": matched,
        "action": "preserve_and_flag" if matched else "preserve",
    }


def _possible_cvt_macros(triples: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    if origin != "freebase":
        return []
    macros = []
    for index, (left, right) in enumerate(zip(triples, triples[1:])):
        middle = left["traversal"]["to"]
        left_components = left["relation_schema"]["components"]
        right_components = right["relation_schema"]["components"]
        same_owner = (
            left_components.get("domain") is not None
            and left_components.get("domain") == right_components.get("domain")
            and left_components.get("type") == right_components.get("type")
        )
        shared_raw_subject = left["subject"] == middle == right["subject"]
        if not (same_owner and shared_raw_subject):
            continue
        owner = f'{left_components["domain"]}.{left_components["type"]}'
        macros.append(
            {
                "kind": "possible_freebase_cvt",
                "middle_variable": middle,
                "triple_indices": [index, index + 1],
                "relation_ids": [left["predicate"], right["predicate"]],
                "syntactic_schema_owner": owner,
                "needs_semantic_description": True,
                "reason": (
                    f"Both raw facts have shared middle variable {middle} as subject and "
                    f"their Freebase IDs have syntactic owner {owner}. This is consistent "
                    "with a reified CVT record, but identifier tokens do not establish the "
                    "macro's meaning."
                ),
            }
        )
    return macros


def render_logical_form(query: dict[str, Any]) -> str:
    """Render a stable, fact-oriented form for a later LLM input."""
    lines = [
        "EXPLICIT LOGICAL QUERY",
        "This is a graph query, not a natural-language question.",
        f'KG: {query["kg"]}; schema origin: {query["schema_origin"]}.',
        "Variable roles:",
    ]
    for variable in query["variables"]:
        binding = f'; bound to {variable["binding"]}' if "binding" in variable else ""
        lines.append(
            f'- {variable["id"]}: role={variable["role"]}; type={variable["type"]}{binding}.'
        )
    lines.extend(
        [
            "Raw subject-predicate-object triples:",
            "Each tuple is directed from subject to object in the KG fact. Traversal may run "
            "with or against that direction.",
        ]
    )
    for index, triple in enumerate(query["triples"]):
        traversal = triple["traversal"]
        metadata = "metadata=true" if triple["metadata"]["is_metadata"] else "metadata=false"
        lines.append(
            f'- T{index}: ({triple["subject"]}, {triple["predicate"]!r}, '
            f'{triple["object"]}); traversal {traversal["from"]} -> '
            f'{traversal["to"]} is {traversal["direction"]}; '
            f'raw_subject_type={triple["subject_type"]}; '
            f'raw_object_type={triple["object_type"]}; '
            f'traversal_source_type={traversal["source_type"]}; '
            f'traversal_target_type={traversal["target_type"]}; '
            f'{metadata}; needs_semantic_description='
            f'{str(triple["needs_semantic_description"]).lower()}.'
        )
    lines.extend(
        [
            "Direction rules:",
            "- forward traversal v_i -> v_j means raw fact (v_i, relation, v_j).",
            "- backward traversal v_i -> v_j means raw fact (v_j, relation, v_i).",
            f'Return {query["answer_variable"]} as the answer variable '
            f'(type={query["answer_type"]}).',
        ]
    )
    if query["semantic_macros"]:
        lines.append("Possible semantic macros (raw triples above remain authoritative):")
        for macro in query["semantic_macros"]:
            indices = ", ".join(f"T{value}" for value in macro["triple_indices"])
            lines.append(
                f'- {macro["kind"]} over {indices} through {macro["middle_variable"]}; '
                f'needs_semantic_description={str(macro["needs_semantic_description"]).lower()}; '
                f'reason={macro["reason"]}'
            )
    lines.append("Do not generate the final natural-language question in this representation step.")
    return "\n".join(lines)


def represent_query(path: dict[str, Any] | PathSpec) -> dict[str, Any]:
    """Convert a traversal path into explicit variables and raw KG triples."""
    if isinstance(path, PathSpec):
        path = path_to_dict(path)
    hops = path.get("hops") or []
    if not hops:
        raise ValueError("query representation requires at least one hop")

    kg = str(path.get("kg") or "unknown")
    origin = schema_origin(kg)
    variables = [
        {
            "id": "v0",
            "role": "anchor",
            "binding": ENTITY_PLACEHOLDER,
            "type": str(path.get("anchor_type") or hops[0].get("source_type") or "entity"),
        }
    ]
    triples = []
    for index, hop in enumerate(hops):
        direction = hop.get("direction")
        if direction not in {"forward", "backward"}:
            raise ValueError(f"hop {index} has invalid direction: {direction!r}")
        source = f"v{index}"
        target = f"v{index + 1}"
        relation_id = str(hop["relation"])
        source_type = str(hop.get("source_type") or "entity")
        target_type = str(hop.get("target_type") or "entity")
        variables.append(
            {
                "id": target,
                "role": "answer" if index == len(hops) - 1 else "intermediate",
                "type": target_type,
            }
        )
        subject, object_ = (source, target) if direction == "forward" else (target, source)
        subject_type, object_type = (
            (source_type, target_type)
            if direction == "forward"
            else (target_type, source_type)
        )
        relation_schema = parse_relation_schema(relation_id, kg)
        metadata = classify_metadata_relation(relation_id)
        triples.append(
            {
                "subject": subject,
                "predicate": relation_id,
                "object": object_,
                "kg": kg,
                "subject_type": subject_type,
                "object_type": object_type,
                "traversal": {
                    "hop_index": index,
                    "from": source,
                    "to": target,
                    "direction": direction,
                    "source_type": source_type,
                    "target_type": target_type,
                },
                "relation_schema": relation_schema,
                "metadata": metadata,
                "needs_semantic_description": (
                    metadata["is_metadata"]
                    or relation_schema["semantic_interpretation"]
                    != "dataset_predicate_label"
                ),
            }
        )

    answer_variable = f"v{len(hops)}"
    answer_type = str(path.get("answer_type") or hops[-1].get("target_type") or "entity")
    query = {
        "version": REPRESENTATION_VERSION,
        "kg": kg,
        "schema_origin": origin,
        "variables": variables,
        "triples": triples,
        "answer_variable": answer_variable,
        "answer_type": answer_type,
        "semantic_macros": _possible_cvt_macros(triples, origin),
    }
    query["logical_form"] = render_logical_form(query)
    return query
