from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class PrimitiveKey:
    relation_id: str
    direction: str
    raw_relation_id: str = field(default="", compare=False)
    raw_direction: str = field(default="", compare=False)

    @property
    def tuple_key(self) -> tuple[str, str]:
        return (self.relation_id, self.direction)

    def display(self) -> str:
        return f"{self.relation_id}/{self.direction}"

    def to_json(self) -> dict[str, str]:
        return {
            "relation_id": self.relation_id,
            "direction": self.direction,
            "raw_relation_id": self.raw_relation_id,
            "raw_direction": self.raw_direction,
            "display": self.display(),
        }


def primitive_key(relation_id: Any, direction: Any) -> PrimitiveKey:
    raw_relation = "" if relation_id is None else str(relation_id)
    raw_direction = "" if direction is None else str(direction)
    return PrimitiveKey(
        relation_id=normalize_relation_id(raw_relation),
        direction=normalize_direction(raw_direction),
        raw_relation_id=raw_relation,
        raw_direction=raw_direction,
    )


def normalize_relation_id(value: str) -> str:
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def normalize_direction(value: str) -> str:
    normalized = str(value).strip().casefold()
    aliases = {
        "fwd": "forward",
        "forwards": "forward",
        "out": "forward",
        "outgoing": "forward",
        "subject_to_object": "forward",
        "subj_to_obj": "forward",
        "back": "backward",
        "bwd": "backward",
        "backwards": "backward",
        "in": "backward",
        "incoming": "backward",
        "object_to_subject": "backward",
        "obj_to_subj": "backward",
    }
    return aliases.get(normalized, normalized)


def key_from_card(card: dict[str, Any]) -> PrimitiveKey:
    return primitive_key(card.get("relation_id", ""), card.get("direction", ""))


def key_from_instance(instance: Any) -> PrimitiveKey:
    return primitive_key(instance.gold_predicate, instance.gold_direction)


def key_from_relation_candidate(candidate: Any) -> PrimitiveKey:
    return primitive_key(candidate.predicate, candidate.direction)


def key_from_program_step(step: dict[str, Any]) -> PrimitiveKey | None:
    if step.get("function") != "Relate":
        return None
    inputs = step.get("inputs", []) or []
    if len(inputs) < 2:
        return None
    return primitive_key(inputs[0], inputs[1])


def possible_relation_normalization_matches(
    loaded_keys: set[PrimitiveKey],
    gold_keys: set[PrimitiveKey],
    limit: int = 20,
) -> list[dict[str, str]]:
    loaded_by_relation = {key.relation_id: key for key in loaded_keys}
    out = []
    for gold in sorted(gold_keys):
        if gold.relation_id in loaded_by_relation and gold not in loaded_keys:
            loaded = loaded_by_relation[gold.relation_id]
            out.append(
                {
                    "gold": gold.display(),
                    "loaded": loaded.display(),
                    "issue": "same normalized relation, different direction",
                }
            )
        if len(out) >= limit:
            break
    return out


def possible_direction_mismatches(
    loaded_keys: set[PrimitiveKey],
    gold_keys: set[PrimitiveKey],
    limit: int = 20,
) -> list[dict[str, str]]:
    loaded_by_relation: dict[str, set[str]] = {}
    for key in loaded_keys:
        loaded_by_relation.setdefault(key.relation_id, set()).add(key.direction)
    out = []
    for gold in sorted(gold_keys):
        directions = loaded_by_relation.get(gold.relation_id, set())
        if directions and gold.direction not in directions:
            out.append(
                {
                    "relation_id": gold.relation_id,
                    "gold_direction": gold.direction,
                    "loaded_directions": ", ".join(sorted(directions)),
                }
            )
        if len(out) >= limit:
            break
    return out


def possible_text_mismatches(
    loaded_keys: set[PrimitiveKey],
    gold_keys: set[PrimitiveKey],
    limit: int = 20,
) -> list[dict[str, str]]:
    loaded_relations = {key.relation_id for key in loaded_keys}
    out = []
    for gold in sorted(gold_keys):
        close = closest_relation(gold.relation_id, loaded_relations)
        if close and close != gold.relation_id:
            out.append({"gold_relation_id": gold.relation_id, "closest_loaded_relation_id": close})
        if len(out) >= limit:
            break
    return out


def closest_relation(target: str, candidates: set[str]) -> str:
    target_tokens = token_set(target)
    best = ("", 0.0)
    for candidate in candidates:
        score = jaccard(target_tokens, token_set(candidate))
        if score > best[1]:
            best = (candidate, score)
    return best[0] if best[1] >= 0.5 else ""


def token_set(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
