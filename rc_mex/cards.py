from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GeneratedCardText:
    predicate_description: str
    argument_1_role: str
    argument_2_role: str
    domain: str
    range: str
    direction: str
    positive_rule: str
    negative_rule: str
    confusable_relations: list[str]
    minimal_decision_test: str
    valid_direction_explanation: str
    invalid_swapped_direction_explanation: str
    confidence: float
    opaque: bool
    opaque_reason: str
    raw_output: str


@dataclass
class RelationCard:
    primitive_id: str
    relation_id: str
    direction: str
    card_variant: str
    obfuscation_mode: str
    entity_evidence_mode: str
    positive_examples_train: list[dict[str, Any]]
    positive_examples_heldout: list[dict[str, Any]]
    hard_negative_examples_train: list[dict[str, Any]]
    hard_negative_examples_heldout: list[dict[str, Any]]
    random_negative_examples_heldout: list[dict[str, Any]]
    swapped_direction_examples_heldout: list[dict[str, Any]]
    description: str
    domain_types: list[str]
    range_types: list[str]
    argument_direction: str
    confidence: float
    opaque_reason: str
    generated: dict[str, Any]

    @property
    def opaque(self) -> bool:
        return bool(self.opaque_reason)

    def to_json(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "card_variant": self.card_variant,
            "obfuscation_mode": self.obfuscation_mode,
            "entity_evidence_mode": self.entity_evidence_mode,
            "positive_examples_train": self.positive_examples_train,
            "positive_examples_heldout": self.positive_examples_heldout,
            "hard_negative_examples_train": self.hard_negative_examples_train,
            "hard_negative_examples_heldout": self.hard_negative_examples_heldout,
            "random_negative_examples_heldout": self.random_negative_examples_heldout,
            "swapped_direction_examples_heldout": self.swapped_direction_examples_heldout,
            "description": self.description,
            "domain_types": self.domain_types,
            "range_types": self.range_types,
            "argument_direction": self.argument_direction,
            "confidence": self.confidence,
            "opaque_reason": self.opaque_reason,
            "generated": self.generated,
        }
