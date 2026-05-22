from __future__ import annotations

from dataclasses import dataclass

from cigr_d_mvp1.kg import KnowledgeGraph

from .schema import Primitive, RelationExample, top_type_names


@dataclass(frozen=True)
class EvidenceCondition:
    condition_id: str
    obfuscation_mode: str
    entity_evidence_mode: str
    relation_mode: str
    entity_mode: str
    include_types: bool


CONDITIONS: dict[str, EvidenceCondition] = {
    "A": EvidenceCondition(
        condition_id="A",
        obfuscation_mode="normal",
        entity_evidence_mode="real_entities_types",
        relation_mode="normal",
        entity_mode="real",
        include_types=True,
    ),
    "B1": EvidenceCondition(
        condition_id="B1",
        obfuscation_mode="anonymized_relation",
        entity_evidence_mode="real_entities_types",
        relation_mode="anonymous",
        entity_mode="real",
        include_types=True,
    ),
    "B2": EvidenceCondition(
        condition_id="B2",
        obfuscation_mode="anonymized_relation",
        entity_evidence_mode="anonymous_entities_types",
        relation_mode="anonymous",
        entity_mode="anonymous",
        include_types=True,
    ),
    "B3": EvidenceCondition(
        condition_id="B3",
        obfuscation_mode="anonymized_relation",
        entity_evidence_mode="anonymous_entities_no_types",
        relation_mode="anonymous",
        entity_mode="anonymous",
        include_types=False,
    ),
    "C": EvidenceCondition(
        condition_id="C",
        obfuscation_mode="misleading_relation",
        entity_evidence_mode="real_entities_types",
        relation_mode="misleading",
        entity_mode="real",
        include_types=True,
    ),
}


class StableMap:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.mapping: dict[str, str] = {}

    def get(self, key: str) -> str:
        if key not in self.mapping:
            self.mapping[key] = f"{self.prefix}_{len(self.mapping) + 1:04d}"
        return self.mapping[key]


@dataclass
class RenderContext:
    relation_labels: dict[str, str]
    misleading_labels: dict[str, str]
    entity_labels: StableMap

    @classmethod
    def from_primitives(cls, primitives: list[Primitive]) -> "RenderContext":
        relation_ids = sorted({primitive.relation_id for primitive in primitives})
        relation_labels = {relation_id: f"R_{idx + 1:04d}" for idx, relation_id in enumerate(relation_ids)}
        if len(relation_ids) <= 1:
            misleading = {relation_id: "misleading_relation" for relation_id in relation_ids}
        else:
            shifted = relation_ids[1:] + relation_ids[:1]
            misleading = dict(zip(relation_ids, shifted, strict=True))
        return cls(
            relation_labels=relation_labels,
            misleading_labels=misleading,
            entity_labels=StableMap("ENTITY"),
        )

    def relation_display(self, relation_id: str, condition: EvidenceCondition) -> str:
        if condition.relation_mode == "normal":
            return relation_id
        if condition.relation_mode == "anonymous":
            return self.relation_labels.get(relation_id, relation_id)
        if condition.relation_mode == "misleading":
            return self.misleading_labels.get(relation_id, relation_id)
        raise ValueError(f"unknown relation mode: {condition.relation_mode}")

    def entity_display(self, entity_id: str, graph: KnowledgeGraph, condition: EvidenceCondition) -> str:
        if condition.entity_mode == "real":
            return graph.entity_name(entity_id)
        if condition.entity_mode == "anonymous":
            return self.entity_labels.get(entity_id)
        raise ValueError(f"unknown entity mode: {condition.entity_mode}")


def render_example(
    graph: KnowledgeGraph,
    context: RenderContext,
    condition: EvidenceCondition,
    example: RelationExample,
    include_hidden: bool = False,
) -> dict:
    row = {
        "head_id": example.head_id if include_hidden else "",
        "tail_id": example.tail_id if include_hidden else "",
        "head": context.entity_display(example.head_id, graph, condition),
        "tail": context.entity_display(example.tail_id, graph, condition),
        "head_types": graph.entity_type_names(example.head_id) if condition.include_types else [],
        "tail_types": graph.entity_type_names(example.tail_id) if condition.include_types else [],
    }
    if include_hidden:
        row.update(
            {
                "relation_id": example.relation_id,
                "direction": example.direction,
                "source_primitive_id": example.source_primitive_id,
            }
        )
    return row


def render_type_summary(
    graph: KnowledgeGraph,
    primitive: Primitive,
    condition: EvidenceCondition,
) -> tuple[list[str], list[str]]:
    if not condition.include_types:
        return [], []
    return (
        top_type_names(graph, primitive.domain_type_counts),
        top_type_names(graph, primitive.range_type_counts),
    )
