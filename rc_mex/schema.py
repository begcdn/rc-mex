from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from cigr_d_mvp1.kg import KnowledgeGraph


@dataclass(frozen=True)
class RelationExample:
    head_id: str
    tail_id: str
    relation_id: str
    direction: str
    source_primitive_id: str

    @property
    def pair(self) -> tuple[str, str]:
        return (self.head_id, self.tail_id)


@dataclass
class Primitive:
    primitive_id: str
    relation_id: str
    direction: str
    examples: list[RelationExample]
    domain_type_counts: dict[str, int]
    range_type_counts: dict[str, int]

    @property
    def extension(self) -> set[tuple[str, str]]:
        return {example.pair for example in self.examples}

    @property
    def head_ids(self) -> set[str]:
        return {example.head_id for example in self.examples}

    @property
    def tail_ids(self) -> set[str]:
        return {example.tail_id for example in self.examples}

    @property
    def cardinality(self) -> int:
        return len(self.examples)


def inventory_primitives(graph: KnowledgeGraph, min_examples: int = 1) -> list[Primitive]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for head_id in sorted(graph.entities):
        for relation in graph.iter_relations(head_id):
            relation_id = str(relation.get("predicate", ""))
            direction = str(relation.get("direction", ""))
            tail_id = str(relation.get("object", ""))
            if relation_id and direction and tail_id:
                grouped[(relation_id, direction)].append((head_id, tail_id))

    primitives: list[Primitive] = []
    for index, ((relation_id, direction), pairs) in enumerate(sorted(grouped.items()), start=1):
        unique_pairs = sorted(set(pairs))
        if len(unique_pairs) < min_examples:
            continue
        primitive_id = f"P{index:05d}"
        examples = [
            RelationExample(
                head_id=head_id,
                tail_id=tail_id,
                relation_id=relation_id,
                direction=direction,
                source_primitive_id=primitive_id,
            )
            for head_id, tail_id in unique_pairs
        ]
        primitives.append(
            Primitive(
                primitive_id=primitive_id,
                relation_id=relation_id,
                direction=direction,
                examples=examples,
                domain_type_counts=type_counts(graph, [head for head, _ in unique_pairs]),
                range_type_counts=type_counts(graph, [tail for _, tail in unique_pairs]),
            )
        )
    return primitives


def type_counts(graph: KnowledgeGraph, entity_ids: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entity_id in entity_ids:
        for type_id in graph.entity_type_ids(entity_id):
            counts[type_id] += 1
    return dict(counts)


def top_type_names(graph: KnowledgeGraph, counts: dict[str, int], limit: int = 8) -> list[str]:
    return [
        graph.entity_name(type_id)
        for type_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def relation_tokens(relation_id: str) -> set[str]:
    token = ""
    tokens: set[str] = set()
    for char in relation_id.casefold():
        if char.isalnum():
            token += char
        else:
            if token:
                tokens.add(token)
            token = ""
    if token:
        tokens.add(token)
    return tokens
