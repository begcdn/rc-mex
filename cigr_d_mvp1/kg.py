from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RelationCandidate:
    predicate: str
    direction: str
    frequency: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.predicate, self.direction)


@dataclass(frozen=True)
class ProofTriple:
    subject_id: str
    predicate: str
    object_id: str
    direction: str


class KnowledgeGraph:
    """Lightweight KQA Pro graph access.

    KQA Pro's kb.json stores each entity with relation entries of the form
    {predicate, object, direction}. We keep both the original record and small
    indexes for candidate generation and witness execution.
    """

    def __init__(self, kb: dict[str, Any]):
        self.concepts: dict[str, dict[str, Any]] = kb.get("concepts", {})
        self.entities: dict[str, dict[str, Any]] = kb.get("entities", {})
        self.name_to_entity_ids: dict[str, list[str]] = defaultdict(list)
        self.name_to_concept_ids: dict[str, list[str]] = defaultdict(list)
        self._concept_ancestors: dict[str, set[str]] = {}

        for entity_id, entity in self.entities.items():
            name = normalize_text(entity.get("name", ""))
            if name:
                self.name_to_entity_ids[name].append(entity_id)

        for concept_id, concept in self.concepts.items():
            name = normalize_text(concept.get("name", ""))
            if name:
                self.name_to_concept_ids[name].append(concept_id)

    def all_entity_ids(self) -> set[str]:
        return set(self.entities.keys())

    def entity_name(self, entity_id: str) -> str:
        if entity_id in self.entities:
            return str(self.entities[entity_id].get("name", entity_id))
        if entity_id in self.concepts:
            return str(self.concepts[entity_id].get("name", entity_id))
        return entity_id

    def find_entities(self, name: str) -> set[str]:
        return set(self.name_to_entity_ids.get(normalize_text(name), []))

    def find_concepts(self, name: str) -> set[str]:
        return set(self.name_to_concept_ids.get(normalize_text(name), []))

    def entity_type_ids(self, entity_id: str, include_ancestors: bool = True) -> set[str]:
        entity = self.entities.get(entity_id)
        if not entity:
            return set()
        direct = set(entity.get("instanceOf", []) or [])
        if not include_ancestors:
            return direct
        out = set(direct)
        for concept_id in direct:
            out.update(self.concept_ancestors(concept_id))
        return out

    def entity_type_names(self, entity_id: str, limit: int = 8) -> list[str]:
        names = [self.entity_name(type_id) for type_id in sorted(self.entity_type_ids(entity_id))]
        return names[:limit]

    def concept_ancestors(self, concept_id: str) -> set[str]:
        if concept_id in self._concept_ancestors:
            return self._concept_ancestors[concept_id]
        seen: set[str] = set()
        queue = deque(self.concepts.get(concept_id, {}).get("instanceOf", []) or [])
        while queue:
            parent = queue.popleft()
            if parent in seen:
                continue
            seen.add(parent)
            queue.extend(self.concepts.get(parent, {}).get("instanceOf", []) or [])
        self._concept_ancestors[concept_id] = seen
        return seen

    def is_instance_of_any(self, entity_id: str, concept_ids: set[str]) -> bool:
        return bool(self.entity_type_ids(entity_id) & concept_ids)

    def iter_relations(self, entity_id: str) -> Iterable[dict[str, Any]]:
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        return entity.get("relations", []) or []

    def candidate_relations(
        self,
        entity_ids: Iterable[str],
        cap: int,
        sample_entities: int,
    ) -> list[RelationCandidate]:
        selected = sorted(set(entity_ids))[:sample_entities]
        counts: Counter[tuple[str, str]] = Counter()
        for entity_id in selected:
            for relation in self.iter_relations(entity_id):
                predicate = str(relation.get("predicate", ""))
                direction = str(relation.get("direction", ""))
                if predicate and direction:
                    counts[(predicate, direction)] += 1
        candidates = [
            RelationCandidate(predicate=p, direction=d, frequency=f)
            for (p, d), f in counts.items()
        ]
        candidates.sort(key=lambda c: (-c.frequency, c.predicate, c.direction))
        return candidates[:cap]

    def follow(
        self,
        entity_ids: Iterable[str],
        predicate: str,
        direction: str,
        max_proofs: int = 20,
    ) -> tuple[set[str], list[ProofTriple]]:
        out: set[str] = set()
        proofs: list[ProofTriple] = []
        for entity_id in sorted(set(entity_ids)):
            for relation in self.iter_relations(entity_id):
                if relation.get("predicate") != predicate:
                    continue
                if relation.get("direction") != direction:
                    continue
                object_id = str(relation.get("object", ""))
                if not object_id:
                    continue
                out.add(object_id)
                if len(proofs) < max_proofs:
                    proofs.append(
                        ProofTriple(
                            subject_id=entity_id,
                            predicate=predicate,
                            object_id=object_id,
                            direction=direction,
                        )
                    )
        return out, proofs

    def attribute_values(self, entity_id: str, key: str) -> list[dict[str, Any]]:
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        values = []
        for attribute in entity.get("attributes", []) or []:
            if attribute.get("key") == key:
                value = attribute.get("value")
                if isinstance(value, dict):
                    values.append(value)
        return values


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())
