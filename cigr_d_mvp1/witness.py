from __future__ import annotations

from dataclasses import dataclass

from .kg import KnowledgeGraph, ProofTriple, RelationCandidate


@dataclass
class WitnessCard:
    candidate_id: str
    predicate: str
    direction: str
    display_relation: str
    returned_entities: list[str]
    returned_types: list[str]
    cardinality: int
    proof_triples: list[str]
    execution_status: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.predicate, self.direction)

    def schema_text(self) -> str:
        return (
            f"{self.candidate_id}. relation={self.display_relation}; "
            f"direction={self.direction}"
        )

    def witness_text(self) -> str:
        entities = ", ".join(self.returned_entities) if self.returned_entities else "<none>"
        types = ", ".join(self.returned_types) if self.returned_types else "<unknown>"
        proofs = " | ".join(self.proof_triples) if self.proof_triples else "<none>"
        return (
            f"{self.candidate_id}. relation={self.display_relation}; direction={self.direction}; "
            f"cardinality={self.cardinality}; returned_entities=[{entities}]; "
            f"returned_types=[{types}]; proof_triples=[{proofs}]; "
            f"status={self.execution_status}"
        )


class StableAnonymizer:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.mapping: dict[str, str] = {}

    def get(self, key: str) -> str:
        if key not in self.mapping:
            self.mapping[key] = f"{self.prefix}_{len(self.mapping) + 1:03d}"
        return self.mapping[key]


def build_witness_cards(
    graph: KnowledgeGraph,
    current_entity_ids: set[str],
    candidates: list[RelationCandidate],
    label_mode: str,
    witness_mode: str,
    returned_sample_size: int,
) -> list[WitnessCard]:
    relation_anonymizer = StableAnonymizer("R")
    entity_anonymizer = StableAnonymizer("ENTITY")
    cards: list[WitnessCard] = []
    for idx, candidate in enumerate(candidates, start=1):
        output_ids, proofs = graph.follow(
            current_entity_ids,
            candidate.predicate,
            candidate.direction,
            max_proofs=returned_sample_size,
        )
        display_relation = (
            candidate.predicate
            if label_mode == "normal"
            else relation_anonymizer.get(candidate.predicate)
        )
        returned_entities, returned_types, proof_lines = render_witness(
            graph=graph,
            output_ids=output_ids,
            proofs=proofs,
            witness_mode=witness_mode,
            returned_sample_size=returned_sample_size,
            display_relation=display_relation,
            entity_anonymizer=entity_anonymizer,
        )
        cards.append(
            WitnessCard(
                candidate_id=f"C{idx:03d}",
                predicate=candidate.predicate,
                direction=candidate.direction,
                display_relation=display_relation,
                returned_entities=returned_entities,
                returned_types=returned_types,
                cardinality=len(output_ids),
                proof_triples=proof_lines,
                execution_status="nonempty" if output_ids else "empty",
            )
        )
    return cards


def render_witness(
    graph: KnowledgeGraph,
    output_ids: set[str],
    proofs: list[ProofTriple],
    witness_mode: str,
    returned_sample_size: int,
    display_relation: str,
    entity_anonymizer: StableAnonymizer,
) -> tuple[list[str], list[str], list[str]]:
    sample_ids = sorted(output_ids)[:returned_sample_size]
    if witness_mode == "real":
        returned_entities = [graph.entity_name(entity_id) for entity_id in sample_ids]
    elif witness_mode == "anon_entities_types":
        returned_entities = [entity_anonymizer.get(entity_id) for entity_id in sample_ids]
    else:
        raise ValueError(f"unknown witness mode: {witness_mode}")

    type_counts: dict[str, int] = {}
    for entity_id in sample_ids:
        for type_name in graph.entity_type_names(entity_id, limit=4):
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
    returned_types = [
        f"{name} ({count})"
        for name, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]

    proof_lines = []
    for proof in proofs[:returned_sample_size]:
        if witness_mode == "real":
            subject = graph.entity_name(proof.subject_id)
            obj = graph.entity_name(proof.object_id)
        else:
            subject = entity_anonymizer.get(proof.subject_id)
            obj = entity_anonymizer.get(proof.object_id)
        proof_lines.append(f"({subject}, {display_relation}, {obj}, {proof.direction})")
    return returned_entities, returned_types, proof_lines
