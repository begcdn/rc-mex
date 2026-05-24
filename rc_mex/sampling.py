from __future__ import annotations

import math
import random
from dataclasses import dataclass

from cigr_d_mvp1.kg import KnowledgeGraph

from .schema import Primitive, RelationExample, relation_tokens


@dataclass
class PrimitiveSamples:
    positive_train: list[RelationExample]
    positive_heldout: list[RelationExample]
    hard_negative_train: list[RelationExample]
    hard_negative_heldout: list[RelationExample]
    random_negative_heldout: list[RelationExample]
    swapped_direction_heldout: list[RelationExample]
    hard_negative_source_ids: list[str]

    def train_entity_ids(self) -> set[str]:
        ids: set[str] = set()
        for example in self.positive_train + self.hard_negative_train:
            ids.add(example.head_id)
            ids.add(example.tail_id)
        return ids

    def heldout_entity_ids(self) -> set[str]:
        ids: set[str] = set()
        for example in (
            self.positive_heldout
            + self.hard_negative_heldout
            + self.random_negative_heldout
            + self.swapped_direction_heldout
        ):
            ids.add(example.head_id)
            ids.add(example.tail_id)
        return ids

    def train_heldout_entity_overlap_rate(self) -> float:
        heldout = self.heldout_entity_ids()
        if not heldout:
            return 0.0
        return len(self.train_entity_ids() & heldout) / len(heldout)


def sample_for_primitive(
    graph: KnowledgeGraph,
    target: Primitive,
    primitives: list[Primitive],
    train_positives: int,
    heldout_positives: int,
    train_negatives: int,
    heldout_negatives: int,
    random_negatives: int,
    seed: int,
) -> PrimitiveSamples:
    rng = random.Random(f"{seed}:{target.primitive_id}")
    positives = list(target.examples)
    rng.shuffle(positives)
    positive_train = positives[:train_positives]
    positive_heldout = positives[train_positives : train_positives + heldout_positives]

    hard_sources = rank_hard_negative_primitives(target, primitives)
    hard_pool = collect_negative_examples(target, hard_sources, train_negatives + heldout_negatives, rng)
    hard_negative_train = hard_pool[:train_negatives]
    hard_negative_heldout = hard_pool[train_negatives : train_negatives + heldout_negatives]

    random_negative_heldout = sample_random_negative_pairs(
        graph=graph,
        target=target,
        count=random_negatives,
        seed=seed,
    )
    swapped_direction_heldout = swapped_direction_examples(target, positive_heldout)
    return PrimitiveSamples(
        positive_train=positive_train,
        positive_heldout=positive_heldout,
        hard_negative_train=hard_negative_train,
        hard_negative_heldout=hard_negative_heldout,
        random_negative_heldout=random_negative_heldout,
        swapped_direction_heldout=swapped_direction_heldout,
        hard_negative_source_ids=[primitive.primitive_id for primitive in hard_sources[:8]],
    )


def rank_hard_negative_primitives(target: Primitive, primitives: list[Primitive]) -> list[Primitive]:
    scored = []
    for candidate in primitives:
        if candidate.primitive_id == target.primitive_id:
            continue
        scored.append((hard_negative_score(target, candidate), candidate.primitive_id, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored]


def hard_negative_score(target: Primitive, candidate: Primitive) -> float:
    domain = jaccard(set(target.domain_type_counts), set(candidate.domain_type_counts))
    range_ = jaccard(set(target.range_type_counts), set(candidate.range_type_counts))
    head_overlap = jaccard(target.head_ids, candidate.head_ids)
    tail_overlap = jaccard(target.tail_ids, candidate.tail_ids)
    endpoint_overlap = jaccard(target.head_ids | target.tail_ids, candidate.head_ids | candidate.tail_ids)
    target_size = max(1, target.cardinality)
    candidate_size = max(1, candidate.cardinality)
    cardinality = 1.0 / (1.0 + abs(math.log(target_size) - math.log(candidate_size)))
    inverse_bonus = 1.0 if (
        target.relation_id == candidate.relation_id and target.direction != candidate.direction
    ) else 0.0
    token_overlap = jaccard(relation_tokens(target.relation_id), relation_tokens(candidate.relation_id))
    return (
        2.0 * domain
        + 2.0 * range_
        + head_overlap
        + tail_overlap
        + endpoint_overlap
        + cardinality
        + 2.0 * inverse_bonus
        + 0.5 * token_overlap
    )


def collect_negative_examples(
    target: Primitive,
    hard_sources: list[Primitive],
    count: int,
    rng: random.Random,
) -> list[RelationExample]:
    target_extension = target.extension
    examples: list[RelationExample] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source in hard_sources:
        shuffled = list(source.examples)
        rng.shuffle(shuffled)
        for example in shuffled:
            key = (example.head_id, example.tail_id, example.relation_id, example.direction)
            if example.pair in target_extension or key in seen:
                continue
            examples.append(example)
            seen.add(key)
            if len(examples) >= count:
                return examples
    return examples


def sample_random_negative_pairs(
    graph: KnowledgeGraph,
    target: Primitive,
    count: int,
    seed: int,
) -> list[RelationExample]:
    rng = random.Random(f"{seed}:{target.primitive_id}:random")
    entity_ids = sorted(graph.entities)
    target_extension = target.extension
    out: list[RelationExample] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(100, count * 100)
    while len(out) < count and attempts < max_attempts and len(entity_ids) >= 2:
        attempts += 1
        head_id = rng.choice(entity_ids)
        tail_id = rng.choice(entity_ids)
        pair = (head_id, tail_id)
        if head_id == tail_id or pair in target_extension or pair in seen:
            continue
        seen.add(pair)
        out.append(
            RelationExample(
                head_id=head_id,
                tail_id=tail_id,
                relation_id="__random_negative__",
                direction="none",
                source_primitive_id="RANDOM",
            )
        )
    return out


def swapped_direction_examples(target: Primitive, positives: list[RelationExample]) -> list[RelationExample]:
    extension = target.extension
    out: list[RelationExample] = []
    for example in positives:
        swapped_pair = (example.tail_id, example.head_id)
        if swapped_pair in extension:
            continue
        out.append(
            RelationExample(
                head_id=example.tail_id,
                tail_id=example.head_id,
                relation_id=example.relation_id,
                direction=f"swapped_from_{example.direction}",
                source_primitive_id=example.source_primitive_id,
            )
        )
    return out


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
