from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .substrate import IndexedSubgraph


@dataclass(frozen=True)
class Hop:
    relation: str
    direction: str


@dataclass(frozen=True)
class Constraint:
    source_index: int
    relation: str
    operator: str
    argument: str
    argument_type: str
    optional: bool = False


@dataclass(frozen=True)
class Order:
    source_index: int
    relation: str
    descending: bool
    offset: int
    count: int


@dataclass(frozen=True)
class ReferenceProgram:
    topic_entity: str
    hops: tuple[Hop, ...]
    constraints: tuple[Constraint, ...] = ()
    order: Order | None = None
    exclude_topic_from_answers: bool = True


@dataclass(frozen=True)
class Execution:
    answers: frozenset[str]
    bindings: tuple[tuple[str, ...], ...]
    traversed_identity_hop: bool


def _comparable(value: str):
    plain = value.strip('"')
    match = re.match(r"^(.+?)\^\^", plain)
    if match:
        plain = match.group(1).strip('"')
    try:
        return datetime.fromisoformat(plain.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return float(plain)
    except ValueError:
        return plain.casefold()


def _matches(value: str, operator: str, argument: str) -> bool:
    left, right = _comparable(value), _comparable(argument)
    try:
        if operator == "Equal":
            return left == right
        if operator == "LessOrEqual":
            return left <= right
        if operator == "GreaterOrEqual":
            return left >= right
        if operator == "LessThan":
            return left < right
        if operator == "GreaterThan":
            return left > right
        if operator == "NotEqual":
            return left != right
    except TypeError:
        return False
    raise ValueError(f"unsupported constraint operator: {operator}")


def _constraint_holds(
    graph: IndexedSubgraph, binding: tuple[str, ...], constraint: Constraint
) -> bool:
    node_position = constraint.source_index + 1
    if node_position >= len(binding):
        return False
    values = graph.values(binding[node_position], constraint.relation)
    if not values:
        return constraint.optional
    return any(_matches(value, constraint.operator, constraint.argument) for value in values)


def execute_reference(
    graph: IndexedSubgraph,
    program: ReferenceProgram,
) -> Execution:
    bindings: set[tuple[str, ...]] = {(program.topic_entity,)}
    traversed_identity = False
    for hop in program.hops:
        next_bindings: set[tuple[str, ...]] = set()
        for binding in bindings:
            for target in graph.traverse((binding[-1],), hop.relation, hop.direction):
                traversed_identity |= target == binding[-1]
                next_bindings.add((*binding, target))
        bindings = next_bindings
        if not bindings:
            break

    for constraint in program.constraints:
        bindings = {
            binding
            for binding in bindings
            if _constraint_holds(graph, binding, constraint)
        }

    ordered = sorted(bindings)
    if program.order is not None:
        order = program.order

        def key(binding: tuple[str, ...]):
            position = order.source_index + 1
            values = (
                graph.values(binding[position], order.relation)
                if position < len(binding)
                else set()
            )
            if not values:
                return (1, "")
            converted = sorted((_comparable(value) for value in values), key=str)
            return (0, converted[-1] if order.descending else converted[0])

        ordered = sorted(ordered, key=key, reverse=order.descending)
        ordered = ordered[order.offset : order.offset + order.count]

    answers = {binding[-1] for binding in ordered}
    if program.exclude_topic_from_answers:
        answers.discard(program.topic_entity)
    return Execution(frozenset(answers), tuple(ordered), traversed_identity)


def answer_set_f1(predicted: Iterable[str], gold: Iterable[str]) -> float:
    predicted_set, gold_set = set(predicted), set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0
