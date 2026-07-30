from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .program import _comparable, _matches
from .substrate import IndexedSubgraph


@dataclass(frozen=True)
class Atom:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class ValueFilter:
    variable: str
    operator: str
    argument: str


@dataclass(frozen=True)
class OptionalRelationFilter:
    source: str
    relation: str
    operator: str
    argument: str


@dataclass(frozen=True)
class QueryOrder:
    variable: str
    descending: bool
    limit: int


@dataclass(frozen=True)
class ConjunctiveProgram:
    select: str
    atoms: tuple[Atom, ...]
    filters: tuple[ValueFilter, ...] = ()
    optional_filters: tuple[OptionalRelationFilter, ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()
    order: QueryOrder | None = None


@dataclass(frozen=True)
class ConjunctiveExecution:
    answers: frozenset[str]
    bindings: tuple[dict[str, str], ...]
    traversed_identity_atom: bool


def is_variable(term: str) -> bool:
    return term.startswith("?")


def _resolve(term: str, binding: dict[str, str]) -> str | None:
    if is_variable(term):
        return binding.get(term)
    return term.removeprefix("ns:")


def _extend_atom(
    graph: IndexedSubgraph,
    binding: dict[str, str],
    atom: Atom,
) -> tuple[list[dict[str, str]], bool, bool]:
    head = _resolve(atom.head, binding)
    tail = _resolve(atom.tail, binding)
    identity = False
    if head is None and tail is None:
        return [], False, False
    if head is not None and tail is not None:
        valid = tail in graph.traverse((head,), atom.relation, "forward")
        return ([binding] if valid else []), True, valid and head == tail

    output = []
    if head is not None:
        for target in graph.traverse((head,), atom.relation, "forward"):
            updated = dict(binding)
            updated[atom.tail] = target
            output.append(updated)
            identity |= target == head
    else:
        for source in graph.traverse((tail,), atom.relation, "backward"):
            updated = dict(binding)
            updated[atom.head] = source
            output.append(updated)
            identity |= source == tail
    return output, True, identity


def _apply_atoms(
    graph: IndexedSubgraph, program: ConjunctiveProgram
) -> tuple[list[dict[str, str]], bool]:
    bindings: list[dict[str, str]] = [{}]
    pending = list(program.atoms)
    identity = False
    while pending and bindings:
        progressed = False
        for atom in list(pending):
            if not any(
                _resolve(atom.head, binding) is not None
                or _resolve(atom.tail, binding) is not None
                for binding in bindings
            ):
                continue
            expanded = []
            for binding in bindings:
                rows, evaluable, atom_identity = _extend_atom(graph, binding, atom)
                if not evaluable:
                    rows = [binding]
                expanded.extend(rows)
                identity |= atom_identity
            bindings = expanded
            pending.remove(atom)
            progressed = True
            break
        if not progressed:
            return [], identity
    return bindings, identity


def _passes_filters(
    graph: IndexedSubgraph,
    binding: dict[str, str],
    program: ConjunctiveProgram,
) -> bool:
    for left, right in program.exclusions:
        if _resolve(left, binding) == _resolve(right, binding):
            return False
    for item in program.filters:
        value = _resolve(item.variable, binding)
        if value is None or not _matches(value, item.operator, item.argument):
            return False
    for item in program.optional_filters:
        source = _resolve(item.source, binding)
        if source is None:
            return False
        values = graph.values(source, item.relation)
        if values and not any(
            _matches(value, item.operator, item.argument) for value in values
        ):
            return False
    return True


def execute_conjunctive(
    graph: IndexedSubgraph, program: ConjunctiveProgram
) -> ConjunctiveExecution:
    bindings, identity = _apply_atoms(graph, program)
    bindings = [
        binding for binding in bindings if _passes_filters(graph, binding, program)
    ]
    if program.order is not None:
        order = program.order
        bindings.sort(
            key=lambda binding: _comparable(binding.get(order.variable, "")),
            reverse=order.descending,
        )
        bindings = bindings[: order.limit]
    answers = {
        answer
        for binding in bindings
        if (answer := _resolve(program.select, binding)) is not None
    }
    return ConjunctiveExecution(frozenset(answers), tuple(bindings), identity)


def exact_f1(predicted: Iterable[str], gold: Iterable[str]) -> tuple[bool, float]:
    from .program import answer_set_f1

    predicted_set, gold_set = set(predicted), set(gold)
    return predicted_set == gold_set, answer_set_f1(predicted_set, gold_set)
