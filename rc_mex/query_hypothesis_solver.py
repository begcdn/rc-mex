"""Exact conjunctive execution over uncertain grounded query hypotheses.

The solver's unit of competition is a grounded query, not an individual
answer binding.  A grounded query keeps every assignment satisfying the same
relation choices and therefore denotes an answer set.  This is the semantic
boundary needed by set-valued KGQA: query hypotheses are ranked first, then
the selected query's full denotation is returned.

Language-to-schema scoring is deliberately injected through ``propose``.
The module knows nothing about benchmark relation vocabularies, encoders, or
entity-linking conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence


Binding = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RelationAtom:
    """A schema-independent binary predicate between two graph terms."""

    atom_id: str
    left: str
    predicate: str
    right: str


@dataclass(frozen=True)
class ConstraintGraph:
    """A conjunctive query with one distinguished answer variable."""

    atoms: tuple[RelationAtom, ...]
    answer_variable: str
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        variable_set = set(self.variables)
        if self.answer_variable not in variable_set:
            raise ValueError("answer_variable must be declared in variables")
        if not self.atoms:
            raise ValueError("constraint graph must contain at least one atom")
        for atom in self.atoms:
            for term in (atom.left, atom.right):
                if term.startswith("?") and term not in variable_set:
                    raise ValueError(f"undeclared variable: {term}")


@dataclass(frozen=True)
class RelationCandidate:
    """One executable grounding proposed for a semantic relation atom."""

    relation_id: str
    direction: str
    log_score: float
    extension: tuple[tuple[str, str], ...]
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str]:
        return self.relation_id, self.direction


@dataclass(frozen=True)
class GroundedAtom:
    atom_id: str
    relation_id: str
    direction: str
    log_score: float
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class QueryHypothesis:
    """A grounded query and all of its satisfying variable assignments."""

    bindings: tuple[Binding, ...]
    grounded_atoms: tuple[GroundedAtom, ...]
    relation_log_score: float = 0.0
    factor_log_score: float = 0.0

    @property
    def score(self) -> float:
        return self.relation_log_score + self.factor_log_score

    def binding_dicts(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(binding) for binding in self.bindings)

    def denotation(self, variable: str) -> frozenset[str]:
        return frozenset(
            value
            for binding in self.binding_dicts()
            if (value := binding.get(variable)) is not None
        )


ProposalFunction = Callable[
    [RelationAtom, QueryHypothesis, Mapping[str, str]],
    Iterable[RelationCandidate],
]
FactorFunction = Callable[[ConstraintGraph, QueryHypothesis, Mapping[str, str]], float]


def _canonical_bindings(bindings: Iterable[Mapping[str, str]]) -> tuple[Binding, ...]:
    return tuple(sorted({tuple(sorted(binding.items())) for binding in bindings}))


def _term_value(term: str, binding: Mapping[str, str], constants: Mapping[str, str]) -> str | None:
    if term in constants:
        return constants[term]
    return binding.get(term)


def _join_candidate(
    hypothesis: QueryHypothesis,
    atom: RelationAtom,
    candidate: RelationCandidate,
    constants: Mapping[str, str],
) -> tuple[Binding, ...]:
    joined: list[dict[str, str]] = []
    for raw_binding in hypothesis.bindings:
        binding = dict(raw_binding)
        expected_left = _term_value(atom.left, binding, constants)
        expected_right = _term_value(atom.right, binding, constants)
        for left_value, right_value in candidate.extension:
            if expected_left is not None and expected_left != left_value:
                continue
            if expected_right is not None and expected_right != right_value:
                continue
            extended = dict(binding)
            if atom.left not in constants:
                if atom.left in extended and extended[atom.left] != left_value:
                    continue
                extended[atom.left] = left_value
            if atom.right not in constants:
                if atom.right in extended and extended[atom.right] != right_value:
                    continue
                extended[atom.right] = right_value
            joined.append(extended)
    return _canonical_bindings(joined)


def _atom_order(graph: ConstraintGraph, constants: Mapping[str, str]) -> tuple[RelationAtom, ...]:
    """Prefer atoms touching known terms while preserving deterministic ties."""

    remaining = list(graph.atoms)
    known = set(constants)
    ordered: list[RelationAtom] = []
    while remaining:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                int(remaining[index].left in known) + int(remaining[index].right in known),
                -index,
            ),
        )
        atom = remaining.pop(best_index)
        ordered.append(atom)
        known.update((atom.left, atom.right))
    return tuple(ordered)


def _deduplicate(hypotheses: Iterable[QueryHypothesis], cap: int) -> list[QueryHypothesis]:
    best: dict[tuple[tuple[tuple[str, str, str], ...], tuple[Binding, ...]], QueryHypothesis] = {}
    for hypothesis in hypotheses:
        grounding_key = tuple(
            (atom.atom_id, atom.relation_id, atom.direction)
            for atom in hypothesis.grounded_atoms
        )
        key = grounding_key, hypothesis.bindings
        incumbent = best.get(key)
        if incumbent is None or hypothesis.score > incumbent.score:
            best[key] = hypothesis
    return sorted(best.values(), key=lambda item: -item.score)[:cap]


def solve_conjunctive_query(
    graph: ConstraintGraph,
    constants: Mapping[str, str],
    propose: ProposalFunction,
    *,
    relation_candidates_per_atom: int = 8,
    hypothesis_cap: int = 1000,
    factor: FactorFunction | None = None,
) -> list[QueryHypothesis]:
    """Ground and execute a constraint graph without splitting answer sets.

    ``propose`` may use language, schema, types, or the currently satisfying
    assignments.  Its candidate scores are query-level relation potentials.
    The solver performs exact natural joins and never divides a query score by
    the number of answer bindings it produces.
    """

    if relation_candidates_per_atom < 1:
        raise ValueError("relation_candidates_per_atom must be positive")
    if hypothesis_cap < 1:
        raise ValueError("hypothesis_cap must be positive")

    hypotheses = [QueryHypothesis(bindings=((),), grounded_atoms=())]
    for atom in _atom_order(graph, constants):
        expanded: list[QueryHypothesis] = []
        for hypothesis in hypotheses:
            candidates = sorted(
                propose(atom, hypothesis, constants),
                key=lambda item: -item.log_score,
            )[:relation_candidates_per_atom]
            for candidate in candidates:
                bindings = _join_candidate(hypothesis, atom, candidate, constants)
                if not bindings:
                    continue
                expanded.append(
                    QueryHypothesis(
                        bindings=bindings,
                        grounded_atoms=hypothesis.grounded_atoms
                        + (
                            GroundedAtom(
                                atom_id=atom.atom_id,
                                relation_id=candidate.relation_id,
                                direction=candidate.direction,
                                log_score=candidate.log_score,
                                metadata=candidate.metadata,
                            ),
                        ),
                        relation_log_score=hypothesis.relation_log_score + candidate.log_score,
                    )
                )
        hypotheses = _deduplicate(expanded, hypothesis_cap)
        if not hypotheses:
            return []

    if factor is not None:
        hypotheses = [
            QueryHypothesis(
                bindings=hypothesis.bindings,
                grounded_atoms=hypothesis.grounded_atoms,
                relation_log_score=hypothesis.relation_log_score,
                factor_log_score=factor(graph, hypothesis, constants),
            )
            for hypothesis in hypotheses
        ]
    return sorted(hypotheses, key=lambda item: -item.score)


def rank_denotations(
    graph: ConstraintGraph,
    hypotheses: Sequence[QueryHypothesis],
) -> list[tuple[frozenset[str], QueryHypothesis]]:
    """Keep the best grounded query for each distinct answer denotation."""

    best: dict[frozenset[str], QueryHypothesis] = {}
    for hypothesis in hypotheses:
        denotation = hypothesis.denotation(graph.answer_variable)
        if not denotation:
            continue
        incumbent = best.get(denotation)
        if incumbent is None or hypothesis.score > incumbent.score:
            best[denotation] = hypothesis
    return sorted(best.items(), key=lambda item: -item[1].score)
