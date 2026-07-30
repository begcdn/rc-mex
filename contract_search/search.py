from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .conjunctive import Atom, ConjunctiveProgram, execute_conjunctive
from .substrate import IndexedSubgraph, QuestionGraph


BEAM_WIDTH = 10
MAX_HOPS = 4
SCORED_EXPANSION_CAP = 500
PER_STATE_ACTION_CAP = 50
SCORER_BATCH_SIZE = 16


class StateScorer(Protocol):
    calls: int

    def score(self, question: str, states: Sequence[str]) -> list[float]: ...


def relation_text(relation: str) -> str:
    return " ".join(relation.split(".")[-2:]).replace("_", " ")


def serialize_program(program: ConjunctiveProgram, topic_entities: tuple[str, ...]) -> str:
    anchors = {entity: f"ANCHOR_{index + 1}" for index, entity in enumerate(topic_entities)}

    def term(value: str) -> str:
        return anchors.get(value, value)

    lines = [
        f"{term(atom.head)} --[{relation_text(atom.relation)}]--> {term(atom.tail)}"
        for atom in program.atoms
    ]
    if program.order is not None:
        direction = "largest/latest" if program.order.descending else "smallest/earliest"
        lines.append(f"select {direction} {program.order.variable}")
    lines.append(f"answer variable: {program.select}")
    return "\n".join(lines)


def program_record(program: ConjunctiveProgram) -> dict:
    return {
        "select": program.select,
        "atoms": [
            {"head": atom.head, "relation": atom.relation, "tail": atom.tail}
            for atom in program.atoms
        ],
        "filters": [
            {
                "variable": item.variable,
                "operator": item.operator,
                "argument": item.argument,
            }
            for item in program.filters
        ],
        "optional_filters": [
            {
                "source": item.source,
                "relation": item.relation,
                "operator": item.operator,
                "argument": item.argument,
            }
            for item in program.optional_filters
        ],
        "exclusions": [list(item) for item in program.exclusions],
        "order": (
            {
                "variable": program.order.variable,
                "descending": program.order.descending,
                "limit": program.order.limit,
            }
            if program.order
            else None
        ),
    }


def _proposed_atom_records(
    programs: Sequence[ConjunctiveProgram],
) -> list[dict[str, str]]:
    atoms = {
        (atom.head, atom.relation, atom.tail)
        for program in programs
        for atom in program.atoms
    }
    return [
        {"head": head, "relation": relation, "tail": tail}
        for head, relation, tail in sorted(atoms)
    ]


class LexicalScorer:
    """CPU-only smoke-test scorer. It is never used for a reported gate."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, question: str, states: Sequence[str]) -> list[float]:
        self.calls += 1
        query = set(re.findall(r"[a-z0-9]+", question.casefold()))
        scores = []
        for state in states:
            words = set(re.findall(r"[a-z0-9]+", state.casefold()))
            overlap = len(query & words) / max(math.sqrt(len(query) * len(words)), 1)
            scores.append(overlap)
        return scores


class TransformerPairScorer:
    def __init__(self, model_path: str, device_name: str = "auto") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.calls = 0

    def score(self, question: str, states: Sequence[str]) -> list[float]:
        import torch

        scores = []
        for start in range(0, len(states), SCORER_BATCH_SIZE):
            batch = list(states[start : start + SCORER_BATCH_SIZE])
            encoded = self.tokenizer(
                [question] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits.float().reshape(-1)
            scores.extend(logits.cpu().tolist())
            self.calls += 1
        return scores


@dataclass(frozen=True)
class SearchState:
    program: ConjunctiveProgram
    answers: frozenset[str]
    score: float
    finished: bool = False

    @property
    def key(self) -> tuple:
        return (
            self.program.select,
            self.program.atoms,
            self.program.order,
            self.finished,
        )


@dataclass(frozen=True)
class SearchResult:
    state: SearchState | None
    scored_expansions: int
    scorer_calls: int
    stopped_by_finish: bool
    cap_reached: bool
    trace: tuple[dict, ...]


def _propose_initial(graph: IndexedSubgraph, topics: tuple[str, ...]) -> list[ConjunctiveProgram]:
    proposals = []
    for topic in dict.fromkeys(topics):
        for relation, direction in sorted(graph.relations(topic)):
            atom = (
                Atom(topic, relation, "?x")
                if direction == "forward"
                else Atom("?x", relation, topic)
            )
            proposals.append(ConjunctiveProgram(select="?x", atoms=(atom,)))
    return proposals


def _relation_relevance(question: str, program: ConjunctiveProgram) -> float:
    query = set(re.findall(r"[a-z0-9]+", question.casefold()))
    relation_words = {
        word
        for atom in program.atoms
        for word in re.findall(r"[a-z0-9]+", relation_text(atom.relation))
    }
    return len(query & relation_words) / max(len(relation_words), 1)


def _propose_extensions(
    graph: IndexedSubgraph,
    state: SearchState,
    topics: tuple[str, ...],
    question: str,
) -> list[ConjunctiveProgram]:
    answer_var = state.program.select
    proposals: dict[tuple, ConjunctiveProgram] = {}
    next_var = f"?v{len(state.program.atoms) + 1}"
    for answer in state.answers:
        for relation, direction in graph.relations(answer):
            atom = (
                Atom(answer_var, relation, next_var)
                if direction == "forward"
                else Atom(next_var, relation, answer_var)
            )
            if atom in state.program.atoms:
                continue
            program = ConjunctiveProgram(
                select=next_var,
                atoms=(*state.program.atoms, atom),
            )
            proposals[(program.select, program.atoms)] = program

            for topic in topics:
                if topic == answer:
                    continue
                targets = graph.traverse((answer,), relation, direction)
                if topic not in targets:
                    continue
                constraint = (
                    Atom(answer_var, relation, topic)
                    if direction == "forward"
                    else Atom(topic, relation, answer_var)
                )
                if constraint in state.program.atoms:
                    continue
                constrained = ConjunctiveProgram(
                    select=answer_var,
                    atoms=(*state.program.atoms, constraint),
                )
                proposals[(constrained.select, constrained.atoms)] = constrained
    ranked = sorted(
        proposals.values(),
        key=lambda program: (
            -_relation_relevance(question, program),
            tuple((atom.relation, atom.head, atom.tail) for atom in program.atoms),
        ),
    )
    return ranked[:PER_STATE_ACTION_CAP]


def _materialize(
    graph: IndexedSubgraph,
    programs: Sequence[ConjunctiveProgram],
) -> list[tuple[ConjunctiveProgram, frozenset[str]]]:
    output = []
    for program in programs:
        execution = execute_conjunctive(graph, program)
        if not execution.answers or execution.traversed_identity_atom:
            continue
        output.append((program, execution.answers))
    return output


def search_question(
    row: QuestionGraph,
    scorer: StateScorer,
    beam_width: int = BEAM_WIDTH,
    max_hops: int = MAX_HOPS,
    expansion_cap: int = SCORED_EXPANSION_CAP,
) -> SearchResult:
    graph = IndexedSubgraph(row.triples)
    scored = 0
    trace = []
    initial = _materialize(graph, _propose_initial(graph, row.topic_entities))
    if not initial:
        return SearchResult(None, 0, scorer.calls, False, False, ())

    initial = sorted(
        initial,
        key=lambda item: (
            -_relation_relevance(row.question, item[0]),
            tuple(
                (atom.relation, atom.head, atom.tail)
                for atom in item[0].atoms
            ),
        ),
    )[:expansion_cap]
    texts = [serialize_program(program, row.topic_entities) for program, _ in initial]
    scores = scorer.score(row.question, texts)
    scored += len(initial)
    beam = [
        SearchState(program, answers, score)
        for (program, answers), score in zip(initial, scores, strict=True)
    ]
    beam = sorted(beam, key=lambda state: state.score, reverse=True)[:beam_width]
    trace.append(
        {
            "round": 1,
            "scored": len(initial),
            "proposed_atoms": _proposed_atom_records(
                [program for program, _ in initial]
            ),
            "beam": [
                {
                    "score": state.score,
                    "answers": sorted(state.answers),
                    "program": serialize_program(state.program, row.topic_entities),
                    "program_state": program_record(state.program),
                }
                for state in beam
            ],
        }
    )

    for depth in range(2, max_hops + 1):
        remaining = expansion_cap - scored
        if remaining <= 0:
            break
        candidates: list[tuple[ConjunctiveProgram, frozenset[str], bool]] = []
        for state in beam:
            candidates.append((state.program, state.answers, True))
            for program, answers in _materialize(
                graph,
                _propose_extensions(
                    graph, state, row.topic_entities, row.question
                ),
            ):
                candidates.append((program, answers, False))
        deduplicated = {}
        for program, answers, finished in candidates:
            key = (program.select, program.atoms, program.order, finished)
            deduplicated.setdefault(key, (program, answers, finished))
        candidates = list(deduplicated.values())[:remaining]
        if not candidates:
            break
        texts = [
            serialize_program(program, row.topic_entities)
            + ("\n[FINISH]" if finished else "")
            for program, _, finished in candidates
        ]
        scores = scorer.score(row.question, texts)
        scored += len(candidates)
        ranked = sorted(
            (
                SearchState(program, answers, score, finished)
                for (program, answers, finished), score in zip(
                    candidates, scores, strict=True
                )
            ),
            key=lambda state: state.score,
            reverse=True,
        )
        trace.append(
            {
                "round": depth,
                "scored": len(candidates),
                "proposed_atoms": _proposed_atom_records(
                    [program for program, _, _ in candidates]
                ),
                "beam": [
                    {
                        "score": state.score,
                        "answers": sorted(state.answers),
                        "finished": state.finished,
                        "program": serialize_program(
                            state.program, row.topic_entities
                        ),
                        "program_state": program_record(state.program),
                    }
                    for state in ranked[:beam_width]
                ],
            }
        )
        if ranked[0].finished:
            return SearchResult(
                ranked[0], scored, scorer.calls, True, scored >= expansion_cap, tuple(trace)
            )
        beam = [state for state in ranked if not state.finished][:beam_width]
        if not beam:
            break

    best = max(beam, key=lambda state: state.score) if beam else None
    return SearchResult(
        best, scored, scorer.calls, False, scored >= expansion_cap, tuple(trace)
    )
