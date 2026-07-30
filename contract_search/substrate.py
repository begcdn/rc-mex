from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("Ã", "Â", "â")):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


@dataclass(frozen=True)
class QuestionGraph:
    question_id: str
    question: str
    topic_entities: tuple[str, ...]
    gold_answers: tuple[str, ...]
    triples: tuple[tuple[str, str, str], ...]


class IndexedSubgraph:
    """An ID-preserving question subgraph decoded from GNN-RAG's release."""

    def __init__(self, triples: Iterable[tuple[str, str, str]]) -> None:
        self.forward: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.backward: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for head, relation, tail in triples:
            self.forward[head][relation].add(tail)
            self.backward[tail][relation].add(head)

    def traverse(
        self, entities: Iterable[str], relation: str, direction: str
    ) -> set[str]:
        adjacency = self.forward if direction == "forward" else self.backward
        return {
            target
            for entity in entities
            for target in adjacency.get(entity, {}).get(relation, ())
        }

    def values(self, entity: str, relation: str) -> set[str]:
        return set(self.forward.get(entity, {}).get(relation, ()))

    def relations(self, entity: str) -> set[tuple[str, str]]:
        return {
            *((relation, "forward") for relation in self.forward.get(entity, {})),
            *((relation, "backward") for relation in self.backward.get(entity, {})),
        }


class GNNRAGDataset:
    """Stream a released GNN-RAG split while restoring Freebase identifiers."""

    def __init__(self, folder: Path | str, split: str = "dev") -> None:
        self.folder = Path(folder)
        self.split = split
        self.entities = self._read_dictionary("entities.txt")
        self.relations = self._read_dictionary("relations.txt")
        self.path = self.folder / f"{split}.json"
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _read_dictionary(self, name: str) -> list[str]:
        path = self.folder / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8").splitlines()

    def __iter__(self) -> Iterator[QuestionGraph]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                triples = tuple(
                    (
                        self.entities[head],
                        self.relations[relation],
                        self.entities[tail],
                    )
                    for head, relation, tail in raw["subgraph"]["tuples"]
                )
                yield QuestionGraph(
                    question_id=raw["id"],
                    question=repair_mojibake(raw["question"]),
                    topic_entities=tuple(self.entities[index] for index in raw["entities"]),
                    gold_answers=tuple(answer["kb_id"] for answer in raw["answers"]),
                    triples=triples,
                )
