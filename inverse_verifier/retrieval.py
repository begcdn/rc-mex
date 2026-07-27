from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .data import Hop, PathSpec, path_to_dict, relation_words, write_jsonl


MAX_HOPS = 2
PATH_CAP = 200
SRTK_SCORER = "drt/srtk-scorer"
TYPE_RELATION = "common.topic.notable_types"


def encode_edge(relation: str, direction: str) -> str:
    return f"{relation}::{direction}"


def decode_edge(edge: str) -> tuple[str, str]:
    relation, separator, direction = edge.rpartition("::")
    if not separator or direction not in {"forward", "backward"}:
        raise ValueError(f"Invalid directed relation: {edge}")
    return relation, direction


class LocalQuestionGraph:
    """SRTK knowledge-graph adapter for a pre-extracted question neighborhood."""

    name = "freebase"

    def __init__(self, triples: list[list[str]]) -> None:
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.types: dict[str, set[str]] = defaultdict(set)
        self.triples = triples
        for head, relation, tail in triples:
            if relation == TYPE_RELATION and not tail.startswith(("m.", "g.")):
                self.types[head].add(tail)
                continue
            self.adjacency[head].append((encode_edge(relation, "forward"), tail))
            self.adjacency[tail].append((encode_edge(relation, "backward"), head))

    def get_neighbor_relations(self, src: str, hop: int = 1, limit: int = 10_000):
        if hop != 1:
            raise NotImplementedError("The SRTK adapter expands one relation at a time")
        return tuple(sorted({edge for edge, _ in self.adjacency.get(src, [])}))[:limit]

    def deduce_leaves(self, src: str, path, limit: int = 10_000):
        leaves = {src}
        for edge in path:
            next_leaves: set[str] = set()
            for entity in leaves:
                next_leaves.update(
                    target for candidate, target in self.adjacency.get(entity, []) if candidate == edge
                )
            leaves = next_leaves
            if not leaves:
                break
        return sorted(leaves)[:limit]

    def get_label(self, identifier: str) -> str:
        try:
            relation, direction = decode_edge(identifier)
        except ValueError:
            return identifier
        label = relation_words(relation)
        return label if direction == "forward" else f"inverse of {label}"

    def entity_type(self, entity: str, relation: str = "") -> str:
        return dominant_type(
            Counter(self.types.get(entity, set())), schema_type_hint(relation)
        )


def schema_type_hint(relation: str) -> str:
    """The entity type a Freebase relation id encodes for its subject.

    ``government.us_president.vice_president`` is stated of a US President, so the
    id itself says which of an entity's many types is the relevant one here.
    """
    pieces = relation.removeprefix("ns:").split(".")
    if len(pieces) < 2:
        return ""
    return pieces[-2].replace("_", " ").casefold()


def dominant_type(counts: Counter[str], schema_hint: str = "") -> str:
    """One type for a frontier, not the union of every type it contains.

    Joining every observed type produced answer types like "administrative division
    / body of water" for a path reaching both Ecuador and the Pacific Ocean, which
    the generator wrote straight into the question.

    Freebase gives most entities several unrelated types, and they nearly always
    tie on count, so any arbitrary tie-break decides the rendering. Alphabetical
    order produced "Abraham Lincoln, an artwork" and "Benjamin Franklin, a film
    character". The relation id resolves it when it can: a relation stated of a US
    President selects that type. Otherwise this returns "entity" rather than
    asserting one of several equally supported types, because a vague true
    description beats a confident false one.
    """
    if not counts:
        return "entity"
    best = max(counts.values())
    top = sorted(name for name, count in counts.items() if count == best)
    if len(top) == 1:
        return top[0]
    if schema_hint:
        for name in top:
            if name.casefold() == schema_hint:
                return name
        for name in top:
            if schema_hint in name.casefold() or name.casefold() in schema_hint:
                return name
    return "entity"


def materialize_path(
    graph: LocalQuestionGraph,
    anchor: str,
    directed_relations: tuple[str, ...],
) -> tuple[dict[str, Any], list[str], list[list[str]]]:
    """Execute one relation family and preserve all endpoint bindings and used facts."""
    frontier = {anchor}
    hops = []
    used_triples: set[tuple[str, str, str]] = set()
    first_relation = decode_edge(directed_relations[0])[0] if directed_relations else ""
    source_type = graph.entity_type(anchor, first_relation)
    for edge in directed_relations:
        relation, direction = decode_edge(edge)
        next_frontier: set[str] = set()
        target_types: Counter[str] = Counter()
        for entity in frontier:
            for candidate, target in graph.adjacency.get(entity, []):
                if candidate != edge:
                    continue
                if target not in next_frontier:
                    target_types.update(graph.types.get(target, set()))
                next_frontier.add(target)
                if direction == "forward":
                    used_triples.add((entity, relation, target))
                else:
                    used_triples.add((target, relation, entity))
        target_type = dominant_type(target_types)
        hops.append(Hop(relation, direction, source_type, target_type))
        frontier = next_frontier
        source_type = target_type
    spec = PathSpec(
        anchor, graph.entity_type(anchor, first_relation), tuple(hops), source_type, "webqsp"
    )
    return path_to_dict(spec), sorted(frontier), [list(triple) for triple in sorted(used_triples)]


class SRTKPathRetriever:
    """Thin adapter around SRTK's released iterative relation-path retriever."""

    def __init__(self, scorer_model: str = SRTK_SCORER, device: str = "auto") -> None:
        try:
            from srtk.scorer import Scorer
        except ImportError as exc:
            raise RuntimeError(
                "SRTK is required for PullNet-style retrieval. Install with "
                "`pip install -e '.[retrieval]'`."
            ) from exc
        if device == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)
        self.scorer_model = scorer_model
        self.scorer = Scorer(scorer_model, resolved)

    def retrieve(self, question: str, graph_row: dict[str, Any]) -> dict[str, Any]:
        try:
            from srtk.retrieve import END_REL, Retriever
        except ImportError as exc:
            raise RuntimeError("SRTK retrieval components are unavailable") from exc

        graph = LocalQuestionGraph(graph_row.get("graph", []))
        backend = Retriever(graph, self.scorer, beam_width=PATH_CAP, max_depth=MAX_HOPS)
        families: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for anchor in graph_row.get("q_entity", []):
            for relation_path, score in backend.beam_search_path(question, anchor):
                directed = tuple(edge for edge in relation_path if edge != END_REL)
                if not directed:
                    continue
                path, answers, triples = materialize_path(graph, anchor, directed)
                if not answers:
                    continue
                key = (anchor, directed)
                existing = families.get(key)
                family = {
                    "path": path,
                    "relation_sequence": list(directed),
                    "answers": answers,
                    "retrieval_score": float(score),
                    "supporting_triples": triples,
                }
                if existing is None or family["retrieval_score"] > existing["retrieval_score"]:
                    families[key] = family
        candidates = sorted(
            families.values(), key=lambda row: row["retrieval_score"], reverse=True
        )[:PATH_CAP]
        retrieved_triples = {
            tuple(triple)
            for candidate in candidates
            for triple in candidate["supporting_triples"]
        }
        return {
            "candidate_paths": candidates,
            "retrieved_subgraph": [list(triple) for triple in sorted(retrieved_triples)],
        }


def path_rank(candidates: list[dict[str, Any]], gold_sequences: list[tuple[str, ...]]):
    gold = set(gold_sequences)
    for rank, candidate in enumerate(candidates, 1):
        if tuple(candidate["relation_sequence"]) in gold:
            return rank
    return None


def gold_path_available(
    graph: LocalQuestionGraph,
    topic_entities: list[str],
    gold_sequences: list[tuple[str, ...]],
) -> bool:
    """Evaluation-only check that the supplied graph can execute an annotated path."""
    return any(
        graph.deduce_leaves(anchor, sequence)
        for anchor in topic_entities
        for sequence in gold_sequences
    )


def run_retrieval_probe(
    questions_path: Path,
    graphs_path: Path,
    scorer_model: str,
    output: Path,
    limit: int,
    device: str,
    question_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Measure evidence recall without invoking the inverse verifier or answer reasoner."""
    started = time.time()
    retriever = SRTKPathRetriever(scorer_model, device)
    rows = []
    with graphs_path.open(encoding="utf-8") as handle:
        for line in handle:
            graph_row = json.loads(line)
            metadata = question_metadata.get(graph_row["id"])
            if metadata is None:
                continue
            retrieval = retriever.retrieve(metadata["question"], graph_row)
            candidates = retrieval["candidate_paths"]
            gold_sequences = [tuple(sequence) for sequence in metadata["gold_sequences"]]
            gold_rank = path_rank(candidates, gold_sequences)
            available = gold_path_available(
                LocalQuestionGraph(graph_row.get("graph", [])),
                graph_row.get("q_entity", []),
                gold_sequences,
            )
            candidate_answers = {
                answer for candidate in candidates for answer in candidate["answers"]
            }
            gold_answers = set(graph_row.get("answer", []))
            row = {
                "question_id": graph_row["id"],
                "question": metadata["question"],
                "topic_entities": graph_row.get("q_entity", []),
                "gold_answers": sorted(gold_answers),
                "gold_sequences": metadata["gold_sequences"],
                "gold_path_rank": gold_rank,
                "gold_path_in_available_graph": available,
                "answer_covered": bool(candidate_answers & gold_answers),
                **retrieval,
            }
            rows.append(row)
            print(
                f"{len(rows)}/{limit} {graph_row['id']}: paths={len(candidates)} "
                f"subgraph_triples={len(retrieval['retrieved_subgraph'])} "
                f"gold_rank={gold_rank} answer_covered={row['answer_covered']}",
                flush=True,
            )
            if len(rows) >= limit:
                break

    count = len(rows)
    cutoffs = (5, 10, 20, 50, 100, 200)
    metrics = {
        "questions": count,
        "retriever": "SRTK iterative relation-path retrieval",
        "scorer_model": scorer_model,
        "max_hops": MAX_HOPS,
        "path_cap": PATH_CAP,
        "provided_topic_entities_used": True,
        "final_reasoning_performed": False,
        "path_recall": {
            f"recall_at_{cutoff}": sum(
                row["gold_path_rank"] is not None and row["gold_path_rank"] <= cutoff
                for row in rows
            )
            / max(count, 1)
            for cutoff in cutoffs
        },
        "gold_path_availability_in_supplied_graph": sum(
            row["gold_path_in_available_graph"] for row in rows
        )
        / max(count, 1),
        "path_recall_given_available": sum(
            row["gold_path_rank"] is not None for row in rows
        )
        / max(sum(row["gold_path_in_available_graph"] for row in rows), 1),
        "answer_coverage": sum(row["answer_covered"] for row in rows) / max(count, 1),
        "average_candidate_paths": sum(len(row["candidate_paths"]) for row in rows)
        / max(count, 1),
        "average_subgraph_triples": sum(len(row["retrieved_subgraph"]) for row in rows)
        / max(count, 1),
        "elapsed_seconds": time.time() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "retrieval.jsonl", rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
