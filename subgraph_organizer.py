"""Lossless structural organization of released SubgraphRAG prompts.

This module changes only the presentation of the retrieved triple block. It
does not infer paths, remove duplicates, add labels, or modify predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParsedTriple:
    index: int
    raw: str
    head: str
    relation: str
    tail: str


def _parse_triple(line: str, index: int) -> ParsedTriple:
    if not (line.startswith("(") and line.endswith(")")):
        raise ValueError(f"not a triple line: {line!r}")
    fields = line[1:-1].split(",")
    if len(fields) < 3:
        raise ValueError(f"triple has fewer than three fields: {line!r}")

    # Released predicates are schema names (for example a.b.c or rdf#range).
    # Looking for that slot lets entity display names contain commas without
    # rewriting the original line.
    candidates = [
        pos
        for pos in range(1, len(fields) - 1)
        if re.fullmatch(r"[A-Za-z0-9_:-]+(?:[.#][A-Za-z0-9_:-]+)+", fields[pos])
    ]
    if candidates:
        relation_pos = candidates[0]
        head = ",".join(fields[:relation_pos])
        relation = fields[relation_pos]
        tail = ",".join(fields[relation_pos + 1 :])
    elif len(fields) == 3:
        head, relation, tail = fields
    else:
        raise ValueError(f"cannot identify predicate in triple: {line!r}")
    return ParsedTriple(index, line, head, relation, tail)


def extract_triple_lines(prompt: str) -> list[str]:
    marker = "Triplets:\n"
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("prompt has no Triplets section")
    start += len(marker)
    end = prompt.find("\n\nQuestion:", start)
    if end < 0:
        raise ValueError("prompt has no Question section after Triplets")
    lines = prompt[start:end].splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Triplets section contains a blank line")
    if any(not line.startswith("(") or not line.endswith(")") for line in lines):
        raise ValueError("Triplets section contains a non-triple line")
    return lines


def _normalised_tokens(text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"\w+", text, flags=re.UNICODE))


def _question_positions(question: str, entity: str) -> tuple[int, ...]:
    """Return starts of complete normalized phrase matches."""
    entity_tokens = _normalised_tokens(entity)
    if not entity_tokens or (len(entity_tokens) == 1 and len(entity_tokens[0]) < 2):
        return ()
    question_tokens = _normalised_tokens(question)
    width = len(entity_tokens)
    return tuple(
        start
        for start in range(len(question_tokens) - width + 1)
        if question_tokens[start : start + width] == entity_tokens
    )


def _question_matches_entity(question: str, entity: str) -> bool:
    """Match a complete normalized phrase, never an arbitrary substring."""
    return bool(_question_positions(question, entity))


def _matched_roots(question: str, component: set[str]) -> list[str]:
    matched = [
        entity
        for entity in component
        if _question_matches_entity(question, entity)
    ]
    return sorted(
        matched,
        key=lambda entity: (
            _question_positions(question, entity)[0],
            -len(_normalised_tokens(entity)),
            -len(entity),
            entity,
        ),
    )


def _fallback_root(component: set[str], adjacency: dict[str, list[int]]) -> str:
    return max(
        component,
        key=lambda entity: (len(adjacency[entity]), -min(adjacency[entity]), entity),
    )


def _roots(question: str, component: set[str], adjacency: dict[str, list[int]]) -> list[str]:
    matched = _matched_roots(question, component)
    return matched or [_fallback_root(component, adjacency)]


def _component_sort_key(question: str, adjacency: dict[str, list[int]], roots: list[str]) -> tuple:
    positions = _question_positions(question, roots[0])
    if positions:
        return (0, positions[0], -len(_normalised_tokens(roots[0])), roots[0])
    return (1, -len(adjacency[roots[0]]), min(min(indices) for indices in adjacency.values()), roots[0])


def _components(triples: list[ParsedTriple]) -> list[tuple[set[str], list[int]]]:
    adjacency: dict[str, list[int]] = defaultdict(list)
    entities: set[str] = set()
    for triple in triples:
        entities.update((triple.head, triple.tail))
        adjacency[triple.head].append(triple.index)
        adjacency[triple.tail].append(triple.index)

    edge_by_index = {triple.index: triple for triple in triples}
    entity_components: list[set[str]] = []
    unseen = set(entities)
    while unseen:
        start = min(unseen)
        component: set[str] = set()
        queue = [start]
        unseen.remove(start)
        while queue:
            entity = queue.pop()
            component.add(entity)
            for edge_index in adjacency[entity]:
                triple = edge_by_index[edge_index]
                other = triple.tail if triple.head == entity else triple.head
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        entity_components.append(component)

    return [
        (component, sorted(index for index, triple in edge_by_index.items() if triple.head in component or triple.tail in component))
        for component in entity_components
    ]


def _ordered_groups(question: str, triples: list[ParsedTriple]) -> list[tuple[str, list[int]]]:
    """Return non-empty BFS emitter groups; each edge index occurs once."""
    by_index = {triple.index: triple for triple in triples}
    component_records: list[tuple[tuple, set[str], list[int], list[str]]] = []
    for component, component_indices in _components(triples):
        adjacency: dict[str, list[int]] = defaultdict(list)
        for index in component_indices:
            triple = by_index[index]
            adjacency[triple.head].append(index)
            adjacency[triple.tail].append(index)
        roots = _roots(question, component, adjacency)
        component_records.append(
            (_component_sort_key(question, adjacency, roots), component, component_indices, roots)
        )

    groups: list[tuple[str, list[int]]] = []
    for _, component, component_indices, roots in sorted(component_records, key=lambda item: item[0]):
        adjacency: dict[str, list[int]] = defaultdict(list)
        for index in component_indices:
            triple = by_index[index]
            adjacency[triple.head].append(index)
            adjacency[triple.tail].append(index)
        seen_entities = set(roots)
        seen_edges: set[int] = set()
        queue = deque(roots)
        while queue:
            emitter = queue.popleft()
            emitted: list[int] = []
            # adjacency was built in released input order. Keep that order
            # whenever this emitter has multiple eligible edges.
            for index in adjacency[emitter]:
                if index in seen_edges:
                    continue
                triple = by_index[index]
                other = triple.tail if triple.head == emitter else triple.head
                seen_edges.add(index)
                emitted.append(index)
                if other not in seen_entities:
                    seen_entities.add(other)
                    queue.append(other)
            if emitted:
                groups.append((emitter, emitted))
        if seen_edges != set(component_indices):
            raise AssertionError("BFS failed to emit every edge in a connected component")
    return groups


def organize_triples(question: str, lines: Iterable[str], *, structured: bool) -> list[str]:
    raw_lines = list(lines)
    triples = [_parse_triple(line, index) for index, line in enumerate(raw_lines)]
    groups = _ordered_groups(question, triples)
    by_index = {triple.index: triple for triple in triples}
    if not structured:
        return [by_index[index].raw for _, indices in groups for index in indices]

    output: list[str] = []
    for emitter, indices in groups:
        output.append(f"[{emitter}]")
        output.extend(by_index[index].raw for index in indices)
    return output


def adjacency_graph_lines(
    question: str,
    lines: Iterable[str],
    *,
    organize_groups: bool,
) -> list[str]:
    """Serialize every triple once as compact directed adjacency groups."""
    raw_lines = list(lines)
    triples = [_parse_triple(line, index) for index, line in enumerate(raw_lines)]
    groups: dict[str, list[ParsedTriple]] = {}
    original_heads = []
    for triple in triples:
        if triple.head not in groups:
            groups[triple.head] = []
            original_heads.append(triple.head)
        groups[triple.head].append(triple)

    if organize_groups:
        emitter_order = [emitter for emitter, _ in _ordered_groups(question, triples)]
        ordered_heads = [head for head in emitter_order if head in groups]
        ordered_heads.extend(head for head in original_heads if head not in set(ordered_heads))
    else:
        ordered_heads = original_heads

    output = ["Directed adjacency:"]
    for head in ordered_heads:
        output.append(f"[{head}]")
        output.extend(
            f"> {triple.relation} > {triple.tail}" for triple in groups[head]
        )
    return output


def decode_adjacency_graph(lines: Iterable[str]) -> list[tuple[str, str, str]]:
    """Decode :func:`adjacency_graph_lines` for losslessness checks."""
    raw_lines = list(lines)
    if not raw_lines or raw_lines[0] != "Directed adjacency:":
        raise ValueError("adjacency graph is missing its section marker")
    head = None
    triples = []
    for line in raw_lines[1:]:
        if line.startswith("[") and line.endswith("]"):
            head = line[1:-1]
            continue
        if head is None or not line.startswith("> "):
            raise ValueError(f"invalid adjacency line: {line!r}")
        relation, separator, tail = line[2:].partition(" > ")
        if not separator:
            raise ValueError(f"invalid adjacency edge: {line!r}")
        triples.append((head, relation, tail))
    return triples


def replace_triples(prompt: str, lines: list[str]) -> str:
    marker = "Triplets:\n"
    start = prompt.find(marker)
    if start < 0:
        raise ValueError("prompt has no Triplets section")
    start += len(marker)
    end = prompt.find("\n\nQuestion:", start)
    if end < 0:
        raise ValueError("prompt has no Question section after Triplets")
    return prompt[:start] + "\n".join(lines) + prompt[end:]


def transform_row(row: dict, *, structured: bool) -> dict:
    prompt = row["all_query"]
    question = row["question"]
    raw_lines = extract_triple_lines(prompt)
    transformed_lines = organize_triples(question, raw_lines, structured=structured)
    transformed = dict(row)
    transformed["user_query"] = replace_triples(row["user_query"], transformed_lines)
    transformed["all_query"] = replace_triples(row["all_query"], transformed_lines)
    return transformed


def run(input_path: Path, reorder_output: Path, structured_output: Path) -> tuple[int, int, int]:
    reorder_output.parent.mkdir(parents=True, exist_ok=True)
    structured_output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    reorder_count = 0
    structured_count = 0
    with input_path.open(encoding="utf-8") as source, reorder_output.open("w", encoding="utf-8") as reorder, structured_output.open("w", encoding="utf-8") as structured:
        for line in source:
            row = json.loads(line)
            json.dump(transform_row(row, structured=False), reorder, ensure_ascii=False)
            reorder.write("\n")
            json.dump(transform_row(row, structured=True), structured, ensure_ascii=False)
            structured.write("\n")
            count += 1
            reorder_count += 1
            structured_count += 1
    return count, reorder_count, structured_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reorder-output", type=Path, required=True)
    parser.add_argument("--structured-output", type=Path, required=True)
    args = parser.parse_args()
    counts = run(args.input, args.reorder_output, args.structured_output)
    print(json.dumps({"input_rows": counts[0], "reorder_rows": counts[1], "structured_rows": counts[2]}))


if __name__ == "__main__":
    main()
