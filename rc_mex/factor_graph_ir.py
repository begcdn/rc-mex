"""Schema-neutral query-graph representation and benchmark adapters.

Gold programs are used by the controlled experiments only to recover linked
entity slots and evaluation labels.  Inference receives neither gold relation
identifiers nor gold query edges.
"""

from __future__ import annotations

from dataclasses import dataclass

from rc_mex.executable_pattern_alignment import best_webqsp_parse, clean_node, pattern_from_sparql


@dataclass(frozen=True)
class QueryGraph:
    edges: tuple[tuple[str, str, str], ...]
    types: tuple[tuple[str, str], ...]
    output: str
    entities: tuple[str, ...]


def replace_node(graph: QueryGraph, old: str, new: str) -> QueryGraph:
    if old == new:
        return graph
    return QueryGraph(
        tuple(
            (
                new if subject == old else subject,
                predicate,
                new if object_ == old else object_,
            )
            for subject, predicate, object_ in graph.edges
        ),
        tuple((new if node == old else node, type_name) for node, type_name in graph.types),
        new if graph.output == old else graph.output,
        tuple(dict.fromkeys(new if entity == old else entity for entity in graph.entities)),
    )


def merge(left: QueryGraph, right: QueryGraph, output: str) -> QueryGraph:
    return QueryGraph(
        tuple(dict.fromkeys(left.edges + right.edges)),
        tuple(dict.fromkeys(left.types + right.types)),
        output,
        tuple(dict.fromkeys(left.entities + right.entities)),
    )


def compile_kqa_program(program: list[dict]) -> tuple[QueryGraph, dict[str, str]] | None:
    """Compile the supported conjunctive KQA Pro fragment for diagnostics."""

    values: dict[int, QueryGraph] = {}
    entity_slots: dict[str, str] = {}
    for index, step in enumerate(program):
        function = step.get("function")
        dependencies = step.get("dependencies") or []
        inputs = step.get("inputs") or []
        if function == "Find" and inputs:
            name = str(inputs[0])
            entity_slots.setdefault(name, f"E{len(entity_slots)}")
            token = entity_slots[name]
            values[index] = QueryGraph((), (), token, (token,))
        elif function == "Relate" and len(dependencies) == 1 and dependencies[0] in values and len(inputs) >= 2:
            source = values[dependencies[0]]
            target = f"raw_v{index}"
            relation, direction = str(inputs[0]), str(inputs[1]).casefold()
            edge = (target, relation, source.output) if direction == "backward" else (source.output, relation, target)
            values[index] = QueryGraph(source.edges + (edge,), source.types, target, source.entities)
        elif function == "FilterConcept" and len(dependencies) == 1 and dependencies[0] in values and inputs:
            source = values[dependencies[0]]
            values[index] = QueryGraph(source.edges, source.types + ((source.output, str(inputs[0])),), source.output, source.entities)
        elif function == "And" and len(dependencies) == 2 and all(dependency in values for dependency in dependencies):
            left, right = values[dependencies[0]], values[dependencies[1]]
            left_entity = left.output.startswith("E")
            right_entity = right.output.startswith("E")
            if right_entity and not left_entity:
                left = replace_node(left, left.output, right.output)
                output = right.output
            else:
                right = replace_node(right, right.output, left.output)
                output = left.output
            values[index] = merge(left, right, output)
        elif function == "What" and len(dependencies) == 1 and dependencies[0] in values:
            values[index] = values[dependencies[0]]
        else:
            return None
    if not values:
        return None
    graph = values[max(values)]
    if not graph.edges or not graph.entities:
        return None
    return canonicalize(graph), entity_slots


def compile_webqsp_question(row: dict) -> tuple[QueryGraph, dict[str, str]] | None:
    """Compile the supported WebQSP conjunctive fragment for diagnostics."""

    parse = best_webqsp_parse(row)
    if not parse or parse.get("Constraints") or parse.get("Time") or parse.get("Order"):
        return None
    topic_mid = str(parse.get("TopicEntityMid", "")).strip()
    topic_name = str(parse.get("TopicEntityName", "")).strip()
    if not topic_mid or not topic_name:
        return None
    pattern = pattern_from_sparql("freebase", str(parse.get("Sparql", "")), [topic_mid])
    if pattern is None or not 1 <= len(pattern.edges) <= 3:
        return None
    anchor = clean_node(topic_mid)
    component = pattern._component_edges()
    nodes = {node for edge in component for node in (edge.subject, edge.object)}
    constants = {node for node in nodes if not node.startswith("?") and node != anchor}
    if constants:
        return None
    mapping = {pattern.answer: "raw_answer", anchor: "E0"}
    for index, node in enumerate(sorted(nodes - set(mapping))):
        mapping[node] = f"raw_v{index}"
    graph = QueryGraph(
        tuple((mapping[edge.subject], str(edge.predicate), mapping[edge.object]) for edge in component),
        (),
        "raw_answer",
        ("E0",),
    )
    return canonicalize(graph), {topic_name: "E0"}


def canonicalize(graph: QueryGraph) -> QueryGraph:
    nodes = {node for edge in graph.edges for node in (edge[0], edge[2])}
    nodes.update(node for node, _ in graph.types)
    mapping = {graph.output: "A", **{entity: entity for entity in graph.entities}}
    adjacency: dict[str, list[tuple[str, str, str]]] = {node: [] for node in nodes}
    for subject, predicate, object_ in graph.edges:
        adjacency.setdefault(subject, []).append((object_, "out", predicate))
        adjacency.setdefault(object_, []).append((subject, "in", predicate))
    distance = {graph.output: 0}
    frontier = [graph.output]
    while frontier:
        node = frontier.pop(0)
        for neighbor, _, _ in sorted(adjacency.get(node, ())):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                frontier.append(neighbor)

    def fingerprint(node: str) -> tuple:
        incident = sorted((direction, predicate) for _, direction, predicate in adjacency.get(node, ()))
        return distance.get(node, 999), tuple(incident), node

    variables = [node for node in nodes if node not in mapping]
    for index, node in enumerate(sorted(variables, key=fingerprint)):
        mapping[node] = f"V{index}"
    edges = tuple(sorted((mapping[subject], predicate, mapping[object_]) for subject, predicate, object_ in graph.edges))
    types = tuple(sorted((mapping[node], type_name) for node, type_name in graph.types))
    return QueryGraph(edges, types, "A", tuple(graph.entities))


def abstract_graph(graph: QueryGraph) -> QueryGraph:
    return QueryGraph(
        tuple((subject, "P", object_) for subject, _, object_ in graph.edges),
        tuple((node, "T") for node, _ in graph.types),
        graph.output,
        graph.entities,
    )


def graph_key(graph: QueryGraph) -> tuple:
    return graph.edges, graph.types, graph.output, graph.entities


def serialize_graph(graph: QueryGraph) -> str:
    parts = [f"answer {graph.output}"]
    parts.extend(f"edge {subject} | {predicate} | {object_}" for subject, predicate, object_ in graph.edges)
    parts.extend(f"type {node} | {type_name}" for node, type_name in graph.types)
    return " ; ".join(parts)


def parse_graph(text: str) -> QueryGraph | None:
    edges: list[tuple[str, str, str]] = []
    types: list[tuple[str, str]] = []
    answer = "A"
    for raw_part in str(text).split(";"):
        part = raw_part.strip()
        if part.casefold().startswith("answer "):
            answer = part.split(None, 1)[1].strip()
        elif part.casefold().startswith("edge "):
            fields = tuple(field.strip() for field in part[5:].split("|"))
            if len(fields) == 3 and all(fields):
                edges.append(fields)
        elif part.casefold().startswith("type "):
            fields = tuple(field.strip() for field in part[5:].split("|"))
            if len(fields) == 2 and all(fields):
                types.append(fields)
    nodes = {node for edge in edges for node in (edge[0], edge[2])}
    nodes.update(node for node, _ in types)
    if answer not in nodes or not edges:
        return None
    return QueryGraph(tuple(edges), tuple(types), answer, tuple(sorted(node for node in nodes if node.startswith("E"))))


def compiler_input(question: str, entity_slots: dict[str, str]) -> str:
    entities = " ; ".join(f"{slot} = {name}" for name, slot in entity_slots.items())
    return f"compile typed query graph: {question} ; linked entities: {entities}"


def semantic_leaf(label: str) -> str:
    return " ".join(str(label).split(".")[-1].replace("_", " ").split()).casefold()
