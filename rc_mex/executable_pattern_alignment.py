"""Cross-schema alignment primitives for executable KG query patterns.

The unit of alignment is a directed query graph, not a benchmark relation id.
This matters when one KG represents a semantic predicate with one edge while
another uses a CVT, statement node, qualifier edge, or inverse direction.

The module intentionally contains no search policy.  It converts supervised
logical forms and runtime path candidates into the same textual pattern space;
a learned encoder can then score ``question <-> executable pattern`` pairs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NODE_TOKEN = r'(?:\?[A-Za-z_][\w]*|(?:(?:ns|wd):|:)[A-Za-z0-9_.-]+|"(?:\\.|[^"\\])*"(?:\^\^[^\s;.]*)?)'
PREDICATE_TOKEN = r'(?:(?:(?:ns|wdt|p|ps|pq):|:)[A-Za-z0-9_.-]+)'
EXPLICIT_TRIPLE = re.compile(
    rf"(?P<subject>{NODE_TOKEN})\s+(?P<predicate>{PREDICATE_TOKEN})\s+(?P<object>{NODE_TOKEN})"
)
SEMICOLON_EDGE = re.compile(
    rf";\s*(?P<predicate>{PREDICATE_TOKEN})\s+(?P<object>{NODE_TOKEN})"
)


@dataclass(frozen=True, order=True)
class PatternEdge:
    subject: str
    predicate: str
    object: str
    predicate_kind: str = "relation"

    def to_json(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "predicate_kind": self.predicate_kind,
        }


@dataclass(frozen=True)
class ExecutablePattern:
    schema: str
    edges: tuple[PatternEdge, ...]
    anchors: tuple[str, ...]
    answer: str
    operators: tuple[str, ...] = ("relation",)

    def _component_edges(self) -> tuple[PatternEdge, ...]:
        """Keep the answer component and discard unrelated nested subqueries."""
        incident: dict[str, list[PatternEdge]] = defaultdict(list)
        for edge in self.edges:
            incident[edge.subject].append(edge)
            incident[edge.object].append(edge)
        if self.answer not in incident:
            return ()
        seen = {self.answer}
        queue = deque([self.answer])
        kept: set[PatternEdge] = set()
        while queue:
            node = queue.popleft()
            for edge in incident[node]:
                kept.add(edge)
                other = edge.object if edge.subject == node else edge.subject
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        return tuple(sorted(kept))

    def _node_roles(self) -> dict[str, str]:
        edges = self._component_edges()
        nodes = {node for edge in edges for node in (edge.subject, edge.object)}
        roles = {self.answer: "answer"}
        for index, anchor in enumerate(sorted(set(self.anchors) & nodes), start=1):
            roles[anchor] = "anchor" if len(self.anchors) == 1 else f"anchor{index}"

        variables = {node for node in nodes if node.startswith("?") and node not in roles}
        constants = nodes - variables - set(roles)

        # Canonicalize intermediate variables by structural distance and their
        # incident predicate multiset.  Raw SPARQL variable names are only a
        # final deterministic tie-break and never enter the rendered text.
        distance = {self.answer: 0}
        queue = deque([self.answer])
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            adjacency[edge.subject].add(edge.object)
            adjacency[edge.object].add(edge.subject)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)

        def fingerprint(node: str) -> tuple:
            incident = sorted(
                ("out" if edge.subject == node else "in", edge.predicate_kind, edge.predicate)
                for edge in edges
                if node in (edge.subject, edge.object)
            )
            return (distance.get(node, 999), tuple(incident), node)

        for index, node in enumerate(sorted(variables, key=fingerprint), start=1):
            roles[node] = f"intermediate{index}"
        for index, node in enumerate(sorted(constants, key=fingerprint), start=1):
            roles[node] = "constraint" if len(constants) == 1 else f"constraint{index}"
        return roles

    def canonical_text(self) -> str:
        roles = self._node_roles()
        rendered = []
        for edge in self._component_edges():
            subject = roles.get(edge.subject, "node")
            object_ = roles.get(edge.object, "node")
            relation = readable_relation(edge.predicate)
            kind = "" if edge.predicate_kind in {"relation", "direct"} else f" {edge.predicate_kind}"
            rendered.append(f"{subject} --[{relation}{kind}]--> {object_}")
        rendered.sort()
        operators = ", ".join(sorted(set(self.operators)))
        return f"operators: {operators}; query graph: " + "; ".join(rendered)

    def signature(self) -> str:
        """Schema-specific identity used for labels, never model input."""
        roles = self._node_roles()
        parts = sorted(
            f"{roles.get(edge.subject, 'node')}|{edge.predicate_kind}|{edge.predicate}|{roles.get(edge.object, 'node')}"
            for edge in self._component_edges()
        )
        return f"{self.schema}::" + "||".join(parts)

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "edges": [edge.to_json() for edge in self._component_edges()],
            "anchors": list(self.anchors),
            "answer": self.answer,
            "operators": list(self.operators),
            "text": self.canonical_text(),
            "signature": self.signature(),
        }


@dataclass(frozen=True)
class AlignmentRecord:
    question_id: str
    question: str
    freebase: ExecutablePattern
    wikidata: ExecutablePattern

    @property
    def group_id(self) -> str:
        return f"{self.freebase.signature()}@@{self.wikidata.signature()}"

    def to_json(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "group_id": self.group_id,
            "freebase": self.freebase.to_json(),
            "wikidata": self.wikidata.to_json(),
        }


def readable_relation(value: str) -> str:
    value = re.sub(r"^(?:(?:ns|wdt|p|ps|pq):|:)", "", str(value))
    value = value.replace("/", " then ").replace(".", " ").replace("_", " ")
    return " ".join(value.split()).casefold()


def clean_node(value: str) -> str:
    value = value.strip()
    if value.startswith(("ns:", "wd:")):
        return value.split(":", 1)[1]
    if value.startswith(":"):
        return value[1:]
    return value


def clean_predicate(value: str) -> tuple[str, str]:
    prefix, predicate = value.split(":", 1)
    kinds = {
        "ns": "relation",
        "wdt": "direct",
        "p": "statement",
        "ps": "value",
        "pq": "qualifier",
    }
    return predicate, kinds.get(prefix, "relation")


def pattern_from_graph_query(schema: str, graph_query: dict) -> ExecutablePattern | None:
    """Convert GrailQA-style graph-query annotations into a canonical pattern."""
    nodes = {
        int(node["nid"]): node
        for node in graph_query.get("nodes") or []
        if "nid" in node
    }
    answer_ids = [
        nid
        for nid, node in nodes.items()
        if int(node.get("question_node", 0) or 0) == 1
    ]
    if len(answer_ids) != 1:
        return None
    answer_id = answer_ids[0]

    def token(nid: int) -> str:
        node = nodes[nid]
        if nid == answer_id:
            return "?answer"
        if str(node.get("node_type", "")).casefold() in {"entity", "literal"}:
            return str(node.get("id") or f"constant_{nid}")
        return f"?node{nid}"

    anchors = tuple(
        token(nid)
        for nid, node in sorted(nodes.items())
        if nid != answer_id
        and str(node.get("node_type", "")).casefold() in {"entity", "literal"}
    )
    if not anchors:
        return None
    edges = []
    for edge in graph_query.get("edges") or []:
        try:
            start = int(edge["start"])
            end = int(edge["end"])
        except (KeyError, TypeError, ValueError):
            continue
        relation = str(edge.get("relation", "")).strip()
        if start in nodes and end in nodes and relation:
            edges.append(PatternEdge(token(start), relation, token(end)))

    answer_class = str(nodes[answer_id].get("class") or "").strip()
    if answer_class and answer_class not in {"common.topic", "type.object"}:
        edges.append(PatternEdge("?answer", "type.object.type", answer_class, "type"))
    if not edges:
        return None
    operators = ("relation", "type_filter") if answer_class else ("relation",)
    return ExecutablePattern(schema, tuple(edges), anchors, "?answer", operators)


def answer_variable(sparql: str) -> str:
    alias = re.search(r"SELECT\s*\(\s*(\?\w+)\s+AS\s+\?\w+\s*\)", sparql, re.IGNORECASE)
    if alias:
        return alias.group(1)
    selected = re.search(r"SELECT\s+(?:DISTINCT\s+)?(\?\w+)", sparql, re.IGNORECASE)
    return selected.group(1) if selected else "?x"


def parse_sparql_edges(sparql: str) -> tuple[PatternEdge, ...]:
    """Extract relation triples, including Wikidata ``;`` shorthand.

    This is deliberately a query-graph extractor rather than a full SPARQL
    evaluator.  It ignores FILTER/VALUES syntax and keeps relation statements
    that can participate in an executable graph pattern.
    """
    compact = re.sub(r"\s+", " ", sparql)
    edges: list[PatternEdge] = []
    seen: set[PatternEdge] = set()
    for match in EXPLICIT_TRIPLE.finditer(compact):
        predicate, kind = clean_predicate(match.group("predicate"))
        edge = PatternEdge(
            clean_node(match.group("subject")),
            predicate,
            clean_node(match.group("object")),
            kind,
        )
        if edge not in seen:
            seen.add(edge)
            edges.append(edge)

        # A semicolon continuation belongs to this subject only until the
        # next statement terminator.  Predicate dots are not followed by
        # whitespace, so the first whitespace-delimited dot is safe here.
        tail = compact[match.end() :]
        terminator = re.search(r"\s*\.\s*|\s*}\s*", tail)
        continuation = tail[: terminator.start()] if terminator else tail
        for extra in SEMICOLON_EDGE.finditer(continuation):
            extra_predicate, extra_kind = clean_predicate(extra.group("predicate"))
            extra_edge = PatternEdge(
                edge.subject,
                extra_predicate,
                clean_node(extra.group("object")),
                extra_kind,
            )
            if extra_edge not in seen:
                seen.add(extra_edge)
                edges.append(extra_edge)
    return tuple(edges)


def pattern_from_sparql(
    schema: str,
    sparql: str,
    anchors: Sequence[str],
    *,
    operators: Sequence[str] = ("relation",),
) -> ExecutablePattern | None:
    edges = parse_sparql_edges(sparql)
    if not edges:
        return None
    pattern = ExecutablePattern(
        schema=schema,
        edges=edges,
        anchors=tuple(clean_node(anchor) for anchor in anchors),
        answer=answer_variable(sparql),
        operators=tuple(operators),
    )
    component = pattern._component_edges()
    component_nodes = {node for edge in component for node in (edge.subject, edge.object)}
    if not component or not (set(pattern.anchors) & component_nodes):
        return None
    return pattern


def pattern_from_runtime_path(path: dict, schema: str = "runtime") -> ExecutablePattern | None:
    """Convert a current search path to the alignment model's input format."""
    relations = path.get("relations") or path.get("steps") or []
    if not relations and path.get("predicate"):
        relations = [{"predicate": path["predicate"], "direction": path.get("direction", "forward")}]
    if not relations:
        return None
    normalized_relations: list[tuple[str, str]] = []
    for raw in relations:
        if isinstance(raw, str):
            predicate, direction = raw, "forward"
        else:
            predicate = str(raw.get("predicate") or raw.get("relation_id") or raw.get("relation") or "")
            direction = str(raw.get("direction", "forward")).casefold()
        if not predicate:
            return None
        segments = [segment.strip() for segment in predicate.split(" / ") if segment.strip()]
        if direction == "backward":
            segments.reverse()
        normalized_relations.extend((segment, direction) for segment in segments)

    edges = []
    current = "anchor"
    for index, (predicate, direction) in enumerate(normalized_relations):
        target = "answer" if index == len(normalized_relations) - 1 else f"?step{index + 1}"
        if direction == "backward":
            edges.append(PatternEdge(target, predicate, current))
        else:
            edges.append(PatternEdge(current, predicate, target))
        current = target
    return ExecutablePattern(schema, tuple(edges), ("anchor",), "answer")


def load_webqsp(path: str | Path) -> dict[str, dict]:
    payload = json.load(open(path, encoding="utf-8"))
    rows = payload.get("Questions", payload) if isinstance(payload, dict) else payload
    return {str(row["QuestionId"]): row for row in rows}


def load_wikiweb(path: str | Path) -> dict[str, dict]:
    rows = json.load(open(path, encoding="utf-8"))
    return {str(row["id"]): row for row in rows}


def best_webqsp_parse(row: dict) -> dict | None:
    parses = row.get("Parses") or []
    complete = [
        parse
        for parse in parses
        if str((parse.get("AnnotatorComment") or {}).get("ParseQuality", "")).casefold() == "complete"
    ]
    return (complete or parses or [None])[0]


def build_parallel_records(
    webqsp_rows: dict[str, dict],
    wikiweb_rows: dict[str, dict],
) -> list[AlignmentRecord]:
    records = []
    for question_id in sorted(webqsp_rows.keys() & wikiweb_rows.keys()):
        freebase_row = webqsp_rows[question_id]
        wikidata_row = wikiweb_rows[question_id]
        parse = best_webqsp_parse(freebase_row)
        if not parse:
            continue
        freebase = pattern_from_sparql(
            "freebase",
            str(parse.get("Sparql", "")),
            [str(parse.get("TopicEntityMid", ""))],
        )
        wiki_input = str(wikidata_row.get("input", ""))
        wiki_sparql = str(wikidata_row.get("output") or wikidata_row.get("sparql") or "")
        anchors = re.findall(r"\bQ\d+\b", wiki_input)
        if not anchors:
            # Public dev/test data omit entity-linker input.  SPARQL writes
            # topic entities before value constraints in these ports; using
            # the first constant is a deterministic representation heuristic,
            # not answer supervision.
            anchors = re.findall(r"\bQ\d+\b", wiki_sparql)[:1]
        wikidata = pattern_from_sparql(
            "wikidata",
            wiki_sparql,
            anchors,
        )
        if freebase and wikidata:
            records.append(
                AlignmentRecord(
                    question_id=question_id,
                    question=str(wikidata_row.get("utterance") or freebase_row.get("RawQuestion") or ""),
                    freebase=freebase,
                    wikidata=wikidata,
                )
            )
    return records


def kqa_simple_patterns(rows: Iterable[dict], max_hops: int = 3) -> list[dict]:
    """Extract leakage-free simple-chain transfer examples from gold KoPL.

    Gold programs are used only to build evaluation labels.  Branching,
    filtering, comparison, and aggregation examples are excluded until their
    runtime candidate representation exists in the architecture.
    """
    examples = []
    allowed = {"Find", "Relate", "FilterConcept", "What", "Count"}
    for index, row in enumerate(rows):
        program = row.get("program") or []
        if not program or any(step.get("function") not in allowed for step in program):
            continue
        finds = [i for i, step in enumerate(program) if step.get("function") == "Find"]
        relates = [i for i, step in enumerate(program) if step.get("function") == "Relate"]
        if len(finds) != 1 or not (1 <= len(relates) <= max_hops):
            continue
        nodes = {finds[0]: "anchor"}
        edges = []
        valid = True
        for step_index in relates:
            step = program[step_index]
            dependencies = step.get("dependencies") or []
            inputs = step.get("inputs") or []
            if len(dependencies) != 1 or len(inputs) < 2 or dependencies[0] not in nodes:
                valid = False
                break
            source = nodes[dependencies[0]]
            target = f"?step{step_index}"
            relation, direction = str(inputs[0]), str(inputs[1]).casefold()
            if direction == "backward":
                edges.append(PatternEdge(target, relation, source))
            else:
                edges.append(PatternEdge(source, relation, target))
            nodes[step_index] = target
        if not valid:
            continue
        answer_node = nodes[relates[-1]]
        operators = ["relation"]
        if any(step.get("function") == "FilterConcept" for step in program):
            operators.append("type_filter")
        if any(step.get("function") == "Count" for step in program):
            operators.append("aggregation")
        pattern = ExecutablePattern("kqa", tuple(edges), ("anchor",), answer_node, tuple(operators))
        examples.append(
            {
                "question_id": f"kqa:{index}",
                "question": str(row.get("question", "")),
                "pattern": pattern,
            }
        )
    return examples
