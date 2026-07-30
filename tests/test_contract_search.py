from __future__ import annotations

import ast
from pathlib import Path

from contract_search.oracle.webqsp import compile_parse
from contract_search.oracle.sparql import compile_sparql
from contract_search.conjunctive import execute_conjunctive
from contract_search.program import execute_reference
from contract_search.substrate import IndexedSubgraph, repair_mojibake
from contract_search.search import LexicalScorer, search_question
from contract_search.substrate import QuestionGraph
from contract_search.audit import _canonical_program, classify_failure


def test_webqsp_compiler_keeps_constraint_and_order() -> None:
    parse = {
        "TopicEntityMid": "m.topic",
        "InferentialChain": ["people.person.children"],
        "Sparql": """
            ns:m.topic ns:people.person.children ?x .
            ?x ns:people.person.gender ns:m.female .
        """,
        "Constraints": [
            {
                "SourceNodeIndex": 0,
                "NodePredicate": "people.person.gender",
                "Operator": "Equal",
                "Argument": "m.female",
                "ArgumentType": "Entity",
            }
        ],
        "Order": {
            "SourceNodeIndex": 0,
            "NodePredicate": "people.person.date_of_birth",
            "SortOrder": "Ascending",
            "Start": 0,
            "Count": 1,
        },
    }
    program = compile_parse(parse)
    assert program is not None
    assert program.constraints[0].relation == "people.person.gender"
    assert program.order is not None
    assert program.order.relation == "people.person.date_of_birth"


def test_reference_execution_applies_constraint_and_order() -> None:
    parse = {
        "TopicEntityMid": "m.topic",
        "InferentialChain": ["people.person.children"],
        "Sparql": "ns:m.topic ns:people.person.children ?x .",
        "Constraints": [
            {
                "SourceNodeIndex": 0,
                "NodePredicate": "people.person.gender",
                "Operator": "Equal",
                "Argument": "m.female",
                "ArgumentType": "Entity",
            }
        ],
        "Order": {
            "SourceNodeIndex": 0,
            "NodePredicate": "people.person.date_of_birth",
            "SortOrder": "Ascending",
            "Start": 0,
            "Count": 1,
        },
    }
    graph = IndexedSubgraph(
        [
            ("m.topic", "people.person.children", "m.older"),
            ("m.topic", "people.person.children", "m.younger"),
            ("m.older", "people.person.gender", "m.female"),
            ("m.younger", "people.person.gender", "m.female"),
            ("m.older", "people.person.date_of_birth", "1980-01-01"),
            ("m.younger", "people.person.date_of_birth", "1990-01-01"),
        ]
    )
    execution = execute_reference(graph, compile_parse(parse))
    assert execution.answers == {"m.older"}


def test_reference_execution_preserves_distinct_entity_ids() -> None:
    graph = IndexedSubgraph(
        [
            ("m.real", "fiction.representations", "m.fictional"),
            ("m.fictional", "people.place_of_death", "m.city"),
        ]
    )
    assert graph.traverse({"m.real"}, "fiction.representations", "forward") == {
        "m.fictional"
    }
    assert graph.traverse({"m.fictional"}, "fiction.representations", "backward") == {
        "m.real"
    }


def test_non_oracle_modules_do_not_import_oracle() -> None:
    package = Path(__file__).parents[1] / "contract_search"
    allowed = {"ceiling.py", "cli.py"}
    offenders = []
    for path in package.glob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "oracle" in (node.module or ""):
                offenders.append(path.name)
            if isinstance(node, ast.Import):
                if any("oracle" in alias.name for alias in node.names):
                    offenders.append(path.name)
    assert not offenders


def test_conjunctive_program_joins_two_anchors() -> None:
    sparql = """
        SELECT DISTINCT ?x
        WHERE {
        ns:m.bolivia ns:location.adjoin_s ?y .
        ?y ns:location.adjoins ?x .
        ?x ns:location.contains ns:m.goias .
        }
    """
    program = compile_sparql(sparql)
    assert program is not None
    graph = IndexedSubgraph(
        [
            ("m.bolivia", "location.adjoin_s", "m.link"),
            ("m.link", "location.adjoins", "m.brazil"),
            ("m.brazil", "location.contains", "m.goias"),
            ("m.link", "location.adjoins", "m.peru"),
        ]
    )
    assert execute_conjunctive(graph, program).answers == {"m.brazil"}


def test_conjunctive_program_applies_comparison_and_superlative() -> None:
    sparql = """
        SELECT DISTINCT ?x
        WHERE {
        ns:m.city ns:travel.attractions ?x .
        ?x ns:architecture.construction_started ?num .
        FILTER (?num >= "1900"^^xsd:dateTime) .
        }
        ORDER BY DESC(?num) LIMIT 1
    """
    program = compile_sparql(sparql)
    assert program is not None
    graph = IndexedSubgraph(
        [
            ("m.city", "travel.attractions", "m.old"),
            ("m.city", "travel.attractions", "m.new"),
            ("m.old", "architecture.construction_started", "1901"),
            ("m.new", "architecture.construction_started", "2001"),
        ]
    )
    assert execute_conjunctive(graph, program).answers == {"m.new"}


def test_baseline_search_only_proposes_executable_non_identity_paths() -> None:
    row = QuestionGraph(
        question_id="toy",
        question="What country is the city in?",
        topic_entities=("m.city",),
        gold_answers=("m.country",),
        triples=(
            ("m.city", "location.location.containedby", "m.country"),
            ("m.city", "metadata.identity", "m.city"),
        ),
    )
    result = search_question(row, LexicalScorer(), max_hops=2, expansion_cap=20)
    assert result.state is not None
    assert result.state.answers == {"m.country"}
    assert all("metadata identity" not in item["program"] for item in result.trace[0]["beam"])


def test_baseline_search_respects_budget_and_does_not_repeat_atoms() -> None:
    row = QuestionGraph(
        question_id="toy-budget",
        question="What country is the city in?",
        topic_entities=("m.city", "m.city"),
        gold_answers=("m.country",),
        triples=(
            ("m.city", "location.location.containedby", "m.country"),
            ("m.country", "location.location.contains", "m.city"),
        ),
    )
    result = search_question(row, LexicalScorer(), max_hops=4, expansion_cap=5)
    assert result.scored_expansions <= 5
    for round_trace in result.trace:
        for item in round_trace["beam"]:
            lines = [
                line
                for line in item["program"].splitlines()
                if "--[" in line
            ]
            assert len(lines) == len(set(lines))


def test_failure_audit_detects_silent_incompleteness() -> None:
    prediction = {
        "exact": False,
        "predicted_answers": ["m.partial"],
        "program_state": {
            "atoms": [
                {"head": "m.topic", "relation": "first.relation", "tail": "?x"}
            ]
        },
        "trace": [
            {
                "proposed_atoms": [
                    {"head": "m.topic", "relation": "first.relation", "tail": "?x"},
                    {"head": "?x", "relation": "second.relation", "tail": "?y"},
                ],
                "beam": [],
            }
        ],
    }
    ceiling = {
        "status": "exact",
        "executed_answers": ["m.gold"],
        "program": {
            "select": "?y",
            "atoms": [
                {"head": "m.topic", "relation": "first.relation", "tail": "?x"},
                {"head": "?x", "relation": "second.relation", "tail": "?y"},
            ],
            "filters": [],
            "optional_filters": [],
            "order": None,
        },
    }
    category, _ = classify_failure(prediction, ceiling)
    assert category == "silent_incompleteness"


def test_program_equivalence_renames_variables_but_not_direction() -> None:
    left = {
        "select": "?answer",
        "atoms": [
            {"head": "m.topic", "relation": "people.children", "tail": "?answer"}
        ],
    }
    renamed = {
        "select": "?x",
        "atoms": [{"head": "m.topic", "relation": "people.children", "tail": "?x"}],
    }
    reversed_path = {
        "select": "?x",
        "atoms": [{"head": "?x", "relation": "people.children", "tail": "m.topic"}],
    }
    assert _canonical_program(left) == _canonical_program(renamed)
    assert _canonical_program(left) != _canonical_program(reversed_path)


def test_substrate_repairs_reversible_utf8_mojibake() -> None:
    assert repair_mojibake("contains GoiÃ¡s") == "contains Goiás"
    assert repair_mojibake("plain ASCII") == "plain ASCII"
