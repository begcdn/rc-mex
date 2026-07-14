import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rc_mex.executable_pattern_alignment import (
    PatternEdge,
    ExecutablePattern,
    kqa_simple_patterns,
    parse_sparql_edges,
    pattern_from_graph_query,
    pattern_from_runtime_path,
    pattern_from_sparql,
)
from rc_mex.run_pattern_alignment import load_bounded_path_examples, load_local_frontier_examples
from rc_mex.run_proof_state_search_smoke import rank_relations_hybrid


class ExecutablePatternAlignmentTests(unittest.TestCase):
    def test_freebase_cvt_pattern_keeps_both_steps(self):
        sparql = """SELECT DISTINCT ?x WHERE {
        ns:m.person ns:people.person.education ?e .
        ?e ns:education.education.institution ?x .
        }"""
        pattern = pattern_from_sparql("freebase", sparql, ["m.person"])
        self.assertIsNotNone(pattern)
        self.assertEqual(len(pattern.edges), 2)
        self.assertIn("people person education", pattern.canonical_text())
        self.assertIn("education education institution", pattern.canonical_text())

    def test_wikidata_statement_and_qualifier_shorthand_is_expanded(self):
        sparql = """SELECT DISTINCT ?x WHERE {
        wd:Q164963 p:cast_member ?statement.
        ?statement ps:cast_member wd:Q312399; pq:character_role ?x.
        }"""
        edges = parse_sparql_edges(sparql)
        self.assertEqual(len(edges), 3)
        pattern = pattern_from_sparql("wikidata", sparql, ["Q164963", "Q312399"])
        self.assertIsNotNone(pattern)
        text = pattern.canonical_text()
        self.assertIn("cast member statement", text)
        self.assertIn("character role qualifier", text)

    def test_variable_names_do_not_change_pattern_text(self):
        first = ExecutablePattern(
            "x",
            (
                PatternEdge("anchor", "spouse", "?a"),
                PatternEdge("?a", "place_of_birth", "?x"),
            ),
            ("anchor",),
            "?x",
        )
        second = ExecutablePattern(
            "x",
            (
                PatternEdge("anchor", "spouse", "?different"),
                PatternEdge("?different", "place_of_birth", "?answer"),
            ),
            ("anchor",),
            "?answer",
        )
        self.assertEqual(first.canonical_text(), second.canonical_text())

    def test_runtime_backward_path_preserves_triple_direction(self):
        pattern = pattern_from_runtime_path(
            {"relations": [{"predicate": "place_of_birth", "direction": "backward"}]}
        )
        self.assertEqual(pattern.edges[0].subject, "answer")
        self.assertEqual(pattern.edges[0].object, "anchor")

    def test_runtime_composite_relation_expands_to_cvt_edges(self):
        pattern = pattern_from_runtime_path(
            {
                "relations": [
                    {
                        "predicate": "people.person.education / education.education.institution",
                        "direction": "forward",
                    }
                ]
            }
        )
        self.assertEqual(
            pattern.canonical_text(),
            "operators: relation; query graph: "
            "anchor --[people person education]--> intermediate1; "
            "intermediate1 --[education education institution]--> answer",
        )

    def test_runtime_backward_composite_reverses_traversal_not_schema_direction(self):
        pattern = pattern_from_runtime_path(
            {
                "relations": [
                    {
                        "predicate": "people.person.education / education.education.institution",
                        "direction": "backward",
                    }
                ]
            }
        )
        self.assertEqual(
            pattern.canonical_text(),
            "operators: relation; query graph: "
            "answer --[people person education]--> intermediate1; "
            "intermediate1 --[education education institution]--> anchor",
        )

    def test_kqa_transfer_extractor_represents_type_filtered_programs(self):
        simple = {
            "question": "Where was Ada born?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Ada"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["place of birth", "forward"]},
                {"function": "What", "dependencies": [1], "inputs": []},
            ],
        }
        filtered = {
            "question": "Which city?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Ada"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["place of birth", "forward"]},
                {"function": "FilterConcept", "dependencies": [1], "inputs": ["city"]},
            ],
        }
        examples = kqa_simple_patterns([simple, filtered])
        self.assertEqual(len(examples), 2)
        self.assertIn("place of birth", examples[0]["pattern"].canonical_text())
        self.assertIn("type_filter", examples[1]["pattern"].operators)

    def test_local_frontier_loader_uses_answers_only_as_offline_labels(self):
        row = {
            "id": "q1",
            "question": "Where was Ada born?",
            "q_entity": ["Ada"],
            "answer": ["London"],
            "graph": [
                ["Ada", "people.person.place_of_birth", "London"],
                ["Ada", "people.person.profession", "Mathematician"],
                ["Paris", "location.contains", "Ada"],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            examples, metadata = load_local_frontier_examples(path)
        self.assertEqual(metadata["direct_edge_examples"], 1)
        self.assertEqual(examples[0]["gold"], {("people.person.place_of_birth", "forward")})
        self.assertTrue(all(candidate["frequency"] == 1 for candidate in examples[0]["candidates"]))
        candidate_text = " ".join(candidate["text"] for candidate in examples[0]["candidates"])
        self.assertNotIn("London", candidate_text)
        self.assertIn("place of birth", candidate_text)

    def test_colon_prefixed_freebase_sparql_is_supported(self):
        pattern = pattern_from_sparql(
            "freebase",
            "SELECT ?x WHERE { :m.ada :people.person.place_of_birth ?x . }",
            ["m.ada"],
        )
        self.assertIsNotNone(pattern)
        self.assertIn("people person place of birth", pattern.canonical_text())

    def test_grail_graph_query_preserves_answer_type_and_edge_direction(self):
        pattern = pattern_from_graph_query(
            "freebase",
            {
                "nodes": [
                    {"nid": 0, "node_type": "class", "class": "theater.play", "question_node": 1},
                    {"nid": 1, "node_type": "entity", "id": "m.production", "question_node": 0},
                ],
                "edges": [{"start": 0, "end": 1, "relation": "theater.play.productions"}],
            },
        )
        self.assertIsNotNone(pattern)
        text = pattern.canonical_text()
        self.assertIn("answer --[theater play productions]--> anchor", text)
        self.assertIn("type object type type", text)
        self.assertIn("theater play", text)

    def test_runtime_ranker_exposes_directed_executable_pattern_text(self):
        frontier = [SimpleNamespace(predicate="people.person.place_of_birth", direction="backward", frequency=2)]
        with patch(
            "rc_mex.run_proof_state_search_smoke.semantic_relation_model_available",
            return_value=False,
        ):
            ranked = rank_relations_hybrid("Who was born here?", frontier)
        self.assertIn("answer --[people person place of birth]--> anchor", ranked[0]["semantic_pattern_text"])

    def test_bounded_path_loader_labels_generated_sequences_not_candidate_text(self):
        row = {
            "id": "q2",
            "question": "Where was Ada's spouse born?",
            "q_entity": ["Ada"],
            "answer": ["Paris"],
            "graph": [
                ["Ada", "people.person.spouse", "Bob"],
                ["Bob", "people.person.place_of_birth", "Paris"],
                ["Bob", "people.person.profession", "Engineer"],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            examples, metadata = load_bounded_path_examples(path)
        self.assertEqual(metadata["shortest_gold_hops"], {"2": 1})
        self.assertIn(
            (
                ("people.person.spouse", "forward"),
                ("people.person.place_of_birth", "forward"),
            ),
            examples[0]["gold"],
        )
        self.assertNotIn("Paris", " ".join(candidate["text"] for candidate in examples[0]["candidates"]))


if __name__ == "__main__":
    unittest.main()
