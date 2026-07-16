import unittest

from rc_mex.execution_conditioned_factor_graph import (
    GroundedCandidate,
    merge_families,
    rank_candidate_pool,
    two_edge_path_grammar,
)
from rc_mex.factor_graph_ir import QueryGraph, compile_kqa_program, parse_graph, serialize_graph


class DummyModels:
    def score_generator(self, question, slots, candidates, semantic_schema):
        return [3.0, 1.0, 2.0]

    def score_denotation(self, question, slots, candidates):
        return [1.0, 3.0, 2.0]

    def score_topology(self, question, slots, candidates):
        return [1.0, 2.0, 3.0]


class ExecutionConditionedFactorGraphTests(unittest.TestCase):
    def test_graph_serialization_round_trip(self):
        graph = QueryGraph((("E0", "parent", "V0"), ("V0", "born in", "A")), (("A", "city"),), "A", ("E0",))
        self.assertEqual(parse_graph(serialize_graph(graph)), graph)

    def test_kqa_compiler_preserves_conjunctive_structure(self):
        program = [
            {"function": "Find", "inputs": ["Ada"], "dependencies": []},
            {"function": "Relate", "inputs": ["educated at", "forward"], "dependencies": [0]},
            {"function": "FilterConcept", "inputs": ["university"], "dependencies": [1]},
            {"function": "What", "inputs": [], "dependencies": [2]},
        ]
        graph, slots = compile_kqa_program(program)
        self.assertEqual(slots, {"Ada": "E0"})
        self.assertEqual(graph.edges, (("E0", "educated at", "A"),))
        self.assertEqual(graph.types, (("A", "university"),))

    def test_two_edge_grammar_covers_orientations_and_type_placements(self):
        rows = two_edge_path_grammar({"topic": "E0"})
        self.assertEqual(len(rows), 16)
        orientations = {
            (
                any(subject == "E0" and object_ == "V0" for subject, _, object_ in row.edges),
                any(subject == "V0" and object_ == "A" for subject, _, object_ in row.edges),
            )
            for row in rows
        }
        self.assertEqual(len(orientations), 4)
        self.assertEqual({len(row.types) for row in rows}, {0, 1, 2})

    def test_family_merge_unions_bindings_for_same_grounding(self):
        family_key = (("r",), ())
        rows = merge_families(
            [
                (family_key[0], family_key[1], {(('?x', 'a'),)}, 1.0, 1),
                (family_key[0], family_key[1], {(('?x', 'b'),)}, 0.5, 1),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], {(('?x', 'a'),), (('?x', 'b'),)})
        self.assertEqual(rows[0][3], 1.0)

    def test_all_selectors_rank_the_same_shared_pool(self):
        graph = QueryGraph((("E0", "r", "A"),), (), "A", ("E0",))
        candidates = [
            GroundedCandidate(0.0, graph, frozenset({"a"}), 1, True),
            GroundedCandidate(0.0, graph, frozenset({"b"}), 1, False),
            GroundedCandidate(0.0, graph, frozenset({"c"}), 1, False),
        ]
        rankings = rank_candidate_pool(DummyModels(), "question", {"topic": "E0"}, candidates, False)
        self.assertEqual(set(rankings), {"learned_product", "grammar_product", "grammar_topology_product", "source_aware"})
        self.assertEqual(len(rankings["learned_product"]), 1)
        self.assertTrue(all(len(rankings[name]) == 3 for name in rankings if name != "learned_product"))
        self.assertEqual(rankings["learned_product"][0][1].answers, frozenset({"a"}))


if __name__ == "__main__":
    unittest.main()
