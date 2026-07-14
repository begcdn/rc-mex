import unittest

from rc_mex.query_hypothesis_solver import (
    ConstraintGraph,
    RelationAtom,
    RelationCandidate,
    rank_denotations,
    solve_conjunctive_query,
)


class QueryHypothesisSolverTests(unittest.TestCase):
    def test_multi_answer_denotation_is_ranked_as_one_query(self):
        graph = ConstraintGraph(
            atoms=(RelationAtom("a1", "anchor", "person influenced whom", "?answer"),),
            answer_variable="?answer",
            variables=("?answer",),
        )

        def propose(atom, hypothesis, constants):
            return [
                RelationCandidate(
                    "influenced",
                    "forward",
                    -0.1,
                    (("ada", "bob"), ("ada", "carol"), ("ada", "dave")),
                ),
                RelationCandidate("spouse", "forward", -0.2, (("ada", "eve"),)),
            ]

        hypotheses = solve_conjunctive_query(graph, {"anchor": "ada"}, propose)
        ranked = rank_denotations(graph, hypotheses)
        self.assertEqual(ranked[0][0], frozenset({"bob", "carol", "dave"}))
        self.assertEqual(len(ranked[0][1].bindings), 3)

    def test_shared_answer_variable_performs_exact_intersection(self):
        graph = ConstraintGraph(
            atoms=(
                RelationAtom("directed", "director", "person directed film", "?film"),
                RelationAtom("starred", "actor", "person starred in film", "?film"),
            ),
            answer_variable="?film",
            variables=("?film",),
        )

        def propose(atom, hypothesis, constants):
            if atom.atom_id == "directed":
                return [
                    RelationCandidate(
                        "directed",
                        "forward",
                        -0.1,
                        (("nolan", "inception"), ("nolan", "dunkirk")),
                    )
                ]
            return [
                RelationCandidate(
                    "starred_in",
                    "forward",
                    -0.1,
                    (("bale", "inception"), ("bale", "the prestige")),
                )
            ]

        hypotheses = solve_conjunctive_query(
            graph,
            {"director": "nolan", "actor": "bale"},
            propose,
        )
        self.assertEqual(hypotheses[0].denotation("?film"), frozenset({"inception"}))

    def test_query_score_is_not_normalized_by_binding_count(self):
        graph = ConstraintGraph(
            atoms=(RelationAtom("a1", "place", "place contains attraction", "?answer"),),
            answer_variable="?answer",
            variables=("?answer",),
        )

        def propose(atom, hypothesis, constants):
            return [
                RelationCandidate(
                    "tourist_attractions",
                    "forward",
                    0.8,
                    tuple(("atlanta", f"attraction-{index}") for index in range(50)),
                ),
                RelationCandidate("capital", "forward", 0.7, (("atlanta", "atlanta"),)),
            ]

        hypotheses = solve_conjunctive_query(graph, {"place": "atlanta"}, propose)
        ranked = rank_denotations(graph, hypotheses)
        self.assertEqual(len(ranked[0][0]), 50)
        self.assertEqual(ranked[0][1].grounded_atoms[0].relation_id, "tourist_attractions")

    def test_factor_ranks_complete_query_without_mutating_bindings(self):
        graph = ConstraintGraph(
            atoms=(RelationAtom("a1", "person", "person died in place", "?answer"),),
            answer_variable="?answer",
            variables=("?answer",),
        )

        def propose(atom, hypothesis, constants):
            return [
                RelationCandidate("cause_of_death", "forward", -0.1, (("mlk", "assassination"),)),
                RelationCandidate("place_of_death", "forward", -0.2, (("mlk", "memphis"),)),
            ]

        def factor(graph, hypothesis, constants):
            relation = hypothesis.grounded_atoms[0].relation_id
            return 0.3 if relation == "place_of_death" else 0.0

        hypotheses = solve_conjunctive_query(
            graph,
            {"person": "mlk"},
            propose,
            factor=factor,
        )
        self.assertEqual(hypotheses[0].denotation("?answer"), frozenset({"memphis"}))
        self.assertEqual(hypotheses[0].bindings, ((('?answer', 'memphis'),),))

    def test_invalid_graph_rejects_undeclared_variables(self):
        with self.assertRaisesRegex(ValueError, "undeclared variable"):
            ConstraintGraph(
                atoms=(RelationAtom("a1", "anchor", "relation", "?missing"),),
                answer_variable="?answer",
                variables=("?answer",),
            )


if __name__ == "__main__":
    unittest.main()
