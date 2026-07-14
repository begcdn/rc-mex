import tempfile
import unittest
from pathlib import Path

from rc_mex.query_hypothesis_solver import (
    ConstraintGraph,
    GroundedAtom,
    QueryHypothesis,
    RelationAtom,
)
from rc_mex.run_query_hypothesis_comparison import (
    aggregate_metrics,
    append_cache,
    compile_constraint_graph,
    evaluate_hypothesis_pool,
    load_cache,
    validate_constraint_graph,
)


class QueryHypothesisComparisonTests(unittest.TestCase):
    def test_compiler_preserves_shared_variables(self):
        raw = {
            "variables": [{"id": "v1", "role": "film"}],
            "answer_variable": "v1",
            "atoms": [
                {"left": "entity_1", "predicate": "person directed film", "right": "v1"},
                {"left": "entity_2", "predicate": "person starred in film", "right": "v1"},
            ],
            "type_constraints": [{"variable": "v1", "type": "film"}],
            "operators": [],
        }
        valid, reason = validate_constraint_graph(raw, 2)
        self.assertTrue(valid, reason)
        graph, constraints = compile_constraint_graph(raw)
        self.assertEqual(graph.answer_variable, "?v1")
        self.assertEqual(graph.atoms[0].right, "?v1")
        self.assertEqual(graph.atoms[1].right, "?v1")
        self.assertEqual(constraints, [{"variable": "?v1", "type": "film"}])

    def test_invalid_graph_rejects_disconnected_answer(self):
        raw = {
            "variables": [{"id": "v1"}, {"id": "v2"}],
            "answer_variable": "v2",
            "atoms": [{"left": "entity_1", "predicate": "relation", "right": "v1"}],
            "operators": [],
        }
        valid, reason = validate_constraint_graph(raw, 1)
        self.assertFalse(valid)
        self.assertEqual(reason, "answer_disconnected_from_entities")

    def test_same_pool_isolates_binding_normalization(self):
        graph = ConstraintGraph(
            atoms=(RelationAtom("a1", "anchor", "artist has genre", "?answer"),),
            answer_variable="?answer",
            variables=("?answer",),
        )
        correct = QueryHypothesis(
            bindings=(
                (("?answer", "rock"),),
                (("?answer", "hard rock"),),
                (("?answer", "blues rock"),),
            ),
            grounded_atoms=(GroundedAtom("a1", "genre", "forward", 0.8),),
            relation_log_score=0.8,
        )
        wrong = QueryHypothesis(
            bindings=((('?answer', 'guitar'),),),
            grounded_atoms=(GroundedAtom("a1", "instrument", "forward", 0.7),),
            relation_log_score=0.7,
        )
        gold = {"rock", "hard rock", "blues rock"}
        result = evaluate_hypothesis_pool(graph, [correct, wrong], gold)
        self.assertEqual(result["per_binding"]["predicted"], ["guitar"])
        self.assertEqual(set(result["query_denotation"]["predicted"]), gold)
        self.assertEqual(result["per_binding"]["f1"], 0.0)
        self.assertEqual(result["query_denotation"]["f1"], 1.0)
        self.assertEqual(result["shared_pool"]["query_hypotheses"], 2)
        self.assertTrue(result["shared_pool"]["exact_denotation_generated"])

    def test_metrics_report_set_valued_gain_separately(self):
        graph = ConstraintGraph(
            atoms=(RelationAtom("a1", "anchor", "artist has genre", "?answer"),),
            answer_variable="?answer",
            variables=("?answer",),
        )
        hypothesis = QueryHypothesis(
            bindings=((('?answer', 'rock'),), (('?answer', 'blues'),)),
            grounded_atoms=(GroundedAtom("a1", "genre", "forward", 0.0),),
        )
        evaluation = evaluate_hypothesis_pool(graph, [hypothesis], {"rock", "blues"})
        row = {
            "dataset": "toy",
            "gold": ["rock", "blues"],
            "valid_graph": True,
            "execution_supported": True,
            "encoders": {
                "base": {**evaluation, "runtime_seconds": 0.01},
                "trained": {**evaluation, "runtime_seconds": 0.01},
            },
        }
        metrics = aggregate_metrics(
            [row],
            {"new_calls": 0, "cache_hits": 1},
            {"base": "base", "trained": "trained"},
        )
        base = metrics["arms"]["base"]
        self.assertEqual(base["per_binding"]["set_valued"]["mean_f1"], 2 / 3)
        self.assertEqual(base["query_denotation"]["set_valued"]["mean_f1"], 1.0)
        self.assertEqual(metrics["pairwise"]["base"]["query_wins"], 1)
        self.assertEqual(metrics["by_dataset"]["toy"]["end_to_end_arms"]["base"]["query_denotation"]["mean_f1"], 1.0)

    def test_constraint_graph_cache_uses_latest_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "constraint_graphs.jsonl"
            append_cache(path, {"key": "cwq:1", "question": "old"})
            append_cache(path, {"key": "cwq:1", "question": "new"})
            cache = load_cache(path)
        self.assertEqual(cache["cwq:1"]["question"], "new")


if __name__ == "__main__":
    unittest.main()
