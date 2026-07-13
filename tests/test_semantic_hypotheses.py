from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from rc_mex.micro_agents import induce_semantic_sketches, select_executable_hypothesis
from rc_mex.run_query_selection import (
    anchor_factor_operator_supported,
    build_anchor_conditioned_joins,
    confident_anchor_factor,
    one_to_one_role_alignment,
    sketch_relation_phrases,
    terminal_relation_label,
)


def llm_result(payload: dict) -> dict:
    return {
        "text": json.dumps(payload),
        "error": "",
        "prompt_tokens": 20,
        "completion_tokens": 10,
    }


class SemanticHypothesisTests(unittest.TestCase):
    def test_factor_role_alignment_does_not_reuse_one_path_twice(self):
        similarities = {
            ("role-a", "path-a"): 0.9,
            ("role-a", "path-b"): 0.8,
            ("role-b", "path-a"): 0.85,
            ("role-b", "path-b"): 0.1,
        }
        with patch(
            "rc_mex.run_query_selection.cosine",
            side_effect=lambda left, right: similarities[(left, right)],
        ):
            score = one_to_one_role_alignment(["role-a", "role-b"], ["path-a", "path-b"])
        self.assertAlmostEqual(score, (0.8 + 0.85) / 2)

    def test_terminal_relation_label_keeps_only_output_role(self):
        self.assertEqual(
            terminal_relation_label("roster / player → coaches / position (reversed)"),
            "position",
        )

    def test_factor_confidence_treats_singleton_as_absolute_not_margin(self):
        sketches = [{"operators": ["relation"], "constraints": []}]
        self.assertIsNone(confident_anchor_factor([{"factor_score": 0.44}], sketches))
        winner = {"factor_score": 0.46}
        self.assertIs(confident_anchor_factor([winner], sketches), winner)

    def test_factor_abstains_when_sketch_activates_unsupported_operator(self):
        temporal = [{"operators": ["chain", "temporal_filter"], "constraints": []}]
        self.assertFalse(anchor_factor_operator_supported(temporal))
        self.assertIsNone(confident_anchor_factor([{"factor_score": 0.9}], temporal))

    def test_anchor_conditioned_join_preserves_anchor_provenance(self):
        class FakeGraph:
            @staticmethod
            def entity_name(entity_id):
                return {"anchor-a": "Anchor A", "anchor-b": "Anchor B"}[entity_id]

        left = {
            "predicate": "domain.left_relation",
            "direction": "forward",
            # The second constraint validates this denotation without making
            # it smaller; provenance-aware joins must still retain it.
            "targets": ["answer"],
            "member_quals": {"answer": {"from": ["left"]}},
        }
        right = {
            "predicate": "domain.right_relation",
            "direction": "backward",
            "targets": ["answer", "right-only"],
            "member_quals": {"answer": {"from": ["right"]}},
        }

        def candidate_paths(_kb, _graph, starts, _question, _descriptions):
            return [left] if starts == {"anchor-a"} else [right]

        with (
            patch("rc_mex.run_query_selection.build_candidate_paths", side_effect=candidate_paths),
            patch("rc_mex.run_query_selection.build_chain_candidates", return_value=[]),
            patch("rc_mex.run_query_selection.semantic_embedding", return_value=[1.0]),
            patch("rc_mex.run_query_selection.cosine", return_value=0.8),
        ):
            candidates = build_anchor_conditioned_joins(
                {}, FakeGraph(), ["anchor-a", "anchor-b"], "question", [], []
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["targets"], ["answer"])
        self.assertEqual(
            [constraint["anchor_id"] for constraint in candidates[0]["anchor_constraints"]],
            ["anchor-a", "anchor-b"],
        )
        self.assertEqual(candidates[0]["member_quals"]["answer"]["from"], ["left", "right"])

    def test_sketches_keep_dynamic_constraints_and_ordered_roles(self):
        payload = {
            "sketches": [
                {
                    "answer_role": "coach during the 1996 season",
                    "answer_types": ["sports coach"],
                    "operators": ["chain", "temporal_filter"],
                    "relation_roles": [
                        {"phrase": "owns sports team", "subject_role": "owner", "object_role": "team"},
                        {"phrase": "team coach", "subject_role": "team", "object_role": "coach"},
                    ],
                    "constraints": [
                        {"kind": "temporal", "description": "coaching tenure includes 1996"},
                        {"kind": "answer_type", "description": "answer is a coach"},
                    ],
                }
            ]
        }
        with patch("rc_mex.micro_agents.call_local_llm", return_value=llm_result(payload)):
            result = induce_semantic_sketches(
                "Who was the 1996 coach of the team owned by Jerry Jones?",
                "Jerry Jones",
            )

        self.assertFalse(result["parse_failed"])
        self.assertEqual(sketch_relation_phrases(result["sketches"], 0), ["owns sports team"])
        self.assertEqual(sketch_relation_phrases(result["sketches"], 1), ["team coach"])
        self.assertEqual(result["sketches"][0]["constraints"][0]["kind"], "temporal")

    def test_invalid_sketch_output_fails_closed_without_inventing_constraints(self):
        with patch(
            "rc_mex.micro_agents.call_local_llm",
            return_value={"text": "not json", "error": "", "prompt_tokens": 1, "completion_tokens": 1},
        ):
            result = induce_semantic_sketches("Where was Ada born?", "Ada")
        self.assertEqual(result["sketches"], [])
        self.assertTrue(result["parse_failed"])

    def test_hypothesis_selector_returns_structured_constraint_decision(self):
        payload = {
            "selected": 2,
            "sketch": 1,
            "constraint_scores": {
                "relation_roles": 0.9,
                "operator_structure": 1.0,
                "answer_type": 0.8,
                "question_constraints": 0.75,
                "execution_evidence": 1.0,
            },
            "unsatisfied_constraints": [],
            "reason": "The second query follows spouse then birthplace and returns cities.",
        }
        with patch("rc_mex.micro_agents.call_local_llm", return_value=llm_result(payload)):
            result = select_executable_hypothesis(
                "In what city was his wife born?",
                "Person",
                [{"answer_types": ["city"], "relation_roles": []}],
                ["spouse", "spouse -> place of birth"],
            )

        self.assertEqual(result["pick"], 1)
        self.assertEqual(result["sketch"], 1)
        self.assertEqual(result["constraint_scores"]["answer_type"], 0.8)
        self.assertIn("birthplace", result["reason"])

    def test_hypothesis_selector_accepts_small_model_markdown_fallback(self):
        text = """**Selected:** 2
**Sketch:** 1
* **relation_roles**: 0.9
* **operator_structure**: 1.0
* **answer_type**: 0.8
* **question_constraints**: 0.7
* **execution_evidence**: 1.0
**Reason:** Query 2 satisfies the complete chain."""
        with patch(
            "rc_mex.micro_agents.call_local_llm",
            return_value={"text": text, "error": "", "prompt_tokens": 5, "completion_tokens": 5},
        ):
            result = select_executable_hypothesis("question", "start", [], ["a", "b"])
        self.assertEqual(result["pick"], 1)
        self.assertFalse(result["parse_failed"])
        self.assertEqual(result["constraint_scores"]["relation_roles"], 0.9)


if __name__ == "__main__":
    unittest.main()
