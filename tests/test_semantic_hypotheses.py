from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from rc_mex.micro_agents import induce_semantic_sketches, select_executable_hypothesis
from rc_mex.run_query_selection import sketch_relation_phrases


def llm_result(payload: dict) -> dict:
    return {
        "text": json.dumps(payload),
        "error": "",
        "prompt_tokens": 20,
        "completion_tokens": 10,
    }


class SemanticHypothesisTests(unittest.TestCase):
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
