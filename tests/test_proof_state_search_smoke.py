from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json
from rc_mex.run_proof_state_search_smoke import main as run_smoke_main


def smoke_kb():
    return {
        "concepts": {
            "person": {"name": "person", "instanceOf": []},
            "film": {"name": "film", "instanceOf": []},
            "genre": {"name": "genre", "instanceOf": []},
            "location": {"name": "location", "instanceOf": []},
        },
        "entities": {
            "nolan": entity("Christopher Nolan", ["person"], [("directed", "inception"), ("produced", "inception")]),
            "inception": entity("Inception", ["film"], [("genre", "science_fiction"), ("filming location", "paris")]),
            "science_fiction": entity("science fiction", ["genre"], []),
            "paris": entity("Paris", ["location"], []),
        },
    }


def entity(name, types, relations):
    return {
        "name": name,
        "instanceOf": types,
        "relations": [
            {"predicate": predicate, "object": obj, "direction": "forward", "qualifiers": {}}
            for predicate, obj in relations
        ],
        "attributes": [],
    }


def smoke_questions():
    return [
        {
            "id": "q1",
            "question": "Which genre is the film directed by Christopher Nolan?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["directed", "forward"]},
                {"function": "Relate", "dependencies": [1], "inputs": ["genre", "forward"]},
            ],
        }
    ]


def real_kqa_style_questions():
    return [
        {
            "id": "q2",
            "question": "tell me the genre of the film directed by Christopher Nolan.",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["directed", "forward"]},
                {"function": "FilterConcept", "dependencies": [1], "inputs": ["film"]},
                {"function": "Relate", "dependencies": [2], "inputs": ["genre", "forward"]},
                {"function": "FilterConcept", "dependencies": [3], "inputs": ["genre"]},
                {"function": "What", "dependencies": [4], "inputs": []},
            ],
        }
    ]


class ProofStateSearchSmokeTests(unittest.TestCase):
    def test_smoke_runner_writes_requested_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            output = root / "proof_state_search_smoke"
            write_json(kb_path, smoke_kb())
            write_json(questions_path, smoke_questions())

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_proof_state_search_smoke",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(questions_path),
                    "--output",
                    str(output),
                    "--max-examples",
                    "1",
                    "--top-k",
                    "2",
                    "--beam-width",
                    "2",
                    "--debug-trace",
                    "--debug-limit",
                    "1",
                ]
                run_smoke_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((output / "predictions.jsonl").exists())
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "error_overlap.json").exists())
            self.assertTrue((output / "error_overlap.md").exists())
            self.assertTrue((output / "behavior_audit.json").exists())
            self.assertTrue((output / "behavior_audit.md").exists())
            self.assertTrue((output / "gold_survival_audit.json").exists())
            self.assertTrue((output / "gold_survival_audit.md").exists())
            self.assertTrue((output / "code_behavior_audit.json").exists())
            self.assertTrue((output / "code_behavior_audit.md").exists())
            self.assertTrue((output / "debug_trace.md").exists())
            self.assertTrue((output / "debug_trace.jsonl").exists())

            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["metrics"]["number_of_selected_questions"], 1)
            self.assertIn("baseline_hits_at_1", metrics["metrics"])
            self.assertIn("proof_state_hits_at_1", metrics["metrics"])
            self.assertIn("two_score_hits_at_1", metrics["metrics"])
            self.assertIn("future_aware_v2_hits_at_1", metrics["metrics"])
            self.assertIn("future_v2_same_first_hop_as_baseline", metrics["metrics"])
            self.assertIn("two_score_proof_state_beam", metrics["scorer_constants"])
            self.assertIn("future_aware_v2_proof_state_beam", metrics["scorer_constants"])

            rows = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["question_id"], "q1")
            self.assertIn("baseline_path_beam", rows[0])
            self.assertIn("soft_proof_state_beam", rows[0])
            self.assertIn("two_score_proof_state_beam", rows[0])
            self.assertIn("future_aware_v2_proof_state_beam", rows[0])
            self.assertTrue(rows[0]["baseline_path_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["soft_proof_state_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["two_score_proof_state_beam"]["candidate_answers"])

            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("Two-Score Proof-State Search Smoke Test", report)
            self.assertIn("Two-Score Wins Over Current Proof-State", report)
            self.assertIn("Proof-State Wins", report)
            self.assertIn("Both Fail", report)
            overlap = load_json(output / "error_overlap.json")
            self.assertIn("summary", overlap)
            self.assertIn("cases", overlap)
            self.assertEqual(overlap["summary"]["diagnostic_future_mode"], "two_score_proof_state_beam")
            overlap_report = (output / "error_overlap.md").read_text(encoding="utf-8")
            self.assertIn("Error Overlap Diagnostic", overlap_report)
            self.assertIn("two_score_proof_state_beam Repeats Baseline Mistakes", overlap_report)
            behavior = load_json(output / "behavior_audit.json")
            self.assertIn("summary", behavior)
            self.assertIn("cases", behavior)
            self.assertIn("behavior_label_counts", behavior["summary"])
            behavior_report = (output / "behavior_audit.md").read_text(encoding="utf-8")
            self.assertIn("Two-Score Behavior Audit", behavior_report)
            survival = load_json(output / "gold_survival_audit.json")
            self.assertIn("summary", survival)
            self.assertIn("cases", survival)
            self.assertIn("stage_counts", survival["summary"])
            survival_report = (output / "gold_survival_audit.md").read_text(encoding="utf-8")
            self.assertIn("Two-Score Gold Survival Audit", survival_report)
            code_behavior = load_json(output / "code_behavior_audit.json")
            self.assertIn("summary", code_behavior)
            self.assertIn("exact_formulas", code_behavior)
            self.assertIn("gate_behavior", code_behavior)
            self.assertIn("mismatch_table", code_behavior)
            code_behavior_report = (output / "code_behavior_audit.md").read_text(encoding="utf-8")
            self.assertIn("Code-Behavior Alignment Audit", code_behavior_report)
            self.assertIn("Exact Formulas From Code", code_behavior_report)
            self.assertIn("Mismatch Table", code_behavior_report)
            trace = (output / "debug_trace.md").read_text(encoding="utf-8")
            self.assertIn("Baseline Hop Trace", trace)
            self.assertIn("Proof-State Hop Trace", trace)
            self.assertIn("Two-Score Proof-State Hop Trace", trace)
            self.assertIn("Future-Aware V2 Proof-State Hop Trace", trace)
            self.assertIn("Why Proof-State Chose This Over Baseline", trace)

    def test_runner_selects_real_kqa_style_two_relate_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            output = root / "proof_state_search_smoke"
            write_json(kb_path, smoke_kb())
            write_json(questions_path, real_kqa_style_questions())

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_proof_state_search_smoke",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(questions_path),
                    "--output",
                    str(output),
                    "--max-examples",
                    "1",
                    "--top-k",
                    "2",
                    "--beam-width",
                    "2",
                ]
                run_smoke_main()
            finally:
                sys.argv = old_argv

            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["metrics"]["number_of_selected_questions"], 1)
            rows = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["question_id"], "q2")
            self.assertEqual(rows[0]["gold_answers"], ["science fiction"])


if __name__ == "__main__":
    unittest.main()
