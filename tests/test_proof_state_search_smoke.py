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
            self.assertTrue((output / "debug_trace.md").exists())
            self.assertTrue((output / "debug_trace.jsonl").exists())

            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["metrics"]["number_of_selected_questions"], 1)
            self.assertIn("baseline_hits_at_1", metrics["metrics"])
            self.assertIn("proof_state_hits_at_1", metrics["metrics"])

            rows = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["question_id"], "q1")
            self.assertIn("baseline_path_beam", rows[0])
            self.assertIn("soft_proof_state_beam", rows[0])
            self.assertTrue(rows[0]["baseline_path_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["soft_proof_state_beam"]["candidate_answers"])

            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("Proof-State Search Smoke Test", report)
            self.assertIn("Proof-State Wins", report)
            self.assertIn("Both Fail", report)
            overlap = load_json(output / "error_overlap.json")
            self.assertIn("summary", overlap)
            self.assertIn("cases", overlap)
            overlap_report = (output / "error_overlap.md").read_text(encoding="utf-8")
            self.assertIn("Future-Aware Error Overlap Diagnostic", overlap_report)
            self.assertIn("Future-Aware Repeats Baseline Mistakes", overlap_report)
            trace = (output / "debug_trace.md").read_text(encoding="utf-8")
            self.assertIn("Baseline Hop Trace", trace)
            self.assertIn("Proof-State Hop Trace", trace)
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
