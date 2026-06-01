from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json

from rc_mex.run_kqa_evidence_probe import main as run_probe_main


def probe_kb():
    return {
        "concepts": {
            "person": {"name": "person", "instanceOf": []},
            "film": {"name": "film", "instanceOf": []},
            "genre": {"name": "genre", "instanceOf": []},
        },
        "entities": {
            "nolan": entity("Christopher Nolan", ["person"], [("directed", "inception"), ("produced", "inception")]),
            "inception": entity("Inception", ["film"], [("genre", "science_fiction"), ("composer", "zimmer")]),
            "science_fiction": entity("science fiction", ["genre"], []),
            "zimmer": entity("Hans Zimmer", ["person"], []),
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


def probe_questions():
    return [
        {
            "id": "q1",
            "question": "Which genre is the film directed by Christopher Nolan?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["directed", "forward"]},
                {"function": "Relate", "dependencies": [1], "inputs": ["genre", "forward"]},
            ],
            "answer": "science fiction",
        }
    ]


class KQAEvidenceProbeTests(unittest.TestCase):
    def test_probe_runner_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            output = root / "probe"
            write_json(kb_path, probe_kb())
            write_json(questions_path, probe_questions())

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_kqa_evidence_probe",
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
                run_probe_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((output / "candidate_paths.jsonl").exists())
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "report.md").exists())
            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["metrics"]["total_questions"], 1)
            rows = [
                json.loads(line)
                for line in (output / "candidate_paths.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertTrue(rows[0]["gold_answer_in_candidate_pool"])
            self.assertGreater(rows[0]["number_of_candidate_answers"], 0)


if __name__ == "__main__":
    unittest.main()
