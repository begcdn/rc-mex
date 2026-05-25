from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json

from rc_mex.run_mvp2 import compute_retrieval_metrics
from rc_mex.run_mvp2 import main as run_mvp2_main
from rc_mex.run_mvp1 import main as run_mvp1_main
from tests.test_rc_mex_mvp1 import rc_mex_kb


def rc_mex_questions():
    return [
        {
            "question": "Which films did Christopher Nolan direct?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["directed", "forward"]},
            ],
            "answer": "Inception|Dunkirk",
        },
        {
            "question": "Where was Christopher Nolan born?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["born_in", "forward"]},
            ],
            "answer": "London",
        },
    ]


class RCMexMVP2Tests(unittest.TestCase):
    def test_retrieval_metrics(self):
        rows = [
            {
                "condition_id": "A",
                "card_variant": "contrastive_hard",
                "gold_card_id": "A::contrastive_hard::directed::forward",
                "gold_in_candidate_pool": True,
                "gold_rank": 1,
                "candidate_count": 3,
            },
            {
                "condition_id": "A",
                "card_variant": "contrastive_hard",
                "gold_card_id": "A::contrastive_hard::born_in::forward",
                "gold_in_candidate_pool": True,
                "gold_rank": 3,
                "candidate_count": 3,
            },
        ]
        metrics = compute_retrieval_metrics(rows)["A/contrastive_hard"]
        self.assertEqual(metrics["candidate_recall"], 1.0)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["recall_at_3"], 1.0)

    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            mvp1_out = root / "mvp1"
            mvp2_out = root / "mvp2"
            write_json(kb_path, rc_mex_kb())
            write_json(questions_path, rc_mex_questions())

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_mvp1",
                    "--kb",
                    str(kb_path),
                    "--output",
                    str(mvp1_out),
                    "--max-primitives",
                    "5",
                    "--min-examples",
                    "1",
                    "--train-positives",
                    "1",
                    "--heldout-positives",
                    "1",
                    "--train-negatives",
                    "1",
                    "--heldout-negatives",
                    "1",
                    "--random-negatives",
                    "1",
                    "--oracle-backend",
                    "mock",
                    "--conditions",
                    "A",
                    "--card-variants",
                    "contrastive_hard",
                ]
                run_mvp1_main()
                sys.argv = [
                    "run_mvp2",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(questions_path),
                    "--cards",
                    str(mvp1_out / "relation_cards.jsonl"),
                    "--output",
                    str(mvp2_out),
                    "--conditions",
                    "A",
                    "--card-variants",
                    "contrastive_hard",
                    "--oracle-backend",
                    "mock",
                    "--max-instances",
                    "2",
                    "--relation-cap",
                    "10",
                ]
                run_mvp2_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((mvp2_out / "retrieval_predictions.jsonl").exists())
            self.assertTrue((mvp2_out / "metrics.json").exists())
            self.assertTrue((mvp2_out / "report.md").exists())
            metrics = load_json(mvp2_out / "metrics.json")
            self.assertGreater(metrics["n_prediction_rows"], 0)
            group = metrics["metrics"]["A/contrastive_hard"]
            self.assertGreaterEqual(group["candidate_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
