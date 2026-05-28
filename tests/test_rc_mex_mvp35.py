from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json

from rc_mex.run_mvp1 import main as run_mvp1_main
from rc_mex.run_mvp35 import main as run_mvp35_main
from tests.test_rc_mex_mvp1 import rc_mex_kb
from tests.test_rc_mex_mvp2 import rc_mex_questions


class RCMexMVP35Tests(unittest.TestCase):
    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            mvp1_out = root / "mvp1"
            mvp35_out = root / "mvp35"
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
                    "A,B1,C",
                    "--card-variants",
                    "contrastive_hard",
                ]
                run_mvp1_main()
                sys.argv = [
                    "run_mvp35",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(questions_path),
                    "--cards",
                    str(mvp1_out / "relation_cards.jsonl"),
                    "--output",
                    str(mvp35_out),
                    "--conditions",
                    "A,B1,C",
                    "--card-variants",
                    "contrastive_hard",
                    "--oracle-backend",
                    "mock",
                    "--max-instances",
                    "2",
                    "--covered-only",
                ]
                run_mvp35_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((mvp35_out / "comparison_predictions.jsonl").exists())
            self.assertTrue((mvp35_out / "metrics.json").exists())
            self.assertTrue((mvp35_out / "debug_examples.md").exists())
            metrics = load_json(mvp35_out / "metrics.json")
            self.assertGreater(metrics["n_rows"], 0)
            self.assertIn("A/contrastive_hard/relation_label", metrics["metrics"])
            self.assertIn("A/contrastive_hard/relation_card", metrics["metrics"])
            self.assertIn("A/contrastive_hard/relation_card_blueprint", metrics["metrics"])
            self.assertIn("robustness_drop", metrics["metrics"])


if __name__ == "__main__":
    unittest.main()
