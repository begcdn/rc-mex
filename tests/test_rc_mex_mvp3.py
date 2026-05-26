from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph
from cigr_d_mvp1.kopl import RelationGroundingInstance

from rc_mex.run_mvp3 import evaluate_prediction
from rc_mex.run_mvp3 import main as run_mvp3_main
from tests.test_rc_mex_mvp1 import rc_mex_kb
from tests.test_rc_mex_mvp2 import rc_mex_questions


class RCMexMVP3Tests(unittest.TestCase):
    def test_marginal_execution_recovers_gold_when_top1_is_wrong(self):
        graph = KnowledgeGraph(rc_mex_kb())
        nolan = next(iter(graph.find_entities("Christopher Nolan")))
        instance = RelationGroundingInstance(
            instance_id="val:0:1",
            question="Which films did Christopher Nolan direct?",
            program_index=0,
            step_index=1,
            current_entity_ids={nolan},
            gold_predicate="directed",
            gold_direction="forward",
            answer=None,
        )
        prediction = {
            "condition_id": "A",
            "card_variant": "contrastive_hard",
            "frontier_mode": "oracle_include_gold",
            "local_gold_in_candidate_pool": True,
            "injected_gold": False,
            "ranked_card_ids": [
                "A::contrastive_hard::produced::forward",
                "A::contrastive_hard::directed::forward",
            ],
        }
        row = evaluate_prediction(graph, instance, prediction, top_k=2, weight_scheme="reciprocal_rank")
        self.assertLess(row["top1_gold_recall"], 1.0)
        self.assertEqual(row["marginal_gold_recall"], 1.0)
        self.assertTrue(row["direct_edge_executable"])
        self.assertFalse(row["gold_relation_in_top1"])
        self.assertTrue(row["gold_relation_in_top3"])
        self.assertIn("top1_precision", row)
        self.assertIn("marginal_precision", row)
        self.assertIn("top1_f1", row)
        self.assertIn("marginal_f1", row)

    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            questions_path = root / "val.json"
            predictions_path = root / "retrieval_predictions.jsonl"
            output = root / "mvp3"
            write_json(kb_path, rc_mex_kb())
            write_json(questions_path, [rc_mex_questions()[0]])
            write_jsonl(
                predictions_path,
                [
                    {
                        "instance_id": "val:0:1",
                        "question": "Which films did Christopher Nolan direct?",
                        "condition_id": "A",
                        "card_variant": "contrastive_hard",
                        "frontier_mode": "local_frontier",
                        "local_gold_in_candidate_pool": True,
                        "injected_gold": False,
                        "gold_card_id": "A::contrastive_hard::directed::forward",
                        "ranked_card_ids": [
                            "A::contrastive_hard::directed::forward",
                            "A::contrastive_hard::produced::forward",
                        ],
                    }
                ],
            )
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_mvp3",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(questions_path),
                    "--retrieval-predictions",
                    str(predictions_path),
                    "--output",
                    str(output),
                    "--conditions",
                    "A",
                    "--card-variants",
                    "contrastive_hard",
                    "--top-k",
                    "2",
                ]
                run_mvp3_main()
            finally:
                sys.argv = old_argv
            self.assertTrue((output / "execution_predictions.jsonl").exists())
            self.assertTrue((output / "metrics.json").exists())
            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["n_execution_rows"], 1)
            group = metrics["metrics"]["A/contrastive_hard"]
            self.assertEqual(group["direct_edge_executable_rate"], 1.0)
            self.assertEqual(group["gold_relation_in_top1_rate"], 1.0)
            self.assertIn("average_top1_result_size", group)
            self.assertIn("average_marginal_result_size", group)
            self.assertIn("__executable_only__", metrics["metrics"])
            self.assertIn("__local_frontier_only__", metrics["metrics"])
            rows = [
                json.loads(line)
                for line in (output / "execution_predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["top1_gold_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
