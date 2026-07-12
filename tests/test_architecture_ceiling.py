from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cigr_d_mvp1.io_utils import load_json, write_json
from rc_mex.run_architecture_ceiling import (
    excluded_question_ids,
    freeze_rows,
    main as run_ceiling_main,
    set_scores,
    structural_ceiling,
)
from rc_mex.run_webqsp_path_family import build_kb


def source_row(question_id: str = "q1") -> dict:
    return {
        "id": question_id,
        "question": "Which genre is the film directed by Christopher Nolan?",
        "answer": ["science fiction"],
        "q_entity": ["Christopher Nolan"],
        "a_entity": ["science fiction"],
        "graph": [
            ["Christopher Nolan", "film.director", "Inception"],
            ["Christopher Nolan", "film.producer", "Inception"],
            ["Inception", "film.genre", "science fiction"],
        ],
    }


class ArchitectureCeilingTests(unittest.TestCase):
    def test_structural_ceiling_finds_two_hop_answer(self):
        kb = build_kb(source_row()["graph"])
        ceiling = structural_ceiling(
            kb,
            {"christopher nolan"},
            {"science fiction"},
            fanout_cap=100,
        )
        self.assertFalse(ceiling["best_one_hop"]["scores"]["has_gold"])
        self.assertTrue(ceiling["best_up_to_two_hop"]["scores"]["has_gold"])
        self.assertEqual(ceiling["best_up_to_two_hop"]["scores"]["f1"], 1.0)
        self.assertEqual(len(ceiling["best_up_to_two_hop"]["query"]), 2)

    def test_freeze_rows_is_deterministic_and_excludes_prior_ids(self):
        rows = [source_row(f"q{i}") for i in range(10)]
        first = freeze_rows(rows, excluded_ids={"q0", "q1"}, offset=0, limit=0, sample_size=3, seed=17)
        second = freeze_rows(rows, excluded_ids={"q0", "q1"}, offset=0, limit=0, sample_size=3, seed=17)
        self.assertEqual([row["id"] for _, row in first], [row["id"] for _, row in second])
        self.assertFalse({"q0", "q1"} & {row["id"] for _, row in first})

    def test_excluded_ids_load_from_prediction_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text('{"question_id":"seen-1"}\n{"id":"seen-2"}\n', encoding="utf-8")
            self.assertEqual(excluded_question_ids([str(path)]), {"seen-1", "seen-2"})

    def test_runner_writes_firewall_and_ceiling_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            output = root / "audit"
            data.write_text(json.dumps(source_row()) + "\n", encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_architecture_ceiling",
                    "--data",
                    str(data),
                    "--output",
                    str(output),
                    "--sample-size",
                    "1",
                ]
                run_ceiling_main()
            finally:
                sys.argv = old_argv

            for name in (
                "split_manifest.json",
                "eval_questions.jsonl",
                "ceiling_questions.jsonl",
                "metrics.json",
                "report.md",
            ):
                self.assertTrue((output / name).exists(), name)
            metrics = load_json(output / "metrics.json")["metrics"]
            self.assertEqual(metrics["questions"], 1)
            self.assertEqual(metrics["structural_ceiling"]["up_to_two_hop_gold_recall"], 1.0)
            self.assertEqual(metrics["subgraph_complete_gold_set_rate"], 1.0)
            self.assertEqual(metrics["subgraph_gold_entity_recall"], 1.0)
            self.assertIsNone(metrics["full_kg_answerability"])
            self.assertIn("Scope Warnings", (output / "report.md").read_text(encoding="utf-8"))

    def test_saved_prediction_adds_menu_and_selector_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "audit"
            data.write_text(json.dumps(source_row()) + "\n", encoding="utf-8")
            prediction = {
                "question_id": "q1",
                "question": source_row()["question"],
                "described_names": ["director"],
                "described_names_2": ["genre"],
                "candidates": [
                    {"predicate": "film.genre", "direction": "forward", "size": 1}
                ],
                "selected": {
                    "predicate": "film.genre",
                    "direction": "forward",
                    "readable": "genre",
                },
                "predicted": ["science fiction"],
            }
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            menu = [
                {
                    "predicate": "film.genre",
                    "direction": "forward",
                    "targets": ["science fiction"],
                    "also": [],
                    "member_quals": {},
                }
            ]
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_architecture_ceiling",
                    "--data",
                    str(data),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                ]
                with patch("rc_mex.run_architecture_ceiling.build_query_menu", return_value=menu):
                    run_ceiling_main()
            finally:
                sys.argv = old_argv

            metrics = load_json(output / "metrics.json")["metrics"]
            self.assertEqual(metrics["generated_menu"]["gold_recall"], 1.0)
            self.assertEqual(metrics["selection_and_answer"]["hits_at_1"], 1.0)
            self.assertEqual(metrics["failure_stage_counts"], {"correct": 1})
            self.assertIn("Conditional Selection", (output / "report.md").read_text(encoding="utf-8"))

    def test_set_scores_preserves_set_semantics(self):
        scores = set_scores({"a", "b"}, {"b", "c"})
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertEqual(scores["f1"], 0.5)
        self.assertFalse(scores["exact_match"])

    def test_unresolved_topic_does_not_break_prediction_aggregation(self):
        from rc_mex.run_architecture_ceiling import aggregate

        row = {
            "prediction_joined": True,
            "menu_matches_stored": None,
            "menu_audit": {
                "candidate_count": 0,
                "best_candidate": {"scores": set_scores(set(), {"gold"})},
                "selected_candidate": None,
            },
            "actual_answer": {"hits_at_1": False, "scores": set_scores(set(), {"gold"})},
            "prediction_flags": {"extended": False},
            "failure_stage": "topic_missing_from_subgraph",
            "start_resolved": False,
            "gold_in_subgraph": False,
            "all_gold_in_subgraph": False,
            "gold_subgraph_recall": 0.0,
            "approx_gold_hops": 0,
            "surface_operator_tags": ["unmarked"],
            "structural_ceiling": {
                "best_one_hop": {"scores": set_scores(set(), {"gold"})},
                "best_up_to_two_hop": {"scores": set_scores(set(), {"gold"})},
            },
        }
        metrics = aggregate([row], predictions_supplied=True)
        self.assertEqual(metrics["generated_menu"]["menu_reconstruction_match_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
