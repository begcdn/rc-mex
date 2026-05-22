from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import write_json
from cigr_d_mvp1.kg import KnowledgeGraph
from cigr_d_mvp1.kopl import extract_relation_grounding_instances
from cigr_d_mvp1.run_mvp1 import main as run_main
from cigr_d_mvp1.witness import build_witness_cards


def mini_kb():
    return {
        "concepts": {
            "film": {"name": "film", "instanceOf": []},
            "person": {"name": "person", "instanceOf": []},
        },
        "entities": {
            "nolan": {
                "name": "Christopher Nolan",
                "instanceOf": ["person"],
                "relations": [
                    {
                        "predicate": "directed",
                        "object": "inception",
                        "direction": "forward",
                        "qualifiers": {},
                    },
                    {
                        "predicate": "directed",
                        "object": "dunkirk",
                        "direction": "forward",
                        "qualifiers": {},
                    },
                    {
                        "predicate": "spouse",
                        "object": "emma",
                        "direction": "forward",
                        "qualifiers": {},
                    },
                ],
                "attributes": [],
            },
            "inception": {
                "name": "Inception",
                "instanceOf": ["film"],
                "relations": [],
                "attributes": [],
            },
            "dunkirk": {
                "name": "Dunkirk",
                "instanceOf": ["film"],
                "relations": [],
                "attributes": [],
            },
            "emma": {
                "name": "Emma Thomas",
                "instanceOf": ["person"],
                "relations": [],
                "attributes": [],
            },
        },
    }


def mini_questions():
    return [
        {
            "question": "Which films did Christopher Nolan direct?",
            "program": [
                {"function": "Find", "dependencies": [], "inputs": ["Christopher Nolan"]},
                {"function": "Relate", "dependencies": [0], "inputs": ["directed", "forward"]},
            ],
            "answer": "Inception|Dunkirk",
        }
    ]


class MVP1Tests(unittest.TestCase):
    def test_extracts_relation_instance(self):
        graph = KnowledgeGraph(mini_kb())
        instances, stats = extract_relation_grounding_instances(mini_questions(), graph, "mini")
        self.assertEqual(stats["instances_created"], 1)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].gold_predicate, "directed")
        self.assertEqual(instances[0].current_entity_ids, {"nolan"})

    def test_candidate_generation_and_witness(self):
        graph = KnowledgeGraph(mini_kb())
        candidates = graph.candidate_relations({"nolan"}, cap=10, sample_entities=10)
        self.assertIn(("directed", "forward"), {candidate.key for candidate in candidates})
        cards = build_witness_cards(
            graph,
            {"nolan"},
            candidates,
            label_mode="anonymous",
            witness_mode="anon_entities_types",
            returned_sample_size=3,
        )
        directed = [card for card in cards if card.predicate == "directed"][0]
        self.assertEqual(directed.display_relation, "R_001")
        self.assertIn("film", " ".join(directed.returned_types))
        self.assertTrue(all(entity.startswith("ENTITY_") for entity in directed.returned_entities))

    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            q_path = root / "val.json"
            out = root / "out"
            write_json(kb_path, mini_kb())
            write_json(q_path, mini_questions())
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_mvp1",
                    "--kb",
                    str(kb_path),
                    "--questions",
                    str(q_path),
                    "--output",
                    str(out),
                    "--max-instances",
                    "1",
                    "--bootstrap-samples",
                    "10",
                    "--judge-backend",
                    "mock",
                ]
                run_main()
            finally:
                sys.argv = old_argv
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
