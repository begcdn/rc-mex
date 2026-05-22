from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json

from rc_mex.debug import (
    compute_primitive_metrics,
    is_metadata_relation,
)
from rc_mex.evidence import CONDITIONS, RenderContext, render_example
from rc_mex.oracle import OPENAI_API_BASE_URL, make_client, parse_bool, parse_confidence
from rc_mex.run_mvp1 import DEFAULT_OPENAI_MODEL
from rc_mex.run_mvp1 import main as run_main
from rc_mex.sampling import sample_for_primitive
from rc_mex.schema import inventory_primitives


def rc_mex_kb():
    return {
        "concepts": {
            "film": {"name": "film", "instanceOf": []},
            "person": {"name": "person", "instanceOf": []},
            "city": {"name": "city", "instanceOf": []},
        },
        "entities": {
            "nolan": entity("Christopher Nolan", ["person"], [
                ("directed", "inception"),
                ("directed", "dunkirk"),
                ("wrote", "inception"),
                ("produced", "inception"),
                ("born_in", "london"),
            ]),
            "spielberg": entity("Steven Spielberg", ["person"], [
                ("directed", "jaws"),
                ("directed", "et"),
                ("produced", "jaws"),
                ("born_in", "cincinnati"),
            ]),
            "gerwig": entity("Greta Gerwig", ["person"], [
                ("directed", "barbie"),
                ("wrote", "barbie"),
                ("born_in", "sacramento"),
            ]),
            "cameron": entity("James Cameron", ["person"], [
                ("directed", "titanic"),
                ("wrote", "titanic"),
            ]),
            "inception": entity("Inception", ["film"], []),
            "dunkirk": entity("Dunkirk", ["film"], []),
            "jaws": entity("Jaws", ["film"], []),
            "et": entity("E.T.", ["film"], []),
            "barbie": entity("Barbie", ["film"], []),
            "titanic": entity("Titanic", ["film"], []),
            "london": entity("London", ["city"], []),
            "cincinnati": entity("Cincinnati", ["city"], []),
            "sacramento": entity("Sacramento", ["city"], []),
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


class RCMexMVP1Tests(unittest.TestCase):
    def test_primitive_inventory(self):
        from cigr_d_mvp1.kg import KnowledgeGraph

        graph = KnowledgeGraph(rc_mex_kb())
        primitives = inventory_primitives(graph, min_examples=1)
        keys = {(primitive.relation_id, primitive.direction) for primitive in primitives}
        self.assertIn(("directed", "forward"), keys)
        self.assertIn(("wrote", "forward"), keys)

    def test_sampling_no_overlap_and_hard_negatives(self):
        from cigr_d_mvp1.kg import KnowledgeGraph

        graph = KnowledgeGraph(rc_mex_kb())
        primitives = inventory_primitives(graph, min_examples=1)
        directed = [primitive for primitive in primitives if primitive.relation_id == "directed"][0]
        samples = sample_for_primitive(
            graph,
            directed,
            primitives,
            train_positives=2,
            heldout_positives=2,
            train_negatives=2,
            heldout_negatives=2,
            random_negatives=2,
            seed=3,
        )
        train_pairs = {example.pair for example in samples.positive_train}
        heldout_pairs = {example.pair for example in samples.positive_heldout}
        self.assertFalse(train_pairs & heldout_pairs)
        self.assertTrue(samples.hard_negative_train)
        self.assertTrue(all(example.pair not in directed.extension for example in samples.hard_negative_train))

    def test_entity_type_ablation_rendering(self):
        from cigr_d_mvp1.kg import KnowledgeGraph

        graph = KnowledgeGraph(rc_mex_kb())
        primitives = inventory_primitives(graph, min_examples=1)
        context = RenderContext.from_primitives(primitives)
        example = primitives[0].examples[0]
        b2 = render_example(graph, context, CONDITIONS["B2"], example)
        b3 = render_example(graph, context, CONDITIONS["B3"], example)
        self.assertTrue(b2["head"].startswith("ENTITY_"))
        self.assertTrue(b2["head_types"])
        self.assertFalse(b3["head_types"])

    def test_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb_path = root / "kb.json"
            out = root / "out"
            write_json(kb_path, rc_mex_kb())
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_mvp1",
                    "--kb",
                    str(kb_path),
                    "--output",
                    str(out),
                    "--max-primitives",
                    "3",
                    "--min-examples",
                    "2",
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
                    "A,B1,B2,B3,C",
                    "--verbose",
                    "--debug-examples-per-primitive",
                    "2",
                ]
                run_main()
            finally:
                sys.argv = old_argv
            self.assertTrue((out / "relation_cards.jsonl").exists())
            self.assertTrue((out / "validation_predictions.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "debug_examples.md").exists())
            self.assertTrue((out / "debug_report.html").exists())
            self.assertTrue((out / "examples_summary.json").exists())
            self.assertTrue((out / "primitive_metrics.jsonl").exists())
            html = (out / "debug_report.html").read_text(encoding="utf-8")
            self.assertIn("Summary Dashboard", html)
            self.assertIn("Primitive Browser", html)
            self.assertIn("Most Important Failures", html)
            metrics = load_json(out / "metrics.json")
            self.assertGreater(metrics["n_cards"], 0)

    def test_openai_backend_defaults(self):
        client = make_client(
            backend="openai",
            model=DEFAULT_OPENAI_MODEL,
            ollama_host=None,
            openai_base_url=None,
            openai_api_key="test-key",
            command=None,
        )
        self.assertEqual(client.model, "gpt-4o-mini")
        self.assertEqual(client.base_url, OPENAI_API_BASE_URL)

    def test_local_model_string_confidence_and_booleans(self):
        self.assertEqual(parse_confidence("low"), 0.25)
        self.assertEqual(parse_confidence("high"), 0.85)
        self.assertEqual(parse_confidence("82%"), 0.82)
        self.assertFalse(parse_bool("false"))
        self.assertFalse(parse_bool("no"))
        self.assertTrue(parse_bool("true"))

    def test_metadata_relation_detection_and_diagnosis(self):
        self.assertTrue(is_metadata_relation("Wikidata property"))
        self.assertTrue(is_metadata_relation("external ID"))
        self.assertFalse(is_metadata_relation("directed"))
        cards = [
            {
                "primitive_id": "P00001",
                "relation_id": "Wikidata property",
                "direction": "backward",
                "condition_id": "A",
                "card_variant": "contrastive_hard",
                "confidence": 0.5,
                "opaque_reason": "",
            }
        ]
        predictions = [
            {
                "primitive_id": "P00001",
                "relation_id": "Wikidata property",
                "direction": "backward",
                "condition_id": "A",
                "card_variant": "contrastive_hard",
                "category": "positive",
                "expected_label": True,
                "predicted_satisfies": False,
                "predicted_direction_correct": False,
                "confidence": 0.9,
                "pair": {"head": "A", "tail": "B", "head_types": [], "tail_types": []},
            },
            {
                "primitive_id": "P00001",
                "relation_id": "Wikidata property",
                "direction": "backward",
                "condition_id": "A",
                "card_variant": "contrastive_hard",
                "category": "hard_negative",
                "expected_label": False,
                "predicted_satisfies": True,
                "predicted_direction_correct": True,
                "confidence": 0.9,
                "pair": {"head": "C", "tail": "D", "head_types": [], "tail_types": []},
            },
        ]
        rows = compute_primitive_metrics(cards, predictions, ["wikidata property"])
        self.assertIn("metadata_relation", rows[0]["diagnosis"])
        self.assertIn("possibly_opaque", rows[0]["diagnosis"])
        self.assertIn("hard_negatives_not_helping", rows[0]["diagnosis"])


if __name__ == "__main__":
    unittest.main()
