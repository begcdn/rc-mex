from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json

from rc_mex.evidence import CONDITIONS, RenderContext, render_example
from rc_mex.oracle import OPENAI_API_BASE_URL, make_client
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
                ]
                run_main()
            finally:
                sys.argv = old_argv
            self.assertTrue((out / "relation_cards.jsonl").exists())
            self.assertTrue((out / "validation_predictions.jsonl").exists())
            self.assertTrue((out / "metrics.json").exists())
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


if __name__ == "__main__":
    unittest.main()
