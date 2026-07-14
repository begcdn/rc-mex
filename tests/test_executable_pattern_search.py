import unittest

import numpy as np

from rc_mex.executable_pattern_search import (
    build_executable_pattern_menu,
    enumerate_pattern_hypotheses,
    retain_depth_families,
)


class ToyEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            text = text.casefold()
            vectors.append(
                np.asarray(
                    [
                        float("spouse" in text),
                        float("place of birth" in text or "born" in text),
                        0.1,
                    ],
                    dtype="float32",
                )
            )
        values = np.stack(vectors)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-8)


def toy_kb():
    return {
        "entities": {
            "ada": {
                "relations": [
                    {"predicate": "spouse", "direction": "forward", "object": "bob"},
                    {"predicate": "profession", "direction": "forward", "object": "mathematician"},
                ]
            },
            "bob": {
                "relations": [
                    {"predicate": "place_of_birth", "direction": "forward", "object": "paris"},
                    {"predicate": "spouse", "direction": "backward", "object": "ada"},
                ]
            },
            "mathematician": {"relations": []},
            "paris": {"relations": []},
        }
    }


class ExecutablePatternSearchTests(unittest.TestCase):
    def test_enumeration_preserves_complete_two_hop_hypothesis(self):
        hypotheses = enumerate_pattern_hypotheses(toy_kb(), {"ada"})
        steps = {hypothesis.steps for hypothesis in hypotheses}
        self.assertIn(
            (("spouse", "forward"), ("place_of_birth", "forward")),
            steps,
        )

    def test_depth_family_retention_does_not_let_two_hop_crowd_out_one_hop(self):
        hypotheses = enumerate_pattern_hypotheses(toy_kb(), {"ada"})
        for index, hypothesis in enumerate(hypotheses):
            hypothesis.score = float(index)
        retained = retain_depth_families(hypotheses, patterns_per_depth=1)
        self.assertEqual({hypothesis.depth for hypothesis in retained}, {1, 2})

    def test_menu_scores_full_pattern_and_executes_its_denotation(self):
        menu = build_executable_pattern_menu(
            toy_kb(),
            {"ada"},
            "Where was Ada's spouse born?",
            ToyEncoder(),
        )
        matching = [
            candidate
            for candidate in menu
            if candidate.get("pattern_steps")
            == [
                {"predicate": "spouse", "direction": "forward"},
                {"predicate": "place_of_birth", "direction": "forward"},
            ]
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["targets"], ["paris"])


if __name__ == "__main__":
    unittest.main()
