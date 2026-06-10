from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from cigr_d_mvp1.io_utils import load_json, write_json
from cigr_d_mvp1.kg import KnowledgeGraph
from rc_mex import run_proof_state_search_smoke as smoke
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
            self.assertTrue((output / "behavior_audit.json").exists())
            self.assertTrue((output / "behavior_audit.md").exists())
            self.assertTrue((output / "gold_survival_audit.json").exists())
            self.assertTrue((output / "gold_survival_audit.md").exists())
            self.assertTrue((output / "code_behavior_audit.json").exists())
            self.assertTrue((output / "code_behavior_audit.md").exists())
            self.assertTrue((output / "debug_trace.md").exists())
            self.assertTrue((output / "debug_trace.jsonl").exists())

            metrics = load_json(output / "metrics.json")
            self.assertEqual(metrics["metrics"]["number_of_selected_questions"], 1)
            self.assertIn("baseline_hits_at_1", metrics["metrics"])
            self.assertIn("proof_state_hits_at_1", metrics["metrics"])
            self.assertIn("two_score_hits_at_1", metrics["metrics"])
            self.assertIn("two_score_fixed_hits_at_1", metrics["metrics"])
            self.assertIn("two_score_wide_hits_at_1", metrics["metrics"])
            self.assertIn("hybrid_relation_proposal_hits_at_1", metrics["metrics"])
            self.assertIn("path_family_hits_at_1", metrics["metrics"])
            self.assertIn("path_family_answer_verifier_hits_at_1", metrics["metrics"])
            self.assertIn("future_aware_v2_hits_at_1", metrics["metrics"])
            self.assertIn("future_v2_same_first_hop_as_baseline", metrics["metrics"])
            self.assertIn("two_score_proof_state_beam", metrics["scorer_constants"])
            self.assertIn("two_score_fixed_proof_state_beam", metrics["scorer_constants"])
            self.assertIn("two_score_wide_proposal_beam", metrics["scorer_constants"])
            self.assertIn("hybrid_relation_proposal_beam", metrics["scorer_constants"])
            self.assertIn("path_family_beam", metrics["scorer_constants"])
            self.assertIn("path_family_answer_verifier", metrics["scorer_constants"])
            self.assertIn("future_aware_v2_proof_state_beam", metrics["scorer_constants"])

            rows = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["question_id"], "q1")
            self.assertIn("baseline_path_beam", rows[0])
            self.assertIn("soft_proof_state_beam", rows[0])
            self.assertIn("two_score_proof_state_beam", rows[0])
            self.assertIn("two_score_fixed_proof_state_beam", rows[0])
            self.assertIn("two_score_wide_proposal_beam", rows[0])
            self.assertIn("hybrid_relation_proposal_beam", rows[0])
            self.assertIn("path_family_beam", rows[0])
            self.assertIn("path_family_answer_verifier", rows[0])
            self.assertIn("future_aware_v2_proof_state_beam", rows[0])
            self.assertTrue(rows[0]["baseline_path_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["soft_proof_state_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["two_score_proof_state_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["two_score_fixed_proof_state_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["two_score_wide_proposal_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["hybrid_relation_proposal_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["path_family_beam"]["candidate_answers"])
            self.assertTrue(rows[0]["path_family_answer_verifier"]["candidate_answers"])
            self.assertIn("family_answer_ranker_diagnostics", rows[0]["path_family_answer_verifier"])

            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("Path-Family Answer Verifier Smoke Test", report)
            self.assertIn("Answer Verifier Wins Over Path-Family", report)
            self.assertIn("Path-Family Wins Over Hybrid Proposal", report)
            self.assertIn("Hybrid Wins Over Wide Lexical Proposal", report)
            self.assertIn("Proof-State Wins", report)
            self.assertIn("Both Fail", report)
            overlap = load_json(output / "error_overlap.json")
            self.assertIn("summary", overlap)
            self.assertIn("cases", overlap)
            self.assertEqual(overlap["summary"]["diagnostic_future_mode"], "path_family_answer_verifier")
            overlap_report = (output / "error_overlap.md").read_text(encoding="utf-8")
            self.assertIn("Error Overlap Diagnostic", overlap_report)
            self.assertIn("path_family_answer_verifier Repeats Baseline Mistakes", overlap_report)
            behavior = load_json(output / "behavior_audit.json")
            self.assertIn("summary", behavior)
            self.assertIn("cases", behavior)
            self.assertIn("behavior_label_counts", behavior["summary"])
            behavior_report = (output / "behavior_audit.md").read_text(encoding="utf-8")
            self.assertIn("Path Family Answer Verifier Behavior Audit", behavior_report)
            survival = load_json(output / "gold_survival_audit.json")
            self.assertIn("summary", survival)
            self.assertIn("cases", survival)
            self.assertIn("stage_counts", survival["summary"])
            survival_report = (output / "gold_survival_audit.md").read_text(encoding="utf-8")
            self.assertIn("Path Family Answer Verifier Gold Survival Audit", survival_report)
            self.assertIn("gold_first_hop_relation_rank_before_proposal", survival["cases"][0])
            self.assertIn("first_hop_relation_label_rank_original_lexical", survival["cases"][0])
            self.assertIn("first_hop_relation_label_rank_hybrid", survival["cases"][0])
            self.assertIn("was_gold_first_hop_in_relation_proposal", survival["cases"][0])
            self.assertIn("gold_first_hop_recovered_by_hybrid_proposal", survival["summary"])
            code_behavior = load_json(output / "code_behavior_audit.json")
            self.assertIn("summary", code_behavior)
            self.assertEqual(code_behavior["summary"]["diagnostic_mode"], "path_family_answer_verifier")
            self.assertIn("exact_formulas", code_behavior)
            self.assertIn("path_family_answer_verifier", code_behavior["exact_formulas"])
            self.assertIn("gate_behavior", code_behavior)
            self.assertIn("mismatch_table", code_behavior)
            code_behavior_report = (output / "code_behavior_audit.md").read_text(encoding="utf-8")
            self.assertIn("Code-Behavior Alignment Audit", code_behavior_report)
            self.assertIn("Exact Formulas From Code", code_behavior_report)
            self.assertIn("Mismatch Table", code_behavior_report)
            trace = (output / "debug_trace.md").read_text(encoding="utf-8")
            self.assertIn("Baseline Hop Trace", trace)
            self.assertIn("Proof-State Hop Trace", trace)
            self.assertIn("Two-Score Proof-State Hop Trace", trace)
            self.assertIn("Two-Score Fixed Proof-State Hop Trace", trace)
            self.assertIn("Two-Score Wide-Proposal Proof-State Hop Trace", trace)
            self.assertIn("Hybrid Relation-Proposal Proof-State Hop Trace", trace)
            self.assertIn("Path-Family Beam Hop Trace", trace)
            self.assertIn("Path-Family Answer Verifier Hop Trace", trace)
            self.assertIn("Future-Aware V2 Proof-State Hop Trace", trace)
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

    def test_role_aware_verifier_distinguishes_answer_roles(self):
        graph = KnowledgeGraph(
            {
                "concepts": {
                    "film": {"name": "film", "instanceOf": []},
                    "film_series": {"name": "film series", "instanceOf": []},
                    "occupation": {"name": "occupation", "instanceOf": []},
                    "human": {"name": "human", "instanceOf": []},
                    "country": {"name": "country", "instanceOf": []},
                    "county": {"name": "county", "instanceOf": []},
                    "admin": {"name": "administrative territorial entity", "instanceOf": []},
                },
                "entities": {
                    "generic_film": entity("Generic Movie", ["film"], [("genre", "genre")]),
                    "film_series": entity("Example Film Series", ["film_series"], [("franchise", "brand")]),
                    "person": entity("Ada Person", ["human"], [("occupation", "occupation_entity")]),
                    "occupation_entity": entity("Software engineer", ["occupation"], [("field of work", "software")]),
                    "country": entity("France", ["country"], [("capital", "city")]),
                    "county": entity("Orange County", ["county", "admin"], [("located in the administrative territorial entity", "state")]),
                    "unknown": entity("Mystery Entity", [], []),
                    "contradict": entity("Random Human", ["human"], [("place of birth", "city")]),
                    "genre": entity("teen film", [], []),
                    "brand": entity("Brand", [], []),
                    "software": entity("software", [], []),
                    "city": entity("Paris", [], []),
                    "state": entity("California", [], []),
                },
            }
        )

        def verify(question: str, entity_id: str) -> dict:
            return smoke.answer_side_verification_role_aware(
                graph=graph,
                question=question,
                answer_type=smoke.guess_answer_type(question),
                entity_id=entity_id,
                old_score=1.0,
                evidence=[],
                support_count=1,
            )

        film_series = verify("What film series has this genre?", "film_series")
        generic_film = verify("What film series has this genre?", "generic_film")
        self.assertGreater(film_series["verified_final_score"], generic_film["verified_final_score"])

        occupation = verify("What occupation is associated with this person?", "occupation_entity")
        person = verify("What occupation is associated with this person?", "person")
        self.assertGreater(occupation["verified_final_score"], person["verified_final_score"])
        self.assertLess(person["role_confusion_penalty"], 0.0)

        county = verify("Which county contains this place?", "county")
        country = verify("Which county contains this place?", "country")
        self.assertGreater(county["verified_final_score"], country["verified_final_score"])

        unknown = verify("What film series has this genre?", "unknown")
        self.assertGreater(unknown["verified_final_score"], 0.5)
        self.assertEqual(unknown["contradiction_penalty"], 0.0)

        contradiction = verify("What film series has this genre?", "contradict")
        self.assertLess(contradiction["contradiction_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
