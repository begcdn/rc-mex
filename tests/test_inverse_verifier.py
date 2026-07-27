from __future__ import annotations

import json
import random
import urllib.error
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from inverse_verifier.data import (
    ENTITY_PLACEHOLDER,
    PathSpec,
    Hop,
    delexicalize_question,
    extract_kqa_examples,
    make_negative_paths,
    parse_sparql_path,
    relation_inventory,
    render_path,
    assign_kqa_splits,
)
from inverse_verifier.comparator_corpus import (
    build_comparator_corpus,
    label_candidates,
    sample_candidates,
)
from inverse_verifier.metrics import ranking_metrics, rouge_l, token_f1
from inverse_verifier.evaluate import generated_question_similarity_scores, similarity_distribution
from inverse_verifier.model import (
    FaithfulInverseDataset,
    TypeAwareGeneratorDataset,
    direct_prompt,
    informative_answer_type,
    normalized_sequence_nll,
    rank_prompt,
    type_compatibility_prompt,
)
from inverse_verifier.synthetic import (
    _expand_bidirectional_evaluation_rows,
    ConcretePath,
    Edge,
    ExecutableGraph,
    canonical_question,
    example_from_path,
    question_covers_path,
    naturalize_corpus,
)
from inverse_verifier.openai_naturalize import (
    _prediction_rejection,
    candidate_items,
    select_negatives,
    validate_question,
    select_rows,
    run_chat_records_sync,
)
from inverse_verifier.query_representation import represent_query
from inverse_verifier.dataset_builder import (
    combine_contrastive_results,
    compact_query,
    validate_glossary,
    validation_rejection,
)
from inverse_verifier.causal_generator import (
    CausalInverseDataset,
    flatten_path_question_pairs,
)
from inverse_verifier.comparator import (
    ANSWER_CHANNELS,
    candidate_answers,
    candidate_specs,
    comparator_answer_evidence,
    comparator_answer_text,
    comparator_input_text,
    comparator_path_text,
    listwise_multi_positive_loss,
    materialize_comparator_data,
)
from inverse_verifier.selector import run_verifier_pipeline
from inverse_verifier.semantic_benchmark import (
    ADJUDICATOR_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    apply_adjudication,
    apply_semantic_labels,
    parse_semantic_results,
    select_adjudication_candidates,
    semantic_adjudication_record,
    semantic_judge_record,
)


def test_openai_naturalization_payload_is_sanitized() -> None:
    row = {
        "question": "Which country was Ada born in?",
        "positive_path": {
            "anchor": "Ada Lovelace",
            "anchor_type": "person",
            "answer_type": "country",
            "hops": [{
                "relation": "people.person.place_of_birth",
                "direction": "forward",
                "source_type": "person",
                "target_type": "place",
            }],
        },
        "negative_paths": [],
        "_naturalization": {"split": "train", "source_index": 0},
    }
    payload = candidate_items([row])[0]
    serialized = json.dumps(payload)
    assert "Ada Lovelace" not in serialized
    assert "Which country" not in serialized
    assert payload["topic"] == "[ENTITY]"


def test_openai_naturalization_validation_rejects_procedural_output() -> None:
    assert validate_question("Which country was [ENTITY] born in?") is None
    assert validate_question("Follow the first hop from [ENTITY]?") == "procedural_language"
    assert validate_question("Where was Ada born?") == "entity_placeholder_count"
    assert _prediction_rejection({"status": "opaque", "question": "", "reason": "mixed"}) == "model_marked_opaque"


def test_openai_naturalization_selects_diverse_negative_types() -> None:
    row = {
        "negative_paths": [
            {"negative_type": "reversed"},
            {"negative_type": "reversed"},
            {"negative_type": "added_hop"},
        ]
    }
    selected = select_negatives(row, 2)
    assert [item["negative_type"] for item in selected] == ["reversed", "added_hop"]


def test_openai_naturalization_reports_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_faithful.jsonl"):
        select_rows(tmp_path, 10, 3)


def test_synchronous_chat_records_resume_without_repeating_calls(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def _request(self, method, endpoint, data):
            self.calls += 1
            request = json.loads(data)
            identifier = json.loads(request["messages"][-1]["content"])["id"]
            content = json.dumps({"items": [{"id": identifier, "status": "valid"}]})
            return {"id": f"response-{identifier}", "choices": [{"message": {"content": content}}]}

    records = [
        {
            "custom_id": f"request-{index}",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": json.dumps({"id": f"item-{index}"})}]
            },
        }
        for index in range(3)
    ]
    client = FakeClient()
    files = run_chat_records_sync(records, tmp_path, client, "test", workers=2)
    assert client.calls == 3
    assert len(files[0].read_text().splitlines()) == 3

    run_chat_records_sync(records, tmp_path, client, "test", workers=2)
    assert client.calls == 3


def test_explicit_query_representation_orients_forward_fact() -> None:
    query = represent_query(
        PathSpec(
            "Ada Lovelace",
            "person",
            (Hop("people.person.place_of_birth", "forward", "person", "city"),),
            "city",
            "webqsp",
        )
    )
    assert query["variables"] == [
        {"id": "v0", "role": "anchor", "binding": "[ENTITY]", "type": "person"},
        {"id": "v1", "role": "answer", "type": "city"},
    ]
    triple = query["triples"][0]
    assert (triple["subject"], triple["predicate"], triple["object"]) == (
        "v0",
        "people.person.place_of_birth",
        "v1",
    )
    assert triple["kg"] == "webqsp"
    assert (triple["subject_type"], triple["object_type"]) == ("person", "city")
    assert (triple["traversal"]["source_type"], triple["traversal"]["target_type"]) == (
        "person",
        "city",
    )
    assert triple["relation_schema"]["components"] == {
        "domain": "people",
        "type": "person",
        "property": "place_of_birth",
    }
    assert triple["relation_schema"]["semantic_interpretation"] == (
        "not_inferred_from_identifier_tokens"
    )
    assert query["answer_variable"] == "v1"


def test_explicit_query_representation_orients_backward_fact() -> None:
    query = represent_query(
        PathSpec(
            "Whitney Houston",
            "musical artist",
            (Hop("music.album.artist", "backward", "musical artist", "album"),),
            "album",
            "webqsp",
        )
    )
    triple = query["triples"][0]
    assert (triple["subject"], triple["predicate"], triple["object"]) == (
        "v1",
        "music.album.artist",
        "v0",
    )
    assert triple["traversal"] == {
        "hop_index": 0,
        "from": "v0",
        "to": "v1",
        "direction": "backward",
        "source_type": "musical artist",
        "target_type": "album",
    }
    assert (triple["subject_type"], triple["object_type"]) == ("album", "musical artist")
    assert "raw fact (v_j, relation, v_i)" in query["logical_form"]


def test_explicit_query_exposes_birthplace_residence_occupation_branch() -> None:
    path = PathSpec(
        "Wendee Lee",
        "human",
        (
            Hop("place of birth", "forward", "human", "city"),
            Hop("residence", "backward", "city", "human"),
            Hop("occupation", "forward", "human", "occupation"),
        ),
        "occupation",
        "kqa_pro",
    )
    query = represent_query(path)
    assert [
        (triple["subject"], triple["predicate"], triple["object"])
        for triple in query["triples"]
    ] == [
        ("v0", "place of birth", "v1"),
        ("v2", "residence", "v1"),
        ("v2", "occupation", "v3"),
    ]
    assert query["schema_origin"] == "kqa_pro"
    assert query["answer_variable"] == "v3"
    assert query["variables"][-1] == {"id": "v3", "role": "answer", "type": "occupation"}
    assert "Do not generate the final natural-language question" in query["logical_form"]


def test_explicit_query_preserves_raw_freebase_cvt_triples() -> None:
    path = PathSpec(
        "Actor A",
        "person",
        (
            Hop("people.place_lived.person", "backward", "person", "entity"),
            Hop("people.place_lived.location", "forward", "entity", "location"),
        ),
        "location",
        "webqsp",
    )
    query = represent_query(path)
    assert [
        (triple["subject"], triple["predicate"], triple["object"])
        for triple in query["triples"]
    ] == [
        ("v1", "people.place_lived.person", "v0"),
        ("v1", "people.place_lived.location", "v2"),
    ]
    assert len(query["semantic_macros"]) == 1
    macro = query["semantic_macros"][0]
    assert macro["kind"] == "possible_freebase_cvt"
    assert macro["middle_variable"] == "v1"
    assert macro["triple_indices"] == [0, 1]
    assert macro["needs_semantic_description"] is True
    assert "identifier tokens do not establish" in macro["reason"]


def test_explicit_query_preserves_and_flags_metadata_relation() -> None:
    path = PathSpec(
        "Book A",
        "book",
        (Hop("common.topic.notable_types", "forward", "book", "type"),),
        "type",
        "webqsp",
    )
    query = represent_query(path)
    triple = query["triples"][0]
    assert triple["predicate"] == "common.topic.notable_types"
    assert triple["metadata"] == {
        "is_metadata": True,
        "matched_patterns": ["common.topic.notable_types"],
        "action": "preserve_and_flag",
    }
    assert "metadata=true" in query["logical_form"]


def test_explicit_query_flags_kwebbase_internal_relation() -> None:
    path = PathSpec(
        "Commander A",
        "military commander",
        (Hop("base.kwebbase.kwtopic.has_sentences", "forward", "military commander", "entity"),),
        "entity",
        "webqsp",
    )
    triple = represent_query(path)["triples"][0]
    assert triple["metadata"]["is_metadata"] is True
    assert triple["needs_semantic_description"] is True


def test_compact_query_preserves_branch_and_relation_meaning() -> None:
    path = {
        "kg": "kqa_pro",
        "anchor_type": "human",
        "answer_type": "occupation",
        "hops": [
            {"relation": "place of birth", "direction": "forward", "source_type": "human", "target_type": "city"},
            {"relation": "residence", "direction": "backward", "source_type": "city", "target_type": "human"},
            {"relation": "occupation", "direction": "forward", "source_type": "human", "target_type": "occupation"},
        ],
    }
    glossary = {
        f"kqa_pro::{relation}": {
            "status": "semantic",
            "description": relation,
            "fact_template": "{subject} " + relation + " {object}",
        }
        for relation in ("place of birth", "residence", "occupation")
    }
    query = compact_query(path, glossary)
    assert [(fact["subject"], fact["object"]) for fact in query["facts"]] == [
        ("v0", "v1"),
        ("v2", "v1"),
        ("v2", "v3"),
    ]
    assert query["return"] == {"variable": "v3", "type": "occupation"}
    assert query["unusable_relations"] == []


def test_glossary_validation_overrides_metadata_and_weak_semantics() -> None:
    evidence = {
        "webqsp::base.kwebbase.kwtopic.has_sentences": {
            "metadata_hint": {"is_metadata": True, "matched_patterns": ["base.kwebbase"]}
        },
        "webqsp::people.person.profession": {
            "metadata_hint": {"is_metadata": False, "matched_patterns": []}
        },
    }
    generated = {
        "webqsp::base.kwebbase.kwtopic.has_sentences": {
            "id": "webqsp::base.kwebbase.kwtopic.has_sentences",
            "status": "semantic",
            "fact_template": "{subject} has {object}",
            "confidence": 0.99,
            "reason": "guessed",
        },
        "webqsp::people.person.profession": {
            "id": "webqsp::people.person.profession",
            "status": "semantic",
            "fact_template": "person profession",
            "confidence": 0.4,
            "reason": "weak",
        },
    }
    glossary = validate_glossary(evidence, generated)
    assert glossary["webqsp::base.kwebbase.kwtopic.has_sentences"]["status"] == "metadata"
    assert glossary["webqsp::people.person.profession"]["status"] == "opaque"


def test_contrastive_validation_requires_intended_path_and_answer_type() -> None:
    path = {
        "kg": "kqa_pro",
        "anchor_type": "book",
        "answer_type": "country",
        "hops": [
            {"relation": "author", "direction": "forward", "source_type": "book", "target_type": "human"},
            {"relation": "citizenship", "direction": "forward", "source_type": "human", "target_type": "country"},
        ],
    }
    glossary = {
        "kqa_pro::author": {"status": "semantic", "description": "book author", "fact_template": "{subject} was authored by {object}"},
        "kqa_pro::citizenship": {"status": "semantic", "description": "person citizenship", "fact_template": "{subject} is a citizen of {object}"},
    }
    accepted = {
        "status": "valid",
        "question": "What country is the author of [ENTITY] a citizen of?",
        "selected_option": "B",
        "intended_option": "B",
        "answer_type_matches": True,
        "all_facts_expressed": True,
        "uses_only_supported_facts": True,
        "is_natural_language_question": True,
        "confidence": 0.94,
    }
    assert validation_rejection(path, accepted, glossary) is None

    wrong_path = dict(accepted, selected_option="A")
    assert validation_rejection(path, wrong_path, glossary) == "wrong_contrastive_path"

    wrong_type = dict(accepted, answer_type_matches=False)
    assert validation_rejection(path, wrong_type, glossary) == "answer_type_mismatch"

    leaked_variable = dict(accepted, question="What country is v1, the author of [ENTITY], a citizen of?")
    assert validation_rejection(path, leaked_variable, glossary) == "internal_notation_in_question"

    vague = dict(accepted, question="What country is associated with the author of [ENTITY]?")
    assert validation_rejection(path, vague, glossary) == "vague_question_wording"


def test_combined_contrastive_result_accepts_only_intended_high_confidence() -> None:
    generated = {
        "good": {"status": "generated", "question": "Who wrote [ENTITY]?"},
        "wrong": {"status": "generated", "question": "Who wrote [ENTITY]?"},
    }
    judgments = {
        "good": {"selected_option": "C", "answer_type_matches": True, "all_facts_expressed": True, "uses_only_supported_facts": True, "is_natural_language_question": True, "confidence": 0.91, "reason": "exact"},
        "wrong": {"selected_option": "A", "answer_type_matches": True, "all_facts_expressed": True, "uses_only_supported_facts": True, "is_natural_language_question": True, "confidence": 0.99, "reason": "different path"},
    }
    combined = combine_contrastive_results(generated, judgments, {"good": "C", "wrong": "B"})
    assert combined["good"]["status"] == "valid"
    assert combined["wrong"]["status"] == "reject"


def test_strict_validation_rejects_metadata_even_when_model_accepts() -> None:
    path = {
        "kg": "webqsp",
        "anchor_type": "person",
        "answer_type": "entity",
        "hops": [{
            "relation": "base.kwebbase.kwtopic.has_sentences",
            "direction": "forward",
            "source_type": "person",
            "target_type": "entity",
        }],
    }
    glossary = {
        "webqsp::base.kwebbase.kwtopic.has_sentences": {
            "status": "metadata",
            "description": "stored sentence",
            "fact_template": "{subject} has sentence {object}",
        }
    }
    prediction = {
        "status": "valid",
        "question": "What description is stored for [ENTITY]?",
        "selected_option": "A",
        "intended_option": "A",
        "answer_type_matches": True,
        "all_facts_expressed": True,
        "uses_only_supported_facts": True,
        "is_natural_language_question": True,
        "confidence": 0.99,
    }
    assert validation_rejection(path, prediction, glossary) == "unusable_relation"
from inverse_verifier.selector import (
    answer_metrics,
    candidate_log_entry,
    candidate_score,
    enumerate_path_families,
    answer_set_key,
    evaluation_subset_coverage,
    first_gold_rank,
    gold_equivalent_answer_sets,
    has_answerable_endpoint,
    select_candidate,
)
from inverse_verifier.retrieval import (
    LocalQuestionGraph,
    decode_edge,
    encode_edge,
    dominant_type,
    gold_path_available,
    schema_type_hint,
    materialize_path,
)


def test_extracts_chain_without_answer_or_intermediate_names(tmp_path: Path) -> None:
    source = tmp_path / "kqa.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question": "Which country was the author of Book A born in?",
                    "answer": "Country Z",
                    "program": [
                        {"function": "Find", "inputs": ["Book A"]},
                        {"function": "Relate", "inputs": ["author", "forward"]},
                        {"function": "FilterConcept", "inputs": ["human"]},
                        {"function": "Relate", "inputs": ["place of birth", "forward"]},
                        {"function": "FilterConcept", "inputs": ["country"]},
                        {"function": "What", "inputs": []},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    rows = extract_kqa_examples(source)
    assert len(rows) == 1
    rendered = render_path(rows[0][2])
    assert "Book A" in rendered
    assert "Country Z" not in rendered
    assert "NODE_1" in rendered and "ANSWER" in rendered
    assert 'relation="place of birth"; fact roles=NODE_1:subject,ANSWER:object' in rendered
    assert not render_path(rows[0][2], include_instruction=False).endswith("Question:")


def test_sparql_parser_recovers_forward_and_backward_direction() -> None:
    forward = "ns:m.topic ns:people.person.place_of_birth ?x ."
    backward = "?x ns:people.person.place_of_birth ns:m.topic ."
    assert parse_sparql_path(forward, "m.topic", ["people.person.place_of_birth"]) == [
        ("people.person.place_of_birth", "forward")
    ]
    assert parse_sparql_path(backward, "m.topic", ["people.person.place_of_birth"]) == [
        ("people.person.place_of_birth", "backward")
    ]


def test_hard_negatives_are_structurally_distinct() -> None:
    positive = PathSpec(
        "Book A",
        "written work",
        (
            Hop("author", "forward", "written work", "human"),
            Hop("place of birth", "forward", "human", "country"),
        ),
        "country",
        "toy",
    )
    alternate = PathSpec(
        "Book B",
        "written work",
        (Hop("director", "forward", "written work", "human"),),
        "human",
        "toy",
    )
    inventory = relation_inventory([("a", "q", positive), ("b", "q", alternate)])
    negatives = make_negative_paths(positive, inventory, "fixed")
    categories = {row["negative_type"] for row in negatives}
    assert {"wrong_direction", "wrong_relation", "wrong_order", "missing_hop", "wrong_answer_type"} <= categories
    assert all(row["hops"] != [hop.__dict__ for hop in positive.hops] for row in negatives)


def test_normalized_nll_and_ranking_metrics() -> None:
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]])
    labels = torch.tensor([[0, 1], [0, 1]])
    nll = normalized_sequence_nll(logits, labels)
    assert nll[0] < nll[1]
    rows = [
        {
            "candidates": [
                {"is_positive": True, "negative_type": "positive", "score": 2.0},
                {"is_positive": False, "negative_type": "wrong_order", "score": 1.0},
            ]
        }
    ]
    metrics = ranking_metrics(rows, "score")
    assert metrics["recall_at_1"] == 1.0
    assert metrics["pairwise_by_negative_type"]["wrong_order"] == 1.0
    assert token_f1("who wrote the book", "Who wrote this book?") > 0.5
    assert rouge_l("who wrote the book", "Who wrote this book?") > 0.5


def test_ranking_accepts_any_annotated_positive() -> None:
    rows = [
        {
            "candidates": [
                {"is_positive": True, "negative_type": "positive", "score": 0.1},
                {"is_positive": True, "negative_type": "alternate_positive", "score": 0.9},
                {"is_positive": False, "negative_type": "wrong_relation", "score": 0.8},
            ]
        }
    ]
    metrics = ranking_metrics(rows, "score")
    assert metrics["recall_at_1"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0


def test_ranking_does_not_award_stable_sort_ties() -> None:
    rows = [
        {
            "candidates": [
                {"is_positive": True, "negative_type": "positive", "score": 1.0},
                {"is_positive": False, "negative_type": "wrong_direction", "score": 1.0},
            ]
        }
    ]
    metrics = ranking_metrics(rows, "score")
    assert metrics["recall_at_1"] == 0.5
    assert metrics["strict_recall_at_1"] == 0.0
    assert metrics["top_score_tie_rate"] == 1.0


def test_direct_prompt_sees_question_and_same_oriented_path() -> None:
    path = PathSpec(
        "Book A",
        "written work",
        (Hop("author", "forward", "written work", "person"),),
        "person",
        "toy",
    )
    prompt = direct_prompt("Who wrote Book A?", json.loads(json.dumps(path, default=lambda x: x.__dict__)))
    assert "Who wrote Book A?" in prompt
    assert 'relation="author"; fact roles=START:subject,ANSWER:object' in prompt


def test_joint_prompt_masks_linked_entity_but_keeps_relation_intent() -> None:
    path = PathSpec(
        "Harry Potter",
        "written work",
        (
            Hop("author", "forward", "written work", "person"),
            Hop("place of birth", "forward", "person", "country"),
        ),
        "country",
        "toy",
    )
    path_dict = json.loads(json.dumps(path, default=lambda value: value.__dict__))
    prompt = rank_prompt("Which country was the author of Harry Potter born in?", path_dict)
    assert "Harry Potter" not in prompt
    assert ENTITY_PLACEHOLDER in prompt
    assert "author" in prompt and "place of birth" in prompt


def test_delexicalization_handles_exact_and_conservative_surface_forms() -> None:
    assert delexicalize_question("Who wrote Harry Potter?", "Harry Potter") == "Who wrote [ENTITY]?"
    assert delexicalize_question("What do Jamaican people speak?", "Jamaica") == (
        "What do [ENTITY] people speak?"
    )


def test_generated_similarity_masks_shared_entity_names() -> None:
    class RecordingEncoder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def encode(self, texts, **_kwargs):
            self.calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    encoder = RecordingEncoder()
    path = {"anchor": "Harry Potter"}
    scores = generated_question_similarity_scores(
        encoder,
        ["Who wrote Harry Potter?"],
        ["Who is the author of Harry Potter?"],
        [path],
        batch_size=1,
    )
    assert scores == [2.0]
    assert encoder.calls == [
        ["Who wrote [ENTITY]?"],
        ["Who is the author of [ENTITY]?"],
    ]


def test_similarity_distribution_reports_threshold_rates() -> None:
    metrics = similarity_distribution([0.5, 0.7, 0.8, 0.95])
    assert metrics["examples"] == 4
    assert metrics["at_least_0_8"] == 0.5
    assert metrics["at_least_0_9"] == 0.25


def test_selector_enumerates_relation_families_with_all_answer_bindings() -> None:
    graph_row = {
        "q_entity": ["Book A"],
        "graph": [
            ["Book A", "book.author", "Writer A"],
            ["Book A", "book.author", "Writer B"],
            ["Writer A", "person.birthplace", "London"],
            ["Writer B", "person.birthplace", "Paris"],
            ["Book A", "common.topic.notable_types", "Book"],
        ],
    }
    families = enumerate_path_families(graph_row)
    family = next(
        row
        for row in families
        if row["relation_sequence"]
        == ["book.author::forward", "person.birthplace::forward"]
    )
    assert family["answers"] == ["London", "Paris"]
    assert first_gold_rank([family], [("book.author::forward", "person.birthplace::forward")]) == 1


def test_selector_answer_metrics_support_multi_answer_sets() -> None:
    metrics = answer_metrics(["London", "Paris"], ["Paris", "Rome"])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["exact_match"] == 0.0
    assert metrics["has_correct_answer"] == 1.0


def test_type_aware_generator_uses_separate_generation_and_type_tasks() -> None:
    row = {
        "question": "Which country was the author of Book A born in?",
        "positive_path": {
            "anchor": "Book A",
            "anchor_type": "written work",
            "hops": [
                {
                    "relation": "author",
                    "direction": "forward",
                    "source_type": "written work",
                    "target_type": "person",
                },
                {
                    "relation": "place of birth",
                    "direction": "forward",
                    "source_type": "person",
                    "target_type": "country",
                },
            ],
            "answer_type": "country",
            "kg": "toy",
        },
        "alternate_positive_paths": [],
        "negative_paths": [
            {
                "answer_type": "person",
                "negative_type": "wrong_answer_type",
            }
        ],
    }
    dataset = TypeAwareGeneratorDataset([row])
    generation = dataset[0]
    assert generation["task"] == "question_generation"
    assert generation["target"] == "Which country was the author of [ENTITY] born in?"
    assert "Answer type" not in generation["target"]
    type_example = dataset[1]
    assert type_example["task"] == "answer_type_compatibility"
    assert type_example["target"] in {"yes", "no"}
    assert "Candidate answer type:" in type_example["source"]
    assert informative_answer_type("country")
    assert not informative_answer_type("answer entity")
    assert "yes or no" in type_compatibility_prompt("Where?", "city")


def test_local_retrieval_graph_preserves_relation_direction() -> None:
    graph = LocalQuestionGraph(
        [
            ["Book A", "book.author", "Writer A"],
            ["Writer A", "person.birthplace", "London"],
            ["Book A", "common.topic.notable_types", "Book"],
        ]
    )
    assert encode_edge("book.author", "forward") in graph.get_neighbor_relations("Book A")
    assert encode_edge("book.author", "backward") in graph.get_neighbor_relations("Writer A")
    assert decode_edge("book.author::backward") == ("book.author", "backward")


def test_materialized_candidate_path_keeps_all_answer_bindings_and_subgraph() -> None:
    graph = LocalQuestionGraph(
        [
            ["Book A", "book.author", "Writer A"],
            ["Book A", "book.author", "Writer B"],
            ["Writer A", "person.birthplace", "London"],
            ["Writer B", "person.birthplace", "Paris"],
        ]
    )
    path, answers, triples = materialize_path(
        graph,
        "Book A",
        ("book.author::forward", "person.birthplace::forward"),
    )
    assert answers == ["London", "Paris"]
    assert len(path["hops"]) == 2
    assert ["Writer A", "person.birthplace", "London"] in triples
    assert gold_path_available(
        graph,
        ["Book A"],
        [("book.author::forward", "person.birthplace::forward")],
    )
    assert not gold_path_available(graph, ["Book A"], [("book.publisher::forward",)])


def test_split_assignment_returns_disjoint_heldout_relations() -> None:
    rows = []
    for index in range(100):
        relation = f"relation {index % 10}"
        path = PathSpec(
            f"Anchor {index}",
            "entity",
            (Hop(relation, "forward", "entity", "thing"),),
            "thing",
            "toy",
        )
        rows.append((str(index), f"question {index}", path))
    splits, heldout, _ = assign_kqa_splits(rows)
    train_relations = {
        hop.relation for _, _, path in splits["train"] + splits["dev"] for hop in path.hops
    }
    assert heldout
    assert not (train_relations & heldout)


def test_canonical_question_preserves_every_relation_and_direction() -> None:
    path = PathSpec(
        "Book A",
        "written work",
        (
            Hop("author", "forward", "written work", "person"),
            Hop("place of birth", "forward", "person", "city"),
            Hop("country", "forward", "city", "country"),
        ),
        "country",
        "toy",
    )
    question = canonical_question(path)
    assert question_covers_path(question, path)
    assert question.count("forward") == 3
    assert all(relation in question for relation in ("author", "place of birth", "country"))


def test_executable_synthetic_example_assigns_distinct_questions_to_negatives() -> None:
    graph = ExecutableGraph(
        adjacency={
            "book": [Edge("writer", "author", "forward")],
            "writer": [
                Edge("city", "place of birth", "forward"),
                Edge("book", "author", "backward"),
            ],
            "city": [Edge("country", "country", "forward")],
            "country": [],
        },
        labels={
            "book": "Book A",
            "writer": "Writer A",
            "city": "London",
            "country": "United Kingdom",
        },
        types={
            "book": "written work",
            "writer": "person",
            "city": "city",
            "country": "country",
        },
        kg="toy",
    )
    path = ConcretePath(
        PathSpec(
            "Book A",
            "written work",
            (Hop("author", "forward", "written work", "person"),),
            "person",
            "toy",
        ),
        ("book", "writer"),
    )
    example = example_from_path(graph, path, 0, __import__("random").Random(3))
    assert example is not None
    added = next(row for row in example["negative_paths"] if row["negative_type"] == "added_hop")
    assert added["question"] != example["question"]
    assert "place of birth" in added["question"]
    assert question_covers_path(added["question"], added)


def test_faithful_dataset_generates_negative_intent_and_contrasts_with_gold() -> None:
    row = {
        "question": "What person do you reach from [ENTITY] if you follow author forward to a person?",
        "positive_path": {
            "anchor": "Book A",
            "anchor_type": "book",
            "hops": [{"relation": "author", "direction": "forward", "source_type": "book", "target_type": "person"}],
            "answer_type": "person",
            "kg": "toy",
        },
        "negative_paths": [{
            "anchor": "Book A",
            "anchor_type": "book",
            "hops": [
                {"relation": "author", "direction": "forward", "source_type": "book", "target_type": "person"},
                {"relation": "place of birth", "direction": "forward", "source_type": "person", "target_type": "city"},
            ],
            "answer_type": "city",
            "kg": "toy",
            "negative_type": "added_hop",
            "question": "Which city is the birthplace of the author of [ENTITY]?",
        }],
    }
    dataset = FaithfulInverseDataset([row])
    dataset.set_epoch(1)
    item = dataset[0]
    assert item["generation_target"] == row["negative_paths"][0]["question"]
    assert item["contrast_target"] == row["question"]
    assert "place of birth" in item["negative_source"]


def test_naturalization_uses_multiple_hosts_and_preserves_source_order(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    rows = []
    for index in range(5):
        rows.append(
            {
                "id": f"row-{index}",
                "question": f"canonical {index}",
                "positive_path": {
                    "hops": [
                        {
                            "relation": "author",
                            "direction": "forward",
                            "source_type": "book",
                            "target_type": "person",
                        }
                    ],
                    "answer_type": "person",
                },
                "negative_paths": [],
            }
        )
    for filename in ("train_faithful.jsonl", "dev_faithful.jsonl"):
        selected = rows if filename.startswith("train") else []
        (source / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8"
        )

    called_hosts = []

    def fake_naturalize(batch, _model, host):
        called_hosts.append(host)
        return {
            f"{index}:positive": f"natural {row['id']}"
            for index, row in enumerate(batch)
        }

    monkeypatch.setattr("inverse_verifier.synthetic._ollama_naturalize", fake_naturalize)
    manifest = naturalize_corpus(
        source,
        output,
        host=["http://gpu0:11434", "http://gpu1:11434"],
        batch_size=1,
    )

    written = [
        json.loads(line)
        for line in (output / "train_faithful.jsonl").open(encoding="utf-8")
    ]
    assert [row["id"] for row in written] == [f"row-{index}" for index in range(5)]
    assert [row["question"] for row in written] == [f"natural row-{index}" for index in range(5)]
    assert set(called_hosts) == {"http://gpu0:11434", "http://gpu1:11434"}
    assert manifest["workers"] == 2

    resumed = naturalize_corpus(
        source,
        output,
        host=["http://gpu0:11434", "http://gpu1:11434"],
        batch_size=1,
    )
    assert resumed["rows"] == 5
    assert len(called_hosts) == 5


def test_naturalization_splits_malformed_batches_instead_of_discarding_rows(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    rows = [
        {
            "id": f"row-{index}",
            "question": f"canonical {index}",
            "positive_path": {
                "hops": [{"relation": "author", "direction": "forward"}],
                "answer_type": "person",
            },
            "negative_paths": [],
        }
        for index in range(4)
    ]
    (source / "train_faithful.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (source / "dev_faithful.jsonl").write_text("", encoding="utf-8")

    def malformed_when_batched(batch, _model, _host):
        if len(batch) > 1:
            raise json.JSONDecodeError("truncated", "{}", 1)
        return {"0:positive": f"natural {batch[0]['id']}"}

    monkeypatch.setattr(
        "inverse_verifier.synthetic._ollama_naturalize", malformed_when_batched
    )
    manifest = naturalize_corpus(source, output, batch_size=4)
    written = [
        json.loads(line)
        for line in (output / "train_faithful.jsonl").open(encoding="utf-8")
    ]

    assert [row["question"] for row in written] == [f"natural row-{index}" for index in range(4)]
    assert manifest["naturalized_questions"] == 4
    assert manifest["canonical_fallbacks"] == 0


def test_naturalization_does_not_turn_http_failure_into_training_data(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    row = {
        "question": "canonical",
        "positive_path": {
            "hops": [{"relation": "author", "direction": "forward"}],
            "answer_type": "person",
        },
        "negative_paths": [],
    }
    (source / "train_faithful.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (source / "dev_faithful.jsonl").write_text("", encoding="utf-8")

    def unavailable(*_args):
        raise urllib.error.HTTPError("http://gpu0/api/chat", 404, "missing", {}, None)

    monkeypatch.setattr("inverse_verifier.synthetic._ollama_naturalize", unavailable)
    try:
        naturalize_corpus(source, output)
    except RuntimeError as exc:
        assert "Ollama request failed" in str(exc)
    else:
        raise AssertionError("HTTP failure should stop naturalization")

    assert (output / "train_faithful.jsonl").read_text(encoding="utf-8") == ""


def test_generalization_slices_are_relative_to_actual_training_data() -> None:
    from inverse_verifier.generalization import derive_generalization_slices

    def row(example_id: str, kg: str, relations: list[tuple[str, str]]) -> dict:
        return {
            "example_id": example_id,
            "positive_path": {
                "kg": kg,
                "hops": [
                    {"relation": relation, "direction": direction}
                    for relation, direction in relations
                ],
            },
        }

    train = [
        row("train-a", "kg", [("a", "forward")]),
        row("train-b", "kg", [("b", "forward")]),
        row("train-ab", "kg", [("a", "forward"), ("b", "forward")]),
    ]
    evaluation = {
        "heldout": [
            row("seen", "kg", [("a", "forward")]),
            row("new-relation", "kg", [("c", "forward")]),
            row("new-composition", "kg", [("b", "forward"), ("a", "forward")]),
            row("same-name-other-kg", "other", [("a", "forward")]),
        ]
    }

    slices, coverage = derive_generalization_slices(train, evaluation)

    assert {item["example_id"] for item in slices["strict_unseen_relation"]} == {
        "new-relation",
        "same-name-other-kg",
    }
    assert [item["example_id"] for item in slices["strict_unseen_composition"]] == [
        "new-composition"
    ]
    assert coverage["slice_examples"]["strict_unseen_relation"] == 2


def test_render_path_makes_subject_object_direction_explicit() -> None:
    base = {
        "anchor": "Work",
        "anchor_type": "book",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
        "answer_type": "person",
        "kg": "test",
    }
    forward = render_path(base, mask_anchor=True)
    backward_path = json.loads(json.dumps(base))
    backward_path["hops"][0]["direction"] = "backward"
    backward = render_path(backward_path, mask_anchor=True)

    assert "fact roles=START:subject,ANSWER:object" in forward
    assert "fact roles=ANSWER:subject,START:object" in backward


def test_grounded_render_path_binds_fact_roles_in_both_directions() -> None:
    glossary = {
        "test::author": {
            "status": "semantic",
            "description": "Connects a written work to its author.",
            "subject_role": "written work",
            "object_role": "author",
            "fact_template": "{subject} was written by {object}.",
        }
    }
    forward_path = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "kg": "test",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    backward_path = json.loads(json.dumps(forward_path))
    backward_path["anchor_type"] = "person"
    backward_path["answer_type"] = "book"
    backward_path["hops"][0].update(
        direction="backward", source_type="person", target_type="book"
    )

    forward = render_path(
        forward_path, mask_anchor=True, relation_glossary=glossary
    )
    backward = render_path(
        backward_path, mask_anchor=True, relation_glossary=glossary
    )

    assert "START (type: book; role: written work) was written by ANSWER" in forward
    assert "ANSWER (type: book; role: written work) was written by START" in backward
    assert "Requested answer: ANSWER (type: person)" in forward
    assert "Requested answer: ANSWER (type: book)" in backward


def test_grounded_render_path_fallback_preserves_full_relation_identity() -> None:
    path = {
        "anchor": "Work",
        "anchor_type": "book",
        "answer_type": "topic",
        "kg": "webqsp",
        "hops": [
            {
                "relation": "book.book_subject.works",
                "direction": "forward",
                "source_type": "book",
                "target_type": "topic",
            }
        ],
    }

    rendered = render_path(path, relation_glossary={})

    assert "book / book subject / works" in rendered
    assert "Relation ID: book.book_subject.works" in rendered


def test_causal_training_flattens_each_path_with_its_own_question() -> None:
    positive = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "kg": "test",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    negative = {
        **positive,
        "hops": [{**positive["hops"][0], "direction": "backward"}],
        "question": "Which book did [ENTITY] write?",
        "negative_type": "executable_opposite_direction",
    }

    pairs = flatten_path_question_pairs(
        [
            {
                "question": "Who wrote [ENTITY]?",
                "positive_path": positive,
                "negative_paths": [negative],
            }
        ]
    )

    assert [pair["question"] for pair in pairs] == [
        "Who wrote [ENTITY]?",
        "Which book did [ENTITY] write?",
    ]


def test_causal_dataset_masks_prompt_and_uses_grounded_semantics() -> None:
    class RecordingTokenizer:
        eos_token = "<eos>"

        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "formatted prompt"

        def __call__(self, text, **kwargs):
            if text == "formatted prompt":
                return {"input_ids": [10, 11, 12]}
            assert text == "Who wrote [ENTITY]?<eos>"
            return {"input_ids": [20, 21]}

    tokenizer = RecordingTokenizer()
    path = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "kg": "test",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    glossary = {
        "test::author": {
            "status": "semantic",
            "description": "Connects a written work to its author.",
            "subject_role": "written work",
            "object_role": "author",
            "fact_template": "{subject} was written by {object}.",
        }
    }

    item = CausalInverseDataset(
        [{"path": path, "question": "Who wrote [ENTITY]?"}],
        tokenizer,
        glossary,
    )[0]

    assert item["input_ids"] == [10, 11, 12, 20, 21]
    assert item["labels"] == [-100, -100, -100, 20, 21]
    assert "START (type: book; role: written work) was written by ANSWER" in (
        tokenizer.messages[1]["content"]
    )


def test_direction_repair_excludes_symmetric_relations() -> None:
    from inverse_verifier.training_data import direction_counterfactuals

    path = {
        "kg": "kqa_pro",
        "anchor": "A",
        "anchor_type": "person",
        "answer_type": "person",
        "hops": [
            {
                "relation": "spouse",
                "direction": "forward",
                "source_type": "person",
                "target_type": "person",
            },
            {
                "relation": "place of birth",
                "direction": "forward",
                "source_type": "person",
                "target_type": "city",
            },
        ],
    }
    counterfactuals = direction_counterfactuals(path, {})

    assert len(counterfactuals) == 1
    assert counterfactuals[0]["flipped_hop_index"] == 1
    assert counterfactuals[0]["hops"][0]["direction"] == "forward"
    assert counterfactuals[0]["hops"][1]["direction"] == "backward"

    relative = json.loads(json.dumps(path))
    relative["hops"] = [relative["hops"][0] | {"relation": "relative"}]
    assert direction_counterfactuals(relative, {}) == []


def test_repetition_filter_rejects_only_pathological_repetition() -> None:
    from inverse_verifier.training_data import repetitive_question_reason

    legitimate = (
        "Who is the parent of the person who is a sibling of a sibling of [ENTITY]?"
    )
    malformed = (
        "What is the type of dish that is a dish that is a dish that is a dish "
        "that is a dish that is a dish?"
    )

    assert repetitive_question_reason(legitimate) is None
    assert repetitive_question_reason(malformed) is not None


def test_faithful_dataset_uses_direction_contrasts_only_for_ranking() -> None:
    row = {
        "question": "Who wrote [ENTITY]?",
        "positive_path": {
            "anchor": "Work",
            "anchor_type": "book",
            "answer_type": "person",
            "kg": "test",
            "hops": [
                {
                    "relation": "author",
                    "direction": "forward",
                    "source_type": "book",
                    "target_type": "person",
                }
            ],
        },
        "negative_paths": [
            {
                "anchor": "Work",
                "anchor_type": "book",
                "answer_type": "publisher",
                "kg": "test",
                "negative_type": "sibling_relation",
                "question": "Who published [ENTITY]?",
                "hops": [
                    {
                        "relation": "publisher",
                        "direction": "forward",
                        "source_type": "book",
                        "target_type": "publisher",
                    }
                ],
            }
        ],
        "contrast_only_negative_paths": [
            {
                "anchor": "Work",
                "anchor_type": "book",
                "answer_type": "person",
                "kg": "test",
                "negative_type": "wrong_direction",
                "contrast_only": True,
                "hops": [
                    {
                        "relation": "author",
                        "direction": "backward",
                        "source_type": "book",
                        "target_type": "person",
                    }
                ],
            }
        ],
    }
    dataset = FaithfulInverseDataset([row])
    dataset.set_epoch(0)
    even = dataset[0]
    dataset.set_epoch(1)
    odd = dataset[0]

    assert even["negative_type"] == "wrong_direction"
    assert "ANSWER:subject,START:object" in even["negative_source"]
    assert even["generation_target"] == row["question"]
    assert odd["negative_type"] == "sibling_relation"
    assert odd["generation_target"] == "Who published [ENTITY]?"


def test_bidirectional_pair_trains_both_executable_directions() -> None:
    forward = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "kg": "test",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    backward = {
        "anchor": "Person",
        "anchor_type": "person",
        "answer_type": "book",
        "kg": "test",
        "negative_type": "executable_opposite_direction",
        "question": "Which book did [ENTITY] write?",
        "hops": [
            {
                "relation": "author",
                "direction": "backward",
                "source_type": "person",
                "target_type": "book",
            }
        ],
    }
    dataset = FaithfulInverseDataset(
        [
            {
                "question": "Who wrote [ENTITY]?",
                "positive_path": forward,
                "negative_paths": [backward],
                "bidirectional_pair": True,
            }
        ]
    )

    dataset.set_epoch(0)
    forward_item = dataset[0]
    dataset.set_epoch(1)
    backward_item = dataset[0]

    assert forward_item["generation_target"] == "Who wrote [ENTITY]?"
    assert "START:subject,ANSWER:object" in forward_item["positive_source"]
    assert "ANSWER:subject,START:object" in forward_item["negative_source"]
    assert backward_item["generation_target"] == "Which book did [ENTITY] write?"
    assert "ANSWER:subject,START:object" in backward_item["positive_source"]
    assert "START:subject,ANSWER:object" in backward_item["negative_source"]


def test_direction_pair_validation_rejects_vague_or_identical_questions() -> None:
    from inverse_verifier.training_data import _natural_pair_rejection

    row = {
        "question": "Who wrote [ENTITY]?",
        "canonical_question": "canonical forward",
        "negative_paths": [
            {
                "question": "Which work was written by [ENTITY]?",
                "canonical_question": "canonical backward",
            }
        ],
    }
    assert _natural_pair_rejection(row) is None

    vague = json.loads(json.dumps(row))
    vague["question"] = "Who is associated with [ENTITY]?"
    assert _natural_pair_rejection(vague) == "weakens_specific_relation"

    genuinely_associative = json.loads(json.dumps(vague))
    genuinely_associative["positive_path"] = {
        "explicit_query": "F0: v0 is associated with the sport v1."
    }
    assert _natural_pair_rejection(genuinely_associative) is None

    geographic = json.loads(json.dumps(row))
    geographic["question"] = "Which geographic region contains [ENTITY]?"
    assert _natural_pair_rejection(geographic) is None

    identical = json.loads(json.dumps(row))
    identical["negative_paths"][0]["question"] = identical["question"]
    assert _natural_pair_rejection(identical) == "directions_have_identical_question"


def test_faithfulness_evaluates_both_executable_directions() -> None:
    row = {
        "example_id": "pair-1",
        "question": "Who wrote [ENTITY]?",
        "positive_path": {"anchor": "Book", "hops": [{"direction": "forward"}]},
        "negative_paths": [
            {
                "anchor": "Person",
                "hops": [{"direction": "backward"}],
                "question": "Which book did [ENTITY] write?",
                "negative_type": "executable_opposite_direction",
            }
        ],
        "bidirectional_pair": True,
    }

    forward, backward = _expand_bidirectional_evaluation_rows([row])

    assert forward is row
    assert backward["example_id"] == "pair-1:backward"
    assert backward["question"] == "Which book did [ENTITY] write?"
    assert backward["positive_path"]["hops"][0]["direction"] == "backward"
    assert backward["negative_paths"][0]["hops"][0]["direction"] == "forward"
    assert backward["negative_paths"][0]["negative_type"] == "executable_opposite_direction"


def test_comparator_candidate_sets_support_multiple_positive_paths() -> None:
    positive = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    alternate = {
        **positive,
        "hops": [
            {
                "relation": "written by",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
        "answer_entity": "Writer",
    }
    negative = {
        **positive,
        "hops": [
            {
                "relation": "publisher",
                "direction": "forward",
                "source_type": "book",
                "target_type": "organization",
            }
        ],
        "answer_entity": "Publisher",
        "negative_type": "sibling_relation",
    }
    candidates = candidate_specs(
        {
            "positive_path": positive,
            "positive_answer_entity": "Writer",
            "alternate_positive_paths": [alternate],
            "negative_paths": [negative],
        }
    )

    assert [candidate["is_positive"] for candidate in candidates] == [
        True,
        True,
        False,
    ]
    assert candidates[-1]["negative_type"] == "sibling_relation"
    assert candidates[-1]["answer_entity"] == "Publisher"


def test_comparator_input_ablations_include_only_requested_channels() -> None:
    question = "Who wrote the book?"
    generated = "Which person authored the book?"
    path = "Book --author--> Person"

    question_generated = comparator_input_text(
        question, generated, path, "question_generated"
    )
    assert generated in question_generated
    assert path not in question_generated

    question_path = comparator_input_text(
        question, generated, path, "question_path"
    )
    assert generated not in question_path
    assert path in question_path

    all_channels = comparator_input_text(
        question, generated, path, "question_generated_path"
    )
    assert question in all_channels
    assert generated in all_channels
    assert path in all_channels


def test_comparator_path_serialization_includes_endpoint_and_direction() -> None:
    text = comparator_path_text(
        {
            "anchor": "Harry Potter",
            "anchor_type": "book",
            "answer_type": "person",
            "hops": [
                {
                    "relation": "author",
                    "direction": "forward",
                    "source_type": "book",
                    "target_type": "person",
                }
            ],
        },
        "J. K. Rowling",
    )

    assert 'relation="author"' in text
    assert "START:subject" in text
    assert "Candidate endpoint: J. K. Rowling" in text


def test_listwise_loss_assigns_probability_to_any_valid_path() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, 1.0], requires_grad=True)
    labels = torch.tensor([True, True, False, True, False])
    loss = listwise_multi_positive_loss(logits, labels, [(0, 3), (3, 5)])
    expected = (
        torch.logsumexp(logits[:3], dim=0)
        - torch.logsumexp(logits[:2], dim=0)
        + torch.logsumexp(logits[3:], dim=0)
        - logits[3]
    ) / 2

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[2] > 0
    assert logits.grad[4] > 0


def test_cross_encoder_pipeline_requires_a_comparator_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires comparator_model"):
        run_verifier_pipeline(
            tmp_path / "questions.json",
            tmp_path / "graphs.jsonl",
            "generator",
            "retriever",
            tmp_path / "output",
            limit=1,
            comparison_mode="cross_encoder",
        )


def test_comparator_materializes_existing_heldout_predictions(
    tmp_path: Path,
) -> None:
    generator = tmp_path / "generator"
    generator.mkdir()
    (generator / "relation_glossary.json").write_text("{}", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    path = {
        "anchor": "Book",
        "anchor_type": "book",
        "answer_type": "person",
        "kg": "test",
        "hops": [
            {
                "relation": "author",
                "direction": "forward",
                "source_type": "book",
                "target_type": "person",
            }
        ],
    }
    predictions.write_text(
        json.dumps(
            {
                "slice": "unseen_composition",
                "example_id": "example-1",
                "question": "Who wrote Book?",
                "kg": "test",
                "candidates": [
                    {
                        "category": "positive",
                        "is_positive": True,
                        "generated_question": "Who authored Book?",
                        "path": path,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "materialized"
    manifest = materialize_comparator_data(
        predictions,
        str(generator),
        output,
    )
    row = json.loads((output / "unseen_composition.jsonl").read_text())

    assert manifest["source_kind"] == "existing_generator_predictions"
    assert row["original_question"] == "Who wrote Book?"
    assert row["candidates"][0]["negative_type"] == "positive"
    assert "author" in row["candidates"][0]["path_text"]


def test_semantic_judge_sees_questions_but_not_path_labels() -> None:
    row = {
        "example_id": "example-1",
        "original_question": "Who is the father of Ada's mother?",
        "candidates": [
            {
                "generated_question": "Who is Ada's maternal grandfather?",
                "is_positive": False,
                "negative_type": "wrong_direction",
                "path_text": "SECRET PATH",
            }
        ],
    }
    record = semantic_judge_record(row, "judge")
    prompt = record["body"]["messages"][1]["content"]

    assert "Who is Ada's maternal grandfather?" in prompt
    assert "SECRET PATH" not in prompt
    assert "wrong_direction" not in prompt
    assert '"is_positive"' not in prompt
    assert "signed to label X" in JUDGE_SYSTEM_PROMPT
    assert 'who resides or has resided in X' in JUDGE_SYSTEM_PROMPT


def test_semantic_labels_allow_equivalent_wrong_paths_and_multiple_positives() -> None:
    rows = [
        {
            "example_id": "example-1",
            "original_question": "Who is Ada's maternal grandfather?",
            "candidates": [
                {"generated_question": "Who is the father of Ada's mother?", "is_positive": True},
                {"generated_question": "Who is Ada's mother's father?", "is_positive": False},
                {"generated_question": "Who is Ada's father?", "is_positive": False},
            ],
        }
    ]
    judgments = {
        "example-1": {
            "items": [
                {"id": "candidate-0", "equivalent": True, "issue": "equivalent"},
                {"id": "candidate-1", "equivalent": True, "issue": "equivalent"},
                {"id": "candidate-2", "equivalent": False, "issue": "missing_constraint"},
            ]
        }
    }
    scored, unscorable, summary = apply_semantic_labels(rows, judgments)

    assert not unscorable
    assert [candidate["is_positive"] for candidate in scored[0]["candidates"]] == [
        True,
        True,
        False,
    ]
    assert scored[0]["candidates"][1]["path_is_positive"] is False
    assert summary["path_negative_accepted_as_equivalent"] == 1


def test_semantic_benchmark_separates_rows_with_no_equivalent_candidate() -> None:
    rows = [
        {
            "example_id": "example-1",
            "original_question": "Who wrote Book?",
            "candidates": [
                {"generated_question": "Who published Book?", "is_positive": True}
            ],
        }
    ]
    judgments = {
        "example-1": {
            "items": [
                {
                    "id": "candidate-0",
                    "equivalent": False,
                    "issue": "wrong_relation",
                }
            ]
        }
    }
    scored, unscorable, summary = apply_semantic_labels(rows, judgments)

    assert not scored
    assert len(unscorable) == 1
    assert summary["no_equivalent_candidate_sets"] == 1
    assert summary["path_positive_rejected_as_not_equivalent"] == 1


def test_semantic_result_parser_keeps_candidate_ids_scoped_by_question(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    rows = []
    for question_id, equivalent in (("q1", True), ("q2", False)):
        rows.append(
            {
                "custom_id": question_id,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "items": [
                                                {
                                                    "id": "candidate-0",
                                                    "equivalent": equivalent,
                                                }
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    }
                },
            }
        )
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    judgments, errors = parse_semantic_results([path])

    assert not errors
    assert judgments["q1"]["items"][0]["equivalent"] is True
    assert judgments["q2"]["items"][0]["equivalent"] is False


def test_semantic_adjudicator_is_difference_first_and_path_blind() -> None:
    item = {
        "example_id": "kqa:1",
        "candidate_index": 2,
        "original_question": "Who is Ada's partner?",
        "candidate_question": "Who is partnered with Ada?",
        "first_label": True,
        "path_is_positive": False,
        "negative_type": "wrong_direction",
        "selection_reason": "path_semantic_disagreement",
    }

    record = semantic_adjudication_record(item, "judge")
    prompt = record["body"]["messages"][1]["content"]

    assert "Who is Ada's partner?" in prompt
    assert "Who is partnered with Ada?" in prompt
    assert "wrong_direction" not in prompt
    assert "path_is_positive" not in prompt
    assert "symmetric relation" in ADJUDICATOR_SYSTEM_PROMPT
    assert "timeless relation" in ADJUDICATOR_SYSTEM_PROMPT
    assert "Headquarters location" in ADJUDICATOR_SYSTEM_PROMPT


def test_adjudication_selection_includes_disagreements_and_samples_agreements() -> None:
    rows = [
        {
            "example_id": "q1",
            "original_question": "Who wrote Book?",
            "candidates": [
                {
                    "generated_question": "Who authored Book?",
                    "is_positive": True,
                    "path_is_positive": False,
                    "negative_type": "wrong_direction",
                    "semantic_judgment": {"issue": "equivalent"},
                },
                {
                    "generated_question": "Who published Book?",
                    "is_positive": False,
                    "path_is_positive": False,
                    "negative_type": "wrong_relation",
                    "semantic_judgment": {"issue": "wrong_relation"},
                },
            ],
        }
    ]

    selected = select_adjudication_candidates(rows, agreement_sample=1, seed=4)

    assert len(selected) == 2
    assert {
        item["selection_reason"] for item in selected
    } == {"path_semantic_disagreement", "agreement_sample"}


def test_disputed_adjudication_excludes_whole_candidate_set() -> None:
    rows = [
        {
            "example_id": "q1",
            "original_question": "Which event followed X?",
            "candidates": [
                {
                    "generated_question": "Which event follows X?",
                    "is_positive": False,
                    "path_is_positive": True,
                    "semantic_judgment": {"issue": "tense_or_quantifier"},
                }
            ],
        }
    ]
    selected = select_adjudication_candidates(rows, agreement_sample=0)
    judgments = {
        "q1-candidate-0": {
            "equivalent": True,
            "issue": "equivalent",
            "differences": [],
        }
    }

    clean, disputed, audit, summary = apply_adjudication(
        rows, selected, judgments
    )

    assert not clean
    assert disputed[0]["candidates"][0]["adjudication_disputed"] is True
    assert audit[0]["judges_agree"] is False
    assert summary["candidate_sets_excluded_as_disputed"] == 1


def _scored(sequence: list[str], answers: list[str], score: float) -> dict:
    return {
        "relation_sequence": sequence,
        "answers": answers,
        "generated_question": " ".join(sequence),
        "retrieval_score": 0.5,
        "cross_encoder_score": score,
    }


def test_unlabeled_machine_id_endpoints_are_not_answerable() -> None:
    assert has_answerable_endpoint(_scored(["r"], ["Jamaican English"], 1.0))
    assert not has_answerable_endpoint(_scored(["r"], ["m.04gdzf4", "g.1hhc3tsc7"], 1.0))
    assert not has_answerable_endpoint(_scored(["r"], [], 1.0))
    # A single labelled endpoint is enough to keep a mixed CVT frontier.
    assert has_answerable_endpoint(_scored(["r"], ["m.04gdzf4", "Kingston"], 1.0))


def test_endpoint_filter_skips_machine_ids_but_falls_back_when_all_unlabeled() -> None:
    candidates = [
        _scored(["cvt"], ["m.0k5ntfn"], 5.0),
        _scored(["good"], ["Kingston"], 1.0),
    ]
    assert select_candidate(candidates, "cross_encoder_score", "comparator", True)[
        "relation_sequence"
    ] == ["good"]
    assert select_candidate(candidates, "cross_encoder_score", "comparator", False)[
        "relation_sequence"
    ] == ["cvt"]

    only_unlabeled = [_scored(["cvt"], ["m.0k5ntfn"], 5.0)]
    assert select_candidate(only_unlabeled, "cross_encoder_score", "comparator", True)[
        "relation_sequence"
    ] == ["cvt"]


def test_score_sources_combine_proposer_and_verifier_evidence() -> None:
    # The verifier prefers "b"; the proposer strongly prefers "a". Selecting on the
    # verifier alone discards the prior that says "b" sits deep in the pool.
    a = {**_scored(["a"], ["Kingston"], 1.0), "retrieval_score": 9.0}
    b = {**_scored(["b"], ["Nairobi"], 2.0), "retrieval_score": 0.5}
    assert candidate_score(a, "cross_encoder_score", "comparator") == 1.0
    assert candidate_score(a, "cross_encoder_score", "retrieval") == 9.0
    assert candidate_score(a, "cross_encoder_score", "comparator_retrieval") == 10.0

    pick = lambda src: select_candidate([a, b], "cross_encoder_score", src, False)["relation_sequence"]
    assert pick("comparator") == ["b"]
    assert pick("retrieval") == ["a"]
    assert pick("comparator_retrieval") == ["a"]

    with pytest.raises(ValueError, match="unknown selection score source"):
        candidate_score(a, "cross_encoder_score", "nonsense")


def test_missing_retrieval_score_does_not_crash_selection() -> None:
    bare = _scored(["x"], ["Kingston"], 1.0)
    bare.pop("retrieval_score")
    assert candidate_score(bare, "cross_encoder_score", "comparator_retrieval") == 1.0
    assert candidate_score(bare, "cross_encoder_score", "retrieval") == 0.0


def test_candidate_log_records_evaluation_only_fields() -> None:
    entry = candidate_log_entry(
        _scored(["people.person.nationality::forward"], ["Jamaica", "m.0k5ntfn"], 3.2),
        "cross_encoder_score",
        {("people.person.place_of_birth::forward",)},
        {"jamaica"},
    )
    assert entry["matches_gold_path"] is False
    assert entry["answer_overlaps_gold"] is True
    assert entry["unlabeled_answer_count"] == 1
    assert entry["answer_count"] == 2
    assert entry["answers_truncated"] is False
    assert entry["score"] == 3.2


def test_evaluation_subset_coverage_reports_constraint_exclusions(tmp_path) -> None:
    questions = {
        "Questions": [
            {
                "QuestionId": "kept",
                "RawQuestion": "who wrote it?",
                "Parses": [{"InferentialChain": ["book.author"], "Sparql": ""}],
            },
            {
                "QuestionId": "constrained",
                "RawQuestion": "who was president in 2004?",
                "Parses": [
                    {
                        "InferentialChain": ["gov.office_holder"],
                        "Constraints": [{"Operator": "Equal"}],
                        "Sparql": "",
                    }
                ],
            },
            {
                "QuestionId": "ordered",
                "RawQuestion": "what is the largest city?",
                "Parses": [
                    {"InferentialChain": ["loc.city"], "Order": {"SortOrder": "Desc"}, "Sparql": ""}
                ],
            },
        ]
    }
    path = tmp_path / "WebQSP.test.json"
    path.write_text(json.dumps(questions), encoding="utf-8")

    coverage = evaluation_subset_coverage(path)

    assert coverage["source_questions"] == 3
    assert coverage["supported_questions"] == 1
    assert coverage["excluded_by_reason"] == {"constraints": 1, "order": 1}


def test_answer_channels_stay_separable_so_labels_cannot_leak_silently() -> None:
    evidence = comparator_answer_evidence(
        ["Vienna"], "city", 0
    )
    assert evidence == {"type": "city", "count": 1, "unlabeled": 0, "labels": ["Vienna"]}

    # The type channel must not carry the answer name: a comparator shown "Vienna"
    # for "capital of Austria?" can score the pair from world knowledge without
    # verifying the path at all, which would change the hypothesis under test.
    type_only = comparator_answer_text(evidence, ANSWER_CHANNELS["question_generated_answer_type"])
    assert type_only == "type: city"
    assert "Vienna" not in type_only

    count_only = comparator_answer_text(evidence, ANSWER_CHANNELS["question_generated_answer_count"])
    assert "count: 1" in count_only
    assert "Vienna" not in count_only

    labels_only = comparator_answer_text(evidence, ANSWER_CHANNELS["question_generated_answer_labels"])
    assert "Vienna" in labels_only
    assert "type:" not in labels_only

    assert comparator_answer_text(evidence, ()) == ""


def test_comparator_answer_modes_select_one_channel_each() -> None:
    evidence = comparator_answer_evidence(["Jamaican English", "Jamaican Creole"], "human language", 0)
    text = comparator_input_text(
        "what language do they speak?",
        "Which language is spoken there?",
        "path",
        "question_generated_answer_type",
        comparator_answer_text(evidence, ANSWER_CHANNELS["question_generated_answer_type"]),
    )
    assert "[CANDIDATE ANSWERS]" in text
    assert "type: human language" in text
    assert "[CANDIDATE PATH]" not in text
    assert "Jamaican English" not in text

    # Existing modes must stay byte-identical so trained checkpoints keep working.
    assert "[CANDIDATE ANSWERS]" not in comparator_input_text(
        "q", "g", "path-text", "question_generated", "type: x"
    )


def test_candidate_answers_prefers_executed_set_over_single_endpoint() -> None:
    path = {"answer_type": "city", "answer_entity": "Kingston"}
    assert candidate_answers({"answers": ["A", "B"]}, path) == ["A", "B"]
    assert candidate_answers({"answer_entity": "Montego Bay"}, path) == ["Montego Bay"]
    assert candidate_answers({}, path) == ["Kingston"]
    assert candidate_answers({}, {"answer_type": "city"}) == []


def _stub_pipeline(monkeypatch, families):
    """Drive run_verifier_pipeline with stubbed models so result assembly is covered."""
    import inverse_verifier.selector as selector

    class _Retriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def retrieve(self, question, graph_row):
            return {"candidate_paths": families, "retrieved_subgraph": graph_row["graph"]}

    monkeypatch.setattr(selector, "SRTKPathRetriever", _Retriever)
    monkeypatch.setattr(
        selector, "load_seq2seq", lambda model, device: (object(), object(), "cpu")
    )
    monkeypatch.setattr(
        selector, "load_comparator", lambda model, device: (object(), object(), "cpu", "question_generated")
    )
    monkeypatch.setattr(
        selector,
        "generate_joint_questions",
        lambda model, tokenizer, paths, device, batch_size=5: [
            f"question for {path['hops'][0]['relation']}" for path in paths
        ],
    )

    def _score(model, tokenizer, rows, mode, device, batch_size=1):
        return [
            {
                **row,
                "candidates": [
                    {**candidate, "cross_encoder_score": candidate["stub_score"]}
                    for candidate in row["candidates"]
                ],
            }
            for row in rows
        ]

    monkeypatch.setattr(selector, "score_comparator_rows", _score)


def test_pipeline_reports_selection_ablation_and_full_candidate_log(
    tmp_path: Path, monkeypatch
) -> None:
    def family(relation, answers, score):
        return {
            "path": {
                "anchor": "Jamaica",
                "anchor_type": "country",
                "answer_type": "human language",
                "kg": "webqsp",
                "hops": [
                    {
                        "relation": relation,
                        "direction": "forward",
                        "source_type": "country",
                        "target_type": "human language",
                    }
                ],
            },
            "relation_sequence": [f"{relation}::forward"],
            "answers": answers,
            "retrieval_score": score,
            "supporting_triples": [],
            "stub_score": score,
        }

    # The unlabeled CVT path outscores the correct one, so argmax fails where the
    # endpoint filter succeeds.
    families = [
        family("location.country.statistics", ["m.0k5ntfn"], 5.0),
        family("location.country.languages_spoken", ["Jamaican English"], 2.0),
        family("location.country.official_language", ["Jamaican English"], 1.9),
    ]
    _stub_pipeline(monkeypatch, families)

    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "Questions": [
                    {
                        "QuestionId": "q1",
                        "RawQuestion": "what language do they speak in jamaica?",
                        "Parses": [
                            {
                                "InferentialChain": ["location.country.languages_spoken"],
                                "TopicEntityMid": "m.03_r3",
                                "Sparql": "ns:m.03_r3 ns:location.country.languages_spoken ?x .",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    graphs = tmp_path / "graphs.jsonl"
    graphs.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "what language do they speak in jamaica?",
                "answer": ["Jamaican English"],
                "q_entity": ["Jamaica"],
                "graph": [
                    ["Jamaica", "location.country.languages_spoken", "Jamaican English"]
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "run"
    metrics = run_verifier_pipeline(
        questions,
        graphs,
        "generator",
        "retriever",
        output,
        limit=1,
        comparison_mode="cross_encoder",
        comparator_model="comparator",
    )

    variants = metrics["selection_variants"]
    assert variants["comparator"]["answer_exact_match"] == 0.0
    assert variants["comparator_filtered"]["answer_exact_match"] == 1.0
    assert variants["retrieval_filtered"]["answer_exact_match"] == 1.0
    # The unfiltered comparator is primary, so the headline metric reflects the
    # incumbent and every gain shows up only in the ablation block.
    assert metrics["selection_policy"] == "comparator"
    assert metrics["answer_exact_match"] == 0.0
    assert metrics["unanswerable_endpoint_selection_rate"] == 1.0
    assert metrics["evaluation_subset"]["supported_questions"] == 1
    # official_language is not the annotated path but returns exactly what it
    # returns, so the comparator is not marked wrong for choosing it.
    v = row_variants = metrics["selection_variants"]
    assert v["comparator_filtered"]["selected_gold_path_accuracy"] == 1.0
    assert v["comparator"]["selected_gold_path_accuracy"] == 0.0
    assert v["comparator"]["selected_gold_equivalent_accuracy"] == 0.0

    row = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(row["candidate_log"]) == len(families)
    assert [entry["score"] for entry in row["candidate_log"]] == [5.0, 2.0, 1.9]
    assert row["candidate_log"][0]["unlabeled_answer_count"] == 1
    assert row["candidate_log"][1]["matches_gold_path"] is True
    assert row["candidate_log"][2]["matches_gold_path"] is False
    assert row["candidate_log"][2]["answer_overlaps_gold"] is True
    assert "Selection ablation" in (output / "report.md").read_text(encoding="utf-8")


def test_gold_equivalent_credits_a_different_route_to_the_same_answers() -> None:
    # WebQSP annotates one path, but Freebase reaches the same answers many ways.
    # Scoring only exact path identity marks the verifier wrong for being right.
    verified = [
        {"relation_sequence": ["annotated::forward"], "answers": ["Novelist", "Playwright"]},
        {"relation_sequence": ["other::forward"], "answers": ["playwright", " Novelist "]},
        {"relation_sequence": ["wrong::forward"], "answers": ["Nairobi"]},
    ]
    gold = {("annotated::forward",)}

    equivalent = gold_equivalent_answer_sets(verified, gold)

    assert answer_set_key(verified[1]["answers"]) in equivalent
    assert answer_set_key(verified[2]["answers"]) not in equivalent
    # Order and case must not matter when comparing answer sets.
    assert answer_set_key(["B", "a"]) == answer_set_key([" A ", "b"])


def test_gold_equivalent_is_empty_when_no_annotated_path_was_verified() -> None:
    verified = [{"relation_sequence": ["other::forward"], "answers": ["Kingston"]}]
    assert gold_equivalent_answer_sets(verified, {("annotated::forward",)}) == set()


def test_frontier_type_is_dominant_not_a_union_of_everything_reached() -> None:
    # location.location.containedby from the Galapagos reaches a country, a province
    # and an ocean. Joining every type produced "administrative division / body of
    # water", which the generator wrote into the question verbatim.
    graph = LocalQuestionGraph(
        [
            ["Galapagos", "location.location.containedby", "Ecuador"],
            ["Galapagos", "location.location.containedby", "Galapagos Province"],
            ["Galapagos", "location.location.containedby", "Pacific Ocean"],
            ["Ecuador", "common.topic.notable_types", "Location"],
            ["Galapagos Province", "common.topic.notable_types", "Location"],
            ["Pacific Ocean", "common.topic.notable_types", "Body Of Water"],
        ]
    )

    path, answers, _ = materialize_path(
        graph, "Galapagos", ("location.location.containedby::forward",)
    )

    assert path["answer_type"] == "Location"
    assert " / " not in path["answer_type"]
    assert answers == ["Ecuador", "Galapagos Province", "Pacific Ocean"]


def test_dominant_type_prefers_a_clear_majority() -> None:
    assert dominant_type(Counter({"City": 3, "Town": 1})) == "City"
    # A genuine tie with no schema hint is not resolved by guessing.
    assert dominant_type(Counter({"Zebra": 2, "Apple": 2})) == "entity"
    assert dominant_type(Counter()) == "entity"


def _cand(seq, answers, score, question="q"):
    return {
        "relation_sequence": seq,
        "answers": answers,
        "score": score,
        "generated_question": question,
        "path": {
            "anchor": "Jamaica",
            "anchor_type": "Country",
            "answer_type": "Human Language",
            "kg": "webqsp",
            "hops": [
                {
                    "relation": seq[0].rsplit("::", 1)[0],
                    "direction": "forward",
                    "source_type": "Country",
                    "target_type": "Human Language",
                }
            ],
        },
    }


def test_labelling_credits_any_route_that_returns_the_gold_answers() -> None:
    candidates = [
        _cand(["annotated::forward"], ["Kingston"], 1.0),
        _cand(["other::forward"], ["kingston"], 2.0),      # same answers, different path
        _cand(["wrong::forward"], ["Nairobi"], 3.0),
        _cand(["partial::forward"], ["Kingston", "Nairobi"], 0.5),  # superset, not equal
    ]

    labelled = label_candidates(candidates, ["Kingston"], {("annotated::forward",)})

    assert [c["is_positive"] for c in labelled] == [True, True, False, False]
    assert [c["negative_type"] for c in labelled] == [
        "positive", "answer_equivalent", "proposed", "proposed",
    ]


def test_labelling_without_gold_answers_falls_back_to_the_annotated_path() -> None:
    labelled = label_candidates(
        [_cand(["a::forward"], ["X"], 1.0), _cand(["b::forward"], [], 1.0)],
        [],
        {("a::forward",)},
    )
    assert [c["is_positive"] for c in labelled] == [True, False]


def test_negatives_are_sampled_hardest_first() -> None:
    rng = random.Random(0)
    labelled = [
        {**_cand(["gold"], ["A"], 0.1), "is_positive": True, "negative_type": "positive"},
        *[
            {**_cand([f"n{i}"], ["B"], float(i)), "is_positive": False, "negative_type": "proposed"}
            for i in range(20)
        ],
    ]

    sampled = sample_candidates(labelled, rng, hard=3, random_count=0)

    assert sampled is not None
    kept = [c["relation_sequence"][0] for c in sampled if not c["is_positive"]]
    # The three the incumbent comparator scores highest are exactly what it must
    # learn to reject.
    assert kept == ["n19", "n18", "n17"]


def test_groups_without_a_positive_are_dropped_not_forced() -> None:
    rng = random.Random(0)
    negatives = [
        {**_cand(["n"], ["B"], 1.0), "is_positive": False, "negative_type": "proposed"}
    ]
    assert sample_candidates(negatives, rng) is None


def test_comparator_corpus_round_trips_into_trainable_groups(tmp_path: Path) -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "question_id": f"q{index}",
                "question": f"question {index}?",
                "gold_answers": ["Kingston"],
                "gold_sequences": [["annotated::forward"]],
                "candidate_log": [
                    _cand(["annotated::forward"], ["Kingston"], 1.0, "gold question"),
                    *[
                        _cand([f"n{j}::forward"], ["Nairobi"], float(j), f"wrong {j}")
                        for j in range(30)
                    ],
                ],
            }
        )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    manifest = build_comparator_corpus(predictions, tmp_path / "corpus", hard=5, random_count=2)

    assert manifest["candidate_sets"] == 10
    assert manifest["dropped_no_positive"] == 0
    assert manifest["train_candidate_sets"] + manifest["dev_candidate_sets"] == 10
    train = [json.loads(line) for line in (tmp_path / "corpus" / "train.jsonl").read_text().splitlines()]
    group = train[0]
    assert any(c["is_positive"] for c in group["candidates"])
    assert len(group["candidates"]) == 1 + 5 + 2
    assert group["original_question"].startswith("question")
    # Every group must be usable by the listwise loss.
    for row in train:
        assert any(c["is_positive"] for c in row["candidates"])


def test_schema_hint_resolves_which_of_many_freebase_types_applies() -> None:
    # Abraham Lincoln is typed Artwork, Book, Film, Location, US President ... and
    # they all tie on count, so alphabetical order rendered him "an artwork".
    counts = Counter(
        ["Artwork", "Book", "Fictional Character", "Film", "Location", "US President"]
    )
    assert dominant_type(counts, schema_type_hint("government.us_president.vice_president")) == (
        "US President"
    )
    assert schema_type_hint("government.us_president.vice_president") == "us president"
    assert schema_type_hint("people.person.place_of_birth") == "person"
    assert schema_type_hint("bare") == ""


def test_ambiguous_types_render_as_entity_rather_than_a_confident_guess() -> None:
    # No schema hint can separate these, and asserting either one states something
    # false about the entity.
    counts = Counter(["Film character", "U.S. Congressperson"])
    assert dominant_type(counts, schema_type_hint("law.inventor.inventions")) == "entity"
    assert dominant_type(counts) == "entity"
    # An unambiguous majority still wins outright.
    assert dominant_type(Counter({"City": 3, "Town": 1})) == "City"
    assert dominant_type(Counter(["Island Group"])) == "Island Group"
    assert dominant_type(Counter()) == "entity"


def test_anchor_type_uses_its_own_relation_to_disambiguate() -> None:
    graph = LocalQuestionGraph(
        [
            ["Abraham Lincoln", "government.us_president.vice_president", "Hannibal Hamlin"],
            ["Abraham Lincoln", "common.topic.notable_types", "Artwork"],
            ["Abraham Lincoln", "common.topic.notable_types", "US President"],
            ["Hannibal Hamlin", "common.topic.notable_types", "US Vice President"],
        ]
    )

    path, answers, _ = materialize_path(
        graph, "Abraham Lincoln", ("government.us_president.vice_president::forward",)
    )

    assert path["anchor_type"] == "US President"
    assert path["answer_type"] == "US Vice President"
    assert answers == ["Hannibal Hamlin"]
