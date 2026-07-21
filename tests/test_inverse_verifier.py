from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import numpy as np
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
from inverse_verifier.selector import (
    answer_metrics,
    enumerate_path_families,
    first_gold_rank,
)
from inverse_verifier.retrieval import (
    LocalQuestionGraph,
    decode_edge,
    encode_edge,
    gold_path_available,
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
    assert 'NODE_1 --["place of birth"]--> ANSWER' in rendered
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
    assert 'START --["author"]--> ANSWER' in prompt


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
