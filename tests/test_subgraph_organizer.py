import json
from collections import Counter

from subgraph_organizer import (
    adjacency_graph_lines,
    decode_adjacency_graph,
    extract_triple_lines,
    organize_triples,
    replace_triples,
    transform_row,
)


def test_reorder_preserves_duplicate_triples_and_adversarial_names():
    question = "Which city is home to Alpha, Beta?"
    lines = [
        "(Other,place.location.contains,Alpha, Beta)",
        "(Alpha, Beta,place.location.country,Country)",
        "(Alpha, Beta,place.location.country,Country)",
        "(Detached,relation.connects,Elsewhere)",
    ]
    organized = organize_triples(question, lines, structured=False)
    assert Counter(organized) == Counter(lines)
    assert organized.count(lines[2]) == 2
    assert set(organized) == set(lines)


def test_structured_output_adds_only_entity_headings_and_keeps_raw_lines():
    question = "Where is Alpha?"
    lines = [
        "(Alpha,relation.connects,Beta)",
        "(Beta,relation.connects,Gamma)",
        "(Unrelated,relation.connects,Other)",
    ]
    organized = organize_triples(question, lines, structured=True)
    triple_lines = [line for line in organized if line.startswith("(")]
    assert Counter(triple_lines) == Counter(lines)
    assert organized[0] == "[Alpha]"
    assert "[Beta]" in organized
    assert "[Unrelated]" in organized
    assert all(line.startswith("[") or line in lines for line in organized)


def test_both_arms_use_the_same_flat_group_order():
    question = "Where is Alpha?"
    lines = [
        "(Beta,relation.connects,Alpha)",
        "(Alpha,relation.connects,Gamma)",
        "(Detached,relation.connects,Other)",
    ]
    reorder = organize_triples(question, lines, structured=False)
    structured = organize_triples(question, lines, structured=True)
    assert [line for line in structured if line.startswith("(")] == reorder


def test_bfs_root_heading_can_emit_a_raw_tail_edge():
    lines = [
        "(Beta,relation.connects,Alpha)",
        "(Gamma,relation.connects,Beta)",
    ]
    structured = organize_triples("Where is Alpha?", lines, structured=True)
    assert structured[:2] == ["[Alpha]", lines[0]]
    assert structured[2:] == ["[Beta]", lines[1]]


def test_edges_in_one_group_keep_released_input_order():
    lines = [
        "(Root,zz.relation,First)",
        "(Root,aa.relation,Second)",
        "(Root,mm.relation,Third)",
    ]
    structured = organize_triples("Where is Root?", lines, structured=True)
    assert structured == ["[Root]", *lines]


def test_multiple_question_roots_follow_question_order():
    lines = [
        "(Raphael,relation.first,Shared)",
        "(Michelangelo,relation.second,Shared)",
        "(Shared,relation.third,Answer)",
    ]
    structured = organize_triples(
        "What connects Michelangelo and Raphael?", lines, structured=True
    )
    assert structured[:2] == ["[Michelangelo]", lines[1]]
    assert structured[2:4] == ["[Raphael]", lines[0]]
    assert structured[4:] == ["[Shared]", lines[2]]


def test_one_character_entity_does_not_match_an_article():
    lines = [
        "(A,relation.connects,B)",
        "(Anchor,relation.connects,C)",
        "(Anchor,relation.connects,D)",
    ]
    structured = organize_triples("What is a place?", lines, structured=True)
    assert structured[0] == "[Anchor]"


def test_prompt_replacement_preserves_question_and_prompt_instructions():
    row = {
        "id": "x",
        "question": "Where is Alpha?",
        "prediction": "ans: old",
        "ground_truth": ["Beta"],
        "sys_query": "Answer with ans:",
        "user_query": "Triplets:\n(Alpha,relation.connects,Beta)\n\nQuestion:\nWhere is Alpha?",
        "all_query": "Answer with ans:\n\nTriplets:\n(Alpha,relation.connects,Beta)\n\nQuestion:\nWhere is Alpha?",
    }
    transformed = transform_row(row, structured=True)
    assert transformed["question"] == row["question"]
    assert transformed["prediction"] == row["prediction"]
    assert transformed["sys_query"] == row["sys_query"]
    assert transformed["ground_truth"] == row["ground_truth"]
    assert "Answer with ans:" in transformed["all_query"]
    assert "Question:\nWhere is Alpha?" in transformed["all_query"]
    body = transformed["all_query"].split("Triplets:\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert body.splitlines() == [
        "[Alpha]",
        "(Alpha,relation.connects,Beta)",
    ]


def test_replacement_is_lossless_for_both_prompt_fields():
    lines = [
        "(A,relation.same,B)",
        "(A,relation.same,B)",
        "(C,relation.same,D)",
    ]
    for prompt in [
        "Triplets:\n" + "\n".join(lines) + "\n\nQuestion:\nQ",
        "SYS\n\nTriplets:\n" + "\n".join(lines) + "\n\nQuestion:\nQ",
    ]:
        replaced = replace_triples(prompt, organize_triples("Q", lines, structured=False))
        assert Counter(extract_triple_lines(replaced)) == Counter(lines)


def test_adjacency_graph_is_lossless_with_duplicates_and_commas():
    lines = [
        "(Alpha, Beta,relation.connects,Gamma)",
        "(Gamma,relation.answers,Delta)",
        "(Gamma,relation.answers,Delta)",
    ]
    encoded = adjacency_graph_lines("Where is Alpha, Beta?", lines, organize_groups=True)
    assert Counter(decode_adjacency_graph(encoded)) == Counter(
        [
            ("Alpha, Beta", "relation.connects", "Gamma"),
            ("Gamma", "relation.answers", "Delta"),
            ("Gamma", "relation.answers", "Delta"),
        ]
    )


def test_adjacency_arms_have_identical_groups_and_edges_but_different_order():
    lines = [
        "(Far,relation.next,Away)",
        "(Question Entity,relation.first,Middle)",
        "(Middle,relation.second,Answer)",
    ]
    flat = adjacency_graph_lines("Where is Question Entity?", lines, organize_groups=False)
    graph = adjacency_graph_lines("Where is Question Entity?", lines, organize_groups=True)
    assert flat[0] == graph[0] == "Directed adjacency:"
    assert Counter(flat[1:]) == Counter(graph[1:])
    assert flat[1:] != graph[1:]
