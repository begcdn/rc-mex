import json
from pathlib import Path

import subgraph_reader_pilot as pilot
from subgraph_reader_pilot import (
    build_conversation,
    evaluate_graph_runs,
    evaluate_runs,
    first_answer_rank,
    needs_follow_up,
    prepare_graph_arms,
    prepare_structure_pilot,
    prepare_pilot,
    score_prediction,
    select_idle_gpu,
)


def _row(identifier: str, rank: int, answer: str = "Answer") -> dict:
    lines = [f"(Noise {i},relation.other,Elsewhere {i})" for i in range(1, 101)]
    lines[rank - 1] = f"(Anchor,relation.answer,{answer})"
    user_query = "Triplets:\n" + "\n".join(lines) + f"\n\nQuestion:\nWhere is {identifier}?"
    return {
        "id": identifier,
        "question": f"Where is {identifier}?",
        "ground_truth": [answer],
        "a_entity": [answer],
        "prediction": "ans: Answer",
        "sys_query": "Answer from the graph.",
        "user_query": user_query,
        "all_query": "Answer from the graph.\n\n" + user_query,
        "cot_query": "Return ans: lines.",
    }


def test_first_answer_rank_uses_exact_endpoints():
    row = _row("x", 55, answer="York")
    row["user_query"] = row["user_query"].replace(
        "(Noise 1,relation.other,Elsewhere 1)",
        "(New York,relation.other,Elsewhere 1)",
    )
    assert first_answer_rank(row) == 55


def test_sparql_relation_parser_does_not_treat_entity_ids_as_relations():
    sparql = """SELECT ?x WHERE {
?c ns:people.person.parents ns:m.abc123 .
?c ns:people.person.place_of_birth ?x .
FILTER (?x != ?c)
}"""
    assert pilot._sparql_relations(sparql) == [
        "people.person.parents",
        "people.person.place_of_birth",
    ]


def test_prepare_builds_balanced_lossless_arms(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [_row("shallow-a", 1), _row("shallow-b", 8), _row("deep-a", 55), _row("deep-b", 90)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    manifest = prepare_pilot(source, tmp_path / "pilot", per_slice=2, seed=17)
    assert manifest["slice_counts"] == {"shallow_1_10": 2, "deep_51_100": 2}
    arm_rows = {}
    for arm in ("original", "reorder", "structured"):
        path = tmp_path / "pilot" / "inputs" / f"{arm}.jsonl"
        arm_rows[arm] = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["id"] for row in arm_rows[arm]] == [row["id"] for row in arm_rows["original"]]
    assert all("prediction" not in row for row in arm_rows["original"])
    assert all(row["released_prediction"] == "ans: Answer" for row in arm_rows["original"])


def test_conversation_matches_published_dc_role_order():
    row = _row("x", 1)
    initial = build_conversation(row)
    follow_up = build_conversation(row, follow_up=True)
    assert [message["role"] for message in initial] == ["system", "user", "assistant", "user"]
    assert [message["role"] for message in follow_up] == ["system", "user", "assistant", "user", "user"]
    assert follow_up[-1]["content"] == row["cot_query"]


def test_follow_up_rule_matches_published_behavior():
    assert not needs_follow_up("Reasoning\nans: Paris")
    assert needs_follow_up("Paris")
    assert needs_follow_up("ans: not available")
    assert needs_follow_up("ans: no information available")


def test_corrected_subgraphrag_scoring_behavior():
    row = _row("x", 1, answer="The United States")
    assert score_prediction(row, "ans: United States") == {
        "hit_at_1": 1.0,
        "f1": 1.0,
        "no_answer": 0.0,
    }
    assert score_prediction(row, "No answer") == {
        "hit_at_1": 0.0,
        "f1": 0.0,
        "no_answer": 1.0,
    }


def test_hit_at_one_uses_first_emitted_answer_not_longest_answer():
    row = _row("x", 1, answer="Paris")
    scores = score_prediction(row, "ans: Definitely wrong\nans: Paris")
    assert scores["hit_at_1"] == 0.0
    assert scores["f1"] == 2 / 3


def test_evaluation_is_paired_by_id(tmp_path: Path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = [_row("a", 2), _row("b", 80)]
    for row in rows:
        row["pilot_bucket"] = "shallow_1_10" if row["id"] == "a" else "deep_51_100"
        row["answer_evidence_rank"] = 2 if row["id"] == "a" else 80
    predictions = {
        "original": ["ans: Wrong", "ans: Wrong"],
        "reorder": ["ans: Answer", "ans: Wrong"],
        "structured": ["ans: Answer", "ans: Answer"],
    }
    for arm, arm_predictions in predictions.items():
        output_rows = []
        for row, prediction in zip(rows, arm_predictions):
            output = dict(row)
            output["prediction"] = prediction
            output_rows.append(output)
        (run_dir / f"{arm}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in output_rows)
        )

    metrics = evaluate_runs(run_dir, tmp_path / "evaluation", bootstrap_samples=100, seed=3)
    assert metrics["slices"]["overall"]["arms"]["original"]["hit_at_1"] == 0.0
    assert metrics["slices"]["overall"]["arms"]["reorder"]["hit_at_1"] == 0.5
    assert metrics["slices"]["overall"]["arms"]["structured"]["hit_at_1"] == 1.0


def test_suite_loads_model_once_for_all_arms(monkeypatch, tmp_path: Path):
    loads = []
    runs = []

    def fake_load(model, tensor_parallel_size):
        loads.append((model, tensor_parallel_size))
        return "engine", "params", "test-vllm"

    def fake_run(input_path, output_path, llm, params, batch_size, limit):
        runs.append((input_path.name, output_path.name, llm, params, batch_size, limit))
        return {"rows": 2}

    monkeypatch.setattr(pilot, "_load_vllm", fake_load)
    monkeypatch.setattr(pilot, "_run_with_engine", fake_run)
    manifest = pilot.run_suite(
        tmp_path / "inputs",
        tmp_path / "outputs",
        Path("model"),
        batch_size=4,
        tensor_parallel_size=1,
        limit=2,
    )

    assert loads == [(Path("model"), 1)]
    assert [run[0] for run in runs] == ["original.jsonl", "reorder.jsonl", "structured.jsonl"]
    assert manifest["vllm"] == "test-vllm"


def test_all_suite_loads_model_once_for_five_arms(monkeypatch, tmp_path: Path):
    loads = []
    runs = []

    monkeypatch.setattr(
        pilot,
        "_load_vllm",
        lambda model, tensor_parallel_size: (
            loads.append((model, tensor_parallel_size)) or "engine",
            "params",
            "test-vllm",
        ),
    )
    monkeypatch.setattr(
        pilot,
        "_run_with_engine",
        lambda input_path, output_path, llm, params, batch_size, limit: (
            runs.append(input_path.name) or {"rows": 2}
        ),
    )
    pilot.run_all_suite(
        tmp_path / "inputs",
        tmp_path / "outputs",
        Path("model"),
        batch_size=4,
        tensor_parallel_size=1,
        limit=2,
    )
    assert loads == [(Path("model"), 1)]
    assert runs == [f"{arm}.jsonl" for arm in pilot.ALL_ARM_NAMES]


def test_idle_gpu_selection_requires_low_memory_and_prefers_emptiest():
    query = "0, 12000\n1, 17\n2, 490\n3, 8000\n"
    assert select_idle_gpu(query, max_used_mib=500) == 1
    assert select_idle_gpu(query, max_used_mib=10) is None


def test_prepare_graph_arms_is_lossless_and_uses_gold_only_for_metadata(tmp_path: Path):
    row = _row("cwq-id", 55)
    row["pilot_bucket"] = "deep_51_100"
    row["answer_evidence_rank"] = 55
    original = tmp_path / "original.jsonl"
    original.write_text(json.dumps(row) + "\n")
    official = tmp_path / "cwq.json"
    official.write_text(
        json.dumps(
            [
                {
                    "ID": "cwq-id",
                    "compositionality_type": "conjunction",
                    "sparql": "SELECT ?x WHERE { ?c ns:relation.answer ?x . }",
                }
            ]
        )
    )

    manifest = prepare_graph_arms(original, tmp_path / "inputs", official, None)
    assert manifest["cwq_type_counts"] == {"conjunction": 1}
    metadata = json.loads((tmp_path / "inputs" / "graph_metadata.jsonl").read_text())
    assert metadata["gold_relations_present"] is True
    assert metadata["evidence_proxy_complete"] is True
    for arm in ("adjacency_flat", "adjacency_graph"):
        prepared = json.loads((tmp_path / "inputs" / f"{arm}.jsonl").read_text())
        assert "relation.answer" in prepared["user_query"]
        assert "conjunction" not in prepared["user_query"]
        assert "SELECT ?x" not in prepared["user_query"]


def test_graph_evaluation_reports_operator_and_complexity_slices(tmp_path: Path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = [_row("a", 2), _row("b", 80)]
    for row, bucket, rank in zip(rows, ("shallow_1_10", "deep_51_100"), (2, 80)):
        row["pilot_bucket"] = bucket
        row["answer_evidence_rank"] = rank
    arm_predictions = {
        "original": ["ans: Wrong", "ans: Wrong"],
        "reorder": ["ans: Wrong", "ans: Wrong"],
        "structured": ["ans: Wrong", "ans: Wrong"],
        "adjacency_flat": ["ans: Wrong", "ans: Answer"],
        "adjacency_graph": ["ans: Answer", "ans: Answer"],
    }
    for arm, predictions in arm_predictions.items():
        output = []
        for row, prediction in zip(rows, predictions):
            item = dict(row)
            item["prediction"] = prediction
            output.append(item)
        (run_dir / f"{arm}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in output)
        )
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row["id"],
                    "cwq_type": kind,
                    "gold_relation_count": count,
                    "gold_relations": ["relation.answer"],
                    "gold_relations_present": True,
                    "answer_endpoint_present": True,
                    "evidence_proxy_complete": True,
                    "prompt_tokens": {
                        "original": 100,
                        "adjacency_flat": 90,
                        "adjacency_graph": 90,
                    },
                }
            )
            for row, kind, count in zip(rows, ("composition", "conjunction"), (2, 3))
        )
        + "\n"
    )

    metrics = evaluate_graph_runs(
        run_dir, metadata_path, tmp_path / "evaluation", bootstrap_samples=100, seed=5
    )
    assert metrics["slices"]["overall"]["arms"]["adjacency_graph"]["hit_at_1"] == 1.0
    assert metrics["slices"]["cwq_conjunction"]["questions"] == 1
    assert metrics["slices"]["gold_relations_3_plus"]["questions"] == 1


def test_structure_pilot_balances_types_without_putting_sparql_in_prompts(tmp_path: Path):
    source_rows = []
    official_rows = []
    for kind in ("composition", "conjunction"):
        for index in range(2):
            identifier = f"{kind}-{index}"
            source_rows.append(_row(identifier, index + 1))
            official_rows.append(
                {
                    "ID": identifier,
                    "compositionality_type": kind,
                    "sparql": "SELECT ?x WHERE { ?c ns:relation.answer ?x . }",
                }
            )
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    official = tmp_path / "official.json"
    official.write_text(json.dumps(official_rows))

    manifest = prepare_structure_pilot(
        source, tmp_path / "pilot", official, None, per_type=1, seed=11
    )
    assert manifest["selected_type_counts"] == {"composition": 1, "conjunction": 1}
    for arm in pilot.ALL_ARM_NAMES:
        rows = [
            json.loads(line)
            for line in (tmp_path / "pilot" / "inputs" / f"{arm}.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert all("SELECT ?x" not in row["user_query"] for row in rows)
