import json
from pathlib import Path

import reader_scale_campaign as campaign


def _row(identifier: str, answer: str = "Shared") -> dict:
    triples = "\n".join(
        (
            "(Entity A,relation.first,Shared)",
            "(Entity B,relation.second,Shared)",
        )
    )
    user = f"Triplets:\n{triples}\n\nQuestion:\nWhat is shared by Entity A and Entity B?"
    return {
        "id": identifier,
        "question": "What is shared by Entity A and Entity B?",
        "prediction": "ans: Shared",
        "ground_truth": [answer],
        "a_entity": [answer],
        "sys_query": "Answer from the KG.",
        "user_query": user,
        "all_query": "Answer from the KG.\n\n" + user,
        "cot_query": "Return ans: lines.",
    }


def test_prepare_full_campaign_is_lossless_and_imports_gpt_baseline(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    gpt = tmp_path / "gpt.jsonl"
    official = tmp_path / "official.json"
    source.write_text(json.dumps(_row("q1")) + "\n")
    gpt_row = _row("q1")
    gpt_row["prediction"] = "ans: GPT answer"
    gpt.write_text(json.dumps(gpt_row) + "\n")
    official.write_text(
        json.dumps(
            [
                {
                    "ID": "q1",
                    "compositionality_type": "conjunction",
                    "sparql": "SELECT ?x WHERE { ns:m.a ns:relation.first ?x . }",
                }
            ]
        )
    )
    manifest = campaign.prepare_full_campaign(source, gpt, official, tmp_path / "out", None)
    assert manifest["rows"] == 1
    assert manifest["branchable_rows"] == 1
    original = campaign.read_jsonl(tmp_path / "out/inputs/original.jsonl")[0]
    branch = campaign.read_jsonl(tmp_path / "out/inputs/branch_grouped.jsonl")[0]
    assert campaign._triple_multiset(original) == campaign._triple_multiset(branch)
    imported = campaign.read_jsonl(tmp_path / "out/gpt4o_mini/runs/original.jsonl")[0]
    assert imported["prediction"] == "ans: GPT answer"


def test_request_keys_are_short_stable_and_arm_specific():
    first = campaign._request_key("branch_grouped", "a-very-long-question-id" * 10, "primary")
    assert first == campaign._request_key("branch_grouped", "a-very-long-question-id" * 10, "primary")
    assert first != campaign._request_key("reorder", "a-very-long-question-id" * 10, "primary")
    assert len(first) < 40


def test_full_evaluation_reports_scale_interaction(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "id": "q1",
                "cwq_type": "conjunction",
                "branchable": True,
                "answer_endpoint_present": True,
            }
        )
        + "\n"
    )
    local = tmp_path / "local"
    gpt = tmp_path / "gpt"
    local.mkdir()
    gpt.mkdir()
    predictions = {
        "llama32_3b": {"original": "ans: Wrong", "reorder": "ans: Shared", "branch_grouped": "ans: Shared"},
        "gpt4o_mini": {"original": "ans: Shared", "reorder": "ans: Shared", "branch_grouped": "ans: Shared"},
    }
    for model, directory in (("llama32_3b", local), ("gpt4o_mini", gpt)):
        for arm in campaign.LOCAL_ARMS:
            row = _row("q1")
            row["prediction"] = predictions[model][arm]
            (directory / f"{arm}.jsonl").write_text(json.dumps(row) + "\n")
    metrics = campaign.evaluate_full_campaign(local, gpt, metadata, tmp_path / "eval", 100, 3)
    interaction = metrics["slices"]["overall"]["interactions"]["branch_effect_3b_minus_gpt"]["f1"]
    assert interaction["delta"] == 1.0
