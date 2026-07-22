from __future__ import annotations

import copy
import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl


SYMMETRIC_RELATION_PARTS = (
    "adjacent to",
    "different from",
    "diplomatic relation",
    "partner",
    "said to be the same as",
    "shares border with",
    "sibling",
    "spouse",
    "twinned administrative body",
)
SYMMETRIC_RELATIONS = {"relative"}
GENERIC_TYPES = {"", "entity", "thing", "unknown"}


def normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


def is_symmetric_hop(
    kg: str,
    hop: dict[str, Any],
    glossary: dict[str, dict[str, Any]],
) -> bool:
    relation = normalized_words(hop["relation"])
    if relation in SYMMETRIC_RELATIONS or any(
        part in relation for part in SYMMETRIC_RELATION_PARTS
    ):
        return True
    entry = glossary.get(f"{kg}::{hop['relation']}", {})
    subject_role = normalized_words(entry.get("subject_role", ""))
    object_role = normalized_words(entry.get("object_role", ""))
    return bool(subject_role and subject_role == object_role)


def direction_counterfactuals(
    path: dict[str, Any],
    glossary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    kg = path.get("kg", "unknown")
    counterfactuals = []
    for index, hop in enumerate(path["hops"]):
        if is_symmetric_hop(kg, hop, glossary):
            continue
        counterfactual = copy.deepcopy(path)
        changed = counterfactual["hops"][index]
        changed["direction"] = "backward" if changed["direction"] == "forward" else "forward"
        counterfactual.update(
            {
                "negative_type": "wrong_direction",
                "contrast_only": True,
                "flipped_hop_index": index,
            }
        )
        counterfactuals.append(counterfactual)
    return counterfactuals


def repetitive_question_reason(question: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", question.casefold())
    if len(words) < 6:
        return None
    for width in (3, 4, 5):
        phrases = Counter(tuple(words[index : index + width]) for index in range(len(words) - width + 1))
        if phrases and max(phrases.values()) >= 5:
            return f"repeated_{width}_gram"
    content = [word for word in words if word not in {"a", "an", "the", "is", "of", "in", "to", "what", "which", "who"}]
    if content and max(Counter(content).values()) >= 7:
        return "excessive_repeated_content_word"
    return None


def path_has_generic_types(path: dict[str, Any]) -> bool:
    values = [path.get("anchor_type", ""), path.get("answer_type", "")]
    values.extend(hop.get("source_type", "") for hop in path["hops"])
    values.extend(hop.get("target_type", "") for hop in path["hops"])
    return any(normalized_words(value) in GENERIC_TYPES for value in values)


def repair_faithful_corpus(
    source: Path,
    output: Path,
    glossary_path: Path,
) -> dict[str, Any]:
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    rejected = []
    for split in ("train", "dev"):
        repaired = []
        for row in read_jsonl(source / f"{split}_faithful.jsonl"):
            totals["source_rows"] += 1
            positive_reason = repetitive_question_reason(row["question"])
            if positive_reason:
                totals["rejected_positive_question"] += 1
                rejected.append(
                    {"example_id": row["example_id"], "split": split, "reason": positive_reason}
                )
                continue
            clean_negatives = []
            for negative in row["negative_paths"]:
                reason = repetitive_question_reason(negative.get("question", ""))
                if reason:
                    totals["removed_negative_question"] += 1
                    continue
                clean_negatives.append(negative)
            if not clean_negatives:
                totals["rejected_without_natural_negative"] += 1
                rejected.append(
                    {
                        "example_id": row["example_id"],
                        "split": split,
                        "reason": "no_nonrepetitive_natural_negative",
                    }
                )
                continue

            item = copy.deepcopy(row)
            item["negative_paths"] = clean_negatives
            item["contrast_only_negative_paths"] = direction_counterfactuals(
                item["positive_path"], glossary
            )
            item["quality_flags"] = {
                "generic_path_types": path_has_generic_types(item["positive_path"]),
                "direction_counterfactuals": len(item["contrast_only_negative_paths"]),
            }
            totals["accepted_rows"] += 1
            totals["natural_negatives"] += len(clean_negatives)
            totals["direction_counterfactuals"] += len(item["contrast_only_negative_paths"])
            totals["generic_type_rows"] += item["quality_flags"]["generic_path_types"]
            repaired.append(item)
        write_jsonl(output / f"{split}_faithful.jsonl", repaired)
        totals[f"accepted_{split}"] = len(repaired)

    write_jsonl(output / "rejected.jsonl", rejected)
    source_manifest_path = source / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        "version": "direction-balanced-v1",
        "source": str(source),
        "glossary": str(glossary_path),
        "method": {
            "direction_counterfactuals": (
                "flip one traversal direction at a time; use only in contrastive loss"
            ),
            "symmetric_relations_excluded": list(SYMMETRIC_RELATION_PARTS),
            "natural_generation_targets_unchanged": True,
            "repetitive_questions_removed": True,
            "generic_types_preserved_and_flagged": True,
        },
        "counts": dict(totals),
        "source_manifest": source_manifest,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _pair_candidate_score(row: dict[str, Any], preferred_ids: set[str]) -> tuple[int, str]:
    path = row["positive_path"]
    values = [path.get("anchor_type", ""), path.get("answer_type", "")]
    informative = sum(normalized_words(value) not in GENERIC_TYPES for value in values)
    clean_types = sum("/" not in value and len(value) <= 50 for value in values)
    preferred = 1 if row["example_id"] in preferred_ids else 0
    return preferred * 10 + informative * 2 + clean_types, row["example_id"]


def prepare_executable_direction_pairs(
    source: Path,
    base: Path,
    glossary_path: Path,
    output: Path,
) -> dict[str, Any]:
    from .dataset_builder import compact_query, render_compact_query

    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    preferred_ids = {
        row["example_id"]
        for split in ("train", "dev")
        for row in read_jsonl(base / f"{split}_faithful.jsonl")
    }
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for split in ("train", "dev"):
        for row in read_jsonl(source / f"{split}_faithful.jsonl"):
            path = row["positive_path"]
            if len(path["hops"]) != 1:
                continue
            hop = path["hops"][0]
            key = (path.get("kg", row.get("kg", "unknown")), hop["relation"])
            entry = glossary.get(f"{key[0]}::{key[1]}", {})
            if entry.get("status") != "semantic" or is_symmetric_hop(key[0], hop, glossary):
                continue
            grouped.setdefault(key, {"forward": [], "backward": []})[hop["direction"]].append(row)

    prepared = {"train": [], "dev": []}
    for (kg, relation), directions in sorted(grouped.items()):
        if not directions["forward"] or not directions["backward"]:
            continue
        forward = max(directions["forward"], key=lambda row: _pair_candidate_score(row, preferred_ids))
        backward = max(directions["backward"], key=lambda row: _pair_candidate_score(row, preferred_ids))
        digest = hashlib.sha256(f"{kg}::{relation}".encode()).hexdigest()
        split = "dev" if int(digest[:8], 16) % 10 == 0 else "train"
        forward_path = copy.deepcopy(forward["positive_path"])
        backward_path = copy.deepcopy(backward["positive_path"])
        forward_path["explicit_query"] = render_compact_query(
            compact_query(forward_path, glossary)
        )
        backward_path["explicit_query"] = render_compact_query(
            compact_query(backward_path, glossary)
        )
        backward_path.update(
            {
                "negative_type": "executable_opposite_direction",
                "question": backward["question"],
                "answer_entity": backward.get("positive_answer_entity", ""),
            }
        )
        prepared[split].append(
            {
                "example_id": f"direction-pair:{kg}:{digest[:16]}",
                "question": forward["question"],
                "positive_path": forward_path,
                "positive_answer_entity": forward.get("positive_answer_entity", ""),
                "alternate_positive_paths": [],
                "negative_paths": [backward_path],
                "split": f"{split}_faithful",
                "kg": kg,
                "relation_sequence": [f"{relation}::forward"],
                "source_kind": "paired_executable_direction",
                "bidirectional_pair": True,
                "paired_source_ids": [forward["example_id"], backward["example_id"]],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        write_jsonl(output / f"{split}_faithful.jsonl", prepared[split])
    manifest = {
        "version": "executable-direction-pairs-v1",
        "source": str(source),
        "base": str(base),
        "glossary": str(glossary_path),
        "train_pairs": len(prepared["train"]),
        "dev_pairs": len(prepared["dev"]),
        "total_pairs": sum(map(len, prepared.values())),
        "selection": (
            "one executable forward and backward one-hop path per semantic asymmetric relation"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _natural_pair_rejection(row: dict[str, Any]) -> str | None:
    positive = row.get("question", "").strip()
    positive_path = row.get("positive_path", {})
    negative_path = row.get("negative_paths", [{}])[0]
    negative = negative_path.get("question", "").strip()
    positive_hops = positive_path.get("hops", [])
    if positive_hops and normalized_words(positive_hops[0]["relation"]) in SYMMETRIC_RELATIONS:
        return "symmetric_relation_pair"
    for question, path in ((positive, positive_path), (negative, negative_path)):
        if question.count("[ENTITY]") != 1 or not question.endswith("?"):
            return "invalid_question_form"
        lowered = question.casefold()
        if re.search(
            r"\b(?:forward|backward|graph|hop)\b|reverse direction|through the relation",
            lowered,
        ):
            return "exposes_graph_notation"
        explicit = path.get("explicit_query", "").casefold()
        for vague in ("associated with", "related to", "connected to"):
            if vague in lowered and vague not in explicit:
                return "weakens_specific_relation"
    if positive.casefold() == negative.casefold():
        return "directions_have_identical_question"
    if positive == row.get("canonical_question"):
        return "positive_naturalization_fallback"
    if negative == row["negative_paths"][0].get("canonical_question"):
        return "negative_naturalization_fallback"
    return None


def merge_executable_direction_pairs(
    base: Path,
    naturalized_pairs: Path,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rejected = []
    for split in ("train", "dev"):
        base_rows = read_jsonl(base / f"{split}_faithful.jsonl")
        accepted_pairs = []
        for row in read_jsonl(naturalized_pairs / f"{split}_faithful.jsonl"):
            reason = _natural_pair_rejection(row)
            if reason:
                rejected.append({"example_id": row["example_id"], "split": split, "reason": reason})
                counts[f"rejected_{reason}"] += 1
            else:
                accepted_pairs.append(row)
        write_jsonl(output / f"{split}_faithful.jsonl", [*base_rows, *accepted_pairs])
        counts[f"base_{split}"] = len(base_rows)
        counts[f"direction_pairs_{split}"] = len(accepted_pairs)
        counts[f"final_{split}"] = len(base_rows) + len(accepted_pairs)
    write_jsonl(output / "rejected_direction_pairs.jsonl", rejected)
    manifest = {
        "version": "faithful-plus-executable-direction-v1",
        "base": str(base),
        "naturalized_pairs": str(naturalized_pairs),
        "counts": dict(counts),
        "direction_pairs_are_executable": True,
        "counterfactual_direction_flips_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
