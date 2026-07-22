from __future__ import annotations

import copy
import json
import re
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
GENERIC_TYPES = {"", "entity", "thing", "unknown"}


def normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


def is_symmetric_hop(
    kg: str,
    hop: dict[str, Any],
    glossary: dict[str, dict[str, Any]],
) -> bool:
    relation = normalized_words(hop["relation"])
    if any(part in relation for part in SYMMETRIC_RELATION_PARTS):
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
