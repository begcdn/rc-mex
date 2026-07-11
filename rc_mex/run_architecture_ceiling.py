"""Evaluation firewall and architecture-ceiling audit for RoG KGQA data.

This runner does not introduce a new method and never calls an LLM. It freezes
a deterministic evaluation slice and separates the ceilings that are otherwise
collapsed into final answer accuracy:

  1. answer and topic presence in the supplied per-question RoG subgraph;
  2. exhaustive one-hop and bounded two-hop structural reachability;
  3. generated-menu reachability and oracle menu F1 (when predictions from
     ``run_query_selection`` are supplied);
  4. selected-query and final-answer accuracy;
  5. the earliest observable failure stage for each question.

The RoG JSONL files contain per-question subgraphs, not full Freebase, and do
not contain gold SPARQL. The report therefore labels full-KG answerability and
formal operator expressibility as unmeasured instead of inventing proxies.

Typical workflow:

  # Freeze a fresh development slice and measure substrate ceilings.
  python3 -m rc_mex.run_architecture_ceiling \
      --data data/cwq/train.jsonl \
      --output runs/firewall_cwq500 \
      --sample-size 500 --seed 20260711

  # Run the current method on exactly that frozen slice.
  python3 -m rc_mex.run_query_selection \
      --data runs/firewall_cwq500/eval_questions.jsonl \
      --output runs/firewall_cwq500_qsel --max-hops 2

  # Add generated-menu, selector, and final-answer ceilings.
  python3 -m rc_mex.run_architecture_ceiling \
      --data runs/firewall_cwq500/eval_questions.jsonl \
      --predictions runs/firewall_cwq500_qsel/predictions.jsonl \
      --output runs/firewall_cwq500_audit --max-hops 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.run_query_selection import (
    INTERSECTIONS_ADDED,
    MIXED_MENU_CAP,
    build_candidate_paths,
    build_chain_candidates,
    build_intersection_candidates,
    display_relation,
    merge_mixed_menu,
)
from rc_mex.run_webqsp_path_family import approx_gold_hop_count, build_kb


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_question_ids(prediction_paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for path in prediction_paths:
        for row in load_jsonl(path):
            question_id = row.get("question_id", row.get("id"))
            if question_id is not None:
                excluded.add(str(question_id))
    return excluded


def freeze_rows(
    rows: list[dict],
    *,
    excluded_ids: set[str],
    offset: int,
    limit: int,
    sample_size: int,
    seed: int,
) -> list[tuple[int, dict]]:
    eligible = [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("id", index)) not in excluded_ids
    ]
    eligible = eligible[offset:]
    if limit:
        eligible = eligible[:limit]
    if sample_size and sample_size < len(eligible):
        rng = random.Random(seed)
        selected_indices = set(rng.sample(range(len(eligible)), sample_size))
        eligible = [item for position, item in enumerate(eligible) if position in selected_indices]
    return eligible


FAST_ID_PATTERN = re.compile(rb'"id"\s*:\s*("(?:\\.|[^"\\])*")')


def fast_question_id(line: bytes, source_index: int) -> str:
    """Read the top-level id without decoding a multi-megabyte graph row.

    RoG exports place ``id`` before ``graph``. The fallback preserves support
    for unusual key ordering, while normal firewall sampling only parses the
    few lines retained by reservoir sampling.
    """
    match = FAST_ID_PATTERN.search(line[:16384])
    if match:
        return str(json.loads(match.group(1)))
    return str(json.loads(line).get("id", source_index))


def freeze_jsonl_file(
    path: str | Path,
    *,
    excluded_ids: set[str],
    offset: int,
    limit: int,
    sample_size: int,
    seed: int,
) -> tuple[list[tuple[int, dict]], int, str]:
    """Freeze a deterministic slice without materializing a huge source file.

    CWQ's converted training JSONL is roughly 5 GB. Reservoir-sampling raw
    lines avoids decoding every embedded graph merely to select 500 rows.
    The file hash is computed in the same pass.
    """
    rng = random.Random(seed)
    digest = hashlib.sha256()
    selected_lines: list[tuple[int, bytes]] = []
    source_questions = 0
    eligible_after_offset = 0
    considered = 0
    with open(path, "rb") as handle:
        for source_index, line in enumerate(handle):
            digest.update(line)
            source_questions += 1
            question_id = fast_question_id(line, source_index)
            if question_id in excluded_ids:
                continue
            if eligible_after_offset < offset:
                eligible_after_offset += 1
                continue
            eligible_after_offset += 1
            if limit and considered >= limit:
                continue
            considered += 1
            if sample_size:
                if len(selected_lines) < sample_size:
                    selected_lines.append((source_index, line))
                else:
                    replacement = rng.randrange(considered)
                    if replacement < sample_size:
                        selected_lines[replacement] = (source_index, line)
            else:
                selected_lines.append((source_index, line))
    selected_lines.sort(key=lambda item: item[0])
    frozen = [(index, json.loads(line)) for index, line in selected_lines]
    return frozen, source_questions, digest.hexdigest()


def set_scores(predicted: set[str], gold: set[str]) -> dict[str, float | bool]:
    overlap = predicted & gold
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "has_gold": bool(overlap),
        "exact_match": predicted == gold if gold else not predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def grouped_relations(kb: dict, sources: set[str]) -> dict[tuple[str, str], set[str]]:
    groups: dict[tuple[str, str], set[str]] = {}
    for source_id in sources:
        for relation in kb["entities"].get(source_id, {}).get("relations", []):
            key = (str(relation["predicate"]), str(relation["direction"]))
            groups.setdefault(key, set()).add(str(relation["object"]))
    return groups


def topic_filtered(targets: set[str], starts: set[str]) -> set[str]:
    without_topics = targets - starts
    return without_topics or targets


def better_candidate(current: dict | None, candidate: dict) -> dict:
    if current is None:
        return candidate
    current_key = (float(current["scores"]["f1"]), bool(current["scores"]["exact_match"]), -len(current["targets"]))
    candidate_key = (
        float(candidate["scores"]["f1"]),
        bool(candidate["scores"]["exact_match"]),
        -len(candidate["targets"]),
    )
    return candidate if candidate_key > current_key else current


def structural_ceiling(
    kb: dict,
    starts: set[str],
    golds: set[str],
    *,
    fanout_cap: int,
) -> dict:
    one_hop = grouped_relations(kb, starts)
    best_one: dict | None = None
    best_two: dict | None = None
    one_hop_sets = 0
    two_hop_sets = 0
    skipped_fanout = 0

    for (predicate, direction), raw_targets in one_hop.items():
        targets = topic_filtered(set(raw_targets), starts)
        if not targets:
            continue
        one_hop_sets += 1
        candidate = {
            "query": [{"predicate": predicate, "direction": direction}],
            "targets": sorted(targets),
            "scores": set_scores(targets, golds),
        }
        best_one = better_candidate(best_one, candidate)
        best_two = better_candidate(best_two, candidate)

        if len(raw_targets) > fanout_cap:
            skipped_fanout += 1
            continue
        for (predicate2, direction2), raw_targets2 in grouped_relations(kb, set(raw_targets)).items():
            targets2 = set(raw_targets2) - starts
            if not targets2:
                continue
            two_hop_sets += 1
            candidate2 = {
                "query": [
                    {"predicate": predicate, "direction": direction},
                    {"predicate": predicate2, "direction": direction2},
                ],
                "targets": sorted(targets2),
                "scores": set_scores(targets2, golds),
            }
            best_two = better_candidate(best_two, candidate2)

    empty = {"query": [], "targets": [], "scores": set_scores(set(), golds)}
    return {
        "one_hop_candidate_sets": one_hop_sets,
        "two_hop_candidate_sets": two_hop_sets,
        "fanout_branches_skipped": skipped_fanout,
        "best_one_hop": best_one or empty,
        "best_up_to_two_hop": best_two or empty,
    }


def build_query_menu(kb: dict, graph: KnowledgeGraph, starts: set[str], question: str, prediction: dict, max_hops: int) -> list[dict]:
    paths = build_candidate_paths(kb, graph, starts, question, prediction.get("described_names") or [])
    if max_hops >= 2 and paths:
        chains = build_chain_candidates(
            kb,
            graph,
            starts,
            paths,
            prediction.get("described_names_2") or [],
        )
        if chains:
            paths = merge_mixed_menu(paths, chains)
        intersections = build_intersection_candidates(paths)
        if intersections:
            paths = merge_mixed_menu(paths, intersections, cap=MIXED_MENU_CAP + INTERSECTIONS_ADDED)
    return paths


def path_description(path: dict) -> dict:
    description = {
        "predicate": path["predicate"],
        "direction": path["direction"],
        "readable": path.get("chain_label") or display_relation(path["predicate"]),
        "size": len(path.get("targets") or []),
    }
    if path.get("chain_base"):
        description["chain_base"] = path["chain_base"]
    if path.get("intersection_of"):
        description["intersection_of"] = path["intersection_of"]
    return description


def menu_signature(paths: list[dict]) -> list[tuple]:
    return [
        (
            path["predicate"],
            path["direction"],
            len(path.get("targets") or []),
            json.dumps(path.get("chain_base"), sort_keys=True),
        )
        for path in paths
    ]


def stored_menu_signature(prediction: dict) -> list[tuple]:
    return [
        (
            candidate.get("predicate"),
            candidate.get("direction"),
            int(candidate.get("size", 0)),
            json.dumps(candidate.get("chain_base"), sort_keys=True),
        )
        for candidate in prediction.get("candidates") or []
    ]


def find_selected_path(paths: list[dict], selected: dict | None) -> dict | None:
    if not selected:
        return None
    readable = selected.get("readable")
    exact_readable = [
        path
        for path in paths
        if (path.get("chain_label") or display_relation(path["predicate"])) == readable
    ]
    if len(exact_readable) == 1:
        return exact_readable[0]
    matches = [
        path
        for path in paths
        if path["predicate"] == selected.get("predicate")
        and path["direction"] == selected.get("direction")
        and path.get("chain_base") == selected.get("chain_base")
    ]
    return matches[0] if matches else None


def menu_ceiling(paths: list[dict], golds: set[str], selected: dict | None) -> dict:
    best: dict | None = None
    for path in paths:
        targets = set(path.get("targets") or [])
        candidate = {
            "query": path_description(path),
            "targets": sorted(targets),
            "scores": set_scores(targets, golds),
        }
        best = better_candidate(best, candidate)
    selected_path = find_selected_path(paths, selected)
    selected_result = None
    if selected_path is not None:
        targets = set(selected_path.get("targets") or [])
        selected_result = {
            "query": path_description(selected_path),
            "targets": sorted(targets),
            "scores": set_scores(targets, golds),
        }
    return {
        "candidate_count": len(paths),
        "best_candidate": best or {"query": {}, "targets": [], "scores": set_scores(set(), golds)},
        "selected_candidate": selected_result,
    }


OPERATOR_PATTERNS = {
    "count": re.compile(r"\b(how many|number of|count)\b", re.I),
    "superlative_or_ordinal": re.compile(r"\b(first|last|latest|earliest|oldest|youngest|largest|smallest|most|least)\b", re.I),
    "comparative": re.compile(r"\b(more than|less than|before|after|older than|younger than|higher than|lower than)\b", re.I),
    "conjunction": re.compile(r"\b(and|both|as well as)\b", re.I),
    "disjunction": re.compile(r"\b(or|either)\b", re.I),
    "negation": re.compile(r"\b(not|never|without|except)\b", re.I),
    "temporal": re.compile(r"\b(when|year|date|during|between|current|currently|former)\b", re.I),
}


def surface_operator_tags(question: str) -> list[str]:
    tags = [name for name, pattern in OPERATOR_PATTERNS.items() if pattern.search(question)]
    return tags or ["unmarked"]


def earliest_failure(row: dict, has_prediction: bool) -> str:
    if not row["start_resolved"]:
        return "topic_missing_from_subgraph"
    if not row["gold_in_subgraph"]:
        return "gold_missing_from_subgraph"
    if not row["structural_ceiling"]["best_up_to_two_hop"]["scores"]["has_gold"]:
        return "not_reachable_by_bounded_two_hop_chain"
    if not has_prediction:
        return "not_evaluated_without_predictions"
    menu = row.get("menu_audit") or {}
    if not menu.get("best_candidate", {}).get("scores", {}).get("has_gold"):
        return "generated_menu_miss"
    selected = menu.get("selected_candidate")
    if not selected or not selected["scores"]["has_gold"]:
        return "selector_miss"
    if not row["actual_answer"]["hits_at_1"]:
        return "post_selection_or_member_ranking_miss"
    return "correct"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(question_rows: list[dict], predictions_supplied: bool) -> dict:
    n = len(question_rows)
    prediction_rows = [row for row in question_rows if row.get("prediction_joined")]
    failures = Counter(row["failure_stage"] for row in question_rows)
    operator_buckets: dict[str, dict] = {}
    for tag in sorted({tag for row in question_rows for tag in row["surface_operator_tags"]}):
        tagged = [row for row in question_rows if tag in row["surface_operator_tags"]]
        operator_buckets[tag] = {
            "questions": len(tagged),
            "subgraph_answerability": mean(float(row["gold_in_subgraph"]) for row in tagged),
            "two_hop_structural_recall": mean(
                float(row["structural_ceiling"]["best_up_to_two_hop"]["scores"]["has_gold"])
                for row in tagged
            ),
            "actual_hits_at_1": mean(float(row["actual_answer"]["hits_at_1"]) for row in tagged if row.get("prediction_joined")),
        }

    metrics = {
        "questions": n,
        "predictions_supplied": predictions_supplied,
        "predictions_joined": len(prediction_rows),
        "prediction_join_rate": len(prediction_rows) / n if n else 0.0,
        "topic_resolution_rate": mean(float(row["start_resolved"]) for row in question_rows),
        "subgraph_answerability": mean(float(row["gold_in_subgraph"]) for row in question_rows),
        "subgraph_complete_gold_set_rate": mean(float(row["all_gold_in_subgraph"]) for row in question_rows),
        "subgraph_gold_entity_recall": mean(float(row["gold_subgraph_recall"]) for row in question_rows),
        "full_kg_answerability": None,
        "full_kg_answerability_note": "Unavailable: input contains RoG per-question subgraphs, not full Freebase.",
        "gold_program_operator_coverage": None,
        "gold_program_operator_coverage_note": "Unavailable: RoG JSONL does not contain gold SPARQL/logical forms.",
        "approximate_gold_hop_distribution": dict(Counter(str(row["approx_gold_hops"]) for row in question_rows)),
        "structural_ceiling": {
            "one_hop_gold_recall": mean(
                float(row["structural_ceiling"]["best_one_hop"]["scores"]["has_gold"])
                for row in question_rows
            ),
            "one_hop_exact_set_rate": mean(
                float(row["structural_ceiling"]["best_one_hop"]["scores"]["exact_match"])
                for row in question_rows
            ),
            "one_hop_oracle_f1": mean(
                float(row["structural_ceiling"]["best_one_hop"]["scores"]["f1"])
                for row in question_rows
            ),
            "up_to_two_hop_gold_recall": mean(
                float(row["structural_ceiling"]["best_up_to_two_hop"]["scores"]["has_gold"])
                for row in question_rows
            ),
            "up_to_two_hop_exact_set_rate": mean(
                float(row["structural_ceiling"]["best_up_to_two_hop"]["scores"]["exact_match"])
                for row in question_rows
            ),
            "up_to_two_hop_oracle_f1": mean(
                float(row["structural_ceiling"]["best_up_to_two_hop"]["scores"]["f1"])
                for row in question_rows
            ),
        },
        "failure_stage_counts": dict(failures),
        "surface_operator_slices": operator_buckets,
    }
    if prediction_rows:
        menu_gold_rows = [
            row for row in prediction_rows if row["menu_audit"]["best_candidate"]["scores"]["has_gold"]
        ]
        metrics["generated_menu"] = {
            "menu_reconstruction_match_rate": mean(float(row["menu_matches_stored"]) for row in prediction_rows),
            "gold_recall": mean(
                float(row["menu_audit"]["best_candidate"]["scores"]["has_gold"])
                for row in prediction_rows
            ),
            "exact_set_rate": mean(
                float(row["menu_audit"]["best_candidate"]["scores"]["exact_match"])
                for row in prediction_rows
            ),
            "oracle_f1": mean(
                float(row["menu_audit"]["best_candidate"]["scores"]["f1"])
                for row in prediction_rows
            ),
            "average_candidate_count": mean(float(row["menu_audit"]["candidate_count"]) for row in prediction_rows),
        }
        metrics["selection_and_answer"] = {
            "selected_query_contains_gold": mean(
                float(bool(row["menu_audit"]["selected_candidate"] and row["menu_audit"]["selected_candidate"]["scores"]["has_gold"]))
                for row in prediction_rows
            ),
            "selected_query_contains_gold_given_menu_gold": mean(
                float(bool(row["menu_audit"]["selected_candidate"] and row["menu_audit"]["selected_candidate"]["scores"]["has_gold"]))
                for row in menu_gold_rows
            ),
            "hits_at_1": mean(float(row["actual_answer"]["hits_at_1"]) for row in prediction_rows),
            "exact_match": mean(float(row["actual_answer"]["scores"]["exact_match"]) for row in prediction_rows),
            "mean_answer_f1": mean(float(row["actual_answer"]["scores"]["f1"]) for row in prediction_rows),
            "hits_at_1_given_menu_gold": mean(float(row["actual_answer"]["hits_at_1"]) for row in menu_gold_rows),
            "oracle_to_actual_f1_gap": mean(
                float(row["menu_audit"]["best_candidate"]["scores"]["f1"])
                - float(row["actual_answer"]["scores"]["f1"])
                for row in prediction_rows
            ),
        }
    return metrics


def pct(value: float | None) -> str:
    return "unmeasured" if value is None else f"{100 * value:.1f}%"


def write_report(path: Path, manifest: dict, metrics: dict) -> None:
    structural = metrics["structural_ceiling"]
    lines = [
        "# Architecture Ceiling Audit",
        "",
        "## Evaluation Firewall",
        "",
        f"- Source: `{manifest['source_path']}`",
        f"- Source SHA-256: `{manifest['source_sha256']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Frozen questions: **{manifest['selected_questions']}**",
        f"- Previously evaluated IDs excluded: **{manifest['excluded_question_count']}**",
        "- Gold is used only for offline reachability and scoring.",
        "- This audit makes no LLM calls and does not alter candidate generation.",
        "",
        "## Scope Warnings",
        "",
        "- The supplied graph is a RoG per-question subgraph, not full Freebase.",
        "- The converted JSONL has no gold SPARQL, so formal operator expressibility is unmeasured.",
        "- Surface operator tags are diagnostic slices, not gold logical-form labels.",
        "",
        "## Ceiling Ladder",
        "",
        "| Stage | Recall / rate | Oracle F1 | Exact set |",
        "|---|---:|---:|---:|",
        f"| Topic resolved in supplied subgraph | {pct(metrics['topic_resolution_rate'])} | - | - |",
        f"| Gold present in supplied subgraph | {pct(metrics['subgraph_answerability'])} | - | - |",
        f"| Complete gold set present in supplied subgraph | {pct(metrics['subgraph_complete_gold_set_rate'])} | {metrics['subgraph_gold_entity_recall']:.3f} entity recall | - |",
        f"| Exhaustive one-hop relation set | {pct(structural['one_hop_gold_recall'])} | {structural['one_hop_oracle_f1']:.3f} | {pct(structural['one_hop_exact_set_rate'])} |",
        f"| Exhaustive chain up to two hops | {pct(structural['up_to_two_hop_gold_recall'])} | {structural['up_to_two_hop_oracle_f1']:.3f} | {pct(structural['up_to_two_hop_exact_set_rate'])} |",
    ]
    if "generated_menu" in metrics:
        menu = metrics["generated_menu"]
        answer = metrics["selection_and_answer"]
        lines.extend(
            [
                f"| Actual generated query menu | {pct(menu['gold_recall'])} | {menu['oracle_f1']:.3f} | {pct(menu['exact_set_rate'])} |",
                f"| Final selected answers | {pct(answer['hits_at_1'])} Hits@1 | {answer['mean_answer_f1']:.3f} | {pct(answer['exact_match'])} |",
                "",
                "## Conditional Selection",
                "",
                f"- Prediction join rate: **{pct(metrics['prediction_join_rate'])}**",
                f"- Exact menu reconstruction: **{pct(menu['menu_reconstruction_match_rate'])}**",
                f"- Selected query contains gold, given gold is on menu: **{pct(answer['selected_query_contains_gold_given_menu_gold'])}**",
                f"- Final Hits@1, given gold is on menu: **{pct(answer['hits_at_1_given_menu_gold'])}**",
                f"- Oracle-menu to actual-answer F1 gap: **{answer['oracle_to_actual_f1_gap']:.3f}**",
            ]
        )
    lines.extend(["", "## Earliest Failure Stage", ""])
    for name, count in sorted(metrics["failure_stage_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{name}`: **{count}**")
    lines.extend(
        [
            "",
            "## Reading The Result",
            "",
            "- Low subgraph answerability means the supplied retriever sets the ceiling.",
            "- A large subgraph-to-two-hop gap means bounded path structure cannot expose the answer.",
            "- A large two-hop-to-generated-menu gap means semantic grounding or menu construction is starving the selector.",
            "- A high menu oracle with weak conditional selection means ranking is the immediate bottleneck.",
            "- A selected query containing gold but a wrong final answer points to set refinement/member ranking.",
            "",
            "## Surface Operator Slices",
            "",
            "| Tag | Questions | Subgraph answerability | Two-hop recall | Actual Hits@1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tag, values in metrics["surface_operator_slices"].items():
        lines.append(
            f"| {tag} | {values['questions']} | {pct(values['subgraph_answerability'])} | "
            f"{pct(values['two_hop_structural_recall'])} | {pct(values['actual_hits_at_1']) if metrics['predictions_joined'] else '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="RoG WebQSP/CWQ JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", help="Optional predictions.jsonl from rc_mex.run_query_selection.")
    parser.add_argument("--sample-size", type=int, default=0, help="Deterministic random sample; 0 uses all eligible rows.")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-hops", type=int, choices=(1, 2), default=2, help="Query-selection menu shape to reconstruct.")
    parser.add_argument("--fanout-cap", type=int, default=2000, help="Skip exhaustive hop-2 expansion above this hop-1 set size.")
    parser.add_argument(
        "--exclude-predictions",
        action="append",
        default=[],
        help="Predictions JSONL whose question IDs must be excluded from the frozen sample; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(ensure_dir(args.output))
    excluded_ids = excluded_question_ids(args.exclude_predictions)
    frozen, source_question_count, source_sha256 = freeze_jsonl_file(
        args.data,
        excluded_ids=excluded_ids,
        offset=args.offset,
        limit=args.limit,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    predictions = load_jsonl(args.predictions) if args.predictions else []
    predictions_by_id = {str(row.get("question_id", row.get("id"))): row for row in predictions}

    print("[1/4] Freezing evaluation slice")
    print(f"      Source questions: {source_question_count}")
    print(f"      Excluded prior-evaluation IDs: {len(excluded_ids)}")
    print(f"      Frozen questions: {len(frozen)} (seed={args.seed})")
    if not frozen:
        raise SystemExit("No eligible questions remain after exclusions/slicing.")

    manifest = {
        "source_path": str(Path(args.data).resolve()),
        "source_sha256": source_sha256,
        "seed": args.seed,
        "offset": args.offset,
        "limit": args.limit,
        "sample_size": args.sample_size,
        "source_questions": source_question_count,
        "excluded_question_count": len(excluded_ids),
        "excluded_prediction_files": [str(Path(path).resolve()) for path in args.exclude_predictions],
        "selected_questions": len(frozen),
        "selected_source_indices": [index for index, _ in frozen],
        "selected_question_ids": [str(row.get("id", index)) for index, row in frozen],
    }
    write_json(output_dir / "split_manifest.json", manifest)
    with open(output_dir / "eval_questions.jsonl", "w", encoding="utf-8") as handle:
        for _, row in frozen:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[2/4] Measuring subgraph and structural ceilings")
    if args.predictions:
        print(f"      Loaded predictions: {len(predictions)}")
    question_rows: list[dict] = []
    failure_counts: Counter = Counter()
    with open(output_dir / "ceiling_questions.jsonl", "w", encoding="utf-8") as handle:
        for position, (source_index, source) in enumerate(frozen, start=1):
            question_id = str(source.get("id", source_index))
            question = str(source.get("question", ""))
            kb = build_kb(source.get("graph") or [])
            graph = KnowledgeGraph(kb)
            all_entity_ids = set(kb["entities"])
            starts = {
                normalize_text(entity)
                for entity in source.get("q_entity") or []
                if str(entity).strip()
            } & all_entity_ids
            golds = {
                normalize_text(answer)
                for answer in source.get("answer") or []
                if str(answer).strip()
            }
            prediction = predictions_by_id.get(question_id)
            structural = structural_ceiling(kb, starts, golds, fanout_cap=args.fanout_cap) if starts else {
                "one_hop_candidate_sets": 0,
                "two_hop_candidate_sets": 0,
                "fanout_branches_skipped": 0,
                "best_one_hop": {"query": [], "targets": [], "scores": set_scores(set(), golds)},
                "best_up_to_two_hop": {"query": [], "targets": [], "scores": set_scores(set(), golds)},
            }
            row = {
                "question_id": question_id,
                "source_index": source_index,
                "question": question,
                "surface_operator_tags": surface_operator_tags(question),
                "topic_entities": [str(entity) for entity in source.get("q_entity") or []],
                "gold_answers": [str(answer) for answer in source.get("answer") or []],
                "subgraph_triples": len(source.get("graph") or []),
                "subgraph_entities": len(all_entity_ids),
                "start_resolved": bool(starts),
                "gold_in_subgraph": bool(golds & all_entity_ids),
                "all_gold_in_subgraph": bool(golds) and golds <= all_entity_ids,
                "gold_subgraph_recall": len(golds & all_entity_ids) / len(golds) if golds else 0.0,
                "approx_gold_hops": approx_gold_hop_count(kb, starts, golds & all_entity_ids, max_depth=4),
                "structural_ceiling": structural,
                "prediction_joined": prediction is not None,
                "menu_matches_stored": None,
                "menu_audit": None,
                "actual_answer": {
                    "predicted": [],
                    "hits_at_1": False,
                    "scores": set_scores(set(), golds),
                },
            }
            if prediction is not None and starts:
                menu = build_query_menu(kb, graph, starts, question, prediction, args.max_hops)
                row["menu_matches_stored"] = menu_signature(menu) == stored_menu_signature(prediction)
                row["menu_audit"] = menu_ceiling(menu, golds, prediction.get("selected"))
                predicted_ids = {
                    normalize_text(answer)
                    for answer in prediction.get("predicted") or []
                    if str(answer).strip()
                }
                predicted_order = [
                    normalize_text(answer)
                    for answer in prediction.get("predicted") or []
                    if str(answer).strip()
                ]
                row["actual_answer"] = {
                    "predicted": prediction.get("predicted") or [],
                    "hits_at_1": bool(predicted_order and predicted_order[0] in golds),
                    "scores": set_scores(predicted_ids, golds),
                }
            elif prediction is not None:
                row["menu_audit"] = menu_ceiling([], golds, prediction.get("selected"))
            row["failure_stage"] = earliest_failure(row, prediction is not None)
            failure_counts[row["failure_stage"]] += 1
            question_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if position % 25 == 0 or position == len(frozen):
                print(
                    f"      {position}/{len(frozen)} | answerable={sum(r['gold_in_subgraph'] for r in question_rows)} "
                    f"| two-hop={sum(r['structural_ceiling']['best_up_to_two_hop']['scores']['has_gold'] for r in question_rows)}",
                    flush=True,
                )

    print("[3/4] Aggregating bottleneck metrics")
    metrics = aggregate(question_rows, bool(args.predictions))
    write_json(output_dir / "metrics.json", {"args": vars(args), "manifest": manifest, "metrics": metrics})

    print("[4/4] Writing report")
    write_report(output_dir / "report.md", manifest, metrics)
    print(f"      Failure stages: {dict(failure_counts)}")
    print(f"      metrics.json")
    print(f"      report.md")
    print(f"      ceiling_questions.jsonl")
    print(f"      split_manifest.json")
    print(f"      eval_questions.jsonl")
    print(f"Wrote architecture ceiling audit to {output_dir} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
