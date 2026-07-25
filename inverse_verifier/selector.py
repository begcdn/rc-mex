from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .comparator import (
    comparator_answer_text,
    comparator_path_text,
    load_comparator,
    score_comparator_rows,
)
from .data import (
    Hop,
    PathSpec,
    delexicalize_question,
    normalize_space,
    parse_sparql_path,
    path_to_dict,
    relation_sequence,
    unlabeled_answer_count,
    write_jsonl,
)
from .model import generate_joint_questions, load_seq2seq
from .retrieval import (
    MAX_HOPS,
    PATH_CAP,
    LocalQuestionGraph,
    SRTKPathRetriever,
    gold_path_available,
    path_rank,
)


VERIFY_CAP = 100
VERIFY_BATCH_SIZE = 5
ACCEPT_THRESHOLD = 0.85
SEMANTIC_MODEL = "BAAI/bge-small-en-v1.5"
ANSWER_LOG_CAP = 50

# Every variant is scored from the same forward passes, so a single run reports the
# whole selection ablation instead of requiring one run per policy.
SELECTION_VARIANTS: dict[str, dict[str, bool]] = {
    "argmax": {"endpoint_filter": False, "aggregate_answers": False},
    "argmax_filtered": {"endpoint_filter": True, "aggregate_answers": False},
    "vote": {"endpoint_filter": False, "aggregate_answers": True},
    "vote_filtered": {"endpoint_filter": True, "aggregate_answers": True},
}
PRIMARY_SELECTION = "vote_filtered"


def has_answerable_endpoint(candidate: dict[str, Any]) -> bool:
    """True when at least one endpoint carries a surface label a question could name."""
    answers = candidate.get("answers", [])
    return bool(answers) and unlabeled_answer_count(answers) < len(answers)


def answer_set_key(answers: list[str]) -> tuple[str, ...]:
    return tuple(sorted({answer.casefold().strip() for answer in answers if answer.strip()}))


def select_candidate(
    verified: list[dict[str, Any]],
    score_key: str,
    endpoint_filter: bool = True,
    aggregate_answers: bool = True,
) -> dict[str, Any]:
    """Pick one candidate under a selection policy.

    ``endpoint_filter`` drops paths whose endpoints are all unlabeled machine ids.
    ``aggregate_answers`` marginalizes over paths: equivalent Freebase routes that
    return the same answer set reinforce each other instead of splitting the vote.
    Neither policy consults gold.
    """
    pool = verified
    if endpoint_filter:
        answerable = [candidate for candidate in verified if has_answerable_endpoint(candidate)]
        # Reported as endpoint_filter_fallback_rate rather than applied silently.
        pool = answerable or verified
    if not aggregate_answers:
        return max(pool, key=lambda candidate: candidate[score_key])

    ceiling = max(candidate[score_key] for candidate in pool)
    mass: dict[tuple[str, ...], float] = defaultdict(float)
    representative: dict[tuple[str, ...], dict[str, Any]] = {}
    for candidate in pool:
        key = answer_set_key(candidate["answers"])
        mass[key] += math.exp(candidate[score_key] - ceiling)
        best = representative.get(key)
        if best is None or candidate[score_key] > best[score_key]:
            representative[key] = candidate
    winner = max(mass, key=lambda key: (mass[key], representative[key][score_key], key))
    return representative[winner]


def evaluation_subset_coverage(path: Path) -> dict[str, Any]:
    """Record why questions leave the evaluation subset so the slice is never implicit."""
    questions = json.load(path.open(encoding="utf-8"))["Questions"]
    reasons: dict[str, int] = defaultdict(int)
    supported = 0
    for question in questions:
        dropped: set[str] = set()
        kept = False
        for parse in question.get("Parses", []):
            chain = parse.get("InferentialChain") or []
            comment = parse.get("AnnotatorComment") or {}
            if not chain:
                dropped.add("no_inferential_chain")
            elif len(chain) > MAX_HOPS:
                dropped.add("over_max_hops")
            elif comment.get("ParseQuality") == "Incomplete":
                dropped.add("incomplete_parse")
            elif parse.get("Constraints"):
                dropped.add("constraints")
            elif parse.get("Time") is not None:
                dropped.add("time")
            elif parse.get("Order") is not None:
                dropped.add("order")
            else:
                kept = True
        if kept:
            supported += 1
        else:
            for reason in dropped or {"no_parses"}:
                reasons[reason] += 1
    return {
        "source_questions": len(questions),
        "supported_questions": supported,
        "supported_fraction": supported / max(len(questions), 1),
        "excluded_by_reason": dict(sorted(reasons.items())),
    }


def supported_questions(path: Path) -> dict[str, dict[str, Any]]:
    """Select the supported evaluation subset and retain gold only for metrics."""
    output: dict[str, dict[str, Any]] = {}
    for question in json.load(path.open(encoding="utf-8"))["Questions"]:
        sequences = []
        for parse in question.get("Parses", []):
            chain = parse.get("InferentialChain") or []
            comment = parse.get("AnnotatorComment") or {}
            if not 1 <= len(chain) <= MAX_HOPS or comment.get("ParseQuality") == "Incomplete":
                continue
            if parse.get("Constraints") or parse.get("Time") is not None or parse.get("Order") is not None:
                continue
            directed = parse_sparql_path(
                parse.get("Sparql", ""), parse.get("TopicEntityMid", ""), chain
            )
            if directed:
                sequences.append(tuple(f"{relation}::{direction}" for relation, direction in directed))
        if sequences:
            output[question["QuestionId"]] = {
                "question": normalize_space(question["RawQuestion"]),
                "gold_sequences": sorted(set(sequences)),
            }
    return output


def enumerate_path_families(graph_row: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate executable path families and retain every endpoint binding."""
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    type_map: dict[str, set[str]] = defaultdict(set)
    for head, relation, tail in graph_row.get("graph", []):
        adjacency[head].append((tail, relation, "forward"))
        adjacency[tail].append((head, relation, "backward"))
        if relation == "common.topic.notable_types" and not tail.startswith(("m.", "g.")):
            type_map[head].add(tail)

    families: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for anchor in graph_row.get("q_entity", []):
        anchor_type = " / ".join(sorted(type_map.get(anchor, set()))) or "entity"
        frontier = [(anchor, tuple(), frozenset({anchor}))]
        for _ in range(MAX_HOPS):
            next_frontier = []
            for node, hops, seen_nodes in frontier:
                for target, relation, direction in adjacency.get(node, []):
                    if relation == "common.topic.notable_types" or target in seen_nodes:
                        continue
                    source_type = (
                        " / ".join(sorted(type_map.get(node, set())))
                        or (hops[-1].target_type if hops else anchor_type)
                    )
                    target_type = " / ".join(sorted(type_map.get(target, set()))) or "entity"
                    new_hops = hops + (Hop(relation, direction, source_type, target_type),)
                    spec = PathSpec(anchor, anchor_type, new_hops, target_type, "webqsp")
                    sequence = relation_sequence(spec)
                    key = (anchor, sequence)
                    family = families.setdefault(
                        key,
                        {
                            "path": path_to_dict(spec),
                            "relation_sequence": list(sequence),
                            "answers": set(),
                        },
                    )
                    family["answers"].add(target)
                    next_frontier.append((target, new_hops, seen_nodes | {target}))
            frontier = next_frontier
    return [
        {**family, "answers": sorted(family["answers"])} for family in families.values()
    ]


def answer_metrics(predicted: list[str], gold: list[str]) -> dict[str, float]:
    predicted_set = {value.casefold().strip() for value in predicted if value.strip()}
    gold_set = {value.casefold().strip() for value in gold if value.strip()}
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": float(predicted_set == gold_set and bool(gold_set)),
        "has_correct_answer": float(bool(predicted_set & gold_set)),
    }


def first_gold_rank(
    families: list[dict[str, Any]], gold_sequences: list[tuple[str, ...]]
) -> int | None:
    return path_rank(families, gold_sequences)


def candidate_log_entry(
    candidate: dict[str, Any],
    score_key: str,
    gold_sequences: set[tuple[str, ...]],
    gold_answers: set[str],
) -> dict[str, Any]:
    """One compact audit record per verified candidate.

    ``matches_gold_path`` and ``answer_overlaps_gold`` are evaluation-only fields;
    nothing in the selection path reads them.
    """
    answers = candidate.get("answers", [])
    return {
        "relation_sequence": candidate["relation_sequence"],
        "generated_question": candidate["generated_question"],
        "retrieval_score": candidate.get("retrieval_score"),
        "score": candidate[score_key],
        "answer_count": len(answers),
        "unlabeled_answer_count": unlabeled_answer_count(answers),
        "answers": answers[:ANSWER_LOG_CAP],
        "answers_truncated": len(answers) > ANSWER_LOG_CAP,
        "matches_gold_path": tuple(candidate["relation_sequence"]) in gold_sequences,
        "answer_overlaps_gold": bool(
            {answer.casefold().strip() for answer in answers} & gold_answers
        ),
    }


def run_verifier_pipeline(
    questions_path: Path,
    graphs_path: Path,
    model_path: str,
    retriever_model: str,
    output: Path,
    limit: int,
    device: str = "auto",
    comparison_mode: str = "cosine",
    comparator_model: str | None = None,
) -> dict[str, Any]:
    if comparison_mode not in {"cosine", "cross_encoder"}:
        raise ValueError(f"unknown comparison mode: {comparison_mode}")
    if comparison_mode == "cross_encoder" and not comparator_model:
        raise ValueError("cross_encoder comparison requires comparator_model")
    started = time.time()
    metadata = supported_questions(questions_path)
    encoder = (
        SentenceTransformer(SEMANTIC_MODEL, local_files_only=True)
        if comparison_mode == "cosine"
        else None
    )
    generator, tokenizer, model_device = load_seq2seq(model_path, device)
    cross_encoder = cross_tokenizer = cross_device = cross_mode = None
    if comparison_mode == "cross_encoder":
        cross_encoder, cross_tokenizer, cross_device, cross_mode = load_comparator(
            comparator_model or "", device
        )
    retriever = SRTKPathRetriever(retriever_model, device)
    results = []
    with graphs_path.open(encoding="utf-8") as handle:
        for line in handle:
            graph_row = json.loads(line)
            question_metadata = metadata.get(graph_row["id"])
            if question_metadata is None:
                continue
            question = question_metadata["question"]
            retrieval = retriever.retrieve(question, graph_row)
            proposals = retrieval["candidate_paths"]
            gold_sequences = [tuple(sequence) for sequence in question_metadata["gold_sequences"]]
            proposal_gold_rank = first_gold_rank(proposals, gold_sequences)
            available = gold_path_available(
                LocalQuestionGraph(graph_row.get("graph", [])),
                graph_row.get("q_entity", []),
                gold_sequences,
            )

            verified = []
            stopped_on_threshold = False
            for start in range(0, min(len(proposals), VERIFY_CAP), VERIFY_BATCH_SIZE):
                batch = proposals[start : start + VERIFY_BATCH_SIZE]
                generated = generate_joint_questions(
                    generator,
                    tokenizer,
                    [family["path"] for family in batch],
                    model_device,
                    batch_size=VERIFY_BATCH_SIZE,
                )
                reference_intents = [
                    delexicalize_question(question, family["path"]["anchor"])
                    for family in batch
                ]
                generated_intents = [
                    delexicalize_question(text, family["path"]["anchor"])
                    for text, family in zip(generated, batch, strict=True)
                ]
                similarities = [float("nan")] * len(batch)
                if encoder is not None:
                    reference_embeddings = encoder.encode(
                        reference_intents,
                        batch_size=VERIFY_BATCH_SIZE,
                        normalize_embeddings=True,
                    )
                    generated_embeddings = encoder.encode(
                        generated_intents,
                        batch_size=VERIFY_BATCH_SIZE,
                        normalize_embeddings=True,
                    )
                    similarities = np.sum(
                        reference_embeddings * generated_embeddings, axis=1
                    ).tolist()
                for family, generated_question, similarity in zip(
                    batch, generated, similarities, strict=True
                ):
                    verified.append(
                        {
                            **family,
                            "generated_question": generated_question,
                            "semantic_similarity": float(similarity),
                            "path_text": comparator_path_text(
                                family["path"],
                                ", ".join(family.get("answers", [])[:10]),
                                getattr(generator, "_relation_glossary", None),
                            ),
                            "answer_text": comparator_answer_text(
                                family.get("answers", []),
                                family["path"].get("answer_type"),
                                unlabeled_answer_count(family.get("answers", [])),
                            ),
                        }
                    )
                if (
                    comparison_mode == "cosine"
                    and max(item["semantic_similarity"] for item in verified)
                    >= ACCEPT_THRESHOLD
                ):
                    stopped_on_threshold = True
                    break

            if not verified:
                print(f"{len(results) + 1}/{limit} {graph_row['id']}: no candidate paths", flush=True)
                continue

            score_key = "semantic_similarity"
            if comparison_mode == "cross_encoder":
                comparator_row = {
                    "example_id": graph_row["id"],
                    "original_question": question,
                    "candidates": [
                        {
                            **candidate,
                            "is_positive": False,
                            "negative_type": "unlabeled",
                        }
                        for candidate in verified
                    ],
                }
                scored = score_comparator_rows(
                    cross_encoder,
                    cross_tokenizer,
                    [comparator_row],
                    cross_mode,
                    cross_device,
                    batch_size=1,
                )[0]["candidates"]
                verified = [
                    {
                        **candidate,
                        "cross_encoder_score": scored_candidate[
                            "cross_encoder_score"
                        ],
                    }
                    for candidate, scored_candidate in zip(
                        verified, scored, strict=True
                    )
                ]
                score_key = "cross_encoder_score"

            gold_sequence_set = {tuple(sequence) for sequence in gold_sequences}
            gold_answers = graph_row.get("answer", [])
            gold_answer_set = {answer.casefold().strip() for answer in gold_answers}
            variants = {}
            for name, policy in SELECTION_VARIANTS.items():
                choice = select_candidate(verified, score_key, **policy)
                variants[name] = {
                    "relation_sequence": choice["relation_sequence"],
                    "generated_question": choice["generated_question"],
                    "score": choice[score_key],
                    "answer_count": len(choice["answers"]),
                    "selected_is_gold_path": tuple(choice["relation_sequence"])
                    in gold_sequence_set,
                    "answer_metrics": answer_metrics(choice["answers"], gold_answers),
                }
            endpoint_filter_fell_back = not any(
                has_answerable_endpoint(candidate) for candidate in verified
            )

            selected = select_candidate(
                verified, score_key, **SELECTION_VARIANTS[PRIMARY_SELECTION]
            )
            selected_is_gold = variants[PRIMARY_SELECTION]["selected_is_gold_path"]
            answer_row = variants[PRIMARY_SELECTION]["answer_metrics"]
            result = {
                "question_id": graph_row["id"],
                "question": question,
                "topic_entities": graph_row.get("q_entity", []),
                "gold_answers": graph_row.get("answer", []),
                "gold_sequences": question_metadata["gold_sequences"],
                "path_families_generated": len(proposals),
                "proposal_gold_rank": proposal_gold_rank,
                "gold_path_in_available_graph": available,
                "retrieved_subgraph": retrieval["retrieved_subgraph"],
                "paths_verified": len(verified),
                "stopped_on_threshold": stopped_on_threshold,
                "selected_is_gold_path": selected_is_gold,
                "answer_metrics": answer_row,
                "selection_policy": PRIMARY_SELECTION,
                "selection_variants": variants,
                "endpoint_filter_fell_back": endpoint_filter_fell_back,
                "selected": selected,
                "top_verified": sorted(
                    verified, key=lambda item: item[score_key], reverse=True
                )[:10],
                "candidate_log": [
                    candidate_log_entry(
                        candidate, score_key, gold_sequence_set, gold_answer_set
                    )
                    for candidate in sorted(
                        verified, key=lambda item: item[score_key], reverse=True
                    )
                ],
                "false_early_accept": bool(stopped_on_threshold and not selected_is_gold),
                "gold_was_verified": bool(
                    proposal_gold_rank is not None and proposal_gold_rank <= len(verified)
                ),
                "low_confidence_unhandled": bool(
                    comparison_mode == "cosine"
                    and selected["semantic_similarity"] < 0.5
                ),
            }
            results.append(result)
            print(
                f"{len(results)}/{limit} {graph_row['id']}: proposal_gold_rank={proposal_gold_rank} "
                f"verified={len(verified)} "
                f"best={(selected[score_key] if selected else float('nan')):.3f} "
                f"path_correct={selected_is_gold}",
                flush=True,
            )
            if len(results) >= limit:
                break

    recall_cutoffs = (5, 10, 20, 50, 100, 200)
    count = len(results)
    metrics: dict[str, Any] = {
        "questions": count,
        "retriever": "SRTK iterative relation-path retrieval",
        "retriever_model": retriever_model,
        "proposal_cap": PATH_CAP,
        "verify_cap": VERIFY_CAP,
        "verify_batch_size": VERIFY_BATCH_SIZE,
        "accept_threshold": ACCEPT_THRESHOLD if comparison_mode == "cosine" else None,
        "semantic_model": SEMANTIC_MODEL if comparison_mode == "cosine" else None,
        "comparator_model": comparator_model,
        "generator_model": model_path,
        "verification_signal": comparison_mode,
        "gold_topic_entity_used": True,
        "gold_path_or_hop_count_used_during_search": False,
        "selection_policy": PRIMARY_SELECTION,
        "evaluation_subset": evaluation_subset_coverage(questions_path),
        "selection_variants": {
            name: {
                "selected_gold_path_accuracy": sum(
                    row["selection_variants"][name]["selected_is_gold_path"]
                    for row in results
                )
                / max(count, 1),
                **{
                    f"answer_{metric}": sum(
                        row["selection_variants"][name]["answer_metrics"][metric]
                        for row in results
                    )
                    / max(count, 1)
                    for metric in ("f1", "exact_match", "has_correct_answer")
                },
            }
            for name in SELECTION_VARIANTS
        },
        "unanswerable_endpoint_selection_rate": sum(
            not has_answerable_endpoint(row["selected"]) for row in results
        )
        / max(count, 1),
        "endpoint_filter_fallback_rate": sum(
            row["endpoint_filter_fell_back"] for row in results
        )
        / max(count, 1),
        "raw_path_recall": sum(row["proposal_gold_rank"] is not None for row in results)
        / max(count, 1),
        "proposal_recall": {
            f"recall_at_{cutoff}": sum(
                row["proposal_gold_rank"] is not None
                and row["proposal_gold_rank"] <= cutoff
                for row in results
            ) / max(count, 1)
            for cutoff in recall_cutoffs
        },
        "gold_path_availability_in_supplied_graph": sum(
            row["gold_path_in_available_graph"] for row in results
        )
        / max(count, 1),
        "proposal_recall_given_available": sum(
            row["proposal_gold_rank"] is not None for row in results
        )
        / max(sum(row["gold_path_in_available_graph"] for row in results), 1),
        "selected_gold_path_accuracy": sum(row["selected_is_gold_path"] for row in results)
        / max(count, 1),
        "gold_verified_rate": sum(row["gold_was_verified"] for row in results) / max(count, 1),
        "threshold_stop_rate": sum(row["stopped_on_threshold"] for row in results) / max(count, 1),
        "false_early_accept_rate": sum(row["false_early_accept"] for row in results)
        / max(count, 1),
        "low_confidence_rate": sum(row["low_confidence_unhandled"] for row in results)
        / max(count, 1),
        "average_paths_verified": sum(row["paths_verified"] for row in results) / max(count, 1),
        "average_path_families": sum(row["path_families_generated"] for row in results)
        / max(count, 1),
        "average_subgraph_triples": sum(len(row["retrieved_subgraph"]) for row in results)
        / max(count, 1),
        **{
            f"answer_{name}": sum(row["answer_metrics"][name] for row in results) / max(count, 1)
            for name in ("precision", "recall", "f1", "exact_match", "has_correct_answer")
        },
        "elapsed_seconds": time.time() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", results)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# High-Recall Path Proposal and Inverse Verification",
        "",
        "The candidate generator uses SRTK's released iterative relation-path retriever over the controlled topic-entity neighborhood. It returns a wide path set and their union subgraph. The inverse generator then checks paths in batches of five. Gold paths and hop counts are evaluation-only.",
        "",
        f"- Questions: {count}",
        f"- Raw path recall: {metrics['raw_path_recall']:.3f}",
        *[
            f"- Recall@{cutoff}: {metrics['proposal_recall'][f'recall_at_{cutoff}']:.3f}"
            for cutoff in recall_cutoffs
        ],
        f"- Gold path verified before stopping: {metrics['gold_verified_rate']:.3f}",
        f"- Selected gold path: {metrics['selected_gold_path_accuracy']:.3f}",
        f"- False early accept: {metrics['false_early_accept_rate']:.3f}",
        f"- Average paths verified: {metrics['average_paths_verified']:.1f}",
        f"- Answer F1: {metrics['answer_f1']:.3f}",
        f"- Answer exact match: {metrics['answer_exact_match']:.3f}",
        f"- Runtime: {metrics['elapsed_seconds']:.1f}s",
        "",
        "## Selection ablation",
        "",
        f"All variants reuse the same generator and comparator scores; `{PRIMARY_SELECTION}` is reported above.",
        "",
        "| Selection | Selected gold path | Answer EM | Answer F1 |",
        "|---|---:|---:|---:|",
        *[
            f"| `{name}` | {values['selected_gold_path_accuracy']:.3f} | "
            f"{values['answer_exact_match']:.3f} | {values['answer_f1']:.3f} |"
            for name, values in metrics["selection_variants"].items()
        ],
        "",
        "## Evaluation subset",
        "",
        f"- Source questions: {metrics['evaluation_subset']['source_questions']}",
        f"- Supported: {metrics['evaluation_subset']['supported_questions']} "
        f"({metrics['evaluation_subset']['supported_fraction']:.3f})",
        *[
            f"- Excluded ({reason}): {value}"
            for reason, value in metrics["evaluation_subset"]["excluded_by_reason"].items()
        ],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
