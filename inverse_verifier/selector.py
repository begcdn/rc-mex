from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .data import (
    Hop,
    PathSpec,
    delexicalize_question,
    normalize_space,
    parse_sparql_path,
    path_to_dict,
    relation_sequence,
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


def run_verifier_pipeline(
    questions_path: Path,
    graphs_path: Path,
    model_path: str,
    retriever_model: str,
    output: Path,
    limit: int,
    device: str = "auto",
) -> dict[str, Any]:
    started = time.time()
    metadata = supported_questions(questions_path)
    encoder = SentenceTransformer(SEMANTIC_MODEL, local_files_only=True)
    generator, tokenizer, model_device = load_seq2seq(model_path, device)
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
                reference_embeddings = encoder.encode(
                    reference_intents, batch_size=VERIFY_BATCH_SIZE, normalize_embeddings=True
                )
                generated_embeddings = encoder.encode(
                    generated_intents, batch_size=VERIFY_BATCH_SIZE, normalize_embeddings=True
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
                        }
                    )
                if max(item["semantic_similarity"] for item in verified) >= ACCEPT_THRESHOLD:
                    stopped_on_threshold = True
                    break

            if not verified:
                print(f"{len(results) + 1}/{limit} {graph_row['id']}: no candidate paths", flush=True)
                continue

            selected = max(verified, key=lambda item: item["semantic_similarity"])
            selected_is_gold = bool(
                selected and tuple(selected["relation_sequence"]) in set(gold_sequences)
            )
            answer_row = answer_metrics(
                selected["answers"] if selected else [], graph_row.get("answer", [])
            )
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
                "selected": selected,
                "top_verified": sorted(
                    verified, key=lambda item: item["semantic_similarity"], reverse=True
                )[:10],
                "false_early_accept": bool(stopped_on_threshold and not selected_is_gold),
                "gold_was_verified": bool(
                    proposal_gold_rank is not None and proposal_gold_rank <= len(verified)
                ),
                "low_confidence_unhandled": bool(
                    selected["semantic_similarity"] < 0.5
                ),
            }
            results.append(result)
            print(
                f"{len(results)}/{limit} {graph_row['id']}: proposal_gold_rank={proposal_gold_rank} "
                f"verified={len(verified)} "
                f"best={(selected['semantic_similarity'] if selected else float('nan')):.3f} "
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
        "accept_threshold": ACCEPT_THRESHOLD,
        "semantic_model": SEMANTIC_MODEL,
        "generator_model": model_path,
        "verification_signal": "generated-question semantic similarity",
        "gold_topic_entity_used": True,
        "gold_path_or_hop_count_used_during_search": False,
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
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
