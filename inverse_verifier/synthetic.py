from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .data import Hop, PathSpec, path_to_dict, relation_sequence, relation_words, stable_bucket, write_jsonl


METADATA_RELATION_PARTS = (
    "base.kwebbase",
    "common.topic.notable_types",
    "common.topic.notable_for",
    "common.topic.subject_of",
    "common.image",
    "freebase.valuenotation",
    "wikidata property",
    "external id",
    "identifier",
    "described by source",
    "imported from",
    "url",
)
DEPTHS = (1, 2, 3)
NEGATIVE_CAP = 6


@dataclass(frozen=True)
class Edge:
    target: str
    relation: str
    direction: str


@dataclass(frozen=True)
class ConcretePath:
    spec: PathSpec
    nodes: tuple[str, ...]


@dataclass
class ExecutableGraph:
    adjacency: dict[str, list[Edge]]
    labels: dict[str, str]
    types: dict[str, str]
    kg: str

    def type_of(self, node: str) -> str:
        return self.types.get(node, "entity") or "entity"

    def label_of(self, node: str) -> str:
        return self.labels.get(node, node)


def semantic_relation(relation: str) -> bool:
    lowered = relation.casefold()
    return not any(part in lowered for part in METADATA_RELATION_PARTS)


def canonical_question(path: dict[str, Any] | PathSpec, style: int = 0) -> str:
    """Produce a controlled question that cannot silently omit a path hop."""
    if isinstance(path, PathSpec):
        path = path_to_dict(path)
    answer_type = path.get("answer_type") or "entity"
    steps = [
        f'follow "{relation_words(hop["relation"])}" {hop["direction"]} '
        f'to a {hop.get("target_type") or "entity"}'
        for hop in path["hops"]
    ]
    chain = ", then ".join(steps)
    if style % 3 == 0:
        return f"What {answer_type} do you reach from [ENTITY] if you {chain}?"
    if style % 3 == 1:
        return f"Starting at [ENTITY], which {answer_type} results when you {chain}?"
    return f"Which {answer_type} is connected to [ENTITY] by this traversal: {chain}?"


def question_covers_path(question: str, path: dict[str, Any] | PathSpec) -> bool:
    if isinstance(path, PathSpec):
        path = path_to_dict(path)
    lowered = question.casefold()
    return all(
        relation_words(hop["relation"]).casefold() in lowered
        and hop["direction"].casefold() in lowered
        for hop in path["hops"]
    )


def _path_key(path: ConcretePath) -> tuple[str, tuple[str, ...], str, str]:
    return (
        path.spec.kg,
        relation_sequence(path.spec),
        path.spec.anchor_type,
        path.spec.answer_type,
    )


def _make_path(graph: ExecutableGraph, nodes: list[str], edges: list[Edge]) -> ConcretePath:
    hops = []
    for index, edge in enumerate(edges):
        hops.append(
            Hop(
                edge.relation,
                edge.direction,
                graph.type_of(nodes[index]),
                graph.type_of(nodes[index + 1]),
            )
        )
    spec = PathSpec(
        graph.label_of(nodes[0]),
        graph.type_of(nodes[0]),
        tuple(hops),
        graph.type_of(nodes[-1]),
        graph.kg,
    )
    return ConcretePath(spec, tuple(nodes))


def random_walk(
    graph: ExecutableGraph,
    anchor: str,
    depth: int,
    rng: random.Random,
) -> ConcretePath | None:
    nodes = [anchor]
    edges: list[Edge] = []
    for _ in range(depth):
        choices = [
            edge
            for edge in graph.adjacency.get(nodes[-1], [])
            if semantic_relation(edge.relation) and edge.target not in nodes
            and not (
                edges
                and edge.relation == edges[-1].relation
                and edge.direction != edges[-1].direction
            )
        ]
        if not choices:
            return None
        edge = rng.choice(choices)
        edges.append(edge)
        nodes.append(edge.target)
    return _make_path(graph, nodes, edges)


def executable_negatives(
    graph: ExecutableGraph,
    positive: ConcretePath,
    rng: random.Random,
) -> list[tuple[ConcretePath, str]]:
    negatives: list[tuple[ConcretePath, str]] = []
    seen = {_path_key(positive)}

    def add(candidate: ConcretePath | None, category: str) -> None:
        if candidate is None or _path_key(candidate) in seen:
            return
        seen.add(_path_key(candidate))
        negatives.append((candidate, category))

    if len(positive.spec.hops) > 1:
        add(
            ConcretePath(
                PathSpec(
                    positive.spec.anchor,
                    positive.spec.anchor_type,
                    positive.spec.hops[:-1],
                    positive.spec.hops[-2].target_type,
                    positive.spec.kg,
                ),
                positive.nodes[:-1],
            ),
            "missing_hop",
        )

    endpoint = positive.nodes[-1]
    extension_choices = [
        edge
        for edge in graph.adjacency.get(endpoint, [])
        if semantic_relation(edge.relation) and edge.target not in positive.nodes
    ]
    if extension_choices:
        extension = rng.choice(extension_choices)
        add(
            _make_path(
                graph,
                list(positive.nodes) + [extension.target],
                [
                    Edge(positive.nodes[index + 1], hop.relation, hop.direction)
                    for index, hop in enumerate(positive.spec.hops)
                ]
                + [extension],
            ),
            "added_hop",
        )

    for divergence in range(len(positive.spec.hops)):
        prefix_nodes = list(positive.nodes[: divergence + 1])
        original = positive.spec.hops[divergence]
        choices = [
            edge
            for edge in graph.adjacency.get(prefix_nodes[-1], [])
            if semantic_relation(edge.relation)
            and edge.target not in prefix_nodes
            and (edge.relation, edge.direction)
            != (original.relation, original.direction)
        ]
        rng.shuffle(choices)
        for alternate in choices[:2]:
            nodes = prefix_nodes + [alternate.target]
            edges = [
                Edge(positive.nodes[index + 1], hop.relation, hop.direction)
                for index, hop in enumerate(positive.spec.hops[:divergence])
            ] + [alternate]
            while len(edges) < len(positive.spec.hops):
                continuation = [
                    edge
                    for edge in graph.adjacency.get(nodes[-1], [])
                    if semantic_relation(edge.relation) and edge.target not in nodes
                ]
                if not continuation:
                    break
                edge = rng.choice(continuation)
                edges.append(edge)
                nodes.append(edge.target)
            if len(edges) == len(positive.spec.hops):
                category = (
                    "wrong_direction"
                    if alternate.relation == original.relation
                    else "sibling_relation"
                )
                add(_make_path(graph, nodes, edges), category)
            if len(negatives) >= NEGATIVE_CAP:
                return negatives
    return negatives


def example_from_path(
    graph: ExecutableGraph,
    path: ConcretePath,
    index: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    endpoint = path.nodes[-1]
    if graph.type_of(endpoint) == "entity" and graph.label_of(endpoint).startswith(("m.", "g.")):
        return None
    question = canonical_question(path.spec, stable_bucket(f"positive:{graph.kg}:{index}", 3))
    negatives = []
    for negative, category in executable_negatives(graph, path, rng):
        negative_data = path_to_dict(negative.spec)
        negative_data["negative_type"] = category
        negative_data["question"] = canonical_question(
            negative.spec,
            stable_bucket(f"negative:{graph.kg}:{index}:{category}", 3),
        )
        negative_data["answer_entity"] = graph.label_of(negative.nodes[-1])
        negatives.append(negative_data)
    if not negatives or not question_covers_path(question, path.spec):
        return None
    return {
        "example_id": f"synthetic:{graph.kg}:{index}",
        "question": question,
        "positive_path": path_to_dict(path.spec),
        "positive_answer_entity": graph.label_of(path.nodes[-1]),
        "alternate_positive_paths": [],
        "negative_paths": negatives,
        "split": "synthetic",
        "kg": graph.kg,
        "relation_sequence": list(relation_sequence(path.spec)),
        "source_kind": "executable_synthetic",
    }


def load_kqa_graph(path: Path) -> ExecutableGraph:
    kb = json.load(path.open(encoding="utf-8"))
    concept_names = {
        concept_id: concept.get("name", concept_id)
        for concept_id, concept in kb.get("concepts", {}).items()
    }
    labels = {
        entity_id: entity.get("name", entity_id)
        for entity_id, entity in kb.get("entities", {}).items()
    }
    types = {
        entity_id: " / ".join(
            sorted({concept_names.get(value, value) for value in entity.get("instanceOf", [])})
        )
        or "entity"
        for entity_id, entity in kb.get("entities", {}).items()
    }
    adjacency: dict[str, list[Edge]] = defaultdict(list)
    triples = set()
    for entity_id, entity in kb.get("entities", {}).items():
        for relation in entity.get("relations", []):
            if relation.get("direction") == "forward":
                head, tail = entity_id, relation["object"]
            else:
                head, tail = relation["object"], entity_id
            triples.add((head, relation["predicate"], tail))
    for head, relation, tail in triples:
        if head not in labels or tail not in labels:
            continue
        adjacency[head].append(Edge(tail, relation, "forward"))
        adjacency[tail].append(Edge(head, relation, "backward"))
    return ExecutableGraph(dict(adjacency), labels, types, "kqa_pro")


def webqsp_graph(row: dict[str, Any]) -> ExecutableGraph:
    adjacency: dict[str, list[Edge]] = defaultdict(list)
    types: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    triples = set()
    for head, relation, tail in row.get("graph", []):
        labels[head] = head
        labels[tail] = tail
        if relation == "common.topic.notable_types" and not tail.startswith(("m.", "g.")):
            types[head].add(tail)
            continue
        triples.add((head, relation, tail))
    for head, relation, tail in triples:
        adjacency[head].append(Edge(tail, relation, "forward"))
        adjacency[tail].append(Edge(head, relation, "backward"))
    return ExecutableGraph(
        dict(adjacency),
        labels,
        {node: " / ".join(sorted(values)) for node, values in types.items()},
        "webqsp",
    )


def _collect_paths(
    graph: ExecutableGraph,
    anchors: Iterable[str],
    targets: dict[int, int],
    rng: random.Random,
    seen: set[tuple[str, tuple[str, ...], str, str]],
    attempts_per_anchor: int = 4,
) -> list[ConcretePath]:
    collected: list[ConcretePath] = []
    counts: Counter[int] = Counter()
    anchors = list(anchors)
    rng.shuffle(anchors)
    for anchor in anchors:
        for _ in range(attempts_per_anchor):
            needed = [depth for depth in DEPTHS if counts[depth] < targets[depth]]
            if not needed:
                return collected
            depth = rng.choice(needed)
            path = random_walk(graph, anchor, depth, rng)
            if path is None or _path_key(path) in seen:
                continue
            seen.add(_path_key(path))
            collected.append(path)
            counts[depth] += 1
    return collected


def synthesize_corpus(
    kqa_kb: Path,
    webqsp_graphs: Path,
    output: Path,
    total_paths: int = 30_000,
    seed: int = 17,
) -> dict[str, Any]:
    rng = random.Random(seed)
    per_kg = total_paths // 2
    targets = {depth: per_kg // len(DEPTHS) for depth in DEPTHS}
    targets[DEPTHS[-1]] += per_kg - sum(targets.values())
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str, str]] = set()

    kqa = load_kqa_graph(kqa_kb)
    kqa_paths = _collect_paths(kqa, kqa.adjacency, targets, rng, seen, attempts_per_anchor=12)
    for path in kqa_paths:
        example = example_from_path(kqa, path, len(examples), rng)
        if example:
            examples.append(example)

    web_counts = {depth: 0 for depth in DEPTHS}
    with webqsp_graphs.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            graph = webqsp_graph(row)
            remaining = {depth: max(targets[depth] - web_counts[depth], 0) for depth in DEPTHS}
            if not any(remaining.values()):
                break
            paths = _collect_paths(
                graph,
                row.get("q_entity", []),
                remaining,
                rng,
                seen,
                attempts_per_anchor=12,
            )
            for path in paths:
                example = example_from_path(graph, path, len(examples), rng)
                if example:
                    examples.append(example)
                    web_counts[len(path.spec.hops)] += 1

    rng.shuffle(examples)
    train, dev = [], []
    for example in examples:
        sequence = "|".join(example["relation_sequence"])
        destination = dev if stable_bucket(f"synthetic-dev:{sequence}", 10) == 0 else train
        example["split"] = "dev_faithful" if destination is dev else "train_faithful"
        destination.append(example)

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train_faithful.jsonl", train)
    write_jsonl(output / "dev_faithful.jsonl", dev)
    counts = {
        "total": len(examples),
        "train": len(train),
        "dev": len(dev),
        "by_kg": {
            kg: sum(example["kg"] == kg for example in examples)
            for kg in sorted({example["kg"] for example in examples})
        },
        "by_depth": {
            str(depth): sum(len(example["positive_path"]["hops"]) == depth for example in examples)
            for depth in DEPTHS
        },
        "negative_types": {
            category: sum(
                negative["negative_type"] == category
                for example in examples
                for negative in example["negative_paths"]
            )
            for category in sorted(
                {
                    negative["negative_type"]
                    for example in examples
                    for negative in example["negative_paths"]
                }
            )
        },
        "all_questions_cover_every_hop": all(
            question_covers_path(example["question"], example["positive_path"])
            and all(
                question_covers_path(negative["question"], negative)
                for negative in example["negative_paths"]
            )
            for example in examples
        ),
        "seed": seed,
        "requested_paths": total_paths,
    }
    (output / "manifest.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


def _candidate_payload(identifier: str, question: str, path: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": identifier,
        "canonical_question": question,
        "answer_type": path.get("answer_type", "entity"),
        "hops": [
            {
                "relation": relation_words(hop["relation"]),
                "direction": hop["direction"],
                "source_type": hop.get("source_type", "entity"),
                "target_type": hop.get("target_type", "entity"),
            }
            for hop in path["hops"]
        ],
    }
    if path.get("explicit_query"):
        payload["explicit_query"] = path["explicit_query"]
    return payload


def naturalization_prompt(rows: list[dict[str, Any]]) -> str:
    candidates = []
    for row_index, row in enumerate(rows):
        candidates.append(
            _candidate_payload(f"{row_index}:positive", row["question"], row["positive_path"])
        )
        for negative_index, negative in enumerate(row["negative_paths"]):
            candidates.append(
                _candidate_payload(
                    f"{row_index}:negative:{negative_index}",
                    negative["question"],
                    negative,
                )
            )
    return (
        "/no_think\n"
        "Rewrite each controlled knowledge-graph question as one concise, natural English "
        "question. Preserve the complete relation composition and requested endpoint. A forward "
        "hop means the current entity is the relation subject; a backward hop means the next "
        "entity is the relation subject. Every hop must affect the meaning. Never silently omit "
        "an added, reversed, or intermediate hop. Do not mention graph traversal, hop numbers, "
        "forward/backward, IDs, or the answer. Keep [ENTITY] exactly as written. Distinct paths "
        "must receive semantically distinct questions. When explicit_query is present, its "
        "subject/object facts and return variable are authoritative. Never replace a specific "
        "fact with vague associated, related, connected, or 'through the relation' wording.\n"
        "Return only JSON with this shape: "
        '{"items":[{"id":"...","question":"...","covered_hops":'
        '[{"relation":"...","direction":"forward"}]}]}. '
        "In covered_hops, copy every supplied relation and direction exactly and in order.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("naturalizer returned no JSON object")
    return json.loads(text[start : end + 1])


def _ollama_naturalize(
    rows: list[dict[str, Any]],
    model: str,
    host: str,
) -> dict[str, str]:
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": naturalization_prompt(rows)}],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 2048},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    parsed = _extract_json(payload["message"]["content"])
    expected = {}
    for row_index, row in enumerate(rows):
        expected[f"{row_index}:positive"] = [
            (relation_words(hop["relation"]), hop["direction"])
            for hop in row["positive_path"]["hops"]
        ]
        for negative_index, negative in enumerate(row["negative_paths"]):
            expected[f"{row_index}:negative:{negative_index}"] = [
                (relation_words(hop["relation"]), hop["direction"])
                for hop in negative["hops"]
            ]
    accepted = {}
    for item in parsed.get("items", []):
        identifier = item.get("id")
        covered = [
            (hop.get("relation", ""), hop.get("direction", ""))
            for hop in item.get("covered_hops", [])
        ]
        if identifier in expected and covered == expected[identifier] and item.get("question"):
            accepted[identifier] = item["question"].strip()
    return accepted


def naturalize_corpus(
    source: Path,
    output: Path,
    model: str = "qwen3:8b",
    host: str | list[str] = "http://127.0.0.1:11434",
    batch_size: int = 4,
) -> dict[str, Any]:
    hosts = [host] if isinstance(host, str) else host
    hosts = [item.rstrip("/") for item in hosts if item]
    if not hosts:
        raise ValueError("at least one Ollama host is required")

    output.mkdir(parents=True, exist_ok=True)
    totals = {"rows": 0, "naturalized_questions": 0, "canonical_fallbacks": 0}
    for filename in ("train_faithful.jsonl", "dev_faithful.jsonl"):
        rows = [json.loads(line) for line in (source / filename).open(encoding="utf-8")]
        destination = output / filename
        completed_rows = []
        if destination.exists():
            completed_rows = [
                json.loads(line) for line in destination.open(encoding="utf-8") if line.strip()
            ]
            totals["rows"] += len(completed_rows)
            completed_naturalized = sum(
                int(row.get("question") != row.get("canonical_question"))
                + sum(
                    int(negative.get("question") != negative.get("canonical_question"))
                    for negative in row["negative_paths"]
                )
                for row in completed_rows
            )
            totals["naturalized_questions"] += completed_naturalized
            totals["canonical_fallbacks"] += sum(
                1 + len(row["negative_paths"]) for row in completed_rows
            ) - completed_naturalized
        pending = list(range(len(completed_rows), len(rows), batch_size))
        with destination.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
            max_workers=len(hosts)
        ) as executor:
            for wave_start in range(0, len(pending), len(hosts)):
                wave = pending[wave_start : wave_start + len(hosts)]
                jobs = []
                for worker_index, start in enumerate(wave):
                    batch = rows[start : start + batch_size]
                    jobs.append(
                        (
                            start,
                            batch,
                            hosts[worker_index],
                            executor.submit(
                                _naturalize_batch,
                                batch,
                                model,
                                hosts[worker_index],
                                start,
                            ),
                        )
                    )

                for start, batch, worker_host, future in jobs:
                    generated = future.result()
                    for row_index, row in enumerate(batch):
                        row["canonical_question"] = row["question"]
                        positive_id = f"{row_index}:positive"
                        if positive_id in generated:
                            row["question"] = generated[positive_id]
                            totals["naturalized_questions"] += 1
                        else:
                            totals["canonical_fallbacks"] += 1
                        for negative_index, negative in enumerate(row["negative_paths"]):
                            negative["canonical_question"] = negative["question"]
                            negative_id = f"{row_index}:negative:{negative_index}"
                            if negative_id in generated:
                                negative["question"] = generated[negative_id]
                                totals["naturalized_questions"] += 1
                            else:
                                totals["canonical_fallbacks"] += 1
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        totals["rows"] += 1
                    handle.flush()
                    print(
                        f"{filename}: {min(start + len(batch), len(rows))}/{len(rows)} rows "
                        f"via {worker_host}",
                        flush=True,
                    )
    totals.update(
        {"model": model, "source": str(source), "ollama_hosts": hosts, "workers": len(hosts)}
    )
    (output / "manifest.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return totals


def _naturalize_batch(
    batch: list[dict[str, Any]],
    model: str,
    host: str,
    start: int,
    id_offset: int = 0,
) -> dict[str, str]:
    for attempt in range(2):
        try:
            generated = _ollama_naturalize(batch, model, host)
            return _offset_generated_ids(generated, id_offset)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request failed at row {start} via {host}: {exc}"
            ) from exc
        except Exception as exc:
            if attempt == 1 and len(batch) == 1:
                print(
                    f"Naturalization fallback for single row {start} via {host}: {exc}",
                    flush=True,
                )
                return {}
    midpoint = len(batch) // 2
    print(
        f"Malformed response for rows {start}-{start + len(batch) - 1} via {host}; "
        "retrying as smaller requests",
        flush=True,
    )
    left = _naturalize_batch(batch[:midpoint], model, host, start, id_offset)
    right = _naturalize_batch(
        batch[midpoint:],
        model,
        host,
        start + midpoint,
        id_offset + midpoint,
    )
    return left | right


def _offset_generated_ids(generated: dict[str, str], offset: int) -> dict[str, str]:
    if offset == 0:
        return generated
    shifted = {}
    for identifier, question in generated.items():
        row_index, separator, suffix = identifier.partition(":")
        if separator and row_index.isdigit():
            shifted[f"{int(row_index) + offset}:{suffix}"] = question
    return shifted


def evaluate_faithful_generation(
    data: Path,
    model_path: str,
    output: Path,
    semantic_model: str = "BAAI/bge-small-en-v1.5",
    limit: int = 500,
    batch_size: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from .data import delexicalize_question
    from .model import generate_joint_questions, load_seq2seq

    rows = [json.loads(line) for line in data.open(encoding="utf-8")][:limit]
    generator, tokenizer, model_device = load_seq2seq(model_path, device)
    encoder = SentenceTransformer(semantic_model, local_files_only=True)
    predictions = []
    category_results: dict[str, list[bool]] = defaultdict(list)
    positive_scores, negative_scores = [], []
    for row_index, row in enumerate(rows, 1):
        candidates = [
            {"path": row["positive_path"], "category": "positive", "is_positive": True}
        ] + [
            {"path": negative, "category": negative["negative_type"], "is_positive": False}
            for negative in row["negative_paths"]
        ]
        generated = generate_joint_questions(
            generator,
            tokenizer,
            [candidate["path"] for candidate in candidates],
            model_device,
            batch_size=batch_size,
        )
        reference = row["question"]
        reference_intents = [reference] * len(candidates)
        generated_intents = [
            delexicalize_question(question, candidate["path"]["anchor"])
            for question, candidate in zip(generated, candidates, strict=True)
        ]
        reference_embeddings = encoder.encode(reference_intents, normalize_embeddings=True)
        generated_embeddings = encoder.encode(generated_intents, normalize_embeddings=True)
        similarities = np.sum(reference_embeddings * generated_embeddings, axis=1).tolist()
        positive_score = float(similarities[0])
        positive_scores.append(positive_score)
        candidate_rows = []
        for candidate, question, intent, similarity in zip(
            candidates, generated, generated_intents, similarities, strict=True
        ):
            candidate_rows.append(
                {
                    "category": candidate["category"],
                    "is_positive": candidate["is_positive"],
                    "generated_question": question,
                    "generated_intent": intent,
                    "semantic_similarity": float(similarity),
                    "path": candidate["path"],
                }
            )
            if not candidate["is_positive"]:
                negative_scores.append(float(similarity))
                category_results[candidate["category"]].append(positive_score > similarity)
        predictions.append(
            {
                "example_id": row["example_id"],
                "reference_question": reference,
                "positive_beats_all_negatives": all(
                    positive_score > similarity for similarity in similarities[1:]
                ),
                "positive_margin_over_best_negative": positive_score - max(similarities[1:]),
                "candidates": candidate_rows,
            }
        )
        if row_index == 1 or row_index % 25 == 0 or row_index == len(rows):
            print(f"Faithfulness evaluation: {row_index}/{len(rows)}", flush=True)

    pair_count = sum(len(values) for values in category_results.values())
    metrics = {
        "examples": len(predictions),
        "model": model_path,
        "semantic_model": semantic_model,
        "positive_beats_all_negatives": sum(
            row["positive_beats_all_negatives"] for row in predictions
        )
        / max(len(predictions), 1),
        "pairwise_accuracy": sum(sum(values) for values in category_results.values())
        / max(pair_count, 1),
        "mean_positive_similarity": float(np.mean(positive_scores)) if positive_scores else 0.0,
        "mean_negative_similarity": float(np.mean(negative_scores)) if negative_scores else 0.0,
        "mean_margin": float(
            np.mean([row["positive_margin_over_best_negative"] for row in predictions])
        )
        if predictions
        else 0.0,
        "by_negative_type": {
            category: {
                "pairs": len(values),
                "pairwise_accuracy": sum(values) / max(len(values), 1),
            }
            for category, values in sorted(category_results.items())
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", predictions)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [
        "# Faithful Inverse Generation",
        "",
        f"- Examples: {metrics['examples']}",
        f"- Gold question beats every executable negative: {metrics['positive_beats_all_negatives']:.3f}",
        f"- Pairwise accuracy: {metrics['pairwise_accuracy']:.3f}",
        f"- Mean gold similarity: {metrics['mean_positive_similarity']:.3f}",
        f"- Mean negative similarity: {metrics['mean_negative_similarity']:.3f}",
        f"- Mean margin over the strongest negative: {metrics['mean_margin']:.3f}",
        "",
        "## Negative Categories",
        "",
        "| Category | Pairs | Gold Wins |",
        "|---|---:|---:|",
        *[
            f"| {category} | {values['pairs']} | {values['pairwise_accuracy']:.3f} |"
            for category, values in metrics["by_negative_type"].items()
        ],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
