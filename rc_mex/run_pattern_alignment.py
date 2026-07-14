"""Train and evaluate a cross-schema executable-pattern alignment encoder.

The experiment uses parallel WebQuestions parses over Freebase and Wikidata.
It reports question-to-pattern retrieval, direct cross-schema pattern
retrieval, and zero-shot transfer to simple KQA Pro chains.  No gold answer or
relation enters runtime scoring; logical forms are training/evaluation labels.

Commands:
  python3 -m rc_mex.run_pattern_alignment prepare --data-root data/pattern_alignment
  python3 -m rc_mex.run_pattern_alignment train-eval --data-root data/pattern_alignment --output runs/pattern_alignment
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from rc_mex.executable_pattern_alignment import (
    AlignmentRecord,
    build_parallel_records,
    kqa_simple_patterns,
    load_webqsp,
    load_wikiweb,
    pattern_from_graph_query,
    pattern_from_runtime_path,
    pattern_from_sparql,
)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WEBQSP_URL = "https://download.microsoft.com/download/f/5/0/f5012144-a4fb-4084-897f-cfda99c60bdf/WebQSP.zip"
WIKIWEB_URL = "https://github.com/stanford-oval/wikidata-emnlp23/archive/refs/heads/master.zip"
TRAINING_CONFIG = {
    "epochs": 2,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "temperature": 0.05,
    "cross_schema_weight": 0.25,
    "max_sequence_length": 160,
    "seed": 20260714,
}


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=600, context=_ssl_context()) as response, open(destination, "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        print(f"urllib failed ({type(exc).__name__}: {exc}); trying curl")
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-o", str(destination), url], check=True)


def prepare_sources(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        webqsp_zip = temp / "WebQSP.zip"
        wikiweb_zip = temp / "wikiweb.zip"
        print("[1/2] Downloading official WebQSP semantic parses")
        download(WEBQSP_URL, webqsp_zip)
        with zipfile.ZipFile(webqsp_zip) as archive:
            archive.extractall(temp / "webqsp")
        webqsp_data = temp / "webqsp" / "WebQSP" / "data"
        target_webqsp = data_root / "webqsp"
        target_webqsp.mkdir(exist_ok=True)
        for name in ("WebQSP.train.json", "WebQSP.test.json"):
            shutil.copy2(webqsp_data / name, target_webqsp / name)

        print("[2/2] Downloading WikiWebQuestions parallel Wikidata parses")
        download(WIKIWEB_URL, wikiweb_zip)
        with zipfile.ZipFile(wikiweb_zip) as archive:
            archive.extractall(temp / "wikiweb")
        repo = next((temp / "wikiweb").glob("wikidata-emnlp23-*"))
        target_wikiweb = data_root / "wikiweb"
        shutil.copytree(repo / "WikiWebQuestions", target_wikiweb / "WikiWebQuestions", dirs_exist_ok=True)
        (target_wikiweb / "training_data").mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / "training_data" / "best.json", target_wikiweb / "training_data" / "best.json")
    build_wikidata_property_labels(data_root)
    print(f"Prepared alignment sources under {data_root}")


def property_ids(text: str) -> list[str]:
    return re.findall(r"\b(?:wdt|p|ps|pq):(P\d+)\b", str(text))


def build_wikidata_property_labels(data_root: Path) -> dict[str, str]:
    """Resolve public schema labels once; test questions/answers are unused."""
    wiki_root = data_root / "wikiweb"
    training = json.load(open(wiki_root / "training_data" / "best.json", encoding="utf-8"))
    labels: dict[str, str] = {}
    for row in training:
        raw = re.findall(r"\b(?:wdt|p|ps|pq):(P\d+)\b", str(row.get("sparql", "")))
        named = re.findall(r"\b(?:wdt|p|ps|pq):([A-Za-z_][\w_]*)\b", str(row.get("output", "")))
        if len(raw) == len(named):
            for property_id, name in zip(raw, named):
                labels.setdefault(property_id, name)

    required = set()
    for split in ("train", "dev", "test"):
        for row in json.load(open(wiki_root / "WikiWebQuestions" / f"{split}.json", encoding="utf-8")):
            required.update(property_ids(row.get("sparql", "")))
    missing = sorted(required - labels.keys())
    if missing:
        print(f"      Resolving {len(missing)} unseen Wikidata property labels")
    for start in range(0, len(missing), 50):
        batch = missing[start : start + 50]
        query = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
        )
        request = urllib.request.Request(
            f"https://www.wikidata.org/w/api.php?{query}",
            headers={"User-Agent": "rc-mex-research/1.0 (schema-label-resolution)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120, context=_ssl_context()) as response:
                entities = json.load(response).get("entities", {})
            for property_id in batch:
                label = ((entities.get(property_id) or {}).get("labels") or {}).get("en", {}).get("value")
                labels[property_id] = str(label or property_id).replace(" ", "_")
        except Exception as exc:
            print(f"      Warning: property label lookup failed ({exc}); preserving IDs")
            for property_id in batch:
                labels[property_id] = property_id
    write_json(wiki_root / "property_labels.json", labels)
    return labels


def wikiweb_split(data_root: Path, split: str) -> dict[str, dict]:
    wiki_root = data_root / "wikiweb"
    rows = load_wikiweb(wiki_root / "WikiWebQuestions" / f"{split}.json")
    labels_path = wiki_root / "property_labels.json"
    labels = json.load(open(labels_path, encoding="utf-8")) if labels_path.exists() else {}
    for row in rows.values():
        raw = str(row.get("sparql", ""))
        row["output"] = re.sub(
            r"\b(wdt|p|ps|pq):(P\d+)\b",
            lambda match: f"{match.group(1)}:{labels.get(match.group(2), match.group(2))}",
            raw,
        )
    return rows


def load_parallel_splits(data_root: Path) -> dict[str, list[AlignmentRecord]]:
    webqsp_train = load_webqsp(data_root / "webqsp" / "WebQSP.train.json")
    webqsp_test = load_webqsp(data_root / "webqsp" / "WebQSP.test.json")
    wiki_training = load_wikiweb(data_root / "wikiweb" / "training_data" / "best.json")
    wiki_dev = wikiweb_split(data_root, "dev")
    wiki_test = wikiweb_split(data_root, "test")
    return {
        "train": build_parallel_records(webqsp_train, wiki_training),
        "dev": build_parallel_records(webqsp_train, wiki_dev),
        "test": build_parallel_records(webqsp_test, wiki_test),
    }


def encode_train(model, texts: list[str], device):
    features = model.preprocess(texts)
    features = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }
    return model(features)["sentence_embedding"]


def multi_positive_loss(logits, positive_mask):
    import torch

    negative_inf = torch.finfo(logits.dtype).min
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, negative_inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator - denominator).mean()


def train_model(
    records: list[AlignmentRecord],
    output: Path,
    *,
    schemas: tuple[str, ...] = ("freebase", "wikidata"),
    cross_schema_weight: float | None = None,
) -> dict:
    import torch
    import torch.nn.functional as functional
    from sentence_transformers import SentenceTransformer

    config = dict(TRAINING_CONFIG)
    config["schemas"] = list(schemas)
    if cross_schema_weight is not None:
        config["cross_schema_weight"] = cross_schema_weight
    if not schemas or any(schema not in {"freebase", "wikidata"} for schema in schemas):
        raise ValueError(f"Unsupported training schemas: {schemas}")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = SentenceTransformer(BASE_MODEL, local_files_only=True)
    model.max_seq_length = config["max_sequence_length"]
    device = model.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    shuffled = list(records)
    history = []
    print(f"Training records: {len(shuffled)} | device: {device}")
    model.train()
    for epoch in range(config["epochs"]):
        random.Random(config["seed"] + epoch).shuffle(shuffled)
        epoch_losses = []
        for start in range(0, len(shuffled), config["batch_size"]):
            batch = shuffled[start : start + config["batch_size"]]
            if len(batch) < 2:
                continue
            questions = encode_train(model, [record.question for record in batch], device)
            questions = functional.normalize(questions, dim=1)
            schema_embeddings = {
                schema: functional.normalize(
                    encode_train(
                        model,
                        [getattr(record, schema).canonical_text() for record in batch],
                        device,
                    ),
                    dim=1,
                )
                for schema in schemas
            }
            patterns = torch.cat([schema_embeddings[schema] for schema in schemas], dim=0)
            groups = [record.group_id for record in batch]
            positive = torch.tensor(
                [[left == right for right in groups * len(schemas)] for left in groups],
                dtype=torch.bool,
                device=device,
            )
            q_to_pattern = multi_positive_loss(
                questions @ patterns.T / config["temperature"],
                positive,
            )
            pattern_to_q = multi_positive_loss(
                patterns @ questions.T / config["temperature"],
                positive.T,
            )
            loss = q_to_pattern + 0.5 * pattern_to_q
            if len(schemas) == 2 and config["cross_schema_weight"]:
                freebase = schema_embeddings["freebase"]
                wikidata = schema_embeddings["wikidata"]
                cross_schema = (1.0 - (freebase * wikidata).sum(dim=1)).mean()
                loss = loss + config["cross_schema_weight"] * cross_schema
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "steps": len(epoch_losses)})
        print(f"  epoch {epoch + 1}/{config['epochs']}: loss={mean_loss:.4f}")
    model.save_pretrained(str(output / "model"))
    return {"config": config, "history": history, "device": str(device)}


def train_pattern_tower(
    records: list[AlignmentRecord],
    output: Path,
    *,
    cross_schema_weight: float = 0.0,
) -> dict:
    """Train schema patterns into a frozen, general question embedding space."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from sentence_transformers import SentenceTransformer

    config = dict(TRAINING_CONFIG)
    config.update(
        {
            "architecture": "frozen_question_trainable_pattern_tower",
            "cross_schema_weight": cross_schema_weight,
            "schemas": ["freebase", "wikidata"],
        }
    )
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    question_model = SentenceTransformer(BASE_MODEL, local_files_only=True)
    pattern_model = SentenceTransformer(BASE_MODEL, local_files_only=True)
    question_model.max_seq_length = config["max_sequence_length"]
    pattern_model.max_seq_length = config["max_sequence_length"]
    device = pattern_model.device
    question_texts = sorted({record.question for record in records})
    question_vectors = question_model.encode(
        question_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_cache = dict(zip(question_texts, question_vectors))
    optimizer = torch.optim.AdamW(pattern_model.parameters(), lr=config["learning_rate"])
    shuffled = list(records)
    history = []
    print(f"Training pattern tower: {len(shuffled)} records | device: {device}")
    pattern_model.train()
    for epoch in range(config["epochs"]):
        random.Random(config["seed"] + epoch).shuffle(shuffled)
        epoch_losses = []
        for start in range(0, len(shuffled), config["batch_size"]):
            batch = shuffled[start : start + config["batch_size"]]
            if len(batch) < 2:
                continue
            questions = torch.as_tensor(
                np.stack([question_cache[record.question] for record in batch]),
                dtype=torch.float32,
                device=device,
            )
            freebase = functional.normalize(
                encode_train(pattern_model, [record.freebase.canonical_text() for record in batch], device),
                dim=1,
            )
            wikidata = functional.normalize(
                encode_train(pattern_model, [record.wikidata.canonical_text() for record in batch], device),
                dim=1,
            )
            patterns = torch.cat([freebase, wikidata], dim=0)
            groups = [record.group_id for record in batch]
            positive = torch.tensor(
                [[left == right for right in groups + groups] for left in groups],
                dtype=torch.bool,
                device=device,
            )
            q_to_pattern = multi_positive_loss(
                questions @ patterns.T / config["temperature"],
                positive,
            )
            pattern_to_q = multi_positive_loss(
                patterns @ questions.T / config["temperature"],
                positive.T,
            )
            loss = q_to_pattern + 0.5 * pattern_to_q
            if cross_schema_weight:
                loss = loss + cross_schema_weight * (1.0 - (freebase * wikidata).sum(dim=1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pattern_model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss, "steps": len(epoch_losses)})
        print(f"  epoch {epoch + 1}/{config['epochs']}: loss={mean_loss:.4f}")
    model_dir = output / "dual_model"
    pattern_model.save_pretrained(str(model_dir / "pattern"))
    write_json(
        model_dir / "config.json",
        {
            "architecture": config["architecture"],
            "question_model": BASE_MODEL,
            "pattern_model": "pattern",
            "max_sequence_length": config["max_sequence_length"],
        },
    )
    return {"config": config, "history": history, "device": str(device)}


class DualEncoder:
    """Inference wrapper for a frozen question tower and learned pattern tower."""

    def __init__(self, model_dir: str | Path):
        from sentence_transformers import SentenceTransformer

        model_dir = Path(model_dir)
        config = json.load(open(model_dir / "config.json", encoding="utf-8"))
        self.question_model = SentenceTransformer(config["question_model"], local_files_only=True)
        self.pattern_model = SentenceTransformer(
            str(model_dir / config["pattern_model"]),
            local_files_only=True,
        )
        self.question_model.max_seq_length = int(config["max_sequence_length"])
        self.pattern_model.max_seq_length = int(config["max_sequence_length"])

    def encode_questions(self, texts: list[str]):
        return self.question_model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_patterns(self, texts: list[str]):
        return self.pattern_model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


def encode(model, texts: list[str], role: str = "pattern"):
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype="float32")
    if role == "question" and hasattr(model, "encode_questions"):
        return model.encode_questions(texts)
    if role == "pattern" and hasattr(model, "encode_patterns"):
        return model.encode_patterns(texts)
    return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def retrieval_metrics(query_embeddings, candidate_embeddings, gold_sets: list[set[int]]) -> tuple[dict, list[int]]:
    import numpy as np

    scores = query_embeddings @ candidate_embeddings.T
    order = np.argsort(-scores, axis=1)
    ranks = []
    for ranked, gold in zip(order, gold_sets):
        ranks.append(min(index + 1 for index, candidate in enumerate(ranked) if int(candidate) in gold))
    count = len(ranks)
    return (
        {
            "count": count,
            "recall_at_1": sum(rank <= 1 for rank in ranks) / count if count else 0.0,
            "recall_at_5": sum(rank <= 5 for rank in ranks) / count if count else 0.0,
            "recall_at_10": sum(rank <= 10 for rank in ranks) / count if count else 0.0,
            "mrr": sum(1.0 / rank for rank in ranks) / count if count else 0.0,
            "candidate_count": len(candidate_embeddings),
        },
        ranks,
    )


def question_pattern_eval(
    model,
    records: list[AlignmentRecord],
    schema: str,
    query_records: list[AlignmentRecord] | None = None,
) -> tuple[dict, list[int]]:
    """Retrieve patterns from ``records`` for all questions in ``query_records``.

    Keeping the candidate inventory fixed while filtering queries is important:
    an unseen-pattern evaluation must not become easier merely because it has
    fewer distractors.
    """
    patterns = [getattr(record, schema) for record in records]
    by_signature = {}
    for pattern in patterns:
        by_signature.setdefault(pattern.signature(), pattern.canonical_text())
    signatures = sorted(by_signature)
    indices = {signature: index for index, signature in enumerate(signatures)}
    queries = records if query_records is None else query_records
    return retrieval_metrics(
        encode(model, [record.question for record in queries], "question"),
        encode(model, [by_signature[signature] for signature in signatures], "pattern"),
        [{indices[getattr(record, schema).signature()]} for record in queries],
    )


def cross_schema_eval(
    model,
    records: list[AlignmentRecord],
    source: str,
    target: str,
    query_records: list[AlignmentRecord] | None = None,
) -> tuple[dict, list[int]]:
    target_patterns = {}
    paired_targets: dict[str, set[str]] = defaultdict(set)
    source_text = {}
    for record in records:
        source_pattern = getattr(record, source)
        target_pattern = getattr(record, target)
        source_text.setdefault(source_pattern.signature(), source_pattern.canonical_text())
        target_patterns.setdefault(target_pattern.signature(), target_pattern.canonical_text())
        paired_targets[source_pattern.signature()].add(target_pattern.signature())
    if query_records is None:
        source_signatures = sorted(source_text)
    else:
        source_signatures = sorted({getattr(record, source).signature() for record in query_records})
    target_signatures = sorted(target_patterns)
    target_indices = {signature: index for index, signature in enumerate(target_signatures)}
    return retrieval_metrics(
        encode(model, [source_text[signature] for signature in source_signatures], "pattern"),
        encode(model, [target_patterns[signature] for signature in target_signatures], "pattern"),
        [{target_indices[target] for target in paired_targets[source]} for source in source_signatures],
    )


def kqa_eval(model, rows: list[dict]) -> tuple[dict, list[int], list[dict]]:
    examples = kqa_simple_patterns(rows)
    patterns = {}
    for example in examples:
        pattern = example["pattern"]
        patterns.setdefault(pattern.signature(), pattern.canonical_text())
    signatures = sorted(patterns)
    indices = {signature: index for index, signature in enumerate(signatures)}
    metrics, ranks = retrieval_metrics(
        encode(model, [example["question"] for example in examples], "question"),
        encode(model, [patterns[signature] for signature in signatures], "pattern"),
        [{indices[example["pattern"].signature()]} for example in examples],
    )
    return metrics, ranks, examples


def transfer_pattern_examples(rows: list[dict], benchmark: str) -> list[dict]:
    """Create evaluation-only question/pattern pairs for held-out benchmarks."""
    examples = []
    for row in rows:
        if benchmark == "grailqa":
            pattern = pattern_from_graph_query("freebase", row.get("graph_query") or {})
            question = str(row.get("question", ""))
            question_id = str(row.get("qid", ""))
        elif benchmark == "cwq":
            topic = row.get("topic_entity") or {}
            anchors = list(topic) if isinstance(topic, dict) else []
            pattern = pattern_from_sparql("freebase", str(row.get("sparql", "")), anchors)
            question = str(row.get("question", ""))
            question_id = str(row.get("ID", ""))
        else:
            raise ValueError(f"Unsupported transfer benchmark: {benchmark}")
        if pattern is not None and question.strip():
            examples.append({"question_id": question_id, "question": question, "pattern": pattern})
    return examples


def pattern_examples_eval(model, examples: list[dict]) -> tuple[dict, list[int]]:
    patterns = {}
    for example in examples:
        pattern = example["pattern"]
        patterns.setdefault(pattern.signature(), pattern.canonical_text())
    signatures = sorted(patterns)
    indices = {signature: index for index, signature in enumerate(signatures)}
    return retrieval_metrics(
        encode(model, [example["question"] for example in examples], "question"),
        encode(model, [patterns[signature] for signature in signatures], "pattern"),
        [{indices[example["pattern"].signature()]} for example in examples],
    )


def _normal_entity(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def load_local_frontier_examples(path: Path) -> tuple[list[dict], dict]:
    """Load direct-answer relation frontiers from RoG WebQSP subgraphs.

    Gold answers identify the correct local edge only for offline evaluation.
    They never enter candidate text or scoring.  Restricting this diagnostic to
    direct edges avoids pretending that a compressed CVT path is a one-edge
    primitive.
    """
    examples = []
    rows_seen = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_seen += 1
            topics = {_normal_entity(value) for value in row.get("q_entity") or []}
            answers = {_normal_entity(value) for value in row.get("answer") or []}
            if not topics or not answers:
                continue
            candidate_frequency: dict[tuple[str, str], int] = defaultdict(int)
            gold: set[tuple[str, str]] = set()
            for triple in row.get("graph") or []:
                if len(triple) != 3:
                    continue
                head, relation, tail = map(str, triple)
                head_key = _normal_entity(head)
                tail_key = _normal_entity(tail)
                relation = relation.strip()
                if not relation:
                    continue
                if head_key in topics:
                    key = (relation, "forward")
                    candidate_frequency[key] += 1
                    if tail_key in answers:
                        gold.add(key)
                if tail_key in topics:
                    key = (relation, "backward")
                    candidate_frequency[key] += 1
                    if head_key in answers:
                        gold.add(key)
            if not gold:
                continue
            candidate_rows = []
            for relation, direction in sorted(candidate_frequency):
                pattern = pattern_from_runtime_path(
                    {"predicate": relation, "direction": direction},
                    schema="freebase_runtime",
                )
                if pattern is not None:
                    candidate_rows.append(
                        {
                            "key": (relation, direction),
                            "text": pattern.canonical_text(),
                            "frequency": candidate_frequency[(relation, direction)],
                        }
                    )
            if candidate_rows:
                examples.append(
                    {
                        "question_id": str(row.get("id", "")),
                        "question": str(row.get("question", "")),
                        "candidates": candidate_rows,
                        "gold": gold,
                    }
                )
    return examples, {
        "source": str(path),
        "rows_seen": rows_seen,
        "direct_edge_examples": len(examples),
    }


def load_bounded_path_examples(path: Path, max_hops: int = 2) -> tuple[list[dict], dict]:
    """Enumerate one/two-hop executable relation sequences in supplied subgraphs.

    Answer entities label successful generated sequences for offline analysis.
    Candidate construction and model inputs remain answer-blind.
    """
    from rc_mex.run_webqsp_path_family import build_kb

    examples = []
    rows_seen = 0
    hop_counts: dict[str, int] = defaultdict(int)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_seen += 1
            kb = build_kb(row.get("graph") or [])
            starts = {
                _normal_entity(value)
                for value in row.get("q_entity") or []
                if _normal_entity(value) in kb["entities"]
            }
            gold_entities = {_normal_entity(value) for value in row.get("answer") or []}
            if not starts or not gold_entities:
                continue

            path_targets: dict[tuple[tuple[str, str], ...], set[str]] = defaultdict(set)
            first_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
            for start in starts:
                for relation in kb["entities"][start]["relations"]:
                    key = (str(relation["predicate"]), str(relation["direction"]))
                    first_targets[key].add(str(relation["object"]))
            for key, targets in first_targets.items():
                path_targets[(key,)].update(targets)
                if max_hops < 2:
                    continue
                second_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
                for target in targets:
                    for relation in kb["entities"].get(target, {}).get("relations", []):
                        second_key = (str(relation["predicate"]), str(relation["direction"]))
                        second_targets[second_key].add(str(relation["object"]))
                for second_key, final_targets in second_targets.items():
                    path_targets[(key, second_key)].update(final_targets)

            gold_paths = {
                key
                for key, targets in path_targets.items()
                if targets & gold_entities
            }
            if not gold_paths:
                continue
            shortest = min(len(path) for path in gold_paths)
            hop_counts[str(shortest)] += 1
            candidates = []
            for key in sorted(path_targets):
                pattern = pattern_from_runtime_path(
                    {
                        "relations": [
                            {"predicate": relation, "direction": direction}
                            for relation, direction in key
                        ]
                    },
                    schema="freebase_runtime",
                )
                if pattern is not None:
                    candidates.append({"key": key, "text": pattern.canonical_text()})
            examples.append(
                {
                    "question_id": str(row.get("id", "")),
                    "question": str(row.get("question", "")),
                    "candidates": candidates,
                    "gold": gold_paths,
                    "shortest_gold_hops": shortest,
                }
            )
    return examples, {
        "source": str(path),
        "rows_seen": rows_seen,
        "answer_reaching_examples": len(examples),
        "shortest_gold_hops": dict(hop_counts),
        "max_hops": max_hops,
    }


def local_frontier_eval(model, examples: list[dict]) -> tuple[dict, list[int]]:
    """Rank each question's actual local frontier with the shared encoder."""
    import numpy as np

    if not examples:
        return {
            "count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "candidate_count": 0,
            "average_candidate_count": 0.0,
        }, []
    questions = sorted({example["question"] for example in examples})
    pattern_texts = sorted(
        {candidate["text"] for example in examples for candidate in example["candidates"]}
    )
    question_vectors = dict(zip(questions, encode(model, questions, "question")))
    pattern_vectors = dict(zip(pattern_texts, encode(model, pattern_texts, "pattern")))
    ranks = []
    candidate_counts = []
    for example in examples:
        candidates = example["candidates"]
        scores = np.asarray(
            [question_vectors[example["question"]] @ pattern_vectors[candidate["text"]] for candidate in candidates]
        )
        order = np.argsort(-scores)
        rank = min(
            index + 1
            for index, candidate_index in enumerate(order)
            if candidates[int(candidate_index)]["key"] in example["gold"]
        )
        ranks.append(rank)
        candidate_counts.append(len(candidates))
    count = len(ranks)
    return {
        "count": count,
        "recall_at_1": sum(rank <= 1 for rank in ranks) / count,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / count,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "candidate_count": max(candidate_counts),
        "average_candidate_count": sum(candidate_counts) / count,
    }, ranks


def local_frontier_hybrid_eval(model, examples: list[dict]) -> tuple[dict, list[int]]:
    """Apply the current search ranker's fixed lexical/semantic mixture."""
    import numpy as np

    from rc_mex.run_proof_state_search_smoke import (
        char_ngram_similarity,
        question_relation_term_overlap,
        relation_text_expansions,
    )

    questions = sorted({example["question"] for example in examples})
    pattern_texts = sorted(
        {candidate["text"] for example in examples for candidate in example["candidates"]}
    )
    question_vectors = dict(zip(questions, encode(model, questions, "question")))
    pattern_vectors = dict(zip(pattern_texts, encode(model, pattern_texts, "pattern")))
    ranks = []
    candidate_counts = []
    for example in examples:
        candidates = example["candidates"]
        scores = []
        for candidate in candidates:
            relation, direction = candidate["key"]
            label = f"{relation.replace('_', ' ')} {direction}"
            relation_texts = relation_text_expansions(relation, direction)
            lexical = max(
                [char_ngram_similarity(example["question"], label)]
                + [char_ngram_similarity(example["question"], text) for text in relation_texts]
            )
            overlap = question_relation_term_overlap(example["question"], relation_texts)
            semantic = float(
                question_vectors[example["question"]] @ pattern_vectors[candidate["text"]]
            )
            frequency_bonus = 0.03 * min(1.0, float(candidate["frequency"]) / 5.0)
            scores.append(0.55 * (lexical + 0.05 * overlap) + 0.35 * max(0.0, semantic) + 0.10 * frequency_bonus)
        order = np.argsort(-np.asarray(scores))
        ranks.append(
            min(
                index + 1
                for index, candidate_index in enumerate(order)
                if candidates[int(candidate_index)]["key"] in example["gold"]
            )
        )
        candidate_counts.append(len(candidates))
    count = len(ranks)
    return {
        "count": count,
        "recall_at_1": sum(rank <= 1 for rank in ranks) / count,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / count,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "candidate_count": max(candidate_counts),
        "average_candidate_count": sum(candidate_counts) / count,
    }, ranks


def evaluate_model(
    model,
    splits,
    kqa_rows,
    transfer_sets: dict[str, list[dict]] | None = None,
) -> tuple[dict, dict]:
    metrics = {}
    ranks = {}
    train_signatures = {
        schema: {getattr(record, schema).signature() for record in splits["train"]}
        for schema in ("freebase", "wikidata")
    }
    for split in ("dev", "test"):
        records = splits[split]
        for schema in ("freebase", "wikidata"):
            key = f"{split}_question_to_{schema}"
            metrics[key], ranks[key] = question_pattern_eval(model, records, schema)
            unseen = [
                record
                for record in records
                if getattr(record, schema).signature() not in train_signatures[schema]
            ]
            unseen_key = f"{split}_unseen_question_to_{schema}"
            metrics[unseen_key], ranks[unseen_key] = question_pattern_eval(
                model,
                records,
                schema,
                unseen,
            )
        for source, target in (("freebase", "wikidata"), ("wikidata", "freebase")):
            key = f"{split}_{source}_to_{target}"
            metrics[key], ranks[key] = cross_schema_eval(model, records, source, target)
            unseen = [
                record
                for record in records
                if getattr(record, source).signature() not in train_signatures[source]
                or getattr(record, target).signature() not in train_signatures[target]
            ]
            unseen_key = f"{split}_unseen_{source}_to_{target}"
            metrics[unseen_key], ranks[unseen_key] = cross_schema_eval(
                model,
                records,
                source,
                target,
                unseen,
            )
    if kqa_rows:
        metrics["kqa_zero_shot"], ranks["kqa_zero_shot"], _ = kqa_eval(model, kqa_rows)
    for benchmark, examples in (transfer_sets or {}).items():
        key = f"{benchmark}_zero_shot"
        metrics[key], ranks[key] = pattern_examples_eval(model, examples)
    return metrics, ranks


def report_text(payload: dict) -> str:
    lines = [
        "# Executable Pattern Alignment",
        "",
        "This experiment trains on parallel Freebase/Wikidata logical forms for the same natural questions.",
        "KQA Pro is evaluation-only and therefore measures unseen-schema transfer.",
        "",
        f"Train pairs: {payload['data']['train_pairs']}",
        f"Dev pairs: {payload['data']['dev_pairs']}",
        f"Test pairs: {payload['data']['test_pairs']}",
        "",
        "| Evaluation | Baseline R@1 | Trained R@1 | Delta | Baseline MRR | Trained MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in payload["baseline"]:
        baseline = payload["baseline"][key]
        trained = payload["trained"][key]
        lines.append(
            f"| {key} | {baseline['recall_at_1']:.3f} | {trained['recall_at_1']:.3f} | "
            f"{trained['recall_at_1'] - baseline['recall_at_1']:+.3f} | {baseline['mrr']:.3f} | {trained['mrr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- WebQSP and WikiWebQuestions are parallel schemas, so their cross-schema retrieval is directly supervised.",
            "- KQA Pro is a different schema and is never used for training in this run.",
            "- This evaluates semantic alignment and candidate ranking, not entity linking or complete KGQA.",
            "- A gain on KQA is evidence of transfer; a gain only on the parallel schemas is evidence of memorization or in-domain adaptation.",
        ]
    )
    return "\n".join(lines) + "\n"


def train_and_evaluate(data_root: Path, output: Path) -> None:
    from sentence_transformers import SentenceTransformer

    ensure_dir(output)
    print("[1/4] Parsing parallel Freebase/Wikidata executable patterns")
    splits = load_parallel_splits(data_root)
    for name, records in splits.items():
        print(f"      {name}: {len(records)} aligned pairs")
    if len(splits["train"]) < 100:
        raise SystemExit("Too few aligned training pairs; check the official dataset layout and parser coverage.")

    kqa_path = data_root / "kqa_pro" / "val.json"
    kqa_rows = json.load(open(kqa_path, encoding="utf-8")) if kqa_path.exists() else []
    print(f"      KQA transfer rows available: {len(kqa_rows)}")
    transfer_sets = {}
    for benchmark in ("grailqa", "cwq"):
        path = data_root / "transfer" / f"{benchmark}.json"
        if path.exists():
            rows = json.load(open(path, encoding="utf-8"))
            transfer_sets[benchmark] = transfer_pattern_examples(rows, benchmark)
            print(f"      {benchmark} transfer examples: {len(transfer_sets[benchmark])}")
    frontier_path = next(
        (
            path
            for path in (
                data_root / "webqsp_rog_test.jsonl",
                Path("data/webqsp/test.jsonl"),
            )
            if path.exists()
        ),
        None,
    )
    frontier_examples = []
    frontier_data = None
    if frontier_path is not None:
        print(f"      Loading real local frontiers from {frontier_path}")
        frontier_examples, frontier_data = load_local_frontier_examples(frontier_path)
        print(f"      Direct-edge frontier examples: {len(frontier_examples)}")

    print("[2/4] Evaluating frozen MiniLM baseline")
    baseline_model = SentenceTransformer(BASE_MODEL, local_files_only=True)
    baseline_model.max_seq_length = TRAINING_CONFIG["max_sequence_length"]
    baseline_metrics, baseline_ranks = evaluate_model(baseline_model, splits, kqa_rows, transfer_sets)
    if frontier_examples:
        baseline_metrics["webqsp_local_frontier"], baseline_ranks["webqsp_local_frontier"] = local_frontier_eval(
            baseline_model,
            frontier_examples,
        )

    print("[3/4] Training shared executable-pattern encoder")
    training = train_model(splits["train"], output)
    trained_model = SentenceTransformer(str(output / "model"), local_files_only=True)
    trained_metrics, trained_ranks = evaluate_model(trained_model, splits, kqa_rows, transfer_sets)
    if frontier_examples:
        trained_metrics["webqsp_local_frontier"], trained_ranks["webqsp_local_frontier"] = local_frontier_eval(
            trained_model,
            frontier_examples,
        )

    print("[4/4] Writing compact research outputs")
    payload = {
        "hypothesis": "Parallel executable patterns can teach a shared encoder to align question semantics across KG schemas.",
        "data": {
            "train_pairs": len(splits["train"]),
            "dev_pairs": len(splits["dev"]),
            "test_pairs": len(splits["test"]),
            "kqa_transfer_examples": len(kqa_simple_patterns(kqa_rows)) if kqa_rows else 0,
            "other_transfer_examples": {name: len(rows) for name, rows in transfer_sets.items()},
            "training_schemas": ["freebase", "wikidata"],
            "held_out_schema": "kqa" if kqa_rows else None,
            "local_frontier": frontier_data,
        },
        "training": training,
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }
    write_json(output / "metrics.json", payload)
    (output / "report.md").write_text(report_text(payload), encoding="utf-8")

    examples = []
    for key in trained_ranks:
        before = baseline_ranks[key]
        after = trained_ranks[key]
        examples.append(
            {
                "evaluation": key,
                "improved": sum(new < old for old, new in zip(before, after)),
                "worsened": sum(new > old for old, new in zip(before, after)),
                "unchanged": sum(new == old for old, new in zip(before, after)),
                "mean_rank_before": sum(before) / len(before) if before else math.nan,
                "mean_rank_after": sum(after) / len(after) if after else math.nan,
            }
        )
    with open(output / "rank_changes.jsonl", "w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote model and evaluation to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="download official parallel supervision")
    prepare.add_argument("--data-root", default="data/pattern_alignment")
    train_eval = subparsers.add_parser("train-eval", help="train once and run all transfer evaluations")
    train_eval.add_argument("--data-root", default="data/pattern_alignment")
    train_eval.add_argument("--output", default="runs/pattern_alignment")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_sources(Path(args.data_root))
    else:
        train_and_evaluate(Path(args.data_root), Path(args.output))


if __name__ == "__main__":
    main()
