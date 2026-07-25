from __future__ import annotations

import copy
import difflib
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


KQA_CHAIN_SIGNATURES = {
    ("Find", "Relate", "FilterConcept", "What"),
    ("Find", "Relate", "FilterConcept", "Relate", "FilterConcept", "What"),
}

ENTITY_PLACEHOLDER = "[ENTITY]"

# The general principle is that an answer entity must be nameable: a question cannot
# be asking for something the graph cannot name. Detecting that is dataset-specific.
# In these preprocessed neighborhoods, nodes surfaced as raw machine ids are ones
# whose labels were not resolved, and gold answers are labels, so such endpoints are
# unusable. This is a property of the preprocessing, NOT a fact about Freebase: an
# ``m.``/``g.`` id names ordinary entities as well as mediator/CVT nodes.
UNLABELED_ID = re.compile(r"^[a-z]\.[0-9a-z_]+$")


def unlabeled_answer_count(answers: Iterable[str]) -> int:
    return sum(bool(UNLABELED_ID.match(answer.strip())) for answer in answers)


@dataclass(frozen=True)
class Hop:
    relation: str
    direction: str
    source_type: str
    target_type: str


@dataclass(frozen=True)
class PathSpec:
    anchor: str
    anchor_type: str
    hops: tuple[Hop, ...]
    answer_type: str
    kg: str


@dataclass
class Example:
    example_id: str
    question: str
    positive_path: dict[str, Any]
    alternate_positive_paths: list[dict[str, Any]]
    negative_paths: list[dict[str, Any]]
    split: str
    kg: str
    relation_sequence: list[str]


def stable_bucket(text: str, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % buckets


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def delexicalize_question(question: str, anchor: str) -> str:
    """Replace the linked topic mention while preserving the question intent.

    Exact matching handles most datasets. A conservative fuzzy fallback covers
    surface forms such as ``Jamaican`` for ``Jamaica`` without requiring a
    benchmark-specific alias table.
    """
    if not anchor.strip():
        return normalize_space(question)
    exact = re.compile(r"(?<!\w)" + re.escape(anchor.strip()) + r"(?!\w)", re.IGNORECASE)
    replaced, count = exact.subn(ENTITY_PLACEHOLDER, question)
    if count:
        return normalize_space(replaced)

    anchor_normalized = " ".join(re.findall(r"[a-z0-9]+", anchor.casefold()))
    tokens = list(re.finditer(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", question))
    if not anchor_normalized or not tokens:
        return normalize_space(question)
    anchor_length = max(1, len(anchor_normalized.split()))
    best: tuple[float, int, int] | None = None
    for start in range(len(tokens)):
        for width in range(1, min(len(tokens) - start, anchor_length + 1) + 1):
            end = start + width
            phrase = " ".join(token.group(0).casefold() for token in tokens[start:end])
            score = difflib.SequenceMatcher(None, anchor_normalized, phrase).ratio()
            if best is None or score > best[0]:
                best = (score, tokens[start].start(), tokens[end - 1].end())
    if best is not None and best[0] >= 0.72:
        return normalize_space(question[: best[1]] + ENTITY_PLACEHOLDER + question[best[2] :])
    return normalize_space(question)


def relation_words(relation: str) -> str:
    relation = relation.removeprefix("ns:")
    pieces = relation.split(".")
    if len(pieces) >= 3:
        # Freebase prefixes encode the source schema type; render_path exposes
        # that separately, so the predicate channel contains only the role.
        pieces = [pieces[-1]]
    return normalize_space(" ".join(pieces).replace("_", " ").replace("/", " "))


def full_relation_words(relation: str) -> str:
    """Render every schema segment instead of discarding relation identity."""
    relation = relation.removeprefix("ns:")
    return normalize_space(
        " / ".join(relation.split(".")).replace("_", " ").replace("/", " / ")
    )


def load_relation_glossary(path: Path | str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    glossary_path = Path(path)
    if not glossary_path.exists():
        raise FileNotFoundError(f"relation glossary not found: {glossary_path}")
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("relation glossary must be a JSON object keyed by KG::relation")
    return data


def _grounded_hop_lines(
    path: dict[str, Any],
    hop: dict[str, Any],
    index: int,
    source: str,
    destination: str,
    glossary: dict[str, dict[str, Any]],
) -> list[str]:
    relation = hop["relation"]
    kg = hop.get("kg") or path.get("kg", "unknown")
    entry = glossary.get(f"{kg}::{relation}", {})
    semantic = entry.get("status") == "semantic"
    description = normalize_space(str(entry.get("description", ""))) if semantic else ""
    subject_role = normalize_space(str(entry.get("subject_role", "entity"))) or "entity"
    object_role = normalize_space(str(entry.get("object_role", "entity"))) or "entity"
    template = str(entry.get("fact_template", "")) if semantic else ""

    if hop["direction"] == "forward":
        fact_subject, fact_object = source, destination
        subject_type, object_type = hop["source_type"], hop["target_type"]
    else:
        fact_subject, fact_object = destination, source
        subject_type, object_type = hop["target_type"], hop["source_type"]

    subject = f"{fact_subject} (type: {subject_type}; role: {subject_role})"
    obj = f"{fact_object} (type: {object_type}; role: {object_role})"
    if "{subject}" in template and "{object}" in template:
        try:
            bound_fact = normalize_space(template.format(subject=subject, object=obj))
        except (KeyError, ValueError):
            bound_fact = ""
    else:
        bound_fact = ""
    if not bound_fact:
        bound_fact = f"{subject} --[{full_relation_words(relation)}]--> {obj}."

    lines = [
        f"Hop {index}: traversal {source} -> {destination}.",
        f"  Relation ID: {relation}",
    ]
    if description:
        lines.append(f"  Relation meaning: {description}")
    else:
        lines.append(f"  Relation meaning: {full_relation_words(relation)}")
    lines.extend(
        [
            f"  Canonical roles: subject={subject_role}; object={object_role}.",
            f"  Bound fact: {bound_fact}",
        ]
    )
    return lines


def render_path(
    path: dict[str, Any] | PathSpec,
    style: str = "natural",
    include_instruction: bool = True,
    mask_anchor: bool = False,
    relation_glossary: dict[str, dict[str, Any]] | None = None,
) -> str:
    if isinstance(path, PathSpec):
        path = path_to_dict(path)
    anchor = ENTITY_PLACEHOLDER if mask_anchor else path["anchor"]
    anchor_type = path.get("anchor_type") or "entity"
    lines = []
    if include_instruction:
        lines.append("Write a question for this KG path.")
    lines.append(f'Start entity: "{anchor}" (type: {anchor_type})')
    for index, hop in enumerate(path["hops"], 1):
        source = "START" if index == 1 else f"NODE_{index - 1}"
        destination = "ANSWER" if index == len(path["hops"]) else f"NODE_{index}"
        if relation_glossary is not None:
            lines.extend(
                _grounded_hop_lines(
                    path, hop, index, source, destination, relation_glossary
                )
            )
            continue
        relation = hop["relation"]
        if style == "schema":
            relation = relation_words(relation).replace(" ", "_")
        else:
            relation = relation_words(relation)
        if hop["direction"] == "forward":
            fact_head, fact_tail = source, destination
        else:
            fact_head, fact_tail = destination, source
        lines.append(
            f'Hop {index}: traverse {source} -> {destination}; relation="{relation}"; '
            f"fact roles={fact_head}:subject,{fact_tail}:object; "
            f"types={source}:{hop['source_type']},{destination}:{hop['target_type']}."
        )
    if relation_glossary is not None:
        lines.append(f"Requested answer: ANSWER (type: {path.get('answer_type') or 'entity'}).")
    if include_instruction:
        lines.append("Question:")
    return "\n".join(lines)


def path_to_dict(path: PathSpec) -> dict[str, Any]:
    data = asdict(path)
    data["hops"] = [asdict(hop) for hop in path.hops]
    return data


def extract_kqa_examples(path: Path) -> list[tuple[str, str, PathSpec]]:
    rows = json.load(path.open(encoding="utf-8"))
    examples: list[tuple[str, str, PathSpec]] = []
    for index, row in enumerate(rows):
        program = row.get("program", [])
        signature = tuple(step.get("function") for step in program)
        if signature not in KQA_CHAIN_SIGNATURES:
            continue
        anchor = program[0]["inputs"][0]
        hops: list[Hop] = []
        source_type = "entity"
        for step_index, step in enumerate(program):
            if step["function"] != "Relate":
                continue
            relation, direction = step["inputs"]
            target_type = "entity"
            if step_index + 1 < len(program) and program[step_index + 1]["function"] == "FilterConcept":
                target_type = program[step_index + 1]["inputs"][0]
            hops.append(Hop(relation, direction, source_type, target_type))
            source_type = target_type
        if not hops:
            continue
        spec = PathSpec(anchor, "entity", tuple(hops), hops[-1].target_type, "kqa_pro")
        examples.append((f"kqa:{index}", normalize_space(row["question"]), spec))
    return examples


def parse_sparql_path(sparql: str, topic_mid: str, chain: list[str]) -> list[tuple[str, str]] | None:
    """Recover relation directions from the topic constant to ?x.

    WebQSP's InferentialChain omits direction. The SPARQL is authoritative, so
    this parser builds a small variable graph and finds the chain-aligned route.
    """
    triples: list[tuple[str, str, str]] = []
    triple_re = re.compile(
        r"^\s*(\?[A-Za-z0-9_]+|ns:[A-Za-z0-9_\.]+)\s+"
        r"ns:([A-Za-z0-9_\.]+)\s+"
        r"(\?[A-Za-z0-9_]+|ns:[A-Za-z0-9_\.]+)\s*\."
    )
    for raw_line in sparql.splitlines():
        line = raw_line.split("#", 1)[0]
        match = triple_re.match(line)
        if match:
            triples.append(match.groups())
    start = f"ns:{topic_mid}"
    adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for head, relation, tail in triples:
        adjacency[head].append((tail, relation, "forward"))
        adjacency[tail].append((head, relation, "backward"))
    queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(start, [])])
    seen = {(start, 0)}
    while queue:
        node, path = queue.popleft()
        if len(path) == len(chain):
            if node == "?x":
                return path
            continue
        expected = chain[len(path)]
        for target, relation, direction in adjacency.get(node, []):
            if relation != expected:
                continue
            state = (target, len(path) + 1)
            if state in seen:
                continue
            seen.add(state)
            queue.append((target, path + [(relation, direction)]))
    return None


def freebase_source_type(relation: str) -> str:
    pieces = relation.split(".")
    return " ".join(pieces[:2]).replace("_", " ") if len(pieces) >= 2 else "entity"


def load_webqsp_types(path: Path | None) -> dict[str, dict[str, set[str]]]:
    if path is None or not path.exists():
        return {}
    output: dict[str, dict[str, set[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            entity_types: dict[str, set[str]] = defaultdict(set)
            for head, relation, tail in row.get("graph", []):
                if relation == "common.topic.notable_types" and not tail.startswith(("m.", "g.")):
                    entity_types[head].add(tail)
            output[row["id"]] = entity_types
    return output


def extract_webqsp_examples(
    path: Path,
    graph_path: Path | None = None,
) -> list[tuple[str, str, PathSpec]]:
    rows = json.load(path.open(encoding="utf-8"))["Questions"]
    type_maps = load_webqsp_types(graph_path)
    examples: list[tuple[str, str, PathSpec]] = []
    for question in rows:
        for parse_index, parse in enumerate(question.get("Parses", [])):
            comment = parse.get("AnnotatorComment") or {}
            chain = parse.get("InferentialChain") or []
            if not 1 <= len(chain) <= 3 or comment.get("ParseQuality") == "Incomplete":
                continue
            # The current representation contains a single relation chain. A
            # constraint, time, or ordering clause would be omitted supervision.
            if parse.get("Constraints") or parse.get("Time") is not None or parse.get("Order") is not None:
                continue
            directed = parse_sparql_path(parse.get("Sparql", ""), parse.get("TopicEntityMid", ""), chain)
            if not directed:
                continue
            entity_types = type_maps.get(question["QuestionId"], {})
            answer_names = [answer.get("EntityName", "") for answer in parse.get("Answers", [])]
            answer_types = sorted({kind for name in answer_names for kind in entity_types.get(name, set())})
            answer_type = " / ".join(answer_types) if answer_types else "answer entity"
            topic_name = parse.get("TopicEntityName") or parse.get("PotentialTopicEntityMention") or "ENTITY_0"
            topic_types = sorted(entity_types.get(topic_name, set()))
            hops: list[Hop] = []
            current_type = " / ".join(topic_types) if topic_types else (
                freebase_source_type(directed[0][0]) if directed[0][1] == "forward" else "entity"
            )
            for hop_index, (relation, direction) in enumerate(directed):
                source_type = current_type
                if hop_index == len(directed) - 1:
                    target_type = answer_type
                else:
                    next_relation, next_direction = directed[hop_index + 1]
                    target_type = (
                        freebase_source_type(next_relation)
                        if next_direction == "forward"
                        else "intermediate entity"
                    )
                hops.append(Hop(relation, direction, source_type, target_type))
                current_type = target_type
            spec = PathSpec(
                topic_name,
                hops[0].source_type,
                tuple(hops),
                hops[-1].target_type,
                "webqsp",
            )
            question_id = f'webqsp:{question["QuestionId"]}:{parse_index}'
            examples.append((question_id, normalize_space(question["RawQuestion"]), spec))
    return examples


def relation_inventory(examples: Iterable[tuple[str, str, PathSpec]]) -> dict[str, list[Hop]]:
    inventory: dict[str, list[Hop]] = defaultdict(list)
    for _, _, path in examples:
        for hop in path.hops:
            inventory[hop.target_type.casefold()].append(hop)
    return inventory


def corrupt_path(
    path: PathSpec,
    category: str,
    inventory: dict[str, list[Hop]],
    rng: random.Random,
    symmetric_relations: set[str] | None = None,
) -> PathSpec | None:
    data = copy.deepcopy(path_to_dict(path))
    hops = data["hops"]
    if category == "wrong_direction":
        index = rng.randrange(len(hops))
        if hops[index]["relation"] in (symmetric_relations or set()):
            return None
        hops[index]["direction"] = "backward" if hops[index]["direction"] == "forward" else "forward"
    elif category == "wrong_relation":
        index = rng.randrange(len(hops))
        candidates = [
            hop for hop in inventory.get(hops[index]["target_type"].casefold(), [])
            if hop.relation != hops[index]["relation"]
        ]
        if not candidates:
            return None
        replacement = rng.choice(candidates)
        hops[index]["relation"] = replacement.relation
    elif category == "wrong_order":
        if len(hops) < 2:
            return None
        hops[0], hops[1] = hops[1], hops[0]
        # Preserve the displayed composition after reordering.
        hops[0]["source_type"] = path.anchor_type
        for index in range(1, len(hops)):
            hops[index]["source_type"] = hops[index - 1]["target_type"]
    elif category == "missing_hop":
        if len(hops) < 2:
            return None
        del hops[rng.randrange(len(hops))]
        hops[0]["source_type"] = path.anchor_type
        for index in range(1, len(hops)):
            hops[index]["source_type"] = hops[index - 1]["target_type"]
        data["answer_type"] = hops[-1]["target_type"]
    elif category == "wrong_answer_type":
        types = [name for name in inventory if name != path.answer_type.casefold() and name != "entity"]
        if not types:
            return None
        replacement = rng.choice(types)
        hops[-1]["target_type"] = replacement
        data["answer_type"] = replacement
    else:
        raise ValueError(f"unknown corruption category: {category}")
    return PathSpec(
        data["anchor"],
        data["anchor_type"],
        tuple(Hop(**hop) for hop in hops),
        data["answer_type"],
        data["kg"],
    )


def make_negative_paths(
    path: PathSpec,
    inventory: dict[str, list[Hop]],
    seed_text: str,
    symmetric_relations: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(stable_bucket(seed_text, 2**31 - 1))
    categories = ["wrong_direction", "wrong_relation", "wrong_order", "missing_hop", "wrong_answer_type"]
    negatives: list[dict[str, Any]] = []
    seen = {json.dumps(path_to_dict(path), sort_keys=True)}
    for category in categories:
        negative = corrupt_path(path, category, inventory, rng, symmetric_relations)
        if negative is None:
            continue
        data = path_to_dict(negative)
        key = json.dumps(data, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        data["negative_type"] = category
        negatives.append(data)
    return negatives


def relation_sequence(path: PathSpec) -> tuple[str, ...]:
    return tuple(f"{hop.relation}::{hop.direction}" for hop in path.hops)


def choose_heldout_relations(examples: list[tuple[str, str, PathSpec]]) -> set[str]:
    counts = Counter(hop.relation for _, _, path in examples for hop in path.hops)
    eligible = sorted(relation for relation, count in counts.items() if count >= 4)
    heldout = {relation for relation in eligible if stable_bucket("relation:" + relation, 7) == 0}
    if not heldout and eligible:
        heldout.add(eligible[-1])
    return heldout


def assign_kqa_splits(
    examples: list[tuple[str, str, PathSpec]],
) -> tuple[dict[str, list[tuple[str, str, PathSpec]]], set[str], set[tuple[str, ...]]]:
    heldout_relations = choose_heldout_relations(examples)
    unseen_relation: list[tuple[str, str, PathSpec]] = []
    remaining: list[tuple[str, str, PathSpec]] = []
    for row in examples:
        relations = {hop.relation for hop in row[2].hops}
        (unseen_relation if relations & heldout_relations else remaining).append(row)

    sequence_counts = Counter(relation_sequence(path) for _, _, path in remaining if len(path.hops) > 1)
    unseen_sequences = {
        sequence for sequence, count in sequence_counts.items()
        if count >= 2 and stable_bucket("sequence:" + "|".join(sequence), 5) == 0
    }
    unseen_composition: list[tuple[str, str, PathSpec]] = []
    train_dev: list[tuple[str, str, PathSpec]] = []
    for row in remaining:
        sequence = relation_sequence(row[2])
        if len(sequence) > 1 and sequence in unseen_sequences:
            unseen_composition.append(row)
        else:
            train_dev.append(row)

    train, dev = [], []
    for row in train_dev:
        (dev if stable_bucket("dev:" + row[0], 10) == 0 else train).append(row)
    return {
        "train": train,
        "dev": dev,
        "test_unseen_relation": unseen_relation,
        "test_unseen_composition": unseen_composition,
    }, heldout_relations, unseen_sequences


def materialize(
    rows: Iterable[tuple[str, str, PathSpec]],
    split: str,
    inventory: dict[str, list[Hop]],
    forbidden_relations: set[str] | None = None,
    forbidden_sequences: set[tuple[str, ...]] | None = None,
    symmetric_relations: set[str] | None = None,
) -> list[Example]:
    forbidden_relations = forbidden_relations or set()
    forbidden_sequences = forbidden_sequences or set()
    output = []
    for example_id, question, path in rows:
        negatives = make_negative_paths(path, inventory, example_id, symmetric_relations)
        negatives = [
            candidate
            for candidate in negatives
            if not any(hop["relation"] in forbidden_relations for hop in candidate["hops"])
            and tuple(f'{hop["relation"]}::{hop["direction"]}' for hop in candidate["hops"])
            not in forbidden_sequences
        ]
        output.append(
            Example(
                example_id=example_id,
                question=question,
                positive_path=path_to_dict(path),
                alternate_positive_paths=[],
                negative_paths=negatives,
                split=split,
                kg=path.kg,
                relation_sequence=list(relation_sequence(path)),
            )
        )
    return output


def path_key(path: dict[str, Any] | PathSpec) -> str:
    if isinstance(path, PathSpec):
        path = path_to_dict(path)
    return json.dumps(path, sort_keys=True)


def materialize_webqsp(
    rows: list[tuple[str, str, PathSpec]],
    inventory: dict[str, list[Hop]],
    split_name: str = "test_cross_kg_webqsp",
) -> list[Example]:
    grouped: dict[str, list[tuple[str, str, PathSpec]]] = defaultdict(list)
    for row in rows:
        grouped[row[0].rsplit(":", 1)[0]].append(row)
    output: list[Example] = []
    for group_id, parses in grouped.items():
        question = parses[0][1]
        positives: list[PathSpec] = []
        seen_positive: set[str] = set()
        for _, _, path in parses:
            key = path_key(path)
            if key not in seen_positive:
                positives.append(path)
                seen_positive.add(key)
        negative_rows: list[dict[str, Any]] = []
        seen_negative: set[str] = set()
        for parse_index, positive in enumerate(positives):
            for negative in make_negative_paths(positive, inventory, f"{group_id}:{parse_index}"):
                key = path_key({key: value for key, value in negative.items() if key != "negative_type"})
                if key in seen_positive or key in seen_negative:
                    continue
                seen_negative.add(key)
                negative_rows.append(negative)
        primary = positives[0]
        output.append(
            Example(
                example_id=group_id,
                question=question,
                positive_path=path_to_dict(primary),
                alternate_positive_paths=[path_to_dict(path) for path in positives[1:]],
                negative_paths=negative_rows,
                split=split_name,
                kg="webqsp",
                relation_sequence=list(relation_sequence(primary)),
            )
        )
    return output


def kqa_symmetric_relations(kb_path: Path | None, threshold: float = 0.8) -> set[str]:
    if kb_path is None or not kb_path.exists():
        return set()
    kb = json.load(kb_path.open(encoding="utf-8"))
    pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for entity_id, entity in kb.get("entities", {}).items():
        for relation in entity.get("relations", []):
            if relation.get("direction") == "forward":
                head, tail = entity_id, relation["object"]
            else:
                head, tail = relation["object"], entity_id
            pairs[relation["predicate"]].add((head, tail))
    symmetric = set()
    for relation, extension in pairs.items():
        if len(extension) < 10:
            continue
        reciprocal = sum((tail, head) in extension for head, tail in extension) / len(extension)
        if reciprocal >= threshold:
            symmetric.add(relation)
    return symmetric


def lexical_path_score(question: str, path: PathSpec) -> float:
    question_tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
    path_tokens = set()
    for hop in path.hops:
        path_tokens.update(re.findall(r"[a-z0-9]+", relation_words(hop.relation).casefold()))
        path_tokens.update(re.findall(r"[a-z0-9]+", hop.target_type.casefold()))
    if not path_tokens:
        return 0.0
    return len(question_tokens & path_tokens) / len(path_tokens)


def build_executable_webqsp_examples(
    official_rows: list[tuple[str, str, PathSpec]],
    graph_path: Path | None,
    candidate_cap: int = 200,
) -> tuple[list[Example], dict[str, Any]]:
    if graph_path is None or not graph_path.exists():
        return [], {"questions": 0, "gold_in_local_graph": 0, "gold_after_cap": 0, "candidate_cap": candidate_cap}
    official: dict[str, list[tuple[str, str, PathSpec]]] = defaultdict(list)
    for row in official_rows:
        official[row[0].rsplit(":", 1)[0]].append(row)
    graph_rows = {json.loads(line)["id"]: json.loads(line) for line in graph_path.open(encoding="utf-8")}
    output: list[Example] = []
    diagnostics = {
        "questions": 0,
        "gold_in_local_graph": 0,
        "gold_after_cap": 0,
        "candidate_cap": candidate_cap,
        "average_candidates_before_cap": 0.0,
    }
    candidate_counts = []
    for group_id, parses in official.items():
        question_id = group_id.split(":", 1)[1]
        graph_row = graph_rows.get(question_id)
        if graph_row is None:
            continue
        diagnostics["questions"] += 1
        question = parses[0][1]
        gold_sequences = {relation_sequence(path) for _, _, path in parses}
        hop_limit = max(len(sequence) for sequence in gold_sequences)
        type_map: dict[str, set[str]] = defaultdict(set)
        adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for head, relation, tail in graph_row.get("graph", []):
            adjacency[head].append((tail, relation, "forward"))
            adjacency[tail].append((head, relation, "backward"))
            if relation == "common.topic.notable_types" and not tail.startswith(("m.", "g.")):
                type_map[head].add(tail)
        anchor = parses[0][2].anchor
        anchor_type = " / ".join(sorted(type_map.get(anchor, set()))) or parses[0][2].anchor_type
        patterns: dict[tuple[str, ...], PathSpec] = {}
        frontier = [(anchor, tuple(), tuple(), frozenset({anchor}))]
        for _ in range(hop_limit):
            next_frontier = []
            for node, hops, nodes, seen_nodes in frontier:
                for target, relation, direction in adjacency.get(node, []):
                    if target in seen_nodes or relation == "common.topic.notable_types":
                        continue
                    source_type = (
                        " / ".join(sorted(type_map.get(node, set())))
                        or (hops[-1].target_type if hops else anchor_type)
                    )
                    target_type = " / ".join(sorted(type_map.get(target, set()))) or "entity"
                    new_hops = hops + (Hop(relation, direction, source_type, target_type),)
                    sequence = relation_sequence(PathSpec(anchor, anchor_type, new_hops, target_type, "webqsp"))
                    patterns.setdefault(sequence, PathSpec(anchor, anchor_type, new_hops, target_type, "webqsp"))
                    next_frontier.append((target, new_hops, nodes + (target,), seen_nodes | {target}))
            frontier = next_frontier
        candidate_counts.append(len(patterns))
        if not (set(patterns) & gold_sequences):
            continue
        diagnostics["gold_in_local_graph"] += 1
        ranked_patterns = sorted(
            patterns.items(),
            key=lambda item: (lexical_path_score(question, item[1]), -len(item[0]), item[0]),
            reverse=True,
        )[:candidate_cap]
        retained = dict(ranked_patterns)
        retained_gold = [retained[sequence] for sequence in gold_sequences if sequence in retained]
        if not retained_gold:
            continue
        diagnostics["gold_after_cap"] += 1
        positive_keys = {path_key(path) for path in retained_gold}
        negatives = []
        for path in retained.values():
            if path_key(path) in positive_keys:
                continue
            data = path_to_dict(path)
            data["negative_type"] = "executable_distractor"
            negatives.append(data)
        primary = retained_gold[0]
        output.append(
            Example(
                example_id=group_id,
                question=question,
                positive_path=path_to_dict(primary),
                alternate_positive_paths=[path_to_dict(path) for path in retained_gold[1:]],
                negative_paths=negatives,
                split="test_executable_webqsp",
                kg="webqsp",
                relation_sequence=list(relation_sequence(primary)),
            )
        )
    diagnostics["average_candidates_before_cap"] = (
        sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    )
    diagnostics["local_graph_recall"] = diagnostics["gold_in_local_graph"] / max(diagnostics["questions"], 1)
    diagnostics["capped_candidate_recall"] = diagnostics["gold_after_cap"] / max(diagnostics["questions"], 1)
    return output, diagnostics


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_dataset(
    kqa_train: Path,
    kqa_val: Path,
    webqsp_test: Path,
    output: Path,
    webqsp_graphs: Path | None = None,
    kqa_kb: Path | None = None,
    webqsp_train: Path | None = None,
    webqsp_train_graphs: Path | None = None,
) -> dict[str, Any]:
    kqa_training_pool = extract_kqa_examples(kqa_train)
    kqa_val_rows = extract_kqa_examples(kqa_val)
    webqsp_rows = extract_webqsp_examples(webqsp_test, webqsp_graphs)
    webqsp_training_rows = (
        extract_webqsp_examples(webqsp_train, webqsp_train_graphs)
        if webqsp_train is not None and webqsp_train.exists()
        else []
    )
    kqa_splits, heldout_relations, heldout_sequences = assign_kqa_splits(kqa_training_pool)
    symmetric_relations = kqa_symmetric_relations(kqa_kb)

    # Training negatives must obey the same split firewall as positive paths.
    kqa_inventory = relation_inventory(kqa_training_pool)
    training_inventory = relation_inventory(kqa_splits["train"] + kqa_splits["dev"])
    webqsp_inventory = relation_inventory(webqsp_rows)
    webqsp_training_inventory = relation_inventory(webqsp_training_rows)
    materialized: dict[str, list[Example]] = {}
    for split, rows in kqa_splits.items():
        if split in {"train", "dev"}:
            materialized[split] = materialize(
                rows,
                split,
                training_inventory,
                forbidden_relations=heldout_relations,
                forbidden_sequences=heldout_sequences,
                symmetric_relations=symmetric_relations,
            )
        else:
            materialized[split] = materialize(
                rows, split, kqa_inventory, symmetric_relations=symmetric_relations
            )
    materialized["test_kqa_val"] = materialize(
        kqa_val_rows,
        "test_kqa_val",
        kqa_inventory,
        symmetric_relations=symmetric_relations,
    )
    materialized["test_cross_kg_webqsp"] = materialize_webqsp(webqsp_rows, webqsp_inventory)
    executable_examples, executable_diagnostics = build_executable_webqsp_examples(
        webqsp_rows,
        webqsp_graphs,
    )
    materialized["test_executable_webqsp"] = executable_examples

    webqsp_training_examples = materialize_webqsp(
        webqsp_training_rows,
        webqsp_training_inventory,
        split_name="train_multi_kg",
    )
    webqsp_train_examples, webqsp_dev_examples = [], []
    for example in webqsp_training_examples:
        if stable_bucket("webqsp-dev:" + example.example_id, 10) == 0:
            example.split = "dev_multi_kg"
            webqsp_dev_examples.append(example)
        else:
            webqsp_train_examples.append(example)
    kqa_train_examples = copy.deepcopy(materialized["train"])
    kqa_dev_examples = copy.deepcopy(materialized["dev"])
    for example in kqa_train_examples:
        example.split = "train_multi_kg"
    for example in kqa_dev_examples:
        example.split = "dev_multi_kg"
    materialized["train_multi_kg"] = kqa_train_examples + webqsp_train_examples
    materialized["dev_multi_kg"] = kqa_dev_examples + webqsp_dev_examples

    counts = {}
    for split, rows in materialized.items():
        counts[split] = write_jsonl(output / f"{split}.jsonl", (asdict(row) for row in rows))
    manifest = {
        "version": 1,
        "training_kg": "kqa_pro",
        "cross_kg_test": "webqsp",
        "heldout_kqa_relations": sorted(heldout_relations),
        "heldout_kqa_compositions": [list(sequence) for sequence in sorted(heldout_sequences)],
        "symmetric_kqa_relations_excluded_from_direction_negatives": sorted(symmetric_relations),
        "counts": counts,
        "leakage_boundaries": {
            "answer_entity_names_in_path": False,
            "intermediate_entity_names_in_path": False,
            "kqa_only_regime_uses_webqsp_for_training": False,
            "unseen_relation_examples_used_for_training": False,
            "unseen_composition_examples_used_for_training": False,
        },
        "multi_kg_regime": {
            "training_kgs": ["kqa_pro", "webqsp"],
            "webqsp_test_used_for_training": False,
            "kqa_heldout_relations_used_for_training": False,
            "kqa_heldout_compositions_used_for_training": False,
        },
        "executable_webqsp": executable_diagnostics,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
