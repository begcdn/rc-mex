"""Execution-conditioned query-graph proposals and proof scoring.

The search state is a grounded query family with its complete binding set.
Individual answer entities never compete in the beam.  A bounded structural
grammar supplements learned topology proposals; KG execution then exposes
only locally admissible predicate and type assignments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.factor_graph_ir import (
    QueryGraph,
    abstract_graph,
    canonicalize,
    compiler_input,
    graph_key,
    parse_graph,
    semantic_leaf,
    serialize_graph,
)


TOPOLOGY_BEAM = 10
FAMILY_BEAM = 64

Binding = tuple[tuple[str, str], ...]
Family = tuple[tuple[str | None, ...], tuple[str | None, ...], set[Binding], float, int]


class Runtime(Protocol):
    outgoing: Mapping[str, Mapping[str, set[str]]]
    incoming: Mapping[str, Mapping[str, set[str]]]

    def anchor_ids(self, name: str) -> set[str]: ...
    def answer_name(self, entity_id: str) -> str: ...
    def entity_type_names(self, entity_id: str) -> set[str]: ...


@dataclass(frozen=True)
class GroundedCandidate:
    proposal_score: float
    graph: QueryGraph
    answers: frozenset[str]
    binding_count: int
    learned_topology: bool


def binding_key(assignments: Mapping[str, str]) -> Binding:
    return tuple(sorted(assignments.items()))


def standardized(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale if scale > 1e-8 else 0.0 for value in values]


def score_answers(predicted: set[str], gold: set[str]) -> dict[str, float | bool]:
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": predicted == gold,
        "hits_at_1": bool(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class KQARuntime:
    def __init__(self, kb_path: str | Path):
        import json

        self.raw = json.load(open(kb_path, encoding="utf-8"))
        self.graph = KnowledgeGraph(self.raw)
        self.outgoing = defaultdict(lambda: defaultdict(set))
        self.incoming = defaultdict(lambda: defaultdict(set))
        for source, entity in self.raw["entities"].items():
            for edge in entity.get("relations", []):
                target, predicate = str(edge["object"]), str(edge["predicate"])
                head, tail = (source, target) if str(edge["direction"]) == "forward" else (target, source)
                self.outgoing[head][predicate].add(tail)
                self.incoming[tail][predicate].add(head)

    def anchor_ids(self, name: str) -> set[str]:
        return self.graph.find_entities(name) | self.graph.find_concepts(name)

    def answer_name(self, entity_id: str) -> str:
        return self.graph.entity_name(entity_id)

    def entity_type_names(self, entity_id: str) -> set[str]:
        return {self.graph.entity_name(type_id) for type_id in self.graph.entity_type_ids(entity_id)}


class WebSubgraphRuntime:
    def __init__(self, row: dict):
        self.outgoing = defaultdict(lambda: defaultdict(set))
        self.incoming = defaultdict(lambda: defaultdict(set))
        self.nodes: set[str] = set()
        for raw_head, raw_relation, raw_tail in row.get("graph", []):
            head, relation, tail = str(raw_head), str(raw_relation), str(raw_tail)
            self.outgoing[head][relation].add(tail)
            self.incoming[tail][relation].add(head)
            self.nodes.update((head, tail))

    def anchor_ids(self, name: str) -> set[str]:
        normalized = normalize_text(name)
        return {node for node in self.nodes if normalize_text(node) == normalized}

    def answer_name(self, entity_id: str) -> str:
        return entity_id

    def entity_type_names(self, entity_id: str) -> set[str]:
        return set()


class FactorGraphModels:
    """The four frozen learned potentials used by the controlled evaluation."""

    def __init__(self, model_dir: str | Path):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        required = {
            "topology": model_dir / "topology",
            "slot_scorer": model_dir / "slot_scorer",
            "generator": model_dir / "generator",
            "denotation_ranker": model_dir / "denotation_ranker",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing factor-graph model directories: " + ", ".join(missing))
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.topology_tokenizer = AutoTokenizer.from_pretrained(required["topology"], local_files_only=True)
        self.topology = AutoModelForSeq2SeqLM.from_pretrained(required["topology"], local_files_only=True).to(self.device).eval()
        self.slot_tokenizer = AutoTokenizer.from_pretrained(required["slot_scorer"], local_files_only=True)
        self.slot = AutoModelForSequenceClassification.from_pretrained(required["slot_scorer"], local_files_only=True).to(self.device).eval()
        self.generator_tokenizer = AutoTokenizer.from_pretrained(required["generator"], local_files_only=True)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(required["generator"], local_files_only=True).to(self.device).eval()
        self.denotation_tokenizer = AutoTokenizer.from_pretrained(required["denotation_ranker"], local_files_only=True)
        self.denotation = AutoModelForSequenceClassification.from_pretrained(required["denotation_ranker"], local_files_only=True).to(self.device).eval()

    def topology_beam(self, question: str, slots: dict[str, str]) -> list[tuple[QueryGraph, float]]:
        encoded = self.topology_tokenizer(compiler_input(question, slots), return_tensors="pt", truncation=True, max_length=192).to(self.device)
        with self.torch.no_grad():
            generated = self.topology.generate(
                **encoded,
                max_new_tokens=192,
                num_beams=TOPOLOGY_BEAM,
                num_return_sequences=TOPOLOGY_BEAM,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        rows: list[tuple[QueryGraph, float]] = []
        seen = set()
        for sequence, score in zip(generated.sequences, generated.sequences_scores):
            topology = parse_graph(self.topology_tokenizer.decode(sequence, skip_special_tokens=True))
            if topology is None or any(predicate != "P" for _, predicate, _ in topology.edges):
                continue
            if any(type_name != "T" for _, type_name in topology.types):
                continue
            if graph_key(topology) not in seen:
                seen.add(graph_key(topology))
                rows.append((topology, float(score.detach().cpu())))
        return rows

    def score_slots(self, question: str, texts: list[str]) -> list[float]:
        values: list[float] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            inputs = self.slot_tokenizer([question] * len(batch), batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                values.extend(self.slot(**inputs).logits.squeeze(-1).detach().cpu().tolist())
        return values

    def _sequence_scores(self, model, tokenizer, sources: list[str], targets: list[str]) -> list[float]:
        import torch.nn.functional as functional

        values: list[float] = []
        for start in range(0, len(targets), 64):
            source_batch, target_batch = sources[start : start + 64], targets[start : start + 64]
            inputs = tokenizer(source_batch, padding=True, truncation=True, max_length=192, return_tensors="pt").to(self.device)
            labels = tokenizer(text_target=target_batch, padding=True, truncation=True, max_length=256, return_tensors="pt")["input_ids"].to(self.device)
            labels[labels == tokenizer.pad_token_id] = -100
            with self.torch.no_grad():
                logits = model(**inputs, labels=labels).logits
            losses = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none", ignore_index=-100
            ).reshape(labels.shape)
            counts = (labels != -100).sum(dim=1).clamp_min(1)
            values.extend((-(losses.sum(dim=1) / counts)).detach().cpu().tolist())
        return values

    def score_generator(self, question: str, slots: dict[str, str], candidates: list[GroundedCandidate], semantic_schema: bool) -> list[float]:
        graphs = []
        for candidate in candidates:
            graph = candidate.graph
            if semantic_schema:
                graph = QueryGraph(
                    tuple((subject, semantic_leaf(predicate), object_) for subject, predicate, object_ in graph.edges),
                    graph.types,
                    graph.output,
                    graph.entities,
                )
            graphs.append(graph)
        sources = [compiler_input(question, slots)] * len(graphs)
        return self._sequence_scores(self.generator, self.generator_tokenizer, sources, [serialize_graph(graph) for graph in graphs])

    def score_topology(self, question: str, slots: dict[str, str], candidates: list[GroundedCandidate]) -> list[float]:
        sources = [compiler_input(question, slots)] * len(candidates)
        targets = [serialize_graph(abstract_graph(candidate.graph)) for candidate in candidates]
        return self._sequence_scores(self.topology, self.topology_tokenizer, sources, targets)

    def score_denotation(self, question: str, slots: dict[str, str], candidates: list[GroundedCandidate]) -> list[float]:
        entities = " ; ".join(f"{slot} = {name}" for name, slot in slots.items())
        texts = [
            f"linked entities: {entities} ; complete executable query graph: {serialize_graph(candidate.graph)} ; "
            f"execution result count: {len(candidate.answers)} ; execution result examples: {' ; '.join(sorted(candidate.answers)[:10])}"
            for candidate in candidates
        ]
        values: list[float] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            inputs = self.denotation_tokenizer([question] * len(batch), batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                values.extend(self.denotation(**inputs).logits.squeeze(-1).detach().cpu().tolist())
        return values


def slot_context(topology: QueryGraph, slots: dict[str, str], *, edge_index: int | None = None, type_index: int | None = None, candidate: str) -> str:
    entities = " ; ".join(f"{slot} = {name}" for name, slot in slots.items())
    parts = [f"linked entities: {entities}", f"topology: {serialize_graph(topology)}"]
    if edge_index is not None:
        subject, _, object_ = topology.edges[edge_index]
        parts.extend((f"target edge: {subject} -> {object_}", f"candidate predicate: {candidate.replace('_', ' ').replace('.', ' ')}"))
    else:
        node, _ = topology.types[int(type_index)]
        parts.extend((f"target type node: {node}", f"candidate type: {candidate}"))
    return " ; ".join(parts)


def two_edge_path_grammar(slots: dict[str, str]) -> list[QueryGraph]:
    """Bounded connected path grammar, independent of KG schema labels."""

    if len(slots) != 1:
        return []
    anchor = next(iter(slots.values()))
    rows = []
    for first_reverse, second_reverse in itertools.product((False, True), repeat=2):
        first = ("V0", "P", anchor) if first_reverse else (anchor, "P", "V0")
        second = ("A", "P", "V0") if second_reverse else ("V0", "P", "A")
        for mask in range(4):
            types = tuple((node, "T") for bit, node in enumerate(("A", "V0")) if mask & (1 << bit))
            rows.append(canonicalize(QueryGraph((first, second), types, "A", (anchor,))))
    return rows


def edge_order(topology: QueryGraph) -> tuple[int, ...] | None:
    bound, remaining, order = set(topology.entities), set(range(len(topology.edges))), []
    while remaining:
        choices = []
        for index in remaining:
            subject, _, object_ = topology.edges[index]
            bound_count = int(subject in bound) + int(object_ in bound)
            if bound_count:
                choices.append((-bound_count, index))
        if not choices:
            return None
        _, chosen = min(choices)
        order.append(chosen)
        bound.update((topology.edges[chosen][0], topology.edges[chosen][2]))
        remaining.remove(chosen)
    return tuple(order)


def initial_bindings(runtime: Runtime, topology: QueryGraph, slots: dict[str, str]) -> set[Binding]:
    slot_names = {slot: name for name, slot in slots.items()}
    values = [sorted(runtime.anchor_ids(slot_names.get(slot, ""))) for slot in topology.entities]
    if any(not item for item in values):
        return set()
    return {binding_key(dict(zip(topology.entities, combination))) for combination in itertools.product(*values)}


def relation_expansions(runtime: Runtime, bindings: set[Binding], edge: tuple[str, str, str]) -> dict[str, set[Binding]]:
    subject, _, object_ = edge
    grouped: dict[str, set[Binding]] = defaultdict(set)
    for raw in bindings:
        assignment = dict(raw)
        subject_bound, object_bound = subject in assignment, object_ in assignment
        if subject_bound and object_bound:
            for relation, targets in runtime.outgoing.get(assignment[subject], {}).items():
                if assignment[object_] in targets:
                    grouped[relation].add(raw)
        elif subject_bound:
            for relation, targets in runtime.outgoing.get(assignment[subject], {}).items():
                for target in targets:
                    updated = dict(assignment); updated[object_] = target
                    grouped[relation].add(binding_key(updated))
        elif object_bound:
            for relation, sources in runtime.incoming.get(assignment[object_], {}).items():
                for source in sources:
                    updated = dict(assignment); updated[subject] = source
                    grouped[relation].add(binding_key(updated))
    return grouped


def type_expansions(runtime: Runtime, bindings: set[Binding], node: str) -> dict[str, set[Binding]]:
    grouped: dict[str, set[Binding]] = defaultdict(set)
    for raw in bindings:
        assignment = dict(raw)
        if node in assignment:
            for type_name in runtime.entity_type_names(assignment[node]):
                grouped[type_name].add(raw)
    return grouped


def merge_families(children: Iterable[Family], cap: int = FAMILY_BEAM) -> list[Family]:
    merged: dict[tuple, list] = {}
    for relations, types, bindings, score_sum, score_count in children:
        key = relations, types
        if key not in merged:
            merged[key] = [set(), score_sum, score_count]
        merged[key][0].update(bindings)
        merged[key][1] = max(merged[key][1], score_sum)
    rows = [(relations, types, values[0], values[1], values[2]) for (relations, types), values in merged.items()]
    return sorted(rows, key=lambda row: row[3] / max(row[4], 1), reverse=True)[:cap]


def ground_topology(models: FactorGraphModels, runtime: Runtime, question: str, slots: dict[str, str], topology: QueryGraph, topology_score: float, learned: bool) -> list[GroundedCandidate]:
    order, bindings = edge_order(topology), initial_bindings(runtime, topology, slots)
    if order is None or not bindings:
        return []
    families: list[Family] = [(tuple([None] * len(topology.edges)), tuple([None] * len(topology.types)), bindings, 0.0, 0)]
    for edge_index in order:
        expansions, options = [], set()
        for family in families:
            groups = relation_expansions(runtime, family[2], topology.edges[edge_index])
            expansions.append(groups); options.update(groups)
        if not options:
            return []
        option_list = sorted(options)
        scores = dict(zip(option_list, models.score_slots(question, [slot_context(topology, slots, edge_index=edge_index, candidate=item) for item in option_list])))
        children = []
        for family, groups in zip(families, expansions):
            for relation, next_bindings in groups.items():
                relations = list(family[0]); relations[edge_index] = relation
                children.append((tuple(relations), family[1], next_bindings, family[3] + scores[relation], family[4] + 1))
        families = merge_families(children)
    for type_index, (node, _) in enumerate(topology.types):
        expansions, options = [], set()
        for family in families:
            groups = type_expansions(runtime, family[2], node)
            expansions.append(groups); options.update(groups)
        if not options:
            return []
        option_list = sorted(options)
        scores = dict(zip(option_list, models.score_slots(question, [slot_context(topology, slots, type_index=type_index, candidate=item) for item in option_list])))
        children = []
        for family, groups in zip(families, expansions):
            for type_name, next_bindings in groups.items():
                types = list(family[1]); types[type_index] = type_name
                children.append((family[0], tuple(types), next_bindings, family[3] + scores[type_name], family[4] + 1))
        families = merge_families(children)
    grounded = []
    for relations, types, complete_bindings, score_sum, score_count in families:
        graph = QueryGraph(
            tuple((subject, str(relations[index]), object_) for index, (subject, _, object_) in enumerate(topology.edges)),
            tuple((node, str(types[index])) for index, (node, _) in enumerate(topology.types)),
            topology.output,
            topology.entities,
        )
        answers = frozenset(
            normalize_text(runtime.answer_name(dict(raw)[topology.output]))
            for raw in complete_bindings
            if topology.output in dict(raw)
        )
        grounded.append(GroundedCandidate(topology_score + score_sum / max(score_count, 1), graph, answers, len(complete_bindings), learned))
    return grounded


def build_candidate_pool(models: FactorGraphModels, runtime: Runtime, question: str, slots: dict[str, str]) -> tuple[list[GroundedCandidate], list[QueryGraph]]:
    learned_rows = models.topology_beam(question, slots)
    learned_keys = {graph_key(topology) for topology, _ in learned_rows}
    proposed = list(learned_rows)
    seen = set(learned_keys)
    for topology in two_edge_path_grammar(slots):
        if graph_key(topology) not in seen:
            seen.add(graph_key(topology)); proposed.append((topology, 0.0))
    candidates: dict[tuple, GroundedCandidate] = {}
    for topology, topology_score in proposed:
        learned = graph_key(topology) in learned_keys
        for candidate in ground_topology(models, runtime, question, slots, topology, topology_score, learned):
            key = graph_key(candidate.graph)
            if key not in candidates or candidate.proposal_score > candidates[key].proposal_score:
                candidates[key] = candidate
    return list(candidates.values()), [topology for topology, _ in learned_rows]


def rank_candidate_pool(models: FactorGraphModels, question: str, slots: dict[str, str], candidates: list[GroundedCandidate], semantic_schema: bool) -> dict[str, list[tuple[float, GroundedCandidate]]]:
    if not candidates:
        return {name: [] for name in ("learned_product", "grammar_product", "grammar_topology_product", "source_aware")}
    generator = standardized(models.score_generator(question, slots, candidates, semantic_schema))
    denotation = standardized(models.score_denotation(question, slots, candidates))
    topology = standardized(models.score_topology(question, slots, candidates))
    arms: dict[str, list[tuple[float, GroundedCandidate]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        product = 0.43 * generator[index] + 0.57 * denotation[index]
        topology_product = (generator[index] + denotation[index] + topology[index]) / 3.0
        if candidate.learned_topology:
            arms["learned_product"].append((product, candidate))
        arms["grammar_product"].append((product, candidate))
        arms["grammar_topology_product"].append((topology_product, candidate))
        arms["source_aware"].append((product if candidate.learned_topology else topology_product, candidate))
    for name in arms:
        arms[name].sort(key=lambda row: row[0], reverse=True)
    return dict(arms)
