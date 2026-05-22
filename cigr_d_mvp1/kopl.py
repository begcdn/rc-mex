from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kg import KnowledgeGraph


@dataclass
class ExecValue:
    kind: str
    entity_ids: set[str] | None = None
    scalar: Any = None

    @classmethod
    def entity_set(cls, ids: set[str]) -> "ExecValue":
        return cls(kind="entity_set", entity_ids=ids)

    @classmethod
    def scalar_value(cls, value: Any) -> "ExecValue":
        return cls(kind="scalar", scalar=value)


@dataclass
class RelationGroundingInstance:
    instance_id: str
    question: str
    program_index: int
    step_index: int
    current_entity_ids: set[str]
    gold_predicate: str
    gold_direction: str
    answer: str | None


class UnsupportedProgram(Exception):
    pass


class KoPLExecutor:
    """Small KoPL subset executor for gold-prefix relation grounding.

    MVP1 only needs to execute enough of a gold prefix to recover the current
    entity set before a Relate step. Unsupported prefixes are skipped and
    counted by the extractor.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def execute_step(self, program: list[dict[str, Any]], index: int, cache: dict[int, ExecValue]) -> ExecValue:
        if index in cache:
            return cache[index]
        if index < 0 or index >= len(program):
            raise UnsupportedProgram(f"bad dependency index: {index}")

        step = program[index]
        fn = step.get("function")
        inputs = step.get("inputs", []) or []
        deps = step.get("dependencies", []) or []
        dep_values = [self.execute_step(program, dep, cache) for dep in deps]

        if fn == "Find":
            if not inputs:
                raise UnsupportedProgram("Find without input")
            value = ExecValue.entity_set(self.graph.find_entities(inputs[0]))
        elif fn == "FindAll":
            value = ExecValue.entity_set(self.graph.all_entity_ids())
        elif fn == "Relate":
            source = require_entity_set(dep_values, fn)
            if len(inputs) < 2:
                raise UnsupportedProgram("Relate without predicate/direction")
            output, _ = self.graph.follow(source, inputs[0], inputs[1])
            value = ExecValue.entity_set(output)
        elif fn == "FilterConcept":
            concept_ids = self.graph.find_concepts(inputs[0]) if inputs else set()
            source = dep_values[0].entity_ids if dep_values else self.graph.all_entity_ids()
            if source is None:
                raise UnsupportedProgram("FilterConcept dependency is not an entity set")
            value = ExecValue.entity_set(
                {eid for eid in source if self.graph.is_instance_of_any(eid, concept_ids)}
            )
        elif fn == "And":
            sets = [require_entity_set([value], fn) for value in dep_values]
            if len(sets) != 2:
                raise UnsupportedProgram("And requires two dependencies")
            value = ExecValue.entity_set(sets[0] & sets[1])
        elif fn == "Or":
            sets = [require_entity_set([value], fn) for value in dep_values]
            if len(sets) != 2:
                raise UnsupportedProgram("Or requires two dependencies")
            value = ExecValue.entity_set(sets[0] | sets[1])
        elif fn in {"FilterStr", "FilterNum", "FilterYear", "FilterDate"}:
            value = self._filter_attribute(fn, inputs, dep_values)
        elif fn in {"Count", "QueryName", "QueryAttr", "SelectAmong", "SelectBetween"}:
            raise UnsupportedProgram(f"{fn} returns a non-set value in MVP1 prefixes")
        elif fn and fn.startswith("QFilter"):
            raise UnsupportedProgram(f"{fn} qualifier filters are not supported in MVP1 prefixes")
        else:
            raise UnsupportedProgram(f"unsupported function: {fn}")

        cache[index] = value
        return value

    def execute_relation_prefix(self, program: list[dict[str, Any]], relation_step_index: int) -> set[str]:
        step = program[relation_step_index]
        deps = step.get("dependencies", []) or []
        if len(deps) != 1:
            raise UnsupportedProgram("Relate step must have one entity-set dependency")
        cache: dict[int, ExecValue] = {}
        value = self.execute_step(program, deps[0], cache)
        if value.entity_ids is None:
            raise UnsupportedProgram("Relate dependency did not produce an entity set")
        return value.entity_ids

    def _filter_attribute(self, fn: str, inputs: list[str], dep_values: list[ExecValue]) -> ExecValue:
        if len(inputs) < 2:
            raise UnsupportedProgram(f"{fn} missing key/value inputs")
        source = require_entity_set(dep_values, fn)
        key, expected = inputs[0], inputs[1]
        op = inputs[2] if len(inputs) > 2 else "="
        kept = set()
        for entity_id in source:
            values = self.graph.attribute_values(entity_id, key)
            if any(compare_attribute(value, expected, op) for value in values):
                kept.add(entity_id)
        return ExecValue.entity_set(kept)


def require_entity_set(values: list[ExecValue], fn: str) -> set[str]:
    if len(values) != 1:
        raise UnsupportedProgram(f"{fn} requires one entity-set dependency")
    entity_ids = values[0].entity_ids
    if entity_ids is None:
        raise UnsupportedProgram(f"{fn} dependency is not an entity set")
    return entity_ids


def compare_attribute(value: dict[str, Any], expected: str, op: str) -> bool:
    actual = value.get("value")
    if op in {"=", "=="}:
        return str(actual).casefold() == str(expected).casefold()
    try:
        actual_num = float(actual)
        expected_num = float(expected)
    except (TypeError, ValueError):
        return False
    if op in {">", "greater"}:
        return actual_num > expected_num
    if op in {"<", "less"}:
        return actual_num < expected_num
    if op in {">=", "ge"}:
        return actual_num >= expected_num
    if op in {"<=", "le"}:
        return actual_num <= expected_num
    return False


def extract_relation_grounding_instances(
    samples: list[dict[str, Any]],
    graph: KnowledgeGraph,
    split_name: str,
    max_instances: int | None = None,
    max_questions: int | None = None,
) -> tuple[list[RelationGroundingInstance], dict[str, int]]:
    executor = KoPLExecutor(graph)
    instances: list[RelationGroundingInstance] = []
    stats = {
        "questions_seen": 0,
        "relation_steps_seen": 0,
        "instances_created": 0,
        "unsupported_prefix": 0,
        "empty_current_set": 0,
        "malformed_relate_step": 0,
    }
    for program_index, sample in enumerate(samples):
        if max_questions is not None and stats["questions_seen"] >= max_questions:
            break
        stats["questions_seen"] += 1
        program = sample.get("program", []) or []
        for step_index, step in enumerate(program):
            if step.get("function") != "Relate":
                continue
            stats["relation_steps_seen"] += 1
            inputs = step.get("inputs", []) or []
            if len(inputs) < 2:
                stats["malformed_relate_step"] += 1
                continue
            try:
                current_ids = executor.execute_relation_prefix(program, step_index)
            except UnsupportedProgram:
                stats["unsupported_prefix"] += 1
                continue
            if not current_ids:
                stats["empty_current_set"] += 1
                continue
            instances.append(
                RelationGroundingInstance(
                    instance_id=f"{split_name}:{program_index}:{step_index}",
                    question=sample.get("question", ""),
                    program_index=program_index,
                    step_index=step_index,
                    current_entity_ids=current_ids,
                    gold_predicate=inputs[0],
                    gold_direction=inputs[1],
                    answer=sample.get("answer"),
                )
            )
            stats["instances_created"] += 1
            if max_instances is not None and len(instances) >= max_instances:
                return instances, stats
    return instances, stats
