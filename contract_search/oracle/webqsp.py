from __future__ import annotations

from typing import Any

from inverse_verifier.data import parse_sparql_path

from ..program import Constraint, Hop, Order, ReferenceProgram


def compile_parse(parse: dict[str, Any]) -> ReferenceProgram | None:
    chain = parse.get("InferentialChain") or []
    topic = parse.get("TopicEntityMid") or ""
    if not topic or not chain:
        return None
    directed = parse_sparql_path(parse.get("Sparql", ""), topic, chain)
    if not directed:
        return None

    temporal = parse.get("Time") or {}
    optional_indices = set(temporal.get("AssociatedConstraints") or [])
    constraints = tuple(
        Constraint(
            source_index=int(item["SourceNodeIndex"]),
            relation=item["NodePredicate"],
            operator=item["Operator"],
            argument=item["Argument"],
            argument_type=item["ArgumentType"],
            optional=index in optional_indices,
        )
        for index, item in enumerate(parse.get("Constraints") or [])
    )
    raw_order = parse.get("Order")
    order = None
    if raw_order:
        order = Order(
            source_index=int(raw_order["SourceNodeIndex"]),
            relation=raw_order["NodePredicate"],
            descending=raw_order["SortOrder"] == "Descending",
            offset=int(raw_order.get("Start", 0)),
            count=int(raw_order.get("Count", 1)),
        )
    return ReferenceProgram(
        topic_entity=topic,
        hops=tuple(Hop(relation, direction) for relation, direction in directed),
        constraints=constraints,
        order=order,
    )
