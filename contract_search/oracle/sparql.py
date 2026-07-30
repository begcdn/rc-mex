from __future__ import annotations

import re

from ..conjunctive import (
    Atom,
    ConjunctiveProgram,
    OptionalRelationFilter,
    QueryOrder,
    ValueFilter,
)


TERM = r'(?:\?[A-Za-z0-9_]+|ns:[A-Za-z0-9_\.]+|"(?:[^"\\]|\\.)*"(?:\^\^xsd:[A-Za-z]+)?)'
TRIPLE = re.compile(
    rf"^\s*({TERM})\s+ns:([A-Za-z0-9_\.]+)\s+({TERM})\s*\.\s*$"
)
SELECT = re.compile(r"SELECT\s+DISTINCT\s+(\?[A-Za-z0-9_]+)", re.IGNORECASE)
EXCLUSION = re.compile(r"FILTER\s*\(\s*(\?\w+)\s*!=\s*([^\s\)]+)\s*\)")
COMPARISON = re.compile(
    rf"FILTER\s*\(\s*(\?\w+)\s*(<=|>=|<|>|=|!=)\s*({TERM})\s*\)\s*\.?"
)
ORDER = re.compile(
    r"ORDER BY\s+(?:(DESC|ASC)\s*\(\s*)?(?:xsd:\w+\s*\(\s*)?"
    r"(\?\w+)\s*\)?\s*\)?(?:\s+LIMIT\s+(\d+))?",
    re.IGNORECASE,
)
OPTIONAL = re.compile(
    rf"FILTER\s*\(\s*NOT EXISTS\s*\{{\s*(\?\w+)\s+ns:([A-Za-z0-9_\.]+)"
    rf"\s+\?\w+\s*\}}\s*\|\|\s*EXISTS\s*\{{.*?"
    rf"FILTER\s*\(\s*(?:xsd:\w+\s*\(\s*)?\?\w+\s*\)?\s*"
    rf"(<=|>=|<|>|=|!=)\s*({TERM})\s*\)\s*\}}\s*\)",
    re.IGNORECASE | re.DOTALL,
)

OPERATORS = {
    "=": "Equal",
    "!=": "NotEqual",
    "<": "LessThan",
    ">": "GreaterThan",
    "<=": "LessOrEqual",
    ">=": "GreaterOrEqual",
}


def _constant(value: str) -> str:
    return value.removeprefix("ns:")


def compile_sparql(sparql: str) -> ConjunctiveProgram | None:
    if "#MANUAL SPARQL" in sparql or re.search(r"\bCOUNT\s*\(", sparql, re.IGNORECASE):
        return None
    selected = SELECT.search(sparql)
    if not selected:
        return None

    optional_spans = [match.span() for match in OPTIONAL.finditer(sparql)]

    def inside_optional(start: int) -> bool:
        return any(left <= start < right for left, right in optional_spans)

    atoms = []
    offset = 0
    for line in sparql.splitlines(keepends=True):
        match = TRIPLE.match(line.rstrip("\n"))
        if match and not inside_optional(offset):
            head, relation, tail = match.groups()
            atoms.append(Atom(_constant(head), relation, _constant(tail)))
        offset += len(line)
    if not atoms:
        return None

    exclusions = tuple(
        (left, _constant(right)) for left, right in EXCLUSION.findall(sparql)
    )
    optional_filters = tuple(
        OptionalRelationFilter(
            source=source,
            relation=relation,
            operator=OPERATORS[operator],
            argument=_constant(argument),
        )
        for source, relation, operator, argument in OPTIONAL.findall(sparql)
    )

    masked = OPTIONAL.sub("", sparql)
    filters = tuple(
        ValueFilter(variable, OPERATORS[operator], _constant(argument))
        for variable, operator, argument in COMPARISON.findall(masked)
        if not (operator == "!=" and argument.startswith(("ns:", "?")))
    )
    order_match = ORDER.search(sparql)
    order = None
    if order_match:
        direction, variable, limit = order_match.groups()
        order = QueryOrder(
            variable=variable,
            descending=(direction or "").upper() == "DESC",
            limit=int(limit or 1),
        )
    return ConjunctiveProgram(
        select=selected.group(1),
        atoms=tuple(atoms),
        filters=filters,
        optional_filters=optional_filters,
        exclusions=exclusions,
        order=order,
    )
