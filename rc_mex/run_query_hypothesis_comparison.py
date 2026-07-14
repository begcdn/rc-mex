"""Controlled comparison of binding-level and query-level KGQA inference.

The experiment freezes one schema-independent constraint graph per question.
For each semantic encoder it then generates one shared pool of executable
grounded queries.  Two selectors consume that exact pool:

* ``per_binding`` treats each satisfying answer binding as an independent
  prediction and assigns it a uniform share of the grounded query score;
* ``query_denotation`` ranks the grounded query once and returns its complete
  answer denotation.

This isolates the representation hypothesis.  Parsing, entity anchors,
relation proposals, relation-candidate limits, execution, and type evidence
are identical inside each encoder comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.judges import http_json, normalize_base_url
from cigr_d_mvp1.kg import normalize_text
from rc_mex.executable_pattern_search import ExecutablePatternScorer, _pattern_text
from rc_mex.query_hypothesis_solver import (
    ConstraintGraph,
    QueryHypothesis,
    RelationAtom,
    RelationCandidate,
    rank_denotations,
    solve_conjunctive_query,
)
from rc_mex.run_webqsp_path_family import build_kb


RELATION_CANDIDATES_PER_ATOM = 8
HYPOTHESIS_CAP = 1000
SEMANTIC_TEMPERATURE = 0.05
PARSER_PROMPT_VERSION = "constraint_graph_v1"


CONSTRAINT_GRAPH_PROMPT = """Convert the question into a complete schema-independent constraint graph.

Output JSON with exactly these fields:
{
  "variables": [{"id":"v1","role":"short semantic role"}],
  "answer_variable":"v1",
  "atoms":[{"left":"entity_1","predicate":"narrow relation from left to right","right":"v1","clause":"exact question words covered"}],
  "type_constraints":[{"variable":"v1","type":"specific requested type"}],
  "operators":[],
  "coverage":["question clause 1","question clause 2"]
}

Rules:
- entity_1, entity_2, ... refer only to the supplied linked entities in that order.
- Each left/right value is exactly one identifier, such as entity_1 or v1.
- Declare every variable used by an atom.
- The answer_variable must be one declared variable.
- Every explicit relational clause must be represented by one atom.
- Atom direction is literal: left predicate right expresses meaning from left to right.
- Reuse the same variable when clauses constrain the same unknown; this creates a join.
- Connect chain atoms through an intermediate variable.
- Put answer classes such as city, person, country, film, or occupation in type_constraints.
- Keep time, comparison, count, superlative, and value conditions in operators.
- Do not omit clauses, invent facts, use KG relation IDs, or use vague predicates such as related to.

Example 1:
Question: Where did the illustrator of De Divina Proportione die?
Entities: ["De Divina Proportione"]
Output: {"variables":[{"id":"v1","role":"illustrator"},{"id":"v2","role":"place of death"}],"answer_variable":"v2","atoms":[{"left":"entity_1","predicate":"work was illustrated by person","right":"v1","clause":"illustrator of De Divina Proportione"},{"left":"v1","predicate":"person died in place","right":"v2","clause":"where did the illustrator die"}],"type_constraints":[{"variable":"v2","type":"place"}],"operators":[],"coverage":["illustrator of De Divina Proportione","where did the illustrator die"]}

Example 2:
Question: Which films were directed by Christopher Nolan and starred Christian Bale?
Entities: ["Christopher Nolan","Christian Bale"]
Output: {"variables":[{"id":"v1","role":"film"}],"answer_variable":"v1","atoms":[{"left":"entity_1","predicate":"person directed film","right":"v1","clause":"directed by Christopher Nolan"},{"left":"entity_2","predicate":"person starred in film","right":"v1","clause":"starred Christian Bale"}],"type_constraints":[{"variable":"v1","type":"film"}],"operators":[],"coverage":["films","directed by Christopher Nolan","starred Christian Bale"]}

Question: {question}
Entities: {entities}
Return JSON only.
"""


def load_rows(path: str | Path, limit: int) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
        return rows
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("questions", payload.get("results")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list or JSONL file: {path}")
    return payload[:limit] if limit else payload


def parse_dataset_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def row_key(dataset: str, row: Mapping[str, object], index: int) -> str:
    return f"{dataset}:{row.get('id', index)}"


def linked_entity_names(row: Mapping[str, object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in row.get("q_entity", []) if str(value).strip()))


def gold_answers(row: Mapping[str, object]) -> set[str]:
    return {normalize_text(str(value)) for value in row.get("answer", []) if str(value).strip()}


def parse_json_object(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def call_constraint_parser(model: str, question: str, entities: Sequence[str]) -> tuple[dict | None, str, dict]:
    prompt = CONSTRAINT_GRAPH_PROMPT.replace("{question}", question).replace(
        "{entities}", json.dumps(list(entities), ensure_ascii=False)
    )
    host = normalize_base_url(os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0, "num_predict": 1000},
    }
    started = time.perf_counter()
    response = http_json(f"{host}/api/chat", payload)
    latency = time.perf_counter() - started
    raw = str(response.get("message", {}).get("content", ""))
    usage = {
        "prompt_tokens": int(response.get("prompt_eval_count", 0) or 0),
        "completion_tokens": int(response.get("eval_count", 0) or 0),
        "latency_seconds": latency,
    }
    return parse_json_object(raw), raw, usage


def validate_constraint_graph(graph: object, entity_count: int) -> tuple[bool, str]:
    if not isinstance(graph, dict):
        return False, "not_object"
    variables = graph.get("variables")
    atoms = graph.get("atoms")
    if not isinstance(variables, list) or not isinstance(atoms, list) or not atoms:
        return False, "missing_variables_or_atoms"
    variable_ids = {
        str(item.get("id", ""))
        for item in variables
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    answer = str(graph.get("answer_variable", ""))
    if answer not in variable_ids:
        return False, "answer_not_declared_variable"
    anchors = {f"entity_{index}" for index in range(1, entity_count + 1)}
    allowed = variable_ids | anchors
    adjacency: dict[str, set[str]] = defaultdict(set)
    used_variables: set[str] = set()
    for atom in atoms:
        if not isinstance(atom, dict):
            return False, "bad_atom"
        left = str(atom.get("left", ""))
        right = str(atom.get("right", ""))
        predicate = str(atom.get("predicate", "")).strip()
        if left not in allowed or right not in allowed or not predicate:
            return False, "bad_atom_endpoint_or_predicate"
        adjacency[left].add(right)
        adjacency[right].add(left)
        used_variables.update({value for value in (left, right) if value in variable_ids})
    if not anchors:
        return False, "no_linked_entity"
    reachable = set(anchors)
    queue = deque(anchors)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    if answer not in reachable:
        return False, "answer_disconnected_from_entities"
    if not used_variables <= reachable:
        return False, "unanchored_variable_component"
    operators = graph.get("operators", [])
    if operators is not None and not isinstance(operators, list):
        return False, "bad_operators"
    return True, ""


def canonical_term(term: str, variable_ids: set[str]) -> str:
    return f"?{term}" if term in variable_ids and not term.startswith("?") else term


def compile_constraint_graph(graph: Mapping[str, object]) -> tuple[ConstraintGraph, list[dict]]:
    variable_ids = {
        str(item["id"])
        for item in graph.get("variables", [])
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    atoms = tuple(
        RelationAtom(
            atom_id=f"a{index}",
            left=canonical_term(str(atom["left"]), variable_ids),
            predicate=str(atom["predicate"]),
            right=canonical_term(str(atom["right"]), variable_ids),
        )
        for index, atom in enumerate(graph.get("atoms", []), start=1)
    )
    compiled = ConstraintGraph(
        atoms=atoms,
        answer_variable=canonical_term(str(graph["answer_variable"]), variable_ids),
        variables=tuple(sorted(canonical_term(value, variable_ids) for value in variable_ids)),
    )
    constraints = []
    for item in graph.get("type_constraints", []) or []:
        if not isinstance(item, dict):
            continue
        variable = str(item.get("variable", ""))
        type_phrase = str(item.get("type", "")).strip()
        if variable in variable_ids and type_phrase:
            constraints.append({"variable": canonical_term(variable, variable_ids), "type": type_phrase})
    return compiled, constraints


def softmax(values: Sequence[float], temperature: float = SEMANTIC_TEMPERATURE) -> list[float]:
    if not values:
        return []
    scaled = [float(value) / temperature for value in values]
    pivot = max(scaled)
    exponentials = [math.exp(value - pivot) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def relation_extensions(kb: Mapping[str, object]) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    extensions: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for source, entity in kb.get("entities", {}).items():
        for edge in entity.get("relations", []):
            key = str(edge["predicate"]), str(edge["direction"])
            extensions[key].add((str(source), str(edge["object"])))
    return {key: tuple(sorted(values)) for key, values in extensions.items()}


def endpoint_value(term: str, binding: Mapping[str, str], constants: Mapping[str, str]) -> str | None:
    return constants.get(term, binding.get(term))


def relation_is_executable(
    atom: RelationAtom,
    hypothesis: QueryHypothesis,
    extension: Sequence[tuple[str, str]],
    constants: Mapping[str, str],
) -> bool:
    for binding in hypothesis.binding_dicts():
        expected_left = endpoint_value(atom.left, binding, constants)
        expected_right = endpoint_value(atom.right, binding, constants)
        if any(
            (expected_left is None or expected_left == left)
            and (expected_right is None or expected_right == right)
            for left, right in extension
        ):
            return True
    return False


def type_description(kb: Mapping[str, object], entity_id: str) -> str:
    names = [
        str(kb.get("concepts", {}).get(concept_id, {}).get("name", ""))
        for concept_id in kb.get("entities", {}).get(entity_id, {}).get("instanceOf", [])
    ]
    return "; ".join(name for name in names if name) or "unknown type"


def build_proposer(kb: Mapping[str, object], scorer: ExecutablePatternScorer):
    extensions = relation_extensions(kb)

    def propose(
        atom: RelationAtom,
        hypothesis: QueryHypothesis,
        constants: Mapping[str, str],
    ) -> Iterable[RelationCandidate]:
        keys = [
            key
            for key, extension in extensions.items()
            if relation_is_executable(atom, hypothesis, extension, constants)
        ]
        texts = [_pattern_text((key,)) for key in keys]
        scores = scorer.score(atom.predicate, texts) if texts else []
        probabilities = softmax(scores)
        for key, text, score, probability in zip(keys, texts, scores, probabilities):
            yield RelationCandidate(
                relation_id=key[0],
                direction=key[1],
                log_score=math.log(max(probability, 1e-12)),
                extension=extensions[key],
                metadata={
                    "pattern_text": text,
                    "semantic_score": float(score),
                    "semantic_probability": float(probability),
                },
            )

    return propose


def build_type_factor(kb: Mapping[str, object], scorer: ExecutablePatternScorer, constraints: Sequence[dict]):
    def factor(
        graph: ConstraintGraph,
        hypothesis: QueryHypothesis,
        constants: Mapping[str, str],
    ) -> float:
        log_score = 0.0
        for constraint in constraints:
            values = sorted(hypothesis.denotation(str(constraint["variable"])))
            descriptions = [type_description(kb, value) for value in values]
            similarities = scorer.score(str(constraint["type"]), descriptions) if descriptions else []
            compatibilities = [
                0.5 if description == "unknown type" else min(0.95, max(0.05, (similarity + 1.0) / 2.0))
                for description, similarity in zip(descriptions, similarities)
            ]
            compatibility = sum(compatibilities) / len(compatibilities) if compatibilities else 0.5
            log_score += math.log(max(compatibility, 1e-12))
        return log_score

    return factor


def answer_scores(predicted: set[str], gold: set[str]) -> dict[str, float | bool]:
    overlap = predicted & gold
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold) if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted == gold,
        "hits_at_1": bool(overlap),
    }


def select_per_binding(graph: ConstraintGraph, hypotheses: Sequence[QueryHypothesis]) -> tuple[set[str], list[dict]]:
    """Rank singleton bindings while preserving the same query pool.

    The uniform ``-log |bindings|`` term is the controlled form of the old
    binding-level assumption: a grounded query's score is divided among its
    satisfying assignments.  This is deliberately absent from query-level
    selection.
    """
    best: dict[str, tuple[float, QueryHypothesis]] = {}
    for hypothesis in hypotheses:
        binding_score = hypothesis.score - math.log(max(1, len(hypothesis.bindings)))
        for binding in hypothesis.binding_dicts():
            answer = binding.get(graph.answer_variable)
            if answer and (answer not in best or binding_score > best[answer][0]):
                best[answer] = binding_score, hypothesis
    ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))
    predicted = {ranked[0][0]} if ranked else set()
    diagnostics = [
        {
            "answers": [answer],
            "score": score,
            "query_score": hypothesis.score,
            "binding_count": len(hypothesis.bindings),
            "groundings": grounded_atoms_json(hypothesis),
        }
        for answer, (score, hypothesis) in ranked[:10]
    ]
    return predicted, diagnostics


def grounded_atoms_json(hypothesis: QueryHypothesis) -> list[dict]:
    return [
        {
            "atom_id": atom.atom_id,
            "relation_id": atom.relation_id,
            "direction": atom.direction,
            "log_score": atom.log_score,
            "metadata": dict(atom.metadata),
        }
        for atom in hypothesis.grounded_atoms
    ]


def select_query_denotation(graph: ConstraintGraph, hypotheses: Sequence[QueryHypothesis]) -> tuple[set[str], list[dict]]:
    ranked = rank_denotations(graph, hypotheses)
    predicted = set(ranked[0][0]) if ranked else set()
    diagnostics = [
        {
            "answers": sorted(denotation),
            "score": hypothesis.score,
            "binding_count": len(hypothesis.bindings),
            "groundings": grounded_atoms_json(hypothesis),
        }
        for denotation, hypothesis in ranked[:10]
    ]
    return predicted, diagnostics


def evaluate_hypothesis_pool(
    graph: ConstraintGraph,
    hypotheses: Sequence[QueryHypothesis],
    gold: set[str],
) -> dict:
    ranked_denotations = rank_denotations(graph, hypotheses)
    candidate_denotations = [set(denotation) for denotation, _ in ranked_denotations]
    per_binding, per_binding_top = select_per_binding(graph, hypotheses)
    query_denotation, query_top = select_query_denotation(graph, hypotheses)
    return {
        "shared_pool": {
            "query_hypotheses": len(hypotheses),
            "candidate_denotations": len(candidate_denotations),
            "gold_answer_generated": any(candidate & gold for candidate in candidate_denotations),
            "exact_denotation_generated": any(candidate == gold for candidate in candidate_denotations),
            "oracle_f1": max((float(answer_scores(candidate, gold)["f1"]) for candidate in candidate_denotations), default=0.0),
        },
        "per_binding": {
            "predicted": sorted(per_binding),
            **answer_scores(per_binding, gold),
            "top_candidates": per_binding_top,
        },
        "query_denotation": {
            "predicted": sorted(query_denotation),
            **answer_scores(query_denotation, gold),
            "top_candidates": query_top,
        },
    }


def aggregate_arm(rows: Sequence[dict], encoder: str, selector: str, subset: str = "all") -> dict:
    selected = [
        row
        for row in rows
        if subset == "all"
        or (subset == "singleton" and len(row["gold"]) == 1)
        or (subset == "set_valued" and len(row["gold"]) > 1)
    ]
    count = len(selected)
    method_rows = [row["encoders"][encoder][selector] for row in selected]
    return {
        "count": count,
        "mean_precision": sum(float(item["precision"]) for item in method_rows) / count if count else 0.0,
        "mean_recall": sum(float(item["recall"]) for item in method_rows) / count if count else 0.0,
        "mean_f1": sum(float(item["f1"]) for item in method_rows) / count if count else 0.0,
        "exact_match": sum(bool(item["exact_match"]) for item in method_rows) / count if count else 0.0,
        "hits_at_1": sum(bool(item["hits_at_1"]) for item in method_rows) / count if count else 0.0,
    }


def aggregate_end_to_end_arm(rows: Sequence[dict], encoder: str, selector: str) -> dict:
    """Count invalid parses and unsupported operators as zero-score questions."""
    count = len(rows)
    method_rows = [
        row.get("encoders", {}).get(encoder, {}).get(selector)
        for row in rows
    ]

    def mean(field: str) -> float:
        return (
            sum(float(item[field]) if item is not None else 0.0 for item in method_rows) / count
            if count
            else 0.0
        )

    return {
        "count": count,
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "mean_f1": mean("f1"),
        "exact_match": mean("exact_match"),
        "hits_at_1": mean("hits_at_1"),
    }


def pairwise_counts(rows: Sequence[dict], encoder: str) -> dict[str, int]:
    wins = losses = ties = 0
    for row in rows:
        binding_f1 = float(row["encoders"][encoder]["per_binding"]["f1"])
        query_f1 = float(row["encoders"][encoder]["query_denotation"]["f1"])
        if query_f1 > binding_f1:
            wins += 1
        elif query_f1 < binding_f1:
            losses += 1
        else:
            ties += 1
    return {"query_wins": wins, "query_losses": losses, "ties": ties}


def aggregate_group(rows: Sequence[dict], model_paths: Mapping[str, str]) -> dict:
    valid_rows = [row for row in rows if row["valid_graph"] and row["execution_supported"]]
    group = {
        "questions": len(rows),
        "valid_supported_questions": len(valid_rows),
        "valid_supported_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "arms": {},
        "end_to_end_arms": {},
        "pairwise": {},
        "candidate_generation": {},
    }
    for encoder in model_paths:
        group["arms"][encoder] = {
            selector: {
                subset: aggregate_arm(valid_rows, encoder, selector, subset)
                for subset in ("all", "singleton", "set_valued")
            }
            for selector in ("per_binding", "query_denotation")
        }
        group["end_to_end_arms"][encoder] = {
            selector: aggregate_end_to_end_arm(rows, encoder, selector)
            for selector in ("per_binding", "query_denotation")
        }
        group["pairwise"][encoder] = pairwise_counts(valid_rows, encoder)
        pools = [row["encoders"][encoder]["shared_pool"] for row in valid_rows]
        count = len(pools)
        group["candidate_generation"][encoder] = {
            "count": count,
            "gold_answer_generated_rate": sum(bool(pool["gold_answer_generated"]) for pool in pools) / count if count else 0.0,
            "exact_denotation_generated_rate": sum(bool(pool["exact_denotation_generated"]) for pool in pools) / count if count else 0.0,
            "mean_oracle_f1": sum(float(pool["oracle_f1"]) for pool in pools) / count if count else 0.0,
            "mean_query_hypotheses": sum(int(pool["query_hypotheses"]) for pool in pools) / count if count else 0.0,
            "mean_candidate_denotations": sum(int(pool["candidate_denotations"]) for pool in pools) / count if count else 0.0,
            "mean_runtime_seconds": sum(float(row["encoders"][encoder]["runtime_seconds"]) for row in valid_rows) / count if count else 0.0,
        }
    return group


def aggregate_metrics(rows: Sequence[dict], parser_stats: Mapping[str, object], model_paths: Mapping[str, str]) -> dict:
    overall = aggregate_group(rows, model_paths)
    datasets = sorted({str(row["dataset"]) for row in rows})
    metrics = {
        "experiment": {
            "questions": overall["questions"],
            "valid_supported_questions": overall["valid_supported_questions"],
            "valid_supported_rate": overall["valid_supported_rate"],
            "parser": dict(parser_stats),
            "semantic_models": dict(model_paths),
            "relation_candidates_per_atom": RELATION_CANDIDATES_PER_ATOM,
            "hypothesis_cap": HYPOTHESIS_CAP,
            "semantic_temperature": SEMANTIC_TEMPERATURE,
            "fairness": "Both selectors consume the identical hypothesis pool for each encoder and question.",
            "hits_at_1_definition": "Top grounded query denotation contains at least one gold answer.",
            "supported_scope": "Valid constraint graphs with no unsupported operators.",
            "end_to_end_scope": "All frozen questions; invalid and unsupported questions receive zero.",
        },
        "arms": overall["arms"],
        "end_to_end_arms": overall["end_to_end_arms"],
        "pairwise": overall["pairwise"],
        "candidate_generation": overall["candidate_generation"],
        "by_dataset": {
            dataset: aggregate_group(
                [row for row in rows if str(row["dataset"]) == dataset],
                model_paths,
            )
            for dataset in datasets
        },
    }
    return metrics


def markdown_report(metrics: Mapping[str, object]) -> str:
    experiment = metrics["experiment"]
    lines = [
        "# Query-Hypothesis Comparison",
        "",
        "This controlled experiment changes only the selection unit. Both selectors consume the exact same executable grounded-query pool for each semantic encoder.",
        "",
        f"Questions: **{experiment['questions']}**  ",
        f"Valid relational constraint graphs: **{experiment['valid_supported_questions']}** ({experiment['valid_supported_rate']:.1%})",
        "",
        "## Benchmark Summary",
        "",
        "| Dataset | Encoder | Selector | Supported F1 | End-to-end F1 | Supported Exact | End-to-end Exact |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    groups = {"overall": metrics, **metrics["by_dataset"]}
    for dataset, group in groups.items():
        for encoder, selectors in group["arms"].items():
            for selector, subsets in selectors.items():
                supported = subsets["all"]
                end_to_end = group["end_to_end_arms"][encoder][selector]
                lines.append(
                    f"| {dataset} | {encoder} | {selector} | {supported['mean_f1']:.3f} | "
                    f"{end_to_end['mean_f1']:.3f} | {supported['exact_match']:.3f} | "
                    f"{end_to_end['exact_match']:.3f} |"
                )
    lines.extend([
        "",
        "## Main Metrics",
        "",
        "| Encoder | Selector | Subset | Count | F1 | Precision | Recall | Exact | Hits@1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for encoder, selectors in metrics["arms"].items():
        for selector, subsets in selectors.items():
            for subset, values in subsets.items():
                lines.append(
                    f"| {encoder} | {selector} | {subset} | {values['count']} | "
                    f"{values['mean_f1']:.3f} | {values['mean_precision']:.3f} | {values['mean_recall']:.3f} | "
                    f"{values['exact_match']:.3f} | {values['hits_at_1']:.3f} |"
                )
    lines.extend([
        "",
        "## Candidate Generation",
        "",
        "| Encoder | Gold answer generated | Exact denotation generated | Oracle F1 | Mean hypotheses | Mean denotations | Seconds/question |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for encoder, values in metrics["candidate_generation"].items():
        lines.append(
            f"| {encoder} | {values['gold_answer_generated_rate']:.3f} | {values['exact_denotation_generated_rate']:.3f} | "
            f"{values['mean_oracle_f1']:.3f} | {values['mean_query_hypotheses']:.1f} | "
            f"{values['mean_candidate_denotations']:.1f} | {values['mean_runtime_seconds']:.3f} |"
        )
    lines.extend(["", "## Paired Selection Result", ""])
    for encoder, values in metrics["pairwise"].items():
        lines.append(
            f"- **{encoder}:** query-level wins {values['query_wins']}, losses {values['query_losses']}, ties {values['ties']}."
        )
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "This runner tests whether complete grounded queries should be ranked before projecting their full answer denotations. It does not test entity linking, missing-KG recovery, unsupported operators, or a learned global query scorer.",
        "",
    ])
    return "\n".join(lines)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                cache[str(row["key"])] = row
    return cache


def append_cache(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def load_encoder(path: str):
    from sentence_transformers import SentenceTransformer

    return ExecutablePatternScorer(SentenceTransformer(path, local_files_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=parse_dataset_spec, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=100)
    parser.add_argument("--parser-model", default="qwen3:8b")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--trained-model", default="runs/pattern_alignment_v1/model")
    parser.add_argument("--graph-cache")
    args = parser.parse_args()

    output = ensure_dir(args.output)
    cache_path = Path(args.graph_cache) if args.graph_cache else Path(output) / "constraint_graphs.jsonl"
    datasets = []
    seen_names = set()
    for name, path in args.dataset:
        if name in seen_names:
            raise ValueError(f"Duplicate dataset name: {name}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen_names.add(name)
        rows = load_rows(path, args.limit_per_dataset)
        datasets.extend((name, index, row) for index, row in enumerate(rows))
        print(f"Loaded {len(rows)} questions from {name} ({path})", flush=True)

    cache = load_cache(cache_path)
    parser_calls = 0
    parser_prompt_tokens = 0
    parser_completion_tokens = 0
    parser_latency = 0.0
    frozen = []
    print(f"Freezing constraint graphs with {args.parser_model}", flush=True)
    for position, (dataset, index, row) in enumerate(datasets, start=1):
        key = row_key(dataset, row, index)
        question = str(row.get("question", ""))
        entities = linked_entity_names(row)
        cached = cache.get(key)
        if cached and cached.get("question") == question and cached.get("entities") == entities:
            record = cached
            source = "cache"
        else:
            graph, raw, usage = call_constraint_parser(args.parser_model, question, entities)
            valid, reason = validate_constraint_graph(graph, len(entities))
            record = {
                "key": key,
                "dataset": dataset,
                "question_id": str(row.get("id", index)),
                "question": question,
                "entities": entities,
                "graph": graph,
                "raw": raw,
                "valid": valid,
                "invalid_reason": reason,
                "parser_model": args.parser_model,
                "prompt_version": PARSER_PROMPT_VERSION,
                "usage": usage,
            }
            append_cache(cache_path, record)
            cache[key] = record
            parser_calls += 1
            parser_prompt_tokens += int(usage["prompt_tokens"])
            parser_completion_tokens += int(usage["completion_tokens"])
            parser_latency += float(usage["latency_seconds"])
            source = "model"
        frozen.append((dataset, index, row, record))
        print(
            f"  {position}/{len(datasets)} {key} graph={'valid' if record.get('valid') else 'invalid'} source={source}",
            flush=True,
        )

    model_paths = {"base": args.base_model, "trained": args.trained_model}
    scorers = {}
    for name, path in model_paths.items():
        print(f"Loading {name} semantic encoder: {path}", flush=True)
        scorers[name] = load_encoder(path)

    predictions = []
    print("Executing shared grounded-query pools", flush=True)
    for position, (dataset, index, row, frozen_graph) in enumerate(frozen, start=1):
        gold = gold_answers(row)
        result = {
            "key": frozen_graph["key"],
            "dataset": dataset,
            "question_id": frozen_graph["question_id"],
            "question": frozen_graph["question"],
            "entities": frozen_graph["entities"],
            "gold": sorted(gold),
            "valid_graph": bool(frozen_graph.get("valid")),
            "invalid_reason": str(frozen_graph.get("invalid_reason", "")),
            "execution_supported": False,
            "unsupported_reason": "",
            "encoders": {},
        }
        graph_value = frozen_graph.get("graph")
        operators = graph_value.get("operators", []) if isinstance(graph_value, dict) else []
        if not result["valid_graph"]:
            result["unsupported_reason"] = "invalid_constraint_graph"
        elif operators:
            result["unsupported_reason"] = "operators_not_supported"
        else:
            try:
                graph, type_constraints = compile_constraint_graph(graph_value)
                constants = {
                    f"entity_{entity_index}": normalize_text(name)
                    for entity_index, name in enumerate(frozen_graph["entities"], start=1)
                }
                kb = build_kb(row.get("graph") or [])
                result["execution_supported"] = True
                for encoder_name, scorer in scorers.items():
                    started = time.perf_counter()
                    hypotheses = solve_conjunctive_query(
                        graph,
                        constants,
                        build_proposer(kb, scorer),
                        relation_candidates_per_atom=RELATION_CANDIDATES_PER_ATOM,
                        hypothesis_cap=HYPOTHESIS_CAP,
                        factor=build_type_factor(kb, scorer, type_constraints),
                    )
                    evaluation = evaluate_hypothesis_pool(graph, hypotheses, gold)
                    evaluation["runtime_seconds"] = time.perf_counter() - started
                    result["encoders"][encoder_name] = evaluation
            except (KeyError, TypeError, ValueError) as exc:
                result["execution_supported"] = False
                result["unsupported_reason"] = f"compile_or_execution_error:{exc}"
        predictions.append(result)
        if result["execution_supported"]:
            values = " ".join(
                f"{encoder}:binding={result['encoders'][encoder]['per_binding']['f1']:.3f} "
                f"query={result['encoders'][encoder]['query_denotation']['f1']:.3f}"
                for encoder in model_paths
            )
        else:
            values = result["unsupported_reason"]
        print(f"  {position}/{len(frozen)} {result['key']} {values}", flush=True)

    parser_stats = {
        "model": args.parser_model,
        "prompt_version": PARSER_PROMPT_VERSION,
        "new_calls": parser_calls,
        "cache_hits": len(frozen) - parser_calls,
        "prompt_tokens": parser_prompt_tokens,
        "completion_tokens": parser_completion_tokens,
        "latency_seconds": parser_latency,
    }
    metrics = aggregate_metrics(predictions, parser_stats, model_paths)
    predictions_path = Path(output) / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(Path(output) / "metrics.json", metrics)
    (Path(output) / "report.md").write_text(markdown_report(metrics), encoding="utf-8")
    print(f"Wrote comparison outputs to {output}", flush=True)
    print(json.dumps(metrics["pairwise"], indent=2), flush=True)


if __name__ == "__main__":
    main()
