"""Query selection with anchor-conditioned factor inference.

The proven query-selection path remains the fallback. Questions with exactly
two topic anchors additionally receive a schema-independent semantic sketch.
Each anchor is executed independently and candidate answer denotations are
joined, preserving which evidence came from which anchor. A factor can override
the fallback only when one-to-one relation-role coverage, relation-terminal
type, and executed entity type agree with sufficient confidence. Unsupported
operators and weak evidence abstain to the original method.

Hard-fails at startup if MiniLM or the LLM endpoint is missing (no silent
degradation), and counts empty LLM completions as a run canary.

Usage:
  python3 -m rc_mex.run_query_selection --data data/webqsp/train150.jsonl --output runs/qsel_train150 [--limit 0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter

import numpy as np

from cigr_d_mvp1.io_utils import ensure_dir, write_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text
from rc_mex.diag_relation_description_eval import relation_similarity
from rc_mex.diag_selection_quality_eval import clean_relation
from rc_mex.micro_agents import (
    DEFAULT_MODEL,
    audit_answer,
    describe_relation_chain,
    describe_target_relation,
    induce_semantic_sketches,
    probe_llm_endpoint,
    select_query_path,
    select_query_path_forced,
    select_set_member,
    verify_query_choice,
)
from rc_mex.run_proof_state_search_smoke import (
    rank_relations_hybrid,
    semantic_embedding,
    semantic_relation_model_available,
)
from rc_mex.run_webqsp_path_family import build_kb

QUESTION_CHANNEL_K = 10
DESCRIPTION_CHANNEL_K = 8
RAW_UNION_CAP = 18
DISTINCT_OPTIONS = 10
EXAMPLES_PER_OPTION = 3
REFINE_MIN_SET = 2
REFINE_MAX_SET = 12
HOP2_FANOUT_CAP = 300  # don't expand a second hop from a degenerate hop-1 set
JUNK_PREDICATE_MARKERS = (
    "freebase.valuenotation",
    "common.image",
    "appears_in_topic_gallery",
    "common.topic.webpage",
    "common.topic.article",
    "type.object",
    "dataworld.",
)


def sketch_relation_phrases(sketches: list[dict], position: int) -> list[str]:
    """Relation-language proposal channel at one ordered sketch position."""
    phrases = []
    for sketch in sketches:
        roles = sketch.get("relation_roles") or []
        if position < len(roles):
            phrase = str(roles[position].get("phrase", "")).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def is_junk_predicate(predicate: str) -> bool:
    return any(marker in predicate for marker in JUNK_PREDICATE_MARKERS)


def cosine(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n else 0.0


def question_type_evidence(kb: dict, question: str, top_n: int = 3) -> list[str]:
    """Zero-LLM type channel (measured 68% top-3): embed the question against
    the subgraph's own type inventory."""
    inventory = sorted({c["name"] for cid, c in kb["concepts"].items() if cid.startswith("type:")})
    if not inventory:
        return []
    qv = semantic_embedding(question)
    ranked = sorted(((cosine(qv, semantic_embedding(t)), t) for t in inventory), reverse=True)
    return [t for _, t in ranked[:top_n]]


def entity_type_names(kb: dict, entity_id: str) -> list[str]:
    return [
        kb["concepts"][cid]["name"]
        for cid in kb["entities"].get(entity_id, {}).get("instanceOf", [])
        if cid.startswith("type:")
    ]


def build_candidate_paths(kb, graph, starts, question, described_names):
    """Union of the question-embedding and description-embedding channels,
    each candidate a (predicate, direction) with its full target set."""
    frontier = graph.candidate_relations(sorted(starts), cap=100000, sample_entities=25)
    ranked = rank_relations_hybrid(question, frontier)
    directions = {}
    for c in ranked:
        directions.setdefault(c["relation_id"], []).append(c["direction"])
    channel_a = [
        (c["relation_id"], c["direction"])
        for c in ranked
        if not is_junk_predicate(c["relation_id"])
    ][:QUESTION_CHANNEL_K]

    channel_b = []
    if described_names:
        name_vecs = [semantic_embedding(n) for n in described_names]
        predicates = [p for p in dict.fromkeys(c["relation_id"] for c in ranked) if not is_junk_predicate(p)]
        scored = sorted(((relation_similarity(name_vecs, p), p) for p in predicates), reverse=True)
        for _, predicate in scored[:DESCRIPTION_CHANNEL_K]:
            for direction in dict.fromkeys(directions.get(predicate, [])):
                channel_b.append((predicate, direction))

    # interleave (description channel first: it carries the semantic intent),
    # dedupe by (predicate, direction), cap the raw union
    union, seen = [], set()
    for pair_list in zip(*[iter_pad(channel_b, RAW_UNION_CAP), iter_pad(channel_a, RAW_UNION_CAP)]):
        for pair in pair_list:
            if pair and pair not in seen:
                seen.add(pair)
                union.append(pair)
    union = union[:RAW_UNION_CAP]

    # execute, then merge options whose ANSWER SETS are identical (3.4 of ~13
    # options per question were duplicate sets — pure noise for the selector).
    # Merging never drops a distinct set, so union recall is untouched.
    paths = []
    by_targets: dict[frozenset, dict] = {}
    for predicate, direction in union:
        targets = set()
        member_quals: dict[str, dict] = {}
        for sid in starts:
            for rel in kb["entities"][sid]["relations"]:
                if rel["predicate"] == predicate and rel["direction"] == direction:
                    targets.add(rel["object"])
                    for qname, values in (rel.get("qualifiers") or {}).items():
                        slot = member_quals.setdefault(rel["object"], {}).setdefault(qname, [])
                        slot.extend(v for v in values if v not in slot)
        # A question never asks for its own anchor: topic entities are not
        # answers (observed: 'Lyon' predicted for a question naming Lyon).
        # Guarded — an option that ONLY returns topics stays as-is rather
        # than vanishing (it may still be the least-wrong fallback).
        if targets - starts:
            targets -= starts
        if not targets:
            continue
        key = frozenset(targets)
        if key in by_targets:
            by_targets[key]["also"].append(display_relation(predicate))
            continue
        path = {
            "predicate": predicate,
            "direction": direction,
            "targets": sorted(targets),
            "also": [],
            "member_quals": member_quals,
        }
        by_targets[key] = path
        paths.append(path)
        if len(paths) >= DISTINCT_OPTIONS:
            break
    return paths


def iter_pad(items, n):
    return items + [None] * (n - len(items))


"""Mixed-menu structural constants — set at the measured ceiling knee on CWQ
dev300 (gold-on-menu / oracle F1 / mean size): M4K3cap12 = 87.3%/.699/11.3,
M6K4cap14 = 90.4%/.732/13.6, M8K4cap16 = 90.8%/.737/15.3 (plateau).
Grounding hop-2 with hop-1 names when the intent omits line 2 measured +0.0
— under-triggering is not the binding constraint; no fallback."""
MIXED_MENU_CAP = 14
CHAIN_BASES = 6
CHAINS_PER_BASE = 4


def build_chain_candidates(kb, graph, starts, hop1_paths, hop2_names):
    """Two-hop query candidates: for the first CHAIN_BASES hop-1 options,
    ground the intent's second property name on that base's target-set
    frontier and EXECUTE the chain (hop 2 runs from the full hop-1 target
    set — query semantics). The selector then compares finished result sets;
    depth is never a blind decision."""
    if not hop2_names:
        return []
    name_vecs = [semantic_embedding(n) for n in hop2_names]
    out = []
    for base in hop1_paths[:CHAIN_BASES]:
        starts2 = set(base["targets"])
        if len(starts2) > HOP2_FANOUT_CAP:
            continue
        frontier = graph.candidate_relations(sorted(starts2), cap=100000, sample_entities=25)
        directions: dict[str, list[str]] = {}
        for c in frontier:
            if not is_junk_predicate(c.predicate):
                directions.setdefault(c.predicate, []).append(c.direction)
        scored = sorted(((relation_similarity(name_vecs, p), p) for p in directions), reverse=True)
        base_index = hop1_paths.index(base) + 1  # menu position (hop-1 options render first)
        for _, pred2 in scored[:CHAINS_PER_BASE]:
            for dir2 in dict.fromkeys(directions[pred2]):
                targets = set()
                member_quals: dict[str, dict] = {}
                for sid in starts2:
                    for rel in kb["entities"][sid]["relations"]:
                        if rel["predicate"] == pred2 and rel["direction"] == dir2:
                            targets.add(rel["object"])
                            for qname, values in (rel.get("qualifiers") or {}).items():
                                slot = member_quals.setdefault(rel["object"], {}).setdefault(qname, [])
                                slot.extend(v for v in values if v not in slot)
                targets -= starts
                if not targets or targets == starts2:
                    continue
                out.append(
                    {
                        "predicate": pred2,
                        "direction": dir2,
                        "targets": sorted(targets),
                        "also": [],
                        "member_quals": member_quals,
                        "chain_base": {"predicate": base["predicate"], "direction": base["direction"]},
                        "base_index": base_index,
                        "chain_label": f"{display_relation(base['predicate'])} → {display_relation(pred2)}",
                    }
                )
    return out


INTERSECTIONS_ADDED = 3


def build_intersection_candidates(menu_paths, max_added: int = INTERSECTIONS_ADDED):
    """Conjunction queries: pairwise intersections of menu option sets.

    54% of answerable CWQ has two topic entities whose constraints intersect
    ('team owned by Jerry Jones' AND 'coached by X'); measured oracle F1
    +0.027 on dev300. Coverage never changes (the parts are already on the
    menu) — this sharpens the SET. Smallest intersections first: a
    conjunction narrows."""
    sets = [(set(p["targets"]), p) for p in menu_paths]
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, pa = sets[i]
            b, pb = sets[j]
            x = a & b
            if not x or x == a or x == b:
                continue
            src_indices = (i + 1, j + 1)  # menu positions of the two parents
            quals = {}
            for src in (pa, pb):
                for m, q in (src.get("member_quals") or {}).items():
                    if m in x:
                        for k, v in q.items():
                            slot = quals.setdefault(m, {}).setdefault(k, [])
                            slot.extend(vv for vv in v if vv not in slot)
            la = pa.get("chain_label") or display_relation(pa["predicate"])
            lb = pb.get("chain_label") or display_relation(pb["predicate"])
            out.append(
                {
                    "predicate": pa["predicate"],
                    "direction": pa["direction"],
                    "targets": sorted(x),
                    "also": [],
                    "member_quals": quals,
                    "chain_label": f"both: {la} AND {lb}",
                    "intersection_of": [
                        {"predicate": pa["predicate"], "direction": pa["direction"]},
                        {"predicate": pb["predicate"], "direction": pb["direction"]},
                    ],
                    "src_indices": src_indices,
                }
            )
    out.sort(key=lambda p: len(p["targets"]))
    return out[:max_added]


def merge_mixed_menu(hop1_paths, chain_paths, cap: int = MIXED_MENU_CAP):
    """One menu of finished queries: hop-1 options keep their order (and
    their selection-cache stability); chains join after, deduped against
    everything by identical target set (a chain that reproduces a 1-hop set
    adds nothing but its name)."""
    by_targets: dict[frozenset, dict] = {frozenset(p["targets"]): p for p in hop1_paths}
    merged = list(hop1_paths)
    for c in chain_paths:
        key = frozenset(c["targets"])
        if key in by_targets:
            by_targets[key]["also"].append(c["chain_label"])
            continue
        by_targets[key] = c
        merged.append(c)
        if len(merged) >= cap:
            break
    return merged


def executable_path_label(path: dict) -> str:
    """Readable relation sequence for scoring and provenance diagnostics."""
    if path.get("chain_label"):
        return str(path["chain_label"])
    label = display_relation(path["predicate"])
    if path.get("direction") == "backward":
        label += " (reversed)"
    return label


def build_anchor_conditioned_joins(
    kb,
    graph,
    starts,
    question,
    described_names,
    hop2_names,
    max_candidates: int | None = None,
):
    """Build executable conjunctions without pooling topic-anchor provenance.

    The ordinary candidate builder executes every proposed relation over the
    union of topic entities. That representation cannot distinguish
    ``r1(anchor_a, answer) AND r2(anchor_b, answer)`` from evidence produced by
    the opposite anchor. Here each anchor receives its own bounded one/two-hop
    proposal menu; only their answer sets are joined. The live selector uses
    these joins only through the confidence-gated factor override below.
    """
    anchor_ids = list(dict.fromkeys(starts))
    if len(anchor_ids) != 2:
        return []

    per_anchor = []
    for anchor_id in anchor_ids:
        paths = build_candidate_paths(kb, graph, {anchor_id}, question, described_names)
        chains = build_chain_candidates(kb, graph, {anchor_id}, paths, hop2_names)
        if chains:
            paths = merge_mixed_menu(paths, chains)
        per_anchor.append(paths)

    role_phrases = list(dict.fromkeys([*described_names, *hop2_names]))
    role_vectors = [semantic_embedding(phrase) for phrase in role_phrases]
    question_vector = semantic_embedding(question) if not role_vectors else None
    by_targets = {}
    for left in per_anchor[0]:
        left_targets = set(left["targets"])
        for right in per_anchor[1]:
            right_targets = set(right["targets"])
            targets = left_targets & right_targets
            # Keep non-narrowing joins too. A second anchor can validate the
            # same denotation without shrinking it, and its anchor-specific
            # parent hypothesis may not exist in the pooled menu at all.
            if not targets:
                continue

            left_label = executable_path_label(left)
            right_label = executable_path_label(right)
            if role_vectors:
                path_vectors = [semantic_embedding(left_label), semantic_embedding(right_label)]
                # Symmetric set matching rewards both semantic-role coverage
                # and path specificity. A pair cannot score highly by matching
                # one relation phrase twice while ignoring the other.
                role_coverage = sum(
                    max(cosine(role_vector, path_vector) for path_vector in path_vectors)
                    for role_vector in role_vectors
                ) / len(role_vectors)
                path_specificity = sum(
                    max(cosine(path_vector, role_vector) for role_vector in role_vectors)
                    for path_vector in path_vectors
                ) / len(path_vectors)
                role_score = (role_coverage + path_specificity) / 2
            else:
                role_score = cosine(
                    question_vector,
                    semantic_embedding(f"{left_label} and {right_label}"),
                )
            member_quals = {}
            for member in targets:
                for source in (left, right):
                    for dim, values in (source.get("member_quals") or {}).get(member, {}).items():
                        slot = member_quals.setdefault(member, {}).setdefault(dim, [])
                        slot.extend(value for value in values if value not in slot)

            anchor_constraints = [
                {
                    "anchor_id": anchor_id,
                    "anchor_name": graph.entity_name(anchor_id),
                    "path_label": executable_path_label(path),
                    "predicate": path["predicate"],
                    "direction": path["direction"],
                    **({"chain_base": path["chain_base"]} if path.get("chain_base") else {}),
                }
                for anchor_id, path in zip(anchor_ids, (left, right))
            ]
            candidate = {
                "predicate": left["predicate"],
                "direction": left["direction"],
                "targets": sorted(targets),
                "also": [],
                "member_quals": member_quals,
                "chain_label": (
                    f"both anchors: {anchor_constraints[0]['anchor_name']} via {left_label} "
                    f"AND {anchor_constraints[1]['anchor_name']} via {right_label}"
                ),
                "anchor_constraints": anchor_constraints,
                "anchor_role_score": role_score,
            }
            key = frozenset(targets)
            incumbent = by_targets.get(key)
            if incumbent is None or role_score > incumbent["anchor_role_score"]:
                by_targets[key] = candidate

    candidates = sorted(
        by_targets.values(),
        key=lambda candidate: (-candidate["anchor_role_score"], len(candidate["targets"])),
    )
    return candidates[:max_candidates] if max_candidates is not None else candidates


ANCHOR_FACTOR_TYPE_MIN = 0.35
ANCHOR_FACTOR_MARGIN_MIN = 0.025
ANCHOR_FACTOR_SINGLETON_MIN = 0.45
ANCHOR_FACTOR_UNSUPPORTED_OPERATORS = {
    "aggregation",
    "comparison",
    "difference",
    "ordering",
    "temporal_filter",
    "union",
    "value_filter",
}
ANCHOR_FACTOR_UNSUPPORTED_CONSTRAINTS = {"aggregation", "numeric", "ordering", "temporal"}


def sketch_role_phrases(sketches: list[dict]) -> list[str]:
    return list(
        dict.fromkeys(
            str(role.get("phrase", "")).strip()
            for sketch in sketches
            for role in (sketch.get("relation_roles") or [])
            if str(role.get("phrase", "")).strip()
        )
    )


def sketch_answer_type_text(sketches: list[dict]) -> str:
    answer_types = list(
        dict.fromkeys(
            str(value).strip()
            for sketch in sketches
            for value in (sketch.get("answer_types") or [])
            if str(value).strip()
        )
    )
    if answer_types:
        return " ".join(answer_types)
    return " ".join(
        dict.fromkeys(
            str(sketch.get("answer_role", "")).strip()
            for sketch in sketches
            if str(sketch.get("answer_role", "")).strip()
        )
    )


def terminal_relation_label(path_label: str) -> str:
    """Best schema-side clue for the type produced at a path endpoint."""
    terminal = str(path_label).split("→")[-1].strip()
    if "/" in terminal:
        terminal = terminal.split("/")[-1].strip()
    return re.sub(r"\s*\(reversed\)\s*$", "", terminal).strip()


def one_to_one_role_alignment(role_vectors: list, path_vectors: list) -> float:
    """Match two semantic constraints to two anchor paths without reuse.

    The previous Chamfer-style score allowed both constraints to be explained
    by the same path. For the common two-anchor/two-role case, score the best
    bijection instead. Larger sketches keep the old set-coverage behavior
    because a compressed anchor path can legitimately cover multiple edges.
    """
    if not role_vectors or not path_vectors:
        return 0.0
    matrix = [
        [cosine(role_vector, path_vector) for path_vector in path_vectors]
        for role_vector in role_vectors
    ]
    if len(role_vectors) == len(path_vectors) == 2:
        return max(
            (matrix[0][0] + matrix[1][1]) / 2,
            (matrix[0][1] + matrix[1][0]) / 2,
        )
    role_coverage = sum(max(row) for row in matrix) / len(matrix)
    path_specificity = sum(
        max(matrix[role_index][path_index] for role_index in range(len(role_vectors)))
        for path_index in range(len(path_vectors))
    ) / len(path_vectors)
    return (role_coverage + path_specificity) / 2


def rank_anchor_conditioned_joins(kb, graph, joins: list[dict], sketches: list[dict]) -> list[dict]:
    """Rank joined denotations as anchor-bound factors, without gold labels.

    Three independent signals receive equal weight: one-to-one relation-role
    coverage, compatibility between the path terminal and requested answer
    type, and compatibility between executed answer entities and that type.
    """
    roles = sketch_role_phrases(sketches)
    answer_type = sketch_answer_type_text(sketches)
    if not roles or not answer_type:
        return []
    role_vectors = [semantic_embedding(role) for role in roles]
    answer_vector = semantic_embedding(answer_type)
    ranked = []
    for join in joins:
        path_labels = [
            str(constraint.get("path_label", ""))
            for constraint in (join.get("anchor_constraints") or [])
        ]
        path_vectors = [semantic_embedding(path_label) for path_label in path_labels]
        assignment_score = one_to_one_role_alignment(role_vectors, path_vectors)
        terminal_score = sum(
            cosine(answer_vector, semantic_embedding(terminal_relation_label(path_label)))
            for path_label in path_labels
        ) / len(path_labels) if path_labels else 0.0
        member_scores = []
        for member in join.get("targets") or []:
            evidence = entity_type_names(kb, member) or [graph.entity_name(member)]
            member_scores.append(
                max(cosine(answer_vector, semantic_embedding(value)) for value in evidence)
            )
        answer_type_score = min(member_scores) if member_scores else 0.0
        join["factor_role_alignment"] = assignment_score
        join["factor_terminal_type"] = terminal_score
        join["factor_answer_type"] = answer_type_score
        join["factor_score"] = (assignment_score + terminal_score + answer_type_score) / 3
        if answer_type_score >= ANCHOR_FACTOR_TYPE_MIN:
            ranked.append(join)
    return sorted(ranked, key=lambda join: join["factor_score"], reverse=True)


def anchor_factor_operator_supported(sketches: list[dict]) -> bool:
    """The MVP factor executor represents joins, chains, and type evidence."""
    for sketch in sketches:
        if set(sketch.get("operators") or []) & ANCHOR_FACTOR_UNSUPPORTED_OPERATORS:
            return False
        if any(
            constraint.get("kind") in ANCHOR_FACTOR_UNSUPPORTED_CONSTRAINTS
            for constraint in (sketch.get("constraints") or [])
        ):
            return False
    return True


def confident_anchor_factor(ranked: list[dict], sketches: list[dict]) -> dict | None:
    """Return a factor only when its available evidence supports overriding.

    A singleton has no comparative margin, so it must clear an absolute score;
    multi-candidate pools use the top-vs-runner-up margin measured in the
    frozen first-half A/B. Unsupported operators abstain to the base method.
    """
    if not ranked or not anchor_factor_operator_supported(sketches):
        return None
    if len(ranked) == 1:
        return ranked[0] if ranked[0]["factor_score"] >= ANCHOR_FACTOR_SINGLETON_MIN else None
    margin = ranked[0]["factor_score"] - ranked[1]["factor_score"]
    ranked[0]["factor_margin"] = margin
    return ranked[0] if margin >= ANCHOR_FACTOR_MARGIN_MIN else None


EXTEND_MENU_CAP = 8  # gold extension visible: top-6 41/49, top-10 45/49 (cwq_dev300f probe)


def build_extension_menu(kb, starts, chosen, question, gap_phrase):
    """Round-2 menu for the audit's MISSING verdict: every (predicate,
    direction) executed from the CHOSEN set's full frontier, ranked by the
    better of two grounding channels — the audit's stated gap phrase (post-
    evidence, replaces the intent second-name channel that caused the
    off-menu class) and the question itself. Deduped by target set, capped.
    Topic entities and the frontier itself are never answers here: an
    extension that returns its own sources is the identity, not a hop."""
    frontier = set(chosen["targets"])
    if not frontier or len(frontier) > HOP2_FANOUT_CAP:
        return []
    exclude = starts | frontier
    groups: dict[tuple[str, str], set] = {}
    quals: dict[tuple[str, str], dict] = {}
    for sid in frontier:
        for rel in kb["entities"].get(sid, {}).get("relations", []):
            if is_junk_predicate(rel["predicate"]):
                continue
            key = (rel["predicate"], rel["direction"])
            groups.setdefault(key, set()).add(rel["object"])
            for qname, values in (rel.get("qualifiers") or {}).items():
                slot = quals.setdefault(key, {}).setdefault(rel["object"], {}).setdefault(qname, [])
                slot.extend(v for v in values if v not in slot)
    groups = {k: v - exclude for k, v in groups.items() if v - exclude}
    if not groups:
        return []
    channel_vecs = [semantic_embedding(question)]
    if gap_phrase:
        channel_vecs.append(semantic_embedding(gap_phrase))
    scored = sorted(
        ((max(cosine(v, semantic_embedding(display_relation(prd))) for v in channel_vecs), (prd, drn))
         for prd, drn in groups),
        reverse=True,
    )
    base_label = chosen.get("chain_label") or display_relation(chosen["predicate"])
    out, seen = [], set()
    for _, (prd, drn) in scored:
        key = frozenset(groups[(prd, drn)])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "predicate": prd,
                "direction": drn,
                "targets": sorted(groups[(prd, drn)]),
                "also": [],
                "member_quals": {m: q for m, q in quals.get((prd, drn), {}).items() if m in groups[(prd, drn)]},
                "chain_label": f"{base_label} → {display_relation(prd)}",
                "extension_of": {"predicate": chosen["predicate"], "direction": chosen["direction"]},
            }
        )
        if len(out) >= EXTEND_MENU_CAP:
            break
    return out


def display_relation(predicate: str) -> str:
    """Selector-facing gloss: clean_relation plus collapsing repeated
    composite segments ('venue / venue' -> 'venue'). Kept separate from
    clean_relation so the offline probes' measurements stay comparable."""
    segments = [s.strip() for s in clean_relation(predicate).split("/")]
    deduped = list(dict.fromkeys(s for s in segments if s))
    return " / ".join(deduped)


DATE_PATTERN = __import__("re").compile(r"\b(1[6-9]\d\d|20\d\d)(?:-\d\d)?(?:-\d\d)?\b")
ORDINAL_MIN = __import__("re").compile(r"\b(first|earliest|oldest|original|debut)\b", __import__("re").I)
ORDINAL_MAX = __import__("re").compile(r"\b(last|latest|newest|most recent|final|current|now|today)\b", __import__("re").I)
SCOPE_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "and", "or", "to", "with"}
# Freebase reification bookkeeping legs, not facts about the member. 'has no
# value: To' is the one that CARRIES meaning (open validity interval = the
# fact is current); the rest is noise that can spuriously scope-match the
# question ('is reviewed: Spouse' vs a spouse question) or waste block space.
ARTIFACT_QUAL_DIMS = {"is reviewed", "has value", "has no value"}


def is_current_fact(quals: dict) -> bool:
    """Open-ended validity: Freebase marks an ongoing tenure/marriage/roster
    spot with a 'has no value' leg for its To date."""
    return any(str(v).strip().lower() == "to" for v in quals.get("has no value", []))


def value_match_strength(question_norm: str, value: str) -> tuple[int, int]:
    """How strongly a qualifier VALUE matches the question. Tiered so that a
    more specific constraint wins ('Superman Returns' over 'Superman'):
      (3, len)  full value appears in the question
      (2, n)    >=2 significant words overlap
      (1, len)  one significant word (>=4 chars) overlaps
      (0, 0)    no match"""
    q_words = set(question_norm.split())
    vn = normalize_text(str(value))
    if len(vn) >= 4 and vn in question_norm:
        return (3, len(vn))
    vw = [w for w in vn.split() if w not in SCOPE_STOPWORDS and len(w) > 2]
    overlap = [w for w in vw if w in q_words]
    if len(vw) > 1 and len(overlap) >= 2:
        return (2, len(overlap))
    if overlap and max(len(w) for w in overlap) >= 4:
        return (1, max(len(w) for w in overlap))
    return (0, 0)


def scope_members(question_norm: str, members: list[str], member_quals: dict) -> list[str] | None:
    """Zero-LLM scope filter: keep the members whose qualifiers best satisfy a
    constraint mentioned in the question ('in superman returns').

    Two rules make this discriminating rather than decorative:
    - A qualifier DIMENSION that matches the question for every member is a
      topic echo (arriving via the character makes character=X match on all
      members) — it carries zero information and is ignored.
    - Among matching members, only those at the MAXIMUM match strength stay
      ('Superman Returns' full-value match beats a bare 'Superman' overlap).

    Returns the scoped subset, or None when no informative dimension
    discriminates (caller keeps the full set)."""
    if len(members) <= 1 or not member_quals:
        return None
    strength: dict[str, tuple[int, int]] = {m: (0, 0) for m in members}
    dims = {d for m in members for d in member_quals.get(m, {})} - ARTIFACT_QUAL_DIMS
    for dim in dims:
        per_member = {}
        for m in members:
            best = (0, 0)
            for value in member_quals.get(m, {}).get(dim, []):
                if normalize_text(str(value)) == m:
                    continue  # self-reference (a CVT leg naming the member) is identity, not constraint evidence
                best = max(best, value_match_strength(question_norm, value))
            per_member[m] = best
        matched = [m for m, s in per_member.items() if s > (0, 0)]
        if not matched or len(matched) == len(members):
            continue  # dimension matches nobody or everybody: uninformative
        for m in members:
            strength[m] = max(strength[m], per_member[m])
    top = max(strength.values())
    if top == (0, 0):
        return None
    scoped = [m for m in members if strength[m] == top]
    if scoped and len(scoped) < len(members):
        return scoped
    return None


def member_date(quals: dict, latest: bool, member_name: str = "") -> str | None:
    """Best date evidence for a member: qualifier values first; failing that,
    the member's own name — event entities carry their date there ('2014
    Stanley Cup Finals'), and direct edges have no qualifiers at all. An
    open-ended validity interval outranks every dated one for 'latest'
    questions: the fact that is still true IS the most recent."""
    if latest and is_current_fact(quals):
        return "9999"
    dates = []
    for dim, values in quals.items():
        if dim in ARTIFACT_QUAL_DIMS:
            continue
        for value in values:
            found = DATE_PATTERN.search(str(value))
            if found:
                dates.append(found.group(0))
    if not dates and member_name:
        found = DATE_PATTERN.search(member_name)
        if found:
            dates.append(found.group(0))
    if not dates:
        return None
    return max(dates) if latest else min(dates)


def order_members(kb, targets: list[str], question_types: list[str]) -> list[str]:
    qtypes = set(question_types)
    return sorted(targets, key=lambda t: (not (set(entity_type_names(kb, t)) & qtypes), t))


def member_block(kb, graph, member: str, quals: dict, informative_dims: set[str]) -> str:
    """One member line for the refinement micro-call: name, types, and the
    qualifiers that vary across the set (marriage vs civil union, the film a
    performance belongs to) — the constraint evidence the picker needs."""
    parts = [graph.entity_name(member)]
    types = entity_type_names(kb, member)
    if types:
        parts.append(f"[{', '.join(types[:2])}]")
    shown = []
    if is_current_fact(quals):
        shown.append("current")
    for dim in sorted(informative_dims - ARTIFACT_QUAL_DIMS):
        values = quals.get(dim, [])
        if values:
            shown.append(f"{dim}: {', '.join(str(v) for v in values[:3])}")
        if len(shown) >= 2:
            break
    if shown:
        parts.append(f"({'; '.join(shown)})")
    return " ".join(parts)


def informative_qualifier_dims(members: list[str], member_quals: dict) -> set[str]:
    """Dimensions whose value sets VARY across members — the only ones that
    can justify picking one member over another."""
    dims = {d for m in members for d in member_quals.get(m, {})}
    out = set()
    for dim in dims:
        seen = {tuple(sorted(map(str, member_quals.get(m, {}).get(dim, [])))) for m in members}
        if len(seen) > 1:
            out.add(dim)
    return out


def name_evidence(question_norm: str, topic_norms: list[str], member_name: str) -> int:
    """Question words that appear in the member's NAME but belong to neither
    the topic entity nor the stopword list ('high school' for 'Ringgold High
    School'). This is the question naming the answer type/shape; topic words
    are excluded because sharing the topic's name (a spouse's surname) is an
    echo, not evidence."""
    topic_words = {w for t in topic_norms for w in t.split()}
    mn_words = set(normalize_text(member_name).split())
    matched = sum(
        1
        for w in question_norm.split()
        if w in mn_words and w not in topic_words and w not in SCOPE_STOPWORDS and len(w) > 2
    )
    # one shared word is coincidence-prone ('world' promoting 'The World' for
    # a Dubai question); a type phrase like 'high school' needs two
    return matched if matched >= 2 else 0


def refine_members(kb, graph, question: str, question_types: list[str], chosen: dict, member_call=None, topic_names: list[str] | None = None):
    """The deterministic refinement chain: name-evidence ordering -> scope ->
    ordinal -> member micro-call. member_call(members, blocks) is injected so
    the live runner and the offline replay harness exercise the same code.

    Ordering note: name evidence is applied HERE and not in order_members —
    the selection menus embed the first members as examples, and reordering
    them would perturb every selection prompt (and invalidate its cache).
    Returns (members, flags)."""
    flags = {"scope": False, "ordinal": False, "refined": False, "whole_set": False}
    members = order_members(kb, chosen["targets"], question_types)
    member_quals = chosen.get("member_quals") or {}
    question_norm = normalize_text(question)
    topic_norms = [normalize_text(t) for t in (topic_names or []) if str(t).strip()]
    if len(members) > 1:
        # stable: only name evidence differentiates; ties keep type/name order
        members = sorted(
            members,
            key=lambda m: -name_evidence(question_norm, topic_norms, graph.entity_name(m)),
        )

    # (1) SCOPE: constraint filtering on informative qualifier dimensions.
    # MUTATES the answer set by design — the scoped subset IS the answer.
    scoped = scope_members(question_norm, members, member_quals)
    if scoped:
        members = scoped
        flags["scope"] = True

    # (2) ORDINAL: first/last questions — argmin/argmax over date evidence
    # (qualifiers, else the member names themselves); reorders top-1 only.
    if len(members) > 1:
        latest = bool(ORDINAL_MAX.search(question))
        if latest or ORDINAL_MIN.search(question):
            dated = [(member_date(member_quals.get(m, {}), latest, graph.entity_name(m)), m) for m in members]
            dated = [(d, m) for d, m in dated if d]
            if len(dated) >= 2:
                best = max(dated)[1] if latest else min(dated)[1]
                members = [best] + [m for m in members if m != best]
                flags["ordinal"] = True

    # (3) member micro-call for what the operators didn't settle.
    # member_call returns a member id (promote to top-1), "ALL" (the set is
    # the answer), or None (error/no signal — leave untouched).
    if not flags["ordinal"] and member_call is not None and REFINE_MIN_SET <= len(members) <= REFINE_MAX_SET:
        show_dims = informative_qualifier_dims(members, member_quals)
        blocks = [member_block(kb, graph, m, member_quals.get(m, {}), show_dims) for m in members]
        outcome = member_call(members, blocks)
        if outcome == "ALL":
            flags["whole_set"] = True
        elif outcome in members:
            members = [outcome] + [m for m in members if m != outcome]
            flags["refined"] = True
    return members, flags


def path_block(kb, graph, path, question_types, label_collides: bool = False, topic_name: str = "") -> str:
    members = order_members(kb, path["targets"], question_types)
    examples = ", ".join(graph.entity_name(m) for m in members[:EXAMPLES_PER_OPTION])
    if len(members) > EXAMPLES_PER_OPTION:
        examples += ", ..."
    type_counts = Counter(t for m in members for t in entity_type_names(kb, m))
    type_str = f" [type: {', '.join(t for t, _ in type_counts.most_common(2))}]" if type_counts else ""
    if path.get("chain_label"):
        # Provenance suffixes ('a property of option k's answers') were
        # A/B-tested and HURT the one-token pick (38.3% vs 41.0% on
        # cwq_dev300): the bias arrives without the capacity to apply it.
        # Thinking mode at the seat fixes the same failure and won (+9).
        label = path["chain_label"]
        if path["direction"] == "backward" and not path.get("src_indices"):
            label += " (reversed)"
    else:
        label = display_relation(path["predicate"])
        if path["direction"] == "backward":
            # When the same label appears in both directions on one menu, a
            # bare "(reversed)" leaves the two options indistinguishable
            # ("contains" vs "contains" for the Balkans). Spell the backward
            # one out as "<label> <topic>": its answers <label> the topic.
            if label_collides and topic_name:
                label = f"{label} {topic_name}"
            label += " (reversed)"
    return f"{label} — {len(members)} answer(s): {examples}{type_str}"


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def structural_neighbour(paths, chosen):
    """The alternative most confusable with the chosen option, known from
    STRUCTURE alone: a chain's base (overshoot), the chosen base's extension
    (undershoot), or an intersection's larger parent. None when the pick has
    no structural neighbour — the verify seat only fires when the menu
    geometry says the decision was genuinely close."""
    if chosen.get("src_indices"):
        i = chosen["src_indices"][0] - 1
        return paths[i] if i < len(paths) else None
    if chosen.get("base_index"):
        i = chosen["base_index"] - 1
        return paths[i] if i < len(paths) else None
    for p in paths:
        if p.get("base_index") and paths[p["base_index"] - 1] is chosen:
            return p
    return None


def verify_block(kb, graph, path, question_types, max_members: int = 10) -> str:
    """Richer evidence than the menu block: more members, plus the
    qualifiers that vary — the binary comparison can afford it."""
    members = order_members(kb, path["targets"], question_types)
    quals = path.get("member_quals") or {}
    dims = informative_qualifier_dims(members, quals)
    label = path.get("chain_label") or display_relation(path["predicate"])
    lines = [f"{label} — {len(members)} answer(s):"]
    for m in members[:max_members]:
        lines.append("   " + member_block(kb, graph, m, quals.get(m, {}), dims))
    if len(members) > max_members:
        lines.append(f"   ... and {len(members) - max_members} more")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--intent-model", default=None, help="Model for call 1 (defaults to --model).")
    parser.add_argument("--selector-model", default=None, help="Model for call 2 (defaults to --model).")
    parser.add_argument("--refiner-model", default=None, help="Model for the member-refinement call (defaults to --model).")
    parser.add_argument(
        "--max-hops",
        type=int,
        default=1,
        choices=(1, 2),
        help="2 enables the chain extension (CWQ): intent may sketch a second property; "
        "after hop-1 selection it is grounded on the chosen target set and selected in a "
        "second bounded call. 1 = exact WebQSP behavior, caches untouched.",
    )
    parser.add_argument(
        "--think-select",
        action="store_true",
        help="Enable qwen3 thinking mode at the SELECTION seat only (cache-keyed separately). "
        "A/B lever: the pick is a 14-way decision currently made in one token.",
    )
    parser.add_argument(
        "--recover-abstain",
        action="store_true",
        help="On selection abstain, re-ask the same menu with no escape hatch before the "
        "blind channel-floor fallback. Measured motivation (cwq_dev300e): 22/30 abstains "
        "ended as misses, 17 with gold ON the menu. Needs its own WebQSP A/B before use there.",
    )
    parser.add_argument(
        "--audit-extend",
        action="store_true",
        help="Propose→audit→revise loop, boolean form: after refinement, a strictly "
        "True/False TYPE check on the proposed answers; on False, ONE extension round "
        "executed from the chosen set's frontier (question-grounded, top-8) with a "
        "FORCED round-2 pick — the condition decides, the body commits, no second veto. "
        "Single iteration by design. Motivation: 49/96 cwq_dev300f answerable misses "
        "predicted the correct INTERMEDIATE entity; v1 (gap-phrase audit + abstainable "
        "round 2) measured +4 with 108 over-fires and 55 deadlocked revisions on dev300g.",
    )
    parser.add_argument(
        "--verify-final",
        action="store_true",
        help="Binary verification call between the selected query and its STRUCTURAL "
        "neighbour (chain vs its base / intersection vs parent), each shown with full "
        "evidence. Fires only when menu geometry says the pick was close; never on 1-hop menus.",
    )
    args = parser.parse_args()
    intent_model = args.intent_model or args.model
    selector_model = args.selector_model or args.model
    refiner_model = args.refiner_model or args.model

    if not semantic_relation_model_available():
        sys.exit("FATAL: sentence-transformers/MiniLM unavailable — refusing to run degraded (check PYTHONPATH/HF_HOME/HF_HUB_OFFLINE).")
    probe = probe_llm_endpoint(model=selector_model)
    if not probe["ok"]:
        sys.exit(f"FATAL: LLM endpoint unavailable ({probe['error']}) — query selection requires a local model.")
    print(f"LLM endpoint OK: {probe['url']} model={probe['model']}")

    rows_in = []
    with open(args.data) as fh:
        for line in fh:
            rows_in.append(json.loads(line))
    rows_in = rows_in[args.offset:]
    if args.limit:
        rows_in = rows_in[: args.limit]
    output_dir = ensure_dir(args.output)
    print(f"{len(rows_in)} questions from {args.data}")

    stats = Counter()
    f1_sum = 0.0
    usage = Counter()
    out_rows = []
    started = time.time()
    for index, src in enumerate(rows_in, start=1):
        kb = build_kb(src.get("graph") or [])
        graph = KnowledgeGraph(kb)
        golds = {normalize_text(a) for a in src.get("answer") or [] if str(a).strip()}
        starts = {normalize_text(e) for e in src.get("q_entity") or [] if str(e).strip()} & set(kb["entities"])
        question = str(src.get("question", ""))
        start_name = str((src.get("q_entity") or [""])[0])
        stats["total"] += 1
        stats["gold_in_subgraph"] += bool(golds & set(kb["entities"]))
        row = {
            "question_id": str(src.get("id", index)),
            "question": question,
            "start_entity": start_name,
            "gold_answers": sorted(golds),
            "selected": None,
            "abstained": False,
            "fallback": False,
            "predicted": [],
        }
        if starts:
            anchor_ids = list(
                dict.fromkeys(
                    normalize_text(value)
                    for value in (src.get("q_entity") or [])
                    if normalize_text(value) in starts
                )
            )
            if args.max_hops >= 2:
                described = describe_relation_chain(question, start_name, model=intent_model)
            else:
                described = describe_target_relation(question, start_name, model=intent_model)
            usage["calls"] += 1
            usage["prompt_tokens"] += described["prompt_tokens"]
            usage["completion_tokens"] += described["completion_tokens"]
            described_names = described["names"]
            hop2_names = described.get("names_2") or []
            if not described_names and not described["error"]:
                stats["empty_completion"] += 1
            question_types = question_type_evidence(kb, question)
            paths = build_candidate_paths(kb, graph, starts, question, described_names)
            has_chains = False
            if args.max_hops >= 2 and paths:
                chains = build_chain_candidates(kb, graph, starts, paths, hop2_names)
                if chains:
                    paths = merge_mixed_menu(paths, chains)
                intersections = build_intersection_candidates(paths)
                if intersections:
                    paths = merge_mixed_menu(paths, intersections, cap=MIXED_MENU_CAP + INTERSECTIONS_ADDED)
                has_chains = any(p.get("chain_label") for p in paths)
            sketches = []
            factor_choice = None
            factor_ranked = []
            if len(anchor_ids) == 2:
                stats["factor_attempts"] += 1
                meaning = induce_semantic_sketches(question, start_name, model=intent_model)
                usage["calls"] += 1
                usage["prompt_tokens"] += meaning["prompt_tokens"]
                usage["completion_tokens"] += meaning["completion_tokens"]
                sketches = meaning["sketches"]
                stats["sketch_parse_failed"] += meaning["parse_failed"]
                if not sketches and not meaning["error"]:
                    stats["empty_completion"] += 1
                factor_first = sketch_relation_phrases(sketches, 0)
                factor_second = sketch_relation_phrases(sketches, 1) if args.max_hops >= 2 else []
                factor_joins = build_anchor_conditioned_joins(
                    kb,
                    graph,
                    anchor_ids,
                    question,
                    factor_first,
                    factor_second,
                )
                factor_ranked = rank_anchor_conditioned_joins(kb, graph, factor_joins, sketches)
                factor_choice = confident_anchor_factor(factor_ranked, sketches)
                if factor_choice is not None:
                    stats["factor_overrides"] += 1
                row["semantic_sketches"] = sketches
                row["factor_decision"] = {
                    "candidate_count": len(factor_ranked),
                    "override": factor_choice is not None,
                    "top_score": factor_ranked[0]["factor_score"] if factor_ranked else 0.0,
                    "margin": factor_ranked[0].get("factor_margin", 0.0) if factor_ranked else 0.0,
                    "operator_supported": anchor_factor_operator_supported(sketches),
                }
            row["described_names"] = described_names
            if hop2_names:
                row["described_names_2"] = hop2_names
            row["question_types"] = question_types
            row["candidates"] = [
                {
                    "predicate": p["predicate"],
                    "direction": p["direction"],
                    "size": len(p["targets"]),
                    **({"chain_base": p["chain_base"]} if p.get("chain_base") else {}),
                }
                for p in paths
            ]
            if paths or factor_choice is not None:
                # Only the same-predicate-both-directions pair is truly
                # indistinguishable ("contains" vs "contains (reversed)" for
                # a region). Label-level collisions are far broader (38% of
                # menus) and rewording them all is an unvalidated
                # perturbation; predicate-level is 16% and surgical.
                pred_counts = Counter(p["predicate"] for p in paths)
                blocks = [
                    path_block(
                        kb,
                        graph,
                        p,
                        question_types,
                        label_collides=pred_counts[p["predicate"]] > 1,
                        topic_name=start_name,
                    )
                    for p in paths
                ]
                chosen = factor_choice
                factor_override = factor_choice is not None
                if not factor_override:
                    selection = select_query_path(
                        question,
                        start_name,
                        blocks,
                        model=selector_model,
                        mixed=has_chains,
                        think=args.think_select,
                    )
                    usage["calls"] += 1
                    usage["prompt_tokens"] += selection["prompt_tokens"]
                    usage["completion_tokens"] += selection["completion_tokens"]
                    if not selection["raw_response"] and not selection["error"]:
                        stats["empty_completion"] += 1
                    if selection["pick"] is not None:
                        chosen = paths[selection["pick"]]
                    else:
                        row["abstained"] = selection["abstain"]
                        stats["abstained"] += selection["abstain"]
                        stats["selection_error"] += bool(selection["error"])
                        if selection["error"]:
                            row["selection_error"] = selection["error"][:160]
                        if args.recover_abstain and selection["abstain"] and paths:
                            forced = select_query_path_forced(
                                question, start_name, blocks, model=selector_model, think=args.think_select
                            )
                            usage["calls"] += 1
                            usage["prompt_tokens"] += forced["prompt_tokens"]
                            usage["completion_tokens"] += forced["completion_tokens"]
                            if not forced["raw_response"] and not forced["error"]:
                                stats["empty_completion"] += 1
                            if forced["pick"] is not None:
                                chosen = paths[forced["pick"]]
                                stats["abstain_recovered"] += 1
                                row["abstain_recovered"] = True
                        if chosen is None:
                            chosen = paths[0] if paths else None
                            row["fallback"] = True
                if (
                    args.verify_final
                    and chosen is not None
                    and not row["fallback"]
                    and not factor_override
                ):
                    rival = structural_neighbour(paths, chosen)
                    if rival is not None:
                        verdict = verify_query_choice(
                            question,
                            start_name,
                            verify_block(kb, graph, chosen, question_types),
                            verify_block(kb, graph, rival, question_types),
                            model=selector_model,
                        )
                        usage["calls"] += 1
                        usage["prompt_tokens"] += verdict["prompt_tokens"]
                        usage["completion_tokens"] += verdict["completion_tokens"]
                        if not verdict["raw_response"] and not verdict["error"]:
                            stats["empty_completion"] += 1
                        stats["verify_calls"] += 1
                        if verdict["pick"] == 1:
                            chosen = rival
                            stats["verify_switched"] += 1
                            row["verify_switched"] = True

                if chosen is not None and chosen.get("chain_label"):
                    row["chained"] = True
                    stats["chained"] += 1

                if chosen is not None:

                    def make_member_call(chosen_path):
                        def live_member_call(members, blocks):
                            refinement = select_set_member(
                                question=question,
                                start_entity_name=start_name,
                                property_name=chosen_path.get("chain_label") or display_relation(chosen_path["predicate"]),
                                member_blocks=blocks,
                                model=refiner_model,
                            )
                            usage["calls"] += 1
                            usage["prompt_tokens"] += refinement["prompt_tokens"]
                            usage["completion_tokens"] += refinement["completion_tokens"]
                            if not refinement["raw_response"] and not refinement["error"]:
                                stats["empty_completion"] += 1
                            if refinement["pick"] is not None:
                                return members[refinement["pick"]]
                            if refinement["whole_set"]:
                                return "ALL"
                            return None

                        return live_member_call

                    if factor_override:
                        # The factor score already ranks the complete joined
                        # denotation. Member refinement would collapse a valid
                        # multi-answer set, while audit-extension would leave
                        # the representation whose confidence was measured.
                        members = list(chosen["targets"])
                        rflags = {"scope": False, "ordinal": False, "refined": False, "whole_set": True}
                    else:
                        members, rflags = refine_members(
                            kb,
                            graph,
                            question,
                            question_types,
                            chosen,
                            member_call=make_member_call(chosen),
                            topic_names=[graph.entity_name(s) for s in starts],
                        )
                    if args.audit_extend and members and not factor_override:
                        audit = audit_answer(
                            question,
                            chosen.get("chain_label") or display_relation(chosen["predicate"]),
                            [graph.entity_name(m) for m in members[:8]],
                            model=selector_model,
                        )
                        usage["calls"] += 1
                        usage["prompt_tokens"] += audit["prompt_tokens"]
                        usage["completion_tokens"] += audit["completion_tokens"]
                        if not audit["raw_response"] and not audit["error"]:
                            stats["empty_completion"] += 1
                        stats["audit_calls"] += 1
                        if audit["verdict"] == "missing":
                            stats["audit_missing"] += 1
                            row["audit_missing"] = True
                            ext_paths = build_extension_menu(kb, starts, chosen, question, "")
                            if ext_paths:
                                ext_blocks = [
                                    path_block(kb, graph, p, question_types, topic_name=start_name)
                                    for p in ext_paths
                                ]
                                # The condition already decided to revise; the
                                # body must commit (no second veto). Measured on
                                # cwq_dev300g: 55/62 revision rounds deadlocked
                                # when round 2 could abstain.
                                round2 = select_query_path_forced(
                                    question,
                                    f"the results of \"{chosen.get('chain_label') or display_relation(chosen['predicate'])}\"",
                                    ext_blocks,
                                    model=selector_model,
                                    think=args.think_select,
                                )
                                usage["calls"] += 1
                                usage["prompt_tokens"] += round2["prompt_tokens"]
                                usage["completion_tokens"] += round2["completion_tokens"]
                                if not round2["raw_response"] and not round2["error"]:
                                    stats["empty_completion"] += 1
                                if round2["pick"] is not None:
                                    chosen2 = ext_paths[round2["pick"]]
                                    members2, rflags2 = refine_members(
                                        kb,
                                        graph,
                                        question,
                                        question_types,
                                        chosen2,
                                        member_call=make_member_call(chosen2),
                                        topic_names=[graph.entity_name(s) for s in starts],
                                    )
                                    # The revision only lands when round 2 both
                                    # picked and produced members — never trade a
                                    # concrete answer for nothing.
                                    if members2:
                                        chosen, members, rflags = chosen2, members2, rflags2
                                        stats["extended"] += 1
                                        row["extended"] = True
                    stats["scope_filtered"] += rflags["scope"]
                    stats["ordinal_applied"] += rflags["ordinal"]
                    stats["refined_top1"] += rflags["refined"]
                    stats["refine_whole_set"] += rflags["whole_set"]
                    if rflags["scope"]:
                        row["scope_filtered"] = True
                    if rflags["ordinal"]:
                        row["ordinal_applied"] = True
                    row["refined"] = rflags["refined"]
                    row["selected"] = {
                        "predicate": chosen["predicate"],
                        "direction": chosen["direction"],
                        "readable": chosen.get("chain_label") or display_relation(chosen["predicate"]),
                        "also": chosen.get("also", []),
                        **({"chain_base": chosen["chain_base"]} if chosen.get("chain_base") else {}),
                    }
                    row["predicted"] = [graph.entity_name(m) for m in members]
                    predicted_ids = set(members)
                    top1 = members[0]
                    stats["hits_at_1"] += top1 in golds
                    p = len(predicted_ids & golds) / len(predicted_ids) if predicted_ids else 0.0
                    r = len(predicted_ids & golds) / len(golds) if golds else 0.0
                    f1_sum += f1(p, r)
            else:
                stats["no_candidates"] += 1
        else:
            stats["unresolved_start"] += 1
        out_rows.append(row)
        if index % 10 == 0 or index == len(rows_in):
            elapsed = time.time() - started
            print(
                f"  ... {index}/{len(rows_in)} ({elapsed:.0f}s) hits@1 {stats['hits_at_1']} | "
                f"F1 {f1_sum/max(1,index):.3f} | abstain {stats['abstained']} | empty {stats['empty_completion']}",
                flush=True,
            )

    n = max(1, stats["total"])
    metrics = {
        "hits_at_1": stats["hits_at_1"],
        "hits_at_1_rate": stats["hits_at_1"] / n,
        "mean_answer_f1": f1_sum / n,
        "total": stats["total"],
        "gold_in_subgraph": stats["gold_in_subgraph"],
        "refined_top1": stats["refined_top1"],
        "refine_whole_set": stats["refine_whole_set"],
        "scope_filtered": stats["scope_filtered"],
        "ordinal_applied": stats["ordinal_applied"],
        "chained": stats["chained"],
        "chain_abstained": stats["chain_abstained"],
        "verify_calls": stats["verify_calls"],
        "verify_switched": stats["verify_switched"],
        "abstain_recovered": stats["abstain_recovered"],
        "audit_calls": stats["audit_calls"],
        "audit_missing": stats["audit_missing"],
        "extended": stats["extended"],
        "abstained": stats["abstained"],
        "fallbacks": stats["abstained"] + stats["selection_error"],
        "selection_errors": stats["selection_error"],
        "no_candidates": stats["no_candidates"],
        "unresolved_start": stats["unresolved_start"],
        "empty_completions": stats["empty_completion"],
        "sketch_parse_failures": stats["sketch_parse_failed"],
        "hypothesis_parse_failures": stats["hypothesis_parse_failed"],
        "anchor_factor_attempts": stats["factor_attempts"],
        "anchor_factor_overrides": stats["factor_overrides"],
        "llm_cost_per_question": {
            "avg_llm_calls": usage["calls"] / n,
            "avg_prompt_tokens": usage["prompt_tokens"] / n,
            "avg_completion_tokens": usage["completion_tokens"] / n,
        },
    }
    with open(f"{output_dir}/predictions.jsonl", "w") as out:
        for row in out_rows:
            out.write(json.dumps(row) + "\n")
    write_json(f"{output_dir}/metrics.json", {"args": vars(args), "metrics": metrics})
    print(json.dumps(metrics, indent=2))
    if stats["empty_completion"] > 0.02 * n:
        print(f"CANARY: {stats['empty_completion']} empty completions (> 2%) — check the serving backend before trusting this run.")
    print(f"Wrote {len(out_rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
