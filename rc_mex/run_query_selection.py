"""Query-selection KGQA (method v2) — WebQSP/CWQ RoG subgraphs.

The answer to a KGQA question is the RESULT SET of a small query, not a
ranked entity. Measured basis (WebQSP test substrate): post-CVT, 99% of
answerable questions are one property from the topic entity, and the correct
property's full target set has F1 ceiling 0.897 (78% exact set).

Per question (2 LLM calls, short outputs):
  1. intent    — LLM names 2-3 candidate property names, no schema shown
  2. grounding — embed those names against the relations actually present
                 (per-segment max for composite CVT relations), UNION with
                 the question-embedding channel; junk predicates dropped
  3. execute   — algorithm walks each grounded (predicate, direction),
                 collecting full target sets + type evidence (zero-LLM:
                 question embedding vs subgraph type inventory)
  4. select    — LLM picks ONE property from ~a dozen options, each shown
                 with its answers; explicit abstain option; on abstain or
                 error, fall back to the question-embedding top candidate
  5. answer    — the selected property's full target set (members ordered
                 by type-match, then name)

Hard-fails at startup if MiniLM or the LLM endpoint is missing (no silent
degradation), and counts empty LLM completions as a run canary.

Usage:
  python3 -m rc_mex.run_query_selection --data data/webqsp/train150.jsonl --output runs/qsel_train150 [--limit 0]
"""

from __future__ import annotations

import argparse
import json
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
    describe_relation_chain,
    describe_target_relation,
    probe_llm_endpoint,
    select_query_path,
    select_set_member,
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
        label = path["chain_label"]
        if path["direction"] == "backward":
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
    args = parser.parse_args()
    intent_model = args.intent_model or args.model
    selector_model = args.selector_model or args.model
    refiner_model = args.refiner_model or args.model

    if not semantic_relation_model_available():
        sys.exit("FATAL: sentence-transformers/MiniLM unavailable — refusing to run degraded (check PYTHONPATH/HF_HOME/HF_HUB_OFFLINE).")
    probe = probe_llm_endpoint(model=selector_model)
    if not probe["ok"]:
        sys.exit(f"FATAL: LLM endpoint unavailable ({probe['error']}) — v2 requires 2 calls/question.")
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
            if args.max_hops >= 2:
                described = describe_relation_chain(question, start_name, model=intent_model)
            else:
                described = describe_target_relation(question, start_name, model=intent_model)
            usage["calls"] += 1
            usage["prompt_tokens"] += described["prompt_tokens"]
            usage["completion_tokens"] += described["completion_tokens"]
            if not described["names"] and not described["error"]:
                stats["empty_completion"] += 1
            hop2_names = described.get("names_2") or []
            question_types = question_type_evidence(kb, question)
            paths = build_candidate_paths(kb, graph, starts, question, described["names"])
            has_chains = False
            if args.max_hops >= 2 and paths:
                chains = build_chain_candidates(kb, graph, starts, paths, hop2_names)
                if chains:
                    paths = merge_mixed_menu(paths, chains)
                intersections = build_intersection_candidates(paths)
                if intersections:
                    paths = merge_mixed_menu(paths, intersections, cap=MIXED_MENU_CAP + INTERSECTIONS_ADDED)
                has_chains = any(p.get("chain_label") for p in paths)
            row["described_names"] = described["names"]
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
            if paths:
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
                selection = select_query_path(
                    question, start_name, blocks, model=selector_model, mixed=has_chains, think=args.think_select
                )
                usage["calls"] += 1
                usage["prompt_tokens"] += selection["prompt_tokens"]
                usage["completion_tokens"] += selection["completion_tokens"]
                if not selection["raw_response"] and not selection["error"]:
                    stats["empty_completion"] += 1
                chosen = None
                if selection["pick"] is not None:
                    chosen = paths[selection["pick"]]
                else:
                    row["abstained"] = selection["abstain"]
                    stats["abstained"] += selection["abstain"]
                    stats["selection_error"] += bool(selection["error"])
                    if selection["error"]:
                        row["selection_error"] = selection["error"][:160]
                    chosen = paths[0] if paths else None  # channel floor
                    row["fallback"] = True
                if chosen is not None and chosen.get("chain_label"):
                    row["chained"] = True
                    stats["chained"] += 1

                if chosen is not None:

                    def live_member_call(members, blocks):
                        refinement = select_set_member(
                            question=question,
                            start_entity_name=start_name,
                            property_name=chosen.get("chain_label") or display_relation(chosen["predicate"]),
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

                    members, rflags = refine_members(
                        kb,
                        graph,
                        question,
                        question_types,
                        chosen,
                        member_call=live_member_call,
                        topic_names=[graph.entity_name(s) for s in starts],
                    )
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
        "abstained": stats["abstained"],
        "fallbacks": stats["abstained"] + stats["selection_error"],
        "selection_errors": stats["selection_error"],
        "no_candidates": stats["no_candidates"],
        "unresolved_start": stats["unresolved_start"],
        "empty_completions": stats["empty_completion"],
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
