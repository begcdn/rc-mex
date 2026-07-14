"""Bounded executable-pattern proposal for KGQA.

The search unit is a complete relation-sequence hypothesis with an executable
denotation, not an independently scored edge.  Hypotheses are compared within
depth families so longer patterns do not crowd out valid direct queries merely
because they contain more question terms.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rc_mex.executable_pattern_alignment import pattern_from_runtime_path, readable_relation


MAX_HOPS = 2
PATTERNS_PER_DEPTH = 7
HOP2_FANOUT_CAP = 300


@dataclass
class PatternHypothesis:
    steps: tuple[tuple[str, str], ...]
    targets: frozenset[str]
    member_qualifiers: dict[str, dict]
    pattern_text: str
    score: float = -math.inf

    @property
    def depth(self) -> int:
        return len(self.steps)

    def to_candidate(self) -> dict:
        predicate, direction = self.steps[-1]
        candidate = {
            "predicate": predicate,
            "direction": direction,
            "targets": sorted(self.targets),
            "also": [],
            "member_quals": self.member_qualifiers,
            "pattern_steps": [
                {"predicate": relation, "direction": edge_direction}
                for relation, edge_direction in self.steps
            ],
            "pattern_text": self.pattern_text,
            "pattern_score": self.score,
        }
        if self.depth > 1:
            first_relation, first_direction = self.steps[0]
            candidate["chain_base"] = {
                "predicate": first_relation,
                "direction": first_direction,
            }
            candidate["chain_label"] = " → ".join(
                readable_relation(relation) for relation, _ in self.steps
            )
        return candidate


class ExecutablePatternScorer:
    """Cached batched cosine scorer for questions and executable patterns."""

    def __init__(self, model):
        self.model = model
        self._question_vectors: dict[str, np.ndarray] = {}
        self._pattern_vectors: dict[str, np.ndarray] = {}

    def score(self, question: str, pattern_texts: list[str]) -> list[float]:
        if question not in self._question_vectors:
            self._question_vectors[question] = self.model.encode(
                [question],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        missing = sorted(set(pattern_texts) - self._pattern_vectors.keys())
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._pattern_vectors.update(zip(missing, vectors))
        question_vector = self._question_vectors[question]
        return [
            float(np.dot(question_vector, self._pattern_vectors[text]))
            for text in pattern_texts
        ]


def _edge_groups(kb: dict, sources: Iterable[str]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict] = {}
    for source in sources:
        for relation in kb["entities"].get(source, {}).get("relations", []):
            key = (str(relation["predicate"]), str(relation["direction"]))
            group = groups.setdefault(key, {"targets": set(), "member_quals": {}})
            target = str(relation["object"])
            group["targets"].add(target)
            for qualifier, values in (relation.get("qualifiers") or {}).items():
                slot = group["member_quals"].setdefault(target, {}).setdefault(qualifier, [])
                slot.extend(value for value in values if value not in slot)
    return groups


def _pattern_text(steps: tuple[tuple[str, str], ...]) -> str:
    pattern = pattern_from_runtime_path(
        {
            "relations": [
                {"predicate": relation, "direction": direction}
                for relation, direction in steps
            ]
        }
    )
    if pattern is None:
        raise ValueError(f"Cannot represent executable steps: {steps}")
    return pattern.canonical_text()


def enumerate_pattern_hypotheses(
    kb: dict,
    starts: set[str],
    *,
    max_hops: int = MAX_HOPS,
) -> list[PatternHypothesis]:
    """Enumerate executable linear patterns before any semantic pruning."""
    hypotheses = []
    first_groups = _edge_groups(kb, starts)
    for first_step, first in first_groups.items():
        first_targets = set(first["targets"])
        visible_first = first_targets - starts or first_targets
        hypotheses.append(
            PatternHypothesis(
                steps=(first_step,),
                targets=frozenset(visible_first),
                member_qualifiers=first["member_quals"],
                pattern_text=_pattern_text((first_step,)),
            )
        )
        if max_hops < 2 or len(first_targets) > HOP2_FANOUT_CAP:
            continue
        for second_step, second in _edge_groups(kb, first_targets).items():
            second_targets = set(second["targets"])
            visible_second = second_targets - starts
            if not visible_second:
                continue
            steps = (first_step, second_step)
            hypotheses.append(
                PatternHypothesis(
                    steps=steps,
                    targets=frozenset(visible_second),
                    member_qualifiers=second["member_quals"],
                    pattern_text=_pattern_text(steps),
                )
            )
    return hypotheses


def rank_pattern_hypotheses(question: str, hypotheses: list[PatternHypothesis], model) -> list[PatternHypothesis]:
    """Score full executable patterns against the question in one batch."""
    if not hypotheses:
        return []
    unique_texts = sorted({hypothesis.pattern_text for hypothesis in hypotheses})
    scorer = model if isinstance(model, ExecutablePatternScorer) else ExecutablePatternScorer(model)
    scores = dict(zip(unique_texts, scorer.score(question, unique_texts)))
    for hypothesis in hypotheses:
        hypothesis.score = scores[hypothesis.pattern_text]
    return sorted(hypotheses, key=lambda item: (-item.score, item.steps))


def retain_depth_families(
    hypotheses: list[PatternHypothesis],
    *,
    patterns_per_depth: int = PATTERNS_PER_DEPTH,
) -> list[PatternHypothesis]:
    """Keep a fixed semantic beam independently for each structural depth."""
    by_depth: dict[int, list[PatternHypothesis]] = defaultdict(list)
    for hypothesis in hypotheses:
        by_depth[hypothesis.depth].append(hypothesis)
    retained = []
    for depth in sorted(by_depth):
        ranked = sorted(by_depth[depth], key=lambda item: (-item.score, item.steps))
        retained.extend(ranked[:patterns_per_depth])
    return retained


def build_executable_pattern_menu(
    kb: dict,
    starts: set[str],
    question: str,
    model,
    *,
    max_hops: int = MAX_HOPS,
) -> list[dict]:
    hypotheses = enumerate_pattern_hypotheses(kb, starts, max_hops=max_hops)
    ranked = rank_pattern_hypotheses(question, hypotheses, model)
    retained = retain_depth_families(ranked)
    return [hypothesis.to_candidate() for hypothesis in retained]
