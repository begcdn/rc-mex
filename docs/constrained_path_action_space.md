# Constrained Paths: Diagnosis and Action-Space Design

Status: diagnosis complete, no implementation. Written while the selection-ablation
run was executing. All measurements are from local WebQSP files and use gold
annotations for dataset characterization only, not for search or ranking.

## 1. The general failure

The pipeline evaluates 1,167 of 1,639 WebQSP test questions (71.2%). The excluded
472 are not a random sample: they are the questions whose meaning includes a
restriction, and restriction is exactly the kind of meaning a question-comparison
verifier should be good at checking. Excluding them removes the questions that
most directly test the hypothesis.

## 2. Why the current representation causes it

The candidate object is `PathSpec`: an anchor plus a linear tuple of `Hop`. A
linear chain cannot express a restriction that hangs off a node in the middle of
the chain, so no proposer built on this object can emit a candidate that answers a
constrained question. `supported_questions` drops those questions rather than
allowing the pipeline to fail on them, which is the correct choice for a controlled
measurement but hides the limit.

This is a representational limit, not a coverage gap or a tuning problem.

## 3. What the constraints actually are

Measured over WebQSP test (`data/pattern_alignment/webqsp/WebQSP.test.json`) joined
with the supplied neighborhoods (`data/webqsp/test.jsonl`).

Constraint inventory over 1,815 parses:

| Property | Value |
|---|---|
| Parses with at least one constraint | 471 |
| Exactly one constraint | 363 |
| Operator | `Equal` 419, `LessOrEqual` 100, `GreaterOrEqual` 100 |
| Argument type | `Entity` 408, `Value` 211 |
| Constrained node index | 0 → 543, 1 → 76 |
| Chain length of constrained parses | 2 hops 361, 1 hop 110 |

The paired `LessOrEqual`/`GreaterOrEqual` counts are date ranges and account for
most multi-constraint parses.

The dominant shape is a two-hop chain through a Freebase mediator (CVT) node with
one `Equal`/`Entity` constraint attached to that mediator. This is a chain with a
decoration, not a general conjunctive query.

Of the 322 single `Equal`/`Entity` constraints that are executable in the supplied
graph (constraint entity present as a node **and** constraint predicate present as
an edge — 83.9% of such constraints):

| Constrained node | Recoverable from question text | Count |
|---|---|---:|
| intermediate CVT | yes | 182 |
| intermediate CVT | no | 28 |
| answer node | yes | 81 |
| answer node | no | 31 |

"Recoverable" is token/stem overlap between the constraint entity name and the
question. 263 of 322 (81.7%) pass. This is a lower bound: the failures are
overwhelmingly constraints the question states implicitly rather than lexically —
`gender = Male` for "who is emma stone father?", `type_of_union = Marriage` for
"who is michael j fox wife?". A language model recovers these; a string matcher
cannot.

**This is the central finding.** WebQSP constraints are almost always things the
question actually says. A method that verifies meaning by reconstructing the
question is therefore not structurally disqualified from handling them.

## 4. Selection cannot enforce a restriction (corrected)

An earlier version of this document claimed that answer-node restrictions come
"for free" because the comparator can see the answer type. **That was wrong**, and
the error matters enough to record.

Selecting a path chooses among denotations; it cannot subset one. If a question
asks for basketball teams and the best available path returns every team the
athlete played for, showing the comparator the answer type may help it *recognize*
the mismatch, but executing the selected path still returns the unfiltered set. To
enforce a restriction the system must either execute a filter as part of the
candidate, or score answer entities individually after execution. Question-level
path selection does neither.

The decisive measurement is therefore not "which node is constrained" but "can any
linear path reach the gold denotation at all". Over the 1,628 graph rows joined
with official questions, excluding via the real `supported_questions()`:

| Constraint kind | Excluded | Linear path reaches gold set | **Needs executable filter** |
|---|---:|---:|---:|
| relation-value (`sport = Basketball`) | 258 | 100 (39%) | **158 (61%)** |
| unary type (`notable_types = US County`) | 82 | 29 (35%) | **53 (65%)** |
| value / range (dates) | 68 | 32 (47%) | **36 (53%)** |
| other exclusions (>2 hops, `Order`, incomplete) | 65 | — | — |

Across all excluded constrained questions, 161 of 408 (39%) are reachable by
selection alone and **247 (61%) require filtering the returned denotation**. The
unary/relation-value split does not predict reachability: both sit near 35-39%.

Two consequences:

1. **No coverage claim may be made from a comparator change alone.** At most 39% of
   excluded constrained questions are within reach of better selection, and only if
   the comparator can distinguish the right path among candidates it already sees.
2. **The action space needs an executable filter, not just a richer input to the
   comparator.** That is a change to the candidate object and to execution, and it
   is required before any constrained slice enters the evaluation.

The earlier answer-node / intermediate-node split is retained below only as a
description of constraint shape. It is not a cost model.

## 4b. Shape of the constraints

Of 322 executable single `Equal`/`Entity` constraints, 210 attach to an
intermediate mediator and 112 to the answer node; 263 (81.7%) are lexically
recoverable from the question, and the failures are implicit rather than absent
(`gender = Male` for "who is emma stone father?"). This supports the claim in §5
that constraints are expressible in a reconstructed question. It says nothing about
whether the pipeline can execute them.

## 5. Hypothesis

> Constraint satisfaction is verifiable by question reconstruction. If a candidate
> omits, adds, or misbinds a restriction that the original question expresses, the
> question generated from that candidate differs in meaning from the original, and
> a comparator that reads both can detect the difference.

This is the existing hypothesis applied to a wider action space, not a new one. It
predicts that constrained questions should be *easier* for this method relative to
path-ranking baselines than unconstrained ones, because a dropped constraint is a
large, visible semantic difference.

## 6. Falsifying experiments

**E1 — does answer evidence help at all, and through which channel?**

Train comparators on identical executed-candidate data differing only in input
mode: `question_generated`, `..._answer_type`, `..._answer_count`,
`..._answer_labels`. Evaluate on the currently supported slice only, since no
constrained question is admissible until E3.

- Supports: `type` and/or `count` beat the baseline. Structural answer evidence
  carries signal the two question strings do not.
- Confounded: only `labels` wins. That is consistent with the comparator answering
  from world knowledge rather than verifying, so it must be re-checked under
  entity-disjoint evaluation before any gain is attributed to verification.
- Rejects: no channel beats the baseline.

**E2 — hard-negative mining, independent of constraints.**

Rebuild the comparator corpus from executed SRTK candidates on WebQSP train,
labeling positives by answer-set equivalence. Justified by the train/inference
negative-distribution mismatch alone. Held-out set must come from the same
distribution, since the current synthetic dev is saturated at R@1 0.991 and cannot
discriminate.

**E3 — executable constrained candidates.**

Only after E1/E2. Extend the candidate object with an executable filter and
measure, on the 408 excluded constrained questions, what fraction the extended
proposer reaches and what fraction the verifier then selects correctly. Report
under an explicitly weaker firewall, because binding a filter value requires
linking a second entity mention.

- Supports: constrained recall approaches unconstrained recall, and comparator
  pairwise accuracy on constraint-dropping negatives exceeds its accuracy on
  nearby-relation negatives — the restriction is the easier discrimination.
- Rejects: the proposer cannot reach constrained candidates at usable recall, in
  which case the bottleneck is proposal and the hypothesis is untested, not wrong.

## 7. Relation to prior work

Not yet verified against the papers or their released code in this session; these
are characterizations to check before any claim is written up.

- **Logical-form methods** (RnG-KBQA, TIARA, Pangu, DecAF) generate or rank
  s-expressions/SPARQL, which express constraints natively. They require
  logical-form supervision and are tied to a schema. The present method needs no
  logical-form supervision, so a fair comparison must state that difference.
- **Path-ranking and retrieval methods** (SRTK, RoG, SR+NSM) rank linear relation
  paths and share the exact limit diagnosed here. If that is accurate, constrained
  WebQSP questions are a place where an inverse verifier could show an advantage
  that is not merely a reranking improvement.
- **GNN-RAG** propagates over a dense subgraph and can satisfy a constraint
  implicitly without representing it, but cannot say which restriction it applied.

Confirm each of these against primary sources before using them as a contrast.

## 8. Recommendation (corrected)

1. **Make no coverage claim yet.** Define and implement an executable constrained
   candidate first — a filter that runs during execution and subsets the
   denotation. Until that exists, admitting constrained questions only adds
   questions the pipeline provably cannot answer.
2. **Keep answer evidence as separate ablation arms, not a bundle.** `type`,
   `count`, and `labels` are separate input modes. Answer *labels* let a comparator
   score "capital of Austria?" against "Vienna" from world knowledge without
   verifying the path, which changes the hypothesis under test. Any gain from the
   labels arm must be checked against entity-disjoint evaluation before it is
   attributed to verification.
3. **Comparator retraining on executed candidates remains worth doing** — the
   train/inference negative-distribution mismatch is real and independent of any of
   this — but it must be justified by hard-negative mining, not by a constraint
   coverage claim.
4. **Selection policies stay ablations.** `argmax` is primary until a run shows a
   general improvement.

Do not attempt value/range constraints. Sixty-eight questions, needing comparison
operators and date normalization, cannot support a claim.
