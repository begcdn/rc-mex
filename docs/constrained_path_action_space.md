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

## 4. Two classes, with different costs

Classifying the 472 excluded questions by what would be needed to admit them:

| Class | Questions | % of test | What it needs |
|---|---:|---:|---|
| Supported today | 1,167 | 71.2% | — |
| **A** answer-node restriction | 137 | 8.4% | answer typing; **no action-space change** |
| **B** intermediate-node join | 147 | 9.0% | branched candidates + secondary entity linking |
| **C** multi/value constraint | 16 | 1.0% | ranges, comparison operators |
| Other (>2 hops, `Order`, incomplete) | 172 | 10.5% | out of scope for this design |

### Class A costs nothing new

A Class A constraint restricts which entities in the answer set are admissible:
`notable_types = US County` for "what county is frederick md in?",
`sports_team.sport = Basketball` for "what basketball teams has shaq played for?".

The path stays linear. The restriction is a property of the executed answer set,
and the question states the required type in ordinary English ("what **county**",
"what **basketball** teams"). Nothing needs filtering at proposal time: the
comparator only needs to *see* each candidate's answer type and prefer the
candidate whose type matches the question.

That channel already exists — commit `1a6cc4f` added `question_generated_answer`
and `question_generated_path_answer`, carrying answer type, cardinality, unlabeled
id count, and sample labels. It is untrained because the synthetic corpus stores
one endpoint per path and therefore has no type or cardinality signal to learn.

**Class A and the planned comparator retraining are the same piece of work.**
Retraining on executed SRTK candidates supplies exactly the answer-type signal that
Class A questions need. Coverage rises 71.2% → 79.6% with no change to the
proposer, the path object, or the firewall.

### Class B is a genuine architectural change

A Class B constraint joins the mediator to a second named entity:
`tv.regular_tv_appearance.character = Ken Barlow` for "who plays ken barlow in
coronation street?". This needs two things the current setup does not have:

1. **A branched candidate object.** Minimally, `PathSpec` gains an optional tuple
   of `(node_index, relation, direction, bound_entity)` filters. This is one
   optional field, not a general query language, and the CVT structure of Freebase
   makes it the natural unit.
2. **Secondary entity linking.** The bound entity is a second mention in the
   question. The current setup deliberately supplies the gold topic entity to avoid
   an entity-linking confound; Class B breaks that. Linking secondary mentions with
   gold would be a new and much weaker firewall, and it must be stated as such.

Cost 2 is the real one, and it is why Class B should not be bundled with Class A.

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

**Class A (runnable as soon as the comparator is retrained):**

Admit the 137 Class A questions. Train two comparators on identical executed-candidate
data, differing only in input mode (`question_generated` vs
`question_generated_answer`). Compare on the Class A slice and the currently supported
slice separately.

- Supports: answer-mode wins on the Class A slice and does not regress on the
  supported slice.
- Weakens: answer-mode wins uniformly on both — the gain is generic answer-typing,
  not constraint handling, and the Class A framing adds nothing.
- Rejects: no gain on Class A. Answer-set evidence does not carry the restriction,
  and Class A needs explicit filtering after all.

**Class B (only after Class A resolves):**

Extend the path object with one optional filter tuple. Report Class B under an
explicitly labeled weaker firewall, with gold-linked secondary entities, as a
controlled upper bound rather than an end-to-end result.

- Supports: recall of annotated constrained paths is comparable to unconstrained
  paths, and comparator pairwise accuracy on constraint-dropping negatives exceeds
  its accuracy on nearby-relation negatives.
- Rejects: the proposer cannot reach constrained paths at usable recall, in which
  case the bottleneck is proposal, not verification, and the hypothesis is untested
  rather than wrong.

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

## 8. Recommendation

Do Class A and the comparator retraining as one experiment. They need the same
corpus, they need no entity linking, they preserve the current firewall, and
together they lift coverage 71.2% → 79.6% while testing the answer-evidence channel
that is currently plumbed but untrained.

Defer Class B until Class A reports. It requires secondary entity linking, which
weakens the firewall, and its value depends on whether answer-set evidence already
carries restriction meaning — which Class A answers.

Do not attempt Class C. Sixteen questions cannot support a claim.
