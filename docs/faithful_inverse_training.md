# Faithful Inverse Training

## Motivation

The frozen inverse generator was tested after SRTK retrieval on 100 controlled
WebQSP questions. SRTK proposed the gold relation path for 90% of questions and
for 97.8% of questions where that path existed in the supplied graph. The
inverse verifier selected the gold path only 37% of the time. It stopped early
on a wrong path for 44% of questions.

The failure was concentrated in the generator rather than candidate recall.
Added hops, missing hops, and reversed relations were frequently omitted from
the reconstructed question. Two-hop exact answer accuracy was 12.5%.

## Training mismatch

The retained training corpus contained 2,683 one-hop examples and 660 two-hop
examples. The language-generation loss was applied only to positive paths.
Hard negatives trained a separate ranking head, so the generator itself was not
taught what an added-hop or otherwise incorrect path meant. Added-hop examples
were absent from the synthetic corruption categories.

## New corpus contract

`inverse_verifier synthesize` samples executable paths from KQA Pro and official
WebQSP training neighborhoods. One-, two-, and three-hop paths are balanced.
Immediate inverse relation bounces, metadata edges, cycles, and anonymous
untyped endpoints are excluded from positive examples.

Every positive and negative path receives a controlled question containing:

- every relation in order;
- every traversal direction;
- the endpoint type;
- no answer entity or intermediate entity name.

Hard negatives are executable paths from the same graph. They include added
hops, missing hops, sibling relations, and executable direction alternatives.
Every negative receives its own question rather than sharing the gold target.

The retained dataset builder no longer uses the earlier local-Ollama rewrite. It compiles each
path into explicit logical facts, asks GPT-4o to verbalize the full query, and asks GPT-4o mini to
select the exact represented path from randomized executable alternatives. Questions are accepted
only when the intended path, endpoint type, complete fact set, and natural-language checks agree.
Canonical questions remain audit fields; failed naturalizations are rejected rather than used as
fallback training targets.

## Objective

The `faithful_inverse` objective combines:

```text
generation loss:
  each executable path -> that path's own question

contrastive likelihood loss:
  p(original question | gold path)
  > p(original question | executable hard negative)
```

The model still emits only a question at inference. No gold path, answer, or
relation label is used by candidate search.

## Required evaluation

The first evaluation is not final QA. For a held-out gold path and its
executable negatives, generate every candidate question and report:

- gold question similarity;
- negative question similarity;
- gold-over-every-negative accuracy;
- pairwise accuracy by negative category;
- margin over the strongest negative.

Only if added-hop, missing-hop, direction, and sibling-relation discrimination
improve should the full SRTK retrieval pipeline be rerun.

## Smoke result

A 32-example, one-epoch integration test on a disjoint synthetic development
sample changed:

| Metric | Frozen generator | Faithful smoke |
|---|---:|---:|
| Gold beats every negative | 0.375 | 0.688 |
| Pairwise accuracy | 0.797 | 0.878 |
| Added-hop accuracy | 0.688 | 0.812 |
| Missing-hop accuracy | 0.556 | 0.778 |
| Mean margin over strongest negative | 0.005 | 0.039 |

This verifies objective direction only. It is not evidence of benchmark
improvement and must not be reported as a final result.

## Direction-supervision correction

The first direction-balanced follow-up used mechanically reversed paths only as
contrastive negatives. Frozen evaluation showed that this was the wrong training
intervention. On KQA Pro, wrong-direction pair accuracy fell to `0.073`, while
wrong-relation (`0.871`) and wrong-answer-type (`0.987`) accuracy remained high.
The model learned that a reversed rendering should score poorly, but generation
was never trained to express what the reverse executable query actually means.

The replacement corpus uses only relations for which the source graphs contain
real one-hop examples in both directions. Each pair contains:

- one executable forward path and its natural question;
- one executable backward path for the same relation and its natural question;
- explicit grounded subject/object facts and return variables;
- a relation-disjoint train/development split.

Training alternates the positive direction by epoch. In even epochs the forward
path is generated and ranked above the backward path for the forward question;
in odd epochs the backward path is generated and ranked above the forward path
for the backward question. Thus both directions receive generation supervision,
and neither is treated as an impossible counterfactual.

The retained direction corpus adds 241 training and 24 development pairs to the
validated v8 corpus. One `relative` pair was rejected because the grounded
relation is symmetric. All 266 source pairs were independently reviewed against
their explicit queries; one over-narrow religion paraphrase was corrected. The
final corpus has no train/development relation overlap and no detected structural
or semantic audit failures.

### Frozen executable-direction result

The 24 development relations are disjoint from the paired-direction training
relations. Each is evaluated in both directions, producing 48 decisions. The
original v8 model and the executable-direction model use the same generated-
question similarity evaluator:

| Model | Executable direction accuracy | Positive similarity | Negative similarity | Margin |
|---|---:|---:|---:|---:|
| Original v8 | 0.729 | 0.849 | 0.766 | 0.083 |
| Executable-direction v1 | **0.854** | **0.905** | 0.785 | **0.120** |

The intervention therefore recovers six additional held-out direction decisions
and increases the separation margin by about 45%. This supports the narrow claim
that paired executable supervision improves direction semantics. It does not
solve full candidate ranking: KQA Pro gold-over-all remains `0.139`, WebQSP
gold-over-all remains `0.293`, and strict unseen-composition gold-over-all is
only `0.034`.

The older KQA `wrong_direction` corruption is not used as the primary direction
measure. It flips an arrow without swapping endpoint types, so many negatives
are incoherent queries rather than executable reverse relations. Its low score
must not be conflated with performance on real bidirectional KG queries.

## Grounded generator input contract

A generator-only audit found that executable-direction supervision improved
relative discrimination but did not make the generated questions reliable:
only 97 of 174 judgeable held-out generations were fully faithful. The main
errors were reversed direction, unsupported facts, wrong answer roles, and
wrong relation meanings.

The cause was an input/target mismatch. Naturalized targets were created from a
grounded relation glossary, but the generator input discarded that glossary.
For Freebase predicates with three or more segments, it retained only the final
segment. In the retained glossary, 472 relations participate in 147 resulting
label collisions; for example, 22 distinct predicates render as `team`.

The grounded input contract now gives the generator, for every hop:

- the complete relation ID;
- a natural-language relation meaning;
- canonical subject and object roles;
- a fact template instantiated with `START`, intermediate nodes, and `ANSWER`;
- source, destination, and requested answer types.

Traversal direction changes the variable binding in the instantiated fact. For
example, a backward authorship hop is rendered as `ANSWER (written work) was
written by START (author)`. The same renderer is used for training and
inference. The glossary is copied into the trained checkpoint, so inference on
a known KG does not require an LLM call. A new KG requires one offline glossary
derived from its schema documentation or grounded examples; an opaque relation
ID alone is not enough to recover semantics.
