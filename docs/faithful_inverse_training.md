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

The optional `naturalize` stage uses a local Ollama model to rewrite controlled
questions into natural English. It asks the teacher to account for the exact
ordered relation/direction sequence and keeps the controlled target as an audit
field. Canonical questions are retained when validation fails.

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
