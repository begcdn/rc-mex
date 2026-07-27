# Experimental Audit

A check of the data, the firewall, and the assumptions behind every number
reported so far. Run against the local repository while the WebQSP-train
candidate pass was executing.

## Verified sound

**Train/test firewall.** WebQSP train and test share zero question ids and zero
identical question strings (3,098 / 1,639 questions; 2,826 / 1,628 graphs).

**Generator provenance.** All 580 distinct anchors in the synthetic training
corpus appear in the full WebQSP *train* graphs. The generator never saw test
graphs. `synthesize` defaults to `data/webqsp/train.jsonl`, and the corpus is
consistent with that.

**Gold path definition.** `parse_sparql_path` recovers traversal direction from
SPARQL, and a direction bug there would silently corrupt every path metric.
Executing each annotated path on its supplied graph over all 1,155 supported test
questions:

| outcome | count |
|---|---:|
| exactly the gold answers | 1,018 (88%) |
| partial overlap | 67 (6%) |
| path returns nothing | 70 (6%) |
| **returns something with zero overlap** | **0 (0%)** |

The last row is the direction-bug signature and it is empty. Gold is sound. The
6% returning nothing is graph incompleteness, consistent with the reported 0.92
availability.

**Endpoint filter.** Across 267,972 graph nodes the machine-id pattern matched 2
strings that are not `m.`/`g.` ids (`c.jpg`, `x.jpg`), neither a plausible
answer. No test question has gold answers consisting entirely of machine ids, so
the filter can never render a question unanswerable.

**No gold in selection.** `select_candidate`, `candidate_score` and
`has_answerable_endpoint` take no gold argument; the only occurrence of the word
is a docstring. Gold enters only through metrics and the evaluation-only
`matches_gold_path` / `gold_equivalent` fields.

**Configuration.** The runs load
`deberta_comparator_question_generated_v1`, whose stored `input_mode` is
`question_generated`, matching what is claimed.

**No duplicate questions** in the test set.

**The generator/comparator conclusion does not depend on Claude's labels:**

| labels | generator | comparator on that set |
|---|---:|---:|
| Ziad only (51) | 90% | 58% |
| Claude only (39) | 86% | 47% |
| all 90 | 88% | 53% |

The gap is large under either annotator alone.

## Problems found

### 1. n=100 cannot separate most of what we compared

95% Wilson intervals on the latest run:

| variant | gold path | 95% interval |
|---|---:|---|
| retrieval | 0.50 | [0.40, 0.60] |
| retrieval_filtered | 0.53 | [0.43, 0.62] |
| comparator | 0.41 | [0.32, 0.51] |
| comparator_filtered | 0.43 | [0.34, 0.53] |
| comparator_retrieval_filtered | 0.57 | [0.47, 0.66] |

Every interval is ~0.19 wide. **Any two variants within about 14 points are
indistinguishable as marginals.** Most of the differences discussed across three
runs are inside that band.

What partly rescues this is that the important claims were made with *paired*
win/loss tests on the same questions, which is far more powerful than comparing
two marginals. Those remain valid. But absolute numbers should always be quoted
with the interval.

### 2. Multiple comparisons are not being accounted for

Each run reports 5 variants; there have been 3 runs, plus a 5-point K sweep and 4
answer-channel arms. At 5 comparisons the chance of at least one spurious p<0.05
is 23%; at 20 it is 64%.

Concretely: **the endpoint filter's 5–0 result (p = 0.031) does not survive
correction.** It should be described as suggestive, not established. The
proposer+verifier combination (p = 0.0094 and p = 0.0003) does survive.

### 3. The evaluation subset is not a random sample

The 100 questions are the first 100 supported rows in file order. They are
slightly easier than the remainder — 84% one-hop against 79%, gold path available
92% against 94%. Not a severe skew, but it is a convenience sample and should be
drawn at random with a fixed seed.

### 4. Exact match penalises correct-but-incomplete answers

On the latest run, 11 questions return a right answer without an exact set match,
and 8 of those return a pure subset or superset of gold. They score 0. Answer F1
already captures this, which is why F1 sits ~5 points above EM; EM alone
overstates failure.

## Assumptions that hold but bound the claims

- **Topic entities are supplied**, 1,599 of 1,628 questions with exactly one.
  This is controlled KGQA, not end-to-end, and no entity-linking error is
  measured.
- **6% of annotated paths are not executable** in the supplied neighborhood, so
  0.94 is the ceiling for any path-selection metric here, not 1.0.
- **~17% of annotated paths do not answer their own question** (15 of 90 audited),
  which is a noise floor under every path-match number.
- **The comparator's synthetic dev set is saturated** at 0.991 and cannot
  distinguish input modes; the choice of `question_generated` over
  `question_path` therefore rests on no evidence.

## Recommendations

1. **Quote intervals.** No comparison at n=100 should be reported as a difference
   unless a paired test supports it.
2. **Evaluate on more questions.** 1,155 supported test questions exist; the runs
   use 100. At ~42s per question the full set is ~13 hours, but even 300 would
   halve the interval width.
3. **Randomize the subset** with a fixed seed rather than taking file order.
4. **Report F1 alongside EM** and stop quoting EM alone.
5. **Pre-register the comparison** for the comparator retraining: decide the
   primary metric and the arms before the run, so the multiple-comparison problem
   does not recur.
