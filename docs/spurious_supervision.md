# Spurious Path Supervision in WebQSP

Modern KGQA systems — RoG, PARoG, DAMR, GCR — are trained on paths that reach the
gold answer. This measures how much that signal actually determines.

Measured on 100 WebQSP test questions with the executed candidate pools the
pipeline already produced. No model is run; this is a property of the dataset and
the graph, not of any system.

## The signal barely constrains anything

| | |
|---|---:|
| answerable questions | 87 |
| distinct paths reaching **exactly** the gold answer set | 339 |
| of those, the annotated path | 89 |
| **paths that reach the answer but are not the annotated one** | **74%** |

**74% is a non-identifiability rate, not a spuriousness rate.** Some of those 250
paths are legitimate alternative readings of the same question. How many are
merely coincidental is not measured here; `spurious_labels_to_judge.md` exists to
measure it.

Only **29%** of questions have a unique path to their answer; the rest admit
several, median 2 and up to 30.

This is not an artifact of the retriever. Enumerating **every** path the supplied
graph admits within two hops, rather than only SRTK's proposals, gives the same
figure: 29% unique either way, mean 3.6 paths against SRTK's 3.9. It remains
conditional on the two-hop budget and the pre-extracted neighborhood.

## What the alternatives look like

*"where did edgar allan poe died?"* — 30 paths return Baltimore, including:

- "In which entity did Edgar Allan Poe, an author, pass away?" ← annotated
- "Where did the **fictional character based on** Edgar Allan Poe pass away?"
- "Where did the **author represented in fiction by** Edgar Allan Poe pass away?"

*"what town was martin luther king assassinated in?"* — 9 paths return Memphis,
including "Where did the **artwork depicted in** Martin Luther King, Jr. pass
away?"

*"where is jamarcus russell from?"* — includes "Which city is **located in the
same county as** the city where…"

The relations most used by answer-reaching non-annotated paths are exactly the
ones that create these loops:

| count | relation |
|---:|---|
| 15 | `fictional_universe.person_in_fiction.representations_in_fiction` |
| 14 | `fictional_universe.fictional_character.based_on` |
| 33 | `location.hud_county_place.place` |
| 14 | `book.written_work.subjects` |

These route through a fictional-character node, an artwork node, or a
same-county node and return to the correct entity. The answer is right; the
reasoning is meaningless. This is the established notion of a **spurious logical
form**.

## Why it matters beyond this pipeline

Any system supervised by "this path reached the answer" is trained on a signal
that does not identify the intended reasoning: it admits a median of two and up to
thirty paths per question. Whether that is harmful depends on what fraction are
coincidental rather than genuinely alternative, which is unmeasured.

Exposure also differs by system and must be checked per method rather than
assumed. RoG mines question-to-relation-path pairs and is plausibly exposed;
PARoG trains a planner on SPARQL-derived sub-objectives, which is a different
signal; DAMR learns an adaptive path scorer during search. No claim about any of
their errors follows from this measurement -- it establishes that the supervision
is ambiguous, not that the ambiguity causes any particular system to fail.

It compounds with the separate hand-measured finding that **15 of 90 annotated
paths do not answer their own question**. That figure is a semantic judgement made
by reading question and path, not an execution failure: executing every annotated
path over 1,155 questions returns the gold answers exactly 88% of the time,
partially 6%, nothing 6%, and never a non-empty wrong set. So it is not a snapshot
or executor artifact, though it does rest on 90 hand labels from two annotators.

## A caveat against our own method

The comparator corpus built here labels a candidate positive when it is the
annotated path **or returns exactly the gold answer set**. On the training split
that found 821 answer-equivalent positives against 596 annotated ones, reported at
the time as 40% more training signal.

By the measurement above, a large share of those 821 are spurious. Fine-tuning on
that corpus still improved answer exact match from 0.65 to 0.74, so the added
signal was not net harmful — but the gain cannot be attributed to better
supervision until the spurious ones are separated out.

**This is the experiment that follows:** label answer-reaching paths as
`intent_correct` versus `denotation_correct_only`, retrain on the intent-correct
subset alone, and compare. If it matches or beats 0.74 with far fewer positives,
the spurious ones are noise the model is absorbing. If it drops, they carry useful
signal and the framing needs revising.

## Prior work

The phenomenon is established. Codex identifies Path Spuriousness-aware RL (EACL
2023), which defines a spuriousness metric and modifies the RL reward, and PathISE
(2026), which learns to distinguish informative paths from answer-level labels and
distils that into a path generator. An earlier draft of this document claimed the
observation was uncovered by prior work; that claim is withdrawn. Any contribution
here has to be positioned against those, and would have to be about using inverse
semantic reconstruction to separate intent-equivalent from denotation-equivalent
paths at scale and across schemas.

## Status

Measured on 100 questions from one dataset. To support a claim about the field it
needs:

- the full WebQSP test set and CWQ, not 100 questions;
- confirmation that a published system trained this way actually selects spurious
  paths at a similar rate — RoG does not release predictions, so this requires
  running it;
- the intent versus denotation retraining comparison above.
