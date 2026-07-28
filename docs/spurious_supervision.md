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

Only **28%** of questions have a unique path to their answer. 71% admit several;
the median is 4 and one question admits 30.

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
that is wrong three times out of four in the sense that matters. It cannot learn
to prefer the intended relation over a coincidental one, because the training
signal does not distinguish them. That is a property of the supervision, not of
any particular architecture, so it applies to every system trained this way.

It also compounds with the separate hand-measured finding that **15 of 90
annotated paths do not answer their own question**. The annotation is noisy in one
direction and the answer-based expansion is noisy in the other.

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

## Status

Measured on 100 questions from one dataset. To support a claim about the field it
needs:

- the full WebQSP test set and CWQ, not 100 questions;
- confirmation that a published system trained this way actually selects spurious
  paths at a similar rate — RoG does not release predictions, so this requires
  running it;
- the intent versus denotation retraining comparison above.
