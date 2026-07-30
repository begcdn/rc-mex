# Spurious Path Supervision in WebQSP

Modern KGQA systems — RoG, PARoG, DAMR, GCR — are trained on paths that reach the
gold answer. This measures how much that signal actually determines.

Measured on 100 WebQSP test questions with the executed candidate pools the
pipeline already produced. No model is run; this is a property of the dataset and
the graph, not of any system.

> **Superseded in part.** The figures below were computed on graphs that store
> entity names rather than Freebase ids, which collapses distinct entities into
> self-loops and fills the candidate space with paths containing hops that do
> nothing. See `identity_hops.md`. At least 36.7% of answer-reaching paths are
> affected. Corrected figures are given inline; the phenomenon survives the
> correction at roughly 1.6× smaller magnitude.

## The signal barely constrains anything

| | | corrected |
|---|---:|---:|
| answerable questions | 87 | 300 |
| distinct paths reaching **exactly** the gold answer set | 339 | 2,114 → **1,338** |
| **paths that reach the answer but are not the annotated one** | **74%** | **77.6%** |

The corrected column enumerates every ≤2-hop path on 300 questions and drops
hops that are genuinely reflexive on the node set they act on. Non-identifiability
is real and was not created by the artifact — but the raw path *count* was
inflated from 4.46 to 7.05 per question by empty hops.

**This is a non-uniqueness rate, not an error rate.** It establishes that
reaching the gold answer does not identify the reasoning. It does not establish
that the extra paths are wrong; some are genuine schema aliases. How many are
semantically invalid is currently **unmeasured** — the attempt below is
withdrawn.

The spuriousness rate was measured separately, by labelling all 151 deduplicated
non-annotated answer-reaching paths in `spurious_labels_to_judge.md`. **Those
labels are now suspect**: they were assigned by reading generated questions
without the raw path beside them, and an unknown share describe paths whose first
hop is a no-op — which is why several read as bizarre.

| label | count | share |
|---|---:|---:|
| valid alternative reading | 46 | 30% |
| coincidence — reaches the answer through unrelated reasoning | 103 | 68% |
| question too vague to decide | 2 | 1% |

> **This measurement is withdrawn.** The 68% figure and the "roughly 50% of
> everything that reaches the gold answer is spurious" claim derived from it
> should not be used.
>
> Three defects, any one of which is disqualifying:
>
> 1. **The labels were assigned by reading generated questions with no raw path
>    beside them**, so they confound path spuriousness with generator error. The
>    generator is 88% faithful on annotated paths, but these candidates are
>    selected for being unusual, and that rate does not transfer to them.
> 2. **Several of the paths contain hops that do nothing** — the Poe, King and
>    Obama examples below are self-loops, not detours through a fictional double.
>    They read as absurd because of a graph-representation artifact, not because
>    the reasoning was spurious. See `identity_hops.md`.
> 3. **They are one model's judgements**, audited only by an 18-item sample, with
>    borderline cases resolved toward `y`.
>
> Re-measuring requires labelling raw paths, stratified, with a taxonomy that
> separates direct realization, reliable alternative proof, correlated shortcut,
> and coincidence. The raw paths are available in the `relation_sequence` field
> of `candidate_log` and were simply never joined into the labelling document.

Only **29%** of questions have a unique path to their answer; the rest admit
several, median 2 and up to 30.

This is not an artifact of the retriever. Enumerating **every** path the supplied
graph admits within two hops, rather than only SRTK's proposals, gives the same
figure: 29% unique either way, mean 3.6 paths against SRTK's 3.9. It remains
conditional on the two-hop budget and the pre-extracted neighborhood.

**Corrected.** On 300 questions with identity hops collapsed, **19%** of
questions have a unique path (13% before collapsing). The retriever is still not
the cause — but the graph representation partly is. Uniqueness is rarer than the
earlier 29% suggested, on a larger sample and a stricter enumeration.

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

**Correction.** Most of the examples above are not what they appear to be. In
these graphs `Martin Luther King, Jr. --representations_in_fiction--> Martin
Luther King, Jr.` and `Barack Obama --representations_in_fiction--> Barack Obama`
are literal self-loops, because the real and fictional entities share a display
name and collapse into one node. `location.hud_county_place.place` likewise
returns the entity it started from. So these paths do not traverse a fictional
double or a county detour at all — they are the annotated path with an empty hop
attached. The "systematic Freebase shortcut" reading of them was wrong; see
`identity_hops.md`. Genuinely distinct alternatives remain (77.6% of paths after
collapsing), but they are not these.

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

An earlier version of this section claimed a large share of those 821 are
spurious, citing the 68% figure. That figure is withdrawn, so **the share is
unknown**. Two things are now measured and bound it:

- **37% of the denotation-only positives contain a hop that does nothing**
  (`identity_hops.md`). The pool-wide rate is 4.2%, but positives are the
  population that matters, and answer-set equivalence selects for the artifact:
  a path plus a no-op returns the same answers, so it is labelled positive by
  construction. Annotated supervision is 0.2%. So a large part of the "40% more
  training signal" is duplicates of paths already present.
- Non-uniqueness is real: after collapsing, 77.6% of answer-reaching paths are
  not the annotated one. How many of those express a different question is the
  open quantity.

Fine-tuning on that corpus improved answer exact match from 0.65 to 0.74. The
gain still cannot be attributed to better supervision until intent-equivalent and
denotation-only positives are separated.

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

Measured on 100 questions from one dataset, with the identity-hop correction
applied on 300. To support a claim about the field it needs:

- **id-preserving graphs.** Every number here rests on name-collapsed execution;
  final experiments should not. See `identity_hops.md`.
- **re-labelling with raw paths visible.** The 151 labels were assigned from
  generated questions alone, which confounds path spuriousness with generator
  error and with identity artifacts.
- the full WebQSP test set and CWQ, not 100 questions;
- confirmation that a published system trained this way actually selects spurious
  paths at a similar rate — RoG does not release predictions, so this requires
  running it;
- the intent versus denotation retraining comparison above.
