# Identity Hops in the Name-Based WebQSP Graphs

The graphs this project executes against store **entity display names, not
Freebase machine ids**. Distinct entities that share a name collapse into one
node, which turns the relation between them into a self-loop:

```
Martin Luther King, Jr. --fictional_universe.person_in_fiction.representations_in_fiction--> Martin Luther King, Jr.
Barack Obama            --fictional_universe.person_in_fiction.representations_in_fiction--> Barack Obama
```

Both are literal triples in `data/webqsp/test.jsonl`. Traversing them is a no-op,
so any path that begins or ends with one is execution-equivalent to a shorter
path. The candidate space fills with semantically empty duplicates.

## How common

Measured over 300 test graphs. No model involved.

| | |
|---|---:|
| graphs containing at least one self-loop triple | 99.7% |
| self-loop triples per graph | median 41, mean 84, max 1,167 |
| **questions whose topic entity itself carries a self-loop** | **49.3%** |
| distinct relations appearing as self-loops | 315 |

The most frequent offenders are *work ↔ version* relations, where the two
entities legitimately share a display name:

| count | relation |
|---:|---|
| 2,537 | `music.recording.tracks` |
| 2,414 | `location.hud_county_place.place` |
| 2,274 | `book.book.editions` |
| 1,938 | `music.composition.recordings` |
| 1,383 | `music.recording.song` |
| 732 | `education.educational_institution.campuses` |

Not all 315 are errors. `base.biblioness.bibs_topic.is_really` and
`freebase.equivalent_topic.equivalent_type` are genuine near-identity relations
in Freebase. The list has not been classified into artifact versus real.

## Effect on path statistics

300 questions, full ≤2-hop enumeration of every path returning exactly the gold
answer set. Two collapse rules:

- **strict** — drop only hops that are genuinely reflexive on the node set they
  act on. An unambiguous **no-op in this graph**, which is not the same as an
  artifact: the underlying edge may connect two distinct Freebase entities that
  merely share a display name, and would be meaningful under ids.
- **loose** — drop any hop whose removal leaves the result unchanged. Also
  catches paths that merely coincide on this entity, so it overstates.

| | raw | strict | loose |
|---|---:|---:|---:|
| distinct answer-reaching paths | 2,114 | 1,338 | 773 |
| mean per question | 7.05 | 4.46 | 2.58 |
| questions with a unique path | 13.0% | 19.0% | 33.7% |
| paths that are not the annotated one | 85.8% | 77.6% | 61.2% |

**At least 36.7% of enumerated answer-reaching paths contain a hop that does
nothing here.** Up to 63.4% are execution-redundant in some form.

**The ambiguity survives the correction.** After strict collapsing, 77.6% of
answer-reaching paths are still not the annotated one and only 19% of questions
have a unique path. Path supervision really is under-determined; the effect was
inflated by roughly 1.6×, not manufactured.

That 77.6% is a **non-uniqueness** rate, not an error rate. It says the answer
does not identify the reasoning. It says nothing about how many of those paths
are semantically wrong — some are genuine schema aliases such as
`people.person.parents::backward` for a children relation. Separating the two
requires semantic labels on raw paths, which do not currently exist.

## Provenance

The file schema (`id`, `question`, `answer`, `q_entity`, `a_entity`, `graph`) and
the row counts — **1,628 test, 2,826 train**, verified locally — match the
RoG-released WebQSP subgraph dataset, which is the de-facto standard input for
several published KGQA systems.

**This does not establish that those systems are affected.** They may preserve
Freebase ids internally regardless of what the released file contains. Any claim
beyond "the RoG-format release has this property" requires checking each
system's own loader.

## The fix

Deleting self-loop relations is wrong: some connect genuinely distinct entities
that happen to share a name, and removing them loses real edges.

The correct repair is to **separate identity from display** — execute over
Freebase machine ids, and use names only when a path is shown to a model or a
human. Until an id-preserving graph source is in place, execution-level identity
collapsing is a usable diagnostic but not a solution.

## Effect on the pipeline: none measurable

The statistics above come from **exhaustive** ≤2-hop enumeration. The pipeline
does not enumerate exhaustively — SRTK proposes a ranked shortlist. Those are
very different populations, and the artifact rate differs by an order of
magnitude.

Collapsing reflexive hops in the real candidate pools, merging candidates that
become identical, and re-selecting by the same score, with no retraining
(`identity_audit.py`):

**The denominator matters more than the headline rate.** Across all candidates
the artifact is rare. Across the candidates that get *labelled positive during
training*, it is not:

| population | test pool | train pool |
|---|---:|---:|
| all candidates | 5.3% | 4.2% |
| annotated positives | 0.0% | 0.2% |
| denotation positives | 33.6% | 26.1% |
| **denotation-only positives** | **45.7%** | **37.0%** |

*(n = 9,752 test / 59,049 train candidates; 245 / 1,252 denotation-only positives)*

An earlier version of this document reported only the 4–5% figure and concluded
the corpus was "not substantially padded with empty-hop duplicates." That was the
wrong denominator and the conclusion was wrong. **Answer-set equivalence — the
labelling rule that bought us "40% more training signal" — admits a positive
containing a no-op hop between a quarter and a half of the time.** The extra
signal is substantially duplicates of paths already in the corpus.

Annotated supervision is almost perfectly clean by comparison, at 0.2%.

**Two columns originally reported here were vacuous and have been removed.**

*Answer metrics cannot move.* An empty hop returns the same answers by
definition, so deleting it cannot change any answer. This needed no experiment.

*Selection cannot change under this merge.* Merging keeps the highest-scoring
member of each group, so the global argmax is always its own group's maximum and
always survives. "Zero selections changed" was true by construction, not by
measurement.

## What the artifact actually distorts

The metric it can move is **path match**, because a candidate that is the
annotated path with an empty hop attached is scored as a different sequence and
counted wrong. Comparing collapsed sequences on both sides:

| | exact sequence | collapsed |
|---|---:|---:|
| gold-path match, test pool | 0.414 | **0.424** |

One question of 99 flips. *"where did eleanor roosevelt die?"* selected

```
fictional_universe.fictional_character.based_on::backward -> people.deceased_person.place_of_death::forward
```

which collapses to exactly the annotated path. It was being counted as a wrong
path when it is the right one with a no-op in front.

So the artifact **understates path accuracy**, by about one point on the test
pool (0.414 → 0.424, 1 flip) and 0.8 on the train pool (0.447 → 0.455, 5 flips).

**This does not license "the artifact does not damage answers."** Everything
above re-scores an existing candidate pool with already-computed scores. A path
with the no-op removed would generate a *different question* and receive a
*different comparator score*, and a model trained on cleaned positives would be a
different model. Neither was tested. The only supported statement is that the
artifact does not change the bookkeeping of a run already completed.

So the artifact inflates the *theoretical* path space heavily and the *actual*
candidate pool barely at all. That is itself worth knowing: a measurement made by
enumerating the graph and a measurement made from retriever output will disagree
about this dataset by roughly 8×.

## What this invalidates, and what it does not

Invalidated — all of these were computed by enumeration:

- The non-identifiability and unique-path figures in `spurious_supervision.md`,
  corrected there.
- An unknown share of the 151 semantic labels in `spurious_labels_to_judge.md`,
  assigned by reading generated questions without the raw path beside them. Some
  read as bizarre precisely because their first hop is a no-op.

**Under question, not cleared:**

- The comparator training corpus. 37% of its denotation-only positives contain a
  no-op, and of those, **97.2% collapse to a path already in the same candidate
  pool** and **78.8% collapse directly onto the annotated path**. They are not
  merely execution-equivalent in the abstract; they are duplicates of training
  examples the corpus already contains.
- The 0.53 → 0.74 answer-EM result. Nothing here shows it is wrong, and nothing
  here shows it is unaffected either. Re-scoring a finished run cannot answer
  that; only retraining on cleaned positives can.

## What the models already do

Comparing each no-op path against its clean counterpart, 112 pairs on the
100-question test set, every trained comparator ranks the clean version higher:

| supervision | clean path preferred | mean margin |
|---|---:|---:|
| annotated | 112/112 | **+17.30** |
| hybrid | 112/112 | +12.50 |
| denotation | 112/112 | +11.72 |

No decision flips. But denotation supervision **narrows the rejection margin by
about a third** against annotated supervision. That is a degradation in
separation, not yet in accuracy, and it is the kind of margin loss that shows up
first under harder competition or a different schema.

Note this does not attribute the rejection to inverse reconstruction: these
comparators see the generated question *and* the serialized path, so either could
be doing the work. Separating them needs a path-only versus generated-question-only
comparison.

## Open

**An id-preserving graph source is the right repair and should not be deferred.**
The earlier claim here that it was non-urgent rested on the denominator error
above and is withdrawn.

Reproduce with `inverse_verifier/identity_audit.py`. The enumeration study that
produced the path-statistics table is at `scratchpad/identity_impact.py` and
should be moved into the repository before those figures are published.
