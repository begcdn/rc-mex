# Error Analysis: What the 0.74 Model Gets Wrong

The best configuration — BGE-reranker-v2-m3 fine-tuned on 506 real candidate sets,
`question_generated_path` — answers 74 of 100 WebQSP test questions exactly. This
is what the other 26 are. It is the analysis that should have preceded any claim
that the engineering was finished.

## Decomposition

| category | count | attributable to |
|---|---:|---|
| picked a different path, partial answer overlap | 12 | mixed |
| picked a different path, no overlap | 10 | comparator |
| **picked the annotated path and still scored 0** | **4** | **exact match** |

The last four are not errors. The model selected the annotated path and exact
match scored it zero because the returned set differed slightly:

| question | F1 | scored |
|---|---:|---:|
| what the zip code for seattle washington? | 0.88 | 0 |
| what does jamaican people speak? | 0.67 | 0 |
| what language does cuba speak? | 0.50 | 0 |

Across all 26 failures roughly 10 return partially-correct sets. **Exact match
alone overstates failure by about a third**; F1 already reflects this and sits
6 points higher.

## The genuine failures

Twenty-two questions where a different path won. Sorted by how close the answer
was, the pattern is:

**Dropped qualifying clause.** The question restricts, the chosen path does not.

- *"where did andy murray **started playing tennis**"* → "where was he born?"
- *"where did rudolf virchow **conduct his research**"* → "what does he have sentences about?"

**Near-relation confusion.** Two relations a person would distinguish, the model
does not.

- *"where are samsung **based**"* → "where was Samsung **founded**?"
- *"who wrote jana gana mana"* → "who wrote the **lyrics**?"
- *"what do portuguese people speak"* → "what is the **official** language?"

**Type versus instance.** The question asks for a category, the path returns
members.

- *"what **type of** books did agatha christie write"* → book titles
- *"what **works of art** did da vinci produce"* → art *forms*

**Role confusion.**

- *"**who played** on the jeffersons"* → character names, not actors

**Answer-type violation.** Only one case, and the only mechanically detectable
failure in the whole set.

- *"**when** did annie open"* → returned a work, not a date

## A hypothesis that was tested and rejected

Four failures carried a visibly wrong entity gloss — *"Stonewall Jackson, a
composer"*, *"Leonardo da Vinci, who is a film character"*, *"the book Thomas
Jefferson"*. The obvious inference was that the generator's entity appositions
mislead the comparator, and that removing them is a free win. 47% of generated
questions carry one.

The base rate contradicted it before the experiment ran: the winning question
carries a gloss on **51% of correct answers** but only **27% of wrong** ones.
Glosses are twice as common in successes.

Stripping them, same model and candidates:

| | gold path | answer EM | F1 |
|---|---:|---:|---:|
| original | 0.510 | 0.650 | 0.709 |
| appositions removed | 0.550 | 0.650 | 0.703 |
| paired | 5–1, p=0.22 | 2–2, p=1.00 | — |

Neutral. The hypothesis came from four memorable examples against a base rate
pointing the other way, which is the same failure mode as the earlier
disjunctive-type and model-size hypotheses in this project.

## Revised attribution

| | count |
|---|---:|
| exact match too strict | ~10 |
| generator emits a wrong entity type | ~3 |
| **genuine semantic discrimination failure** | **~13** |

Half the residual errors survive a strong reranker fine-tuned on in-distribution
data. That is the more defensible version of the project's motivating claim: the
semantic failure is real, it is not an artefact of a weak model, and it is not
fixed by better plumbing.

## What follows

There is no cheap win left in the generator, and none in the base model — size and
family were both swept without finding one. The remaining failures are the
research problem itself.

The open question is whether this taxonomy is a property of this pipeline or of
KGQA methods generally. RoG and GNN-RAG publish their predictions. Applying the
same decomposition to their errors is what would turn this from an observation
about one system into a diagnostic about the field, and it requires no new
training.
