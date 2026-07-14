# Executable pattern alignment

## Hypothesis

Question-to-schema grounding should score a natural-language request against a
runtime executable query pattern, rather than classify the request into a
fixed benchmark relation vocabulary.  A pattern may contain one relation, a
multi-edge Freebase CVT, a Wikidata statement/qualifier structure, or an
inverse traversal.  Its representation therefore preserves directed query
graph structure while omitting benchmark entity identifiers.

The first experiment uses WikiWebQuestions and WebQSP because they provide a
rare parallel signal: the same real user question has a Freebase semantic
parse and a Wikidata semantic parse.  This yields direct supervision for
cross-schema equivalence without asking an LLM to invent relation mappings.
KQA Pro simple chains are evaluation-only and test transfer to an unseen
schema.

## Relation to prior work

- Semantic relation linking has encoded question text and schema relations,
  including AMR-enhanced relation linking across DBpedia and Wikidata.
- UniKGQA unifies retrieval and reasoning around question-relation matching
  and propagates matching scores over KG edges.
- TIARA and later semantic parsers retrieve schema items before constrained
  logical-form generation.
- WikiWebQuestions ports WebQuestions from Freebase to Wikidata, providing the
  parallel executable forms used here.

The current experiment does not claim that dense relation retrieval is new.
It tests whether **parallel directed executable-pattern alignment** supplies a
better reusable semantic potential for the architecture than frozen lexical
similarity or benchmark-specific classes.

## Leakage boundary

- Training reads only the WikiWebQuestions train split and matching WebQSP
  semantic parses.
- WikiWebQuestions dev/test and WebQSP test are evaluation-only.
- KQA Pro is never used for model updates.
- Gold logical forms provide labels during training/evaluation; no gold answer
  or relation is supplied to runtime scoring.
- CWQ RoG answer-reaching paths are not treated as exact relation labels.

## Decision rule

Continue toward runtime integration only if the trained encoder improves both:

1. held-out parallel-schema retrieval; and
2. zero-shot KQA question-to-pattern retrieval.

If it improves only Freebase/Wikidata parallel retrieval, it is an in-domain
adapter, not a schema-general component.  If it transfers, the next experiment
replaces the frozen MiniLM semantic potential in candidate proposal while
holding the search and evaluation firewall fixed.

The evaluation additionally reports exact held-out-pattern subsets and a real
WebQSP local-frontier diagnostic.  The latter ranks only relation/direction
candidates adjacent to the topic entity in the supplied question subgraph.
Gold answers label the correct direct edge offline; they are never rendered in
the candidate representation or supplied to the encoder.

## Results from the first controlled iteration

| Evaluation | Frozen R@1 | Trained R@1 |
|---|---:|---:|
| Unseen WebQSP Freebase patterns | 0.234 | 0.355 |
| Unseen WikiWebQuestions Wikidata patterns | 0.087 | 0.292 |
| KQA Pro, zero-shot schema transfer | 0.190 | 0.337 |
| GrailQA, zero-shot benchmark transfer | 0.294 | 0.397 |
| CWQ, zero-shot benchmark transfer | 0.152 | 0.311 |
| WebQSP real local relation frontier | 0.525 | 0.734 |

The current search ranker's fixed lexical/semantic mixture improves from
0.479 to 0.666 R@1 when its semantic term uses the trained directed pattern.
This does not by itself improve the CWQ menu ceiling because complex questions
contain several relation roles and the old architecture commits one edge at a
time.

When complete executable candidates are compared within the same depth family,
R@1 improves from 0.467 to 0.590 for one-hop patterns and from 0.475 to 0.692
for two-hop patterns. Replacing edge-first proposal with a 14-seat pattern
family menu changes the frozen CWQ firewall as follows:

| Menu metric | Previous edge-first menu | Executable-pattern menu |
|---|---:|---:|
| Gold answer present | 0.662 | 0.670 |
| Exact answer-set option | 0.440 | 0.464 |
| Oracle answer F1 | 0.528 | 0.548 |
| Average candidates | 13.4 | 14.4 |

These are answer-blind proposal results over 500 frozen questions. A 50-item
local smoke test with Llama 3.2 3B reached 0.400 Hits@1 and 0.370 mean answer
F1. Its executable menu contained a gold answer on 0.700 of the questions and
an exact gold set on 0.500, so the run primarily validates integration and
exposes a substantial downstream selection gap; it is not an end-to-end model
quality claim. A three-question replay check confirmed that the evaluation
firewall reconstructs the stored runtime menu exactly.

## Rejected hypotheses

- An explicit Freebase/Wikidata cosine term adds only small, inconsistent gains;
  the shared question pivot supplies most of the alignment signal.
- Freezing a separate question tower improves GrailQA slightly but reduces
  WebQSP local-frontier R@1 from 0.734 to 0.635 and CWQ R@1 from 0.311 to 0.277.
- Replacing only the edge-local semantic score does not improve the CWQ menu
  recall. The learned representation must score the complete executable
  hypothesis at the same granularity used during training.

## Architectural role

The functional component now has three boundaries:

1. `executable_pattern_alignment.py` normalizes supervised and runtime query
   graphs into the same structural text representation.
2. `run_pattern_alignment.py` trains and evaluates the semantic potential with
   explicit seen/unseen and cross-benchmark controls.
3. `executable_pattern_search.py` enumerates, scores, executes, and retains
   bounded pattern families before denotation-level selection.

This is not a claim that full KGQA is solved. It is evidence that semantic
grounding improves when uncertainty is represented over complete executable
patterns rather than unrelated local edges.
