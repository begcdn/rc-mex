# Generator-Only Faithfulness Audit

## Question

Does the trained inverse generator faithfully verbalize a supplied KG path,
independently of path ranking or sentence-embedding similarity?

## Protocol

The audit contains 177 held-out generated questions:

- all 48 executable-direction evaluations;
- all 29 strict unseen two-hop compositions;
- 50 KQA Pro validation examples;
- 50 executable WebQSP examples.

Each generated question was compared only with the path's grounded facts,
relation order and direction, and returned answer role. Candidate rankings and
similarity scores were hidden. The first audit flagged 78 unfaithful or partial
cases. All 78, plus 22 randomly sampled passes, received a blinded second audit.
Binary faithful-versus-not agreement was 89%. Eleven binary disagreements were
resolved by a third blinded adjudication.

This is an LLM semantic audit, not a human annotation study. It is adequate for
an engineering decision, but publication claims require a human-labeled sample.

## Results

| Slice | Judgeable | Faithful | Partial | Unfaithful | Strict fidelity |
|---|---:|---:|---:|---:|---:|
| Executable direction | 48 | 21 | 2 | 25 | 43.8% |
| Unseen composition | 29 | 15 | 2 | 12 | 51.7% |
| KQA Pro validation | 49 | 28 | 4 | 17 | 57.1% |
| Executable WebQSP | 48 | 33 | 1 | 14 | 68.8% |
| **Overall** | **174** | **97** | **9** | **68** | **55.7%** |

Including partial but recoverable questions raises the overall rate only to
60.9%. The generator is therefore not reliable enough to act as a semantic path
verifier.

## Main Failures

The dominant semantic errors were wrong direction (41 cases), wrong answer role
(30), and wrong relation meaning (29). Unsupported facts appeared in 41 cases.
Representative substitutions include:

- `narrator` rendered as a fictional-character relation;
- `sexual orientation` rendered as gender;
- `voice type` rendered as profession;
- writing-system use rendered as being "written by";
- inverse location and membership relations expressed in the wrong direction.

The model can also be faithful. For example, it correctly rendered an inverse
editor relation as "Which film was edited by [ENTITY]?" and a two-hop
author-to-derivative path as "What visual artwork is the derivative work of the
written work authored by [ENTITY]?"

## Implementation Mismatch

The natural training targets were built from grounded relation descriptions,
but the model does not receive those descriptions. `render_path` calls
`relation_words`, which reduces Freebase predicates with three or more segments
to only their final segment. In the retained glossary, 1,285 WebQSP relations
collapse to 960 rendered labels; 472 relations participate in 147 label
collisions. For example, 22 distinct relations render simply as `team`.

Direction is represented only by symbolic `START`/`ANSWER` subject-object tags.
The audit shows that the small generator often ignores or misuses those tags.
Thus the model is asked to reconstruct precise natural semantics from an input
that discards semantic information used to create its targets.

## Conclusion

The executable-direction data itself is useful: it improved relative direction
discrimination from 72.9% to 85.4%. But relative ranking concealed absolute
generation errors. The next model experiment should repair the input contract:
provide grounded subject-to-object relation descriptions and explicit natural
roles for every hop, while retaining schema IDs only as optional identifiers.
Changing the downstream ranker cannot repair an unfaithful generator.
