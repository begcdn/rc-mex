# Explicit Query Representation Pilot

## Purpose

This pilot tests one mechanistic hypothesis: inverse naturalization may fail because an ordered
hop list does not make the underlying fact directions and variable roles explicit. The new
conversion is deterministic and uses no LLM. It produces named variables, raw
subject-predicate-object triples, an answer variable, schema-origin annotations, metadata flags,
and a textual logical form for a later naturalization call. It does not produce a question.

The raw triples are authoritative. Optional semantic macros never replace them.

## Representation

- `v0` is bound to `[ENTITY]`; each traversal hop introduces `v{i+1}`.
- A forward hop emits `(v_i, relation, v_{i+1})`.
- A backward hop emits `(v_{i+1}, relation, v_i)`.
- Every triple retains its raw relation ID, KG, traversal direction, raw subject/object types, and
  traversal source/target types. The final variable and answer type are explicit.
- KQA Pro predicates remain dataset labels. Freebase IDs are syntactically split into
  domain/type/property components, with `semantic_interpretation` set to
  `not_inferred_from_identifier_tokens`.
- Relations matching `METADATA_RELATION_PARTS` are preserved and flagged.
- A possible Freebase CVT macro is recorded when consecutive backward/forward traversal hops
  yield two raw facts with the same middle subject and the same syntactically parsed Freebase
  owner. Every such macro has `needs_semantic_description=true` and a reason; no gloss is
  invented.

## Pilot Corpus

The source file was available at
`runs/inverse_verifier/faithful_data_openai_pilot_40/.openai_batch/selected_rows.jsonl`.
The analysis converted all 40 positive paths and, separately, all positive and selected negative
paths in memory. It did not write generated run outputs.

| Measure | Positive paths | All selected candidate paths |
|---|---:|---:|
| Paths | 40 | 157 |
| KQA Pro / WebQSP paths | 20 / 20 | 77 / 80 |
| Raw triples | 78 | 319 |
| Hop counts | 14 one, 14 two, 12 three | 53 one, 58 two, 34 three, 12 four |
| Paths with flagged metadata | 1 | 4 |
| Paths with possible CVT macros | 7 | 22 |
| Possible CVT macros | 7 | 23 |
| Incompletely parsed Freebase IDs | 0 | 0 |

The metadata paths include the internal `base.kwebbase` relation family. Across all selected
candidates, five individual triples are flagged in four paths. Flagging preserves the executable
fact while preventing an internal storage relation from being treated as an ordinary semantic
question relation.

## Examples

The KQA Pro birthplace/residence/occupation path becomes:

```text
(v0, 'place of birth', v1)
(v2, 'residence', v1)
(v2, 'occupation', v3)
return v3
```

This exposes a branch at the birthplace/residence value: the person in `v2`, rather than the
anchor in `v0`, supplies the occupation answer.

A one-hop backward Freebase path becomes:

```text
(v1, 'government.government_agency.jurisdiction', v0)
return v1
```

For the `people.place_lived` candidate, both raw facts remain present:

```text
(v1, 'people.place_lived.person', v0)
(v1, 'people.place_lived.location', v2)
```

The separate macro records only that `v1` may be a reified record shared by the two predicates.
It does not claim that tokenizing `place_lived`, `person`, or `location` supplies a reliable
natural-language meaning.

## Ambiguity And Decision

All seven positive CVT candidates require semantic descriptions. Some are strong structural
reification candidates (`people.place_lived`, `education.education`, `tv.tv_guest_role`, and
`award.award_nomination`). Others pass the same syntactic test around ordinary-looking schema
owners such as `people.person`; generic `entity` types in the pilot do not let deterministic code
separate those cases safely. The count is therefore a candidate upper bound, not a CVT precision
claim.

The pilot supports the narrow representation claim: every path is recoverable as explicit,
fact-directed triples, and the backward branch is no longer hidden. It does not yet show that an
LLM can naturalize the representation faithfully. A matched naturalization comparison should
measure exact hop/fact coverage, direction errors, opaque rates, and downstream verifier ranking;
failure there would reject the idea that query explicitness alone fixes the information problem.

## Initial Matched Naturalization Check

Qwen3 8B was given four compact explicit queries covering birthplace/residence/occupation,
influence/nationality, production-company/distributor/director, and a Freebase education CVT.
It produced a faithful question for all four. In particular, the previously collapsed branch was
rendered as:

> What is the occupation of the person who resides in the city where [ENTITY] was born?

The production/distribution query correctly preserved the shared company and distinct films:

> Who is the director of the film that was distributed by the same company that produced
> [ENTITY]?

A separate attempt that supplied the entire verbose diagnostic object for all four queries caused
the same model to ignore the task and emit unrelated text. This distinguishes two issues: explicit
triples resolve structural ambiguity, while the naturalizer input must be the compact query rather
than the redundant diagnostic representation. Four examples are evidence for proceeding to a
matched 40-path pilot, not evidence of general reliability.
