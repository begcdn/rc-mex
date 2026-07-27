# Where We Stand

Supersedes the earlier version, whose headline conclusions — that the comparator
scores ~53% and that the verifier does not beat its own proposer — were both
overturned.

## Result

On 100 WebQSP test questions, same candidates throughout:

| selector | gold path | answer EM | F1 |
|---|---:|---:|---:|
| no verifier (proposer's own top pick) | 0.53 | 0.53 | 0.585 |
| original DeBERTa comparator | 0.41 | 0.52 | 0.565 |
| BGE-reranker-v2-m3, off the shelf | 0.51 | 0.65 | 0.709 |
| **BGE fine-tuned, `question_generated`** | 0.66 | 0.71 | 0.772 |
| **BGE fine-tuned, `question_path`** | 0.71 | 0.72 | 0.794 |
| **BGE fine-tuned, `question_generated_path`** | 0.70 | **0.74** | 0.801 |

**0.53 → 0.74 exact match.** The verifier decisively beats the no-verifier
baseline for the first time. Two changes produced it: a reranker-class base model
instead of a paraphrase-class one, and fine-tuning on real pipeline candidates
labelled by answer-set equivalence.

## The hypothesis, restated against evidence

> A candidate path is correct if the question generated from it means the same as
> the question the user asked.

| # | Requirement | Status |
|---|---|---|
| 1 | The generator faithfully turns a path into its question | 88%, hand-judged |
| 2 | The correct path is among the candidates | 90% at K=100 |
| 3 | The comparator can rank a correct candidate first | **0.71–0.74 EM** |
| 4 | Meaning-equivalence separates right paths from wrong ones | **supported** — a verifier reading only generated questions reaches 0.71 against 0.53 for none |
| 5 | **The generated question beats the raw path** | **not supported** |

Requirement 5 was never part of the original hypothesis but is what decides the
contribution. Paired over the same 100 questions:

| comparison | metric | W–L | p |
|---|---|---:|---:|
| generated vs path | answer EM | 7–8 | 1.000 |
| generated vs path | gold path | 6–11 | 0.332 |
| both vs generated | answer EM | 7–4 | 0.549 |

Nothing separates them. Generating a question is neither better nor worse than
showing the reranker the serialized path.

Two facts qualify that:

- The arms **pick the same candidate on only 67 of 100 questions**, and at least
  one of them is right on **79**. There is 5–7 points of complementary signal that
  neither arm nor their naive concatenation captures.
- The generated-question arm runs **6× faster** (18s against 115s), because a
  question is short and a serialized path is not.

## Base model: family and training, not size

All off the shelf, same 100 questions:

| model | size | answer EM |
|---|---:|---:|
| bge-reranker-base (v1) | 278M | 0.44 |
| bge-reranker-large (v1) | 560M | 0.43 |
| bge-reranker-v2-m3 | 568M | **0.65** |
| Qwen3-Reranker-0.6B + equivalence instruction | 0.6B | 0.57 |
| QQP roberta-large | 355M | 0.51 |
| NLI deberta-v3-large (bidirectional) | 435M | 0.36 |

Doubling size within v1 changes nothing (0.44 → 0.43); at the same size, v2 gains
22 points. Within this family size is not the lever. That does not license a claim
about modern families — Qwen3-Reranker-4B is the outstanding test.

Instruction-following did not help at 0.6B. Bidirectional entailment, predicted to
be the strongest general prior, was the worst.

**Fine-tuning on 506 examples was worth more than any base-model swap** (0.65 →
0.74), which also settles the earlier question: that corpus size is sufficient to
improve an already-strong ranker.

## Three framings

1. **"A reranker over executed KG paths, fine-tuned on answer-equivalence labels,
   lifts a strong retriever from 0.53 to 0.74."** Fully supported today. The
   answer-equivalence labelling is the novel piece: it found 40% more valid
   training candidates than annotated-path matching, because Freebase reaches one
   answer many ways and 17% of annotated paths do not answer their own question.
2. **"Question generation is an equally good, cheaper, schema-independent way to
   feed that reranker."** Supported on cost and parity, not on accuracy. The
   schema-independence claim is untested and is the one experiment that would make
   this framing real: a generated question is KG-agnostic, a serialized Freebase
   path is not. If the generated-question comparator transfers to another graph and
   the path one does not, the generator earns its place.
3. **"Reconstructing the question is what verifies the path."** Closed. Requirement
   5 fails.

## Still true and still limiting

- **Controlled setting.** Gold topic entities supplied, 71% of questions, two hops.
  Not comparable to the ~88% Hits@1 published on full WebQSP.
- **n = 100**, ±0.10 intervals. Only paired tests carry weight, and ~25 comparisons
  have now been run without correction.
- **17% of annotated paths do not answer their own question**, a noise floor under
  every path-match number.
- **The comparator has never been measured in isolation.** The 500-item labelling
  task exists for this and is unstarted; it would also give the ceiling number that
  decides ranker-versus-filter.

## Next

1. **Decide the framing.** Framing 1 is defensible now.
2. **Cross-KG transfer**, which is the only experiment that rescues framing 2.
3. **Rerank another system's candidates** (RoG) — a plug-in improvement is a far
   easier claim than beating SOTA from scratch.
4. The 500 meaning labels, which remain the only clean measurement of the
   comparator alone.
