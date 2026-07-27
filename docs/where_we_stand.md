# Where We Stand

A consolidation. What is established, what is assumed, what was wrong, and what
decides the next step.

## The hypothesis

> A candidate path is correct if the question generated from it means the same as
> the question the user asked.

For that to work, four things must hold. Three have been measured. The fourth
never has, and it is the one that decides whether the method can work at all.

| # | Requirement | Status |
|---|---|---|
| 1 | The generator faithfully turns a path into its question | **88%**, hand-judged on 90 real cases |
| 2 | The correct path is among the candidates | **90%** at K=100 |
| 3 | The comparator can tell same-meaning from different-meaning | **~53%**, and only as a proxy |
| 4 | **Meaning-equivalence actually separates right paths from wrong ones** | **never tested** |

## Requirement 4 is the real question

The method assumes that a wrong path produces a question that *means something
different*. Where that fails, no comparator can help, because the information
needed is not in the question.

"Where is JaMarcus Russell from?" — nationality and place of birth are both honest
readings of that English sentence. They give different answers. WebQSP picked one.
No amount of meaning-comparison recovers which one was intended, because the
question does not say.

The hand audit already hints at the size of this: **15 of 90 annotated paths do not
answer their own question**, and several disagreements were of exactly this kind.

This makes the labelling task in `comparator_labels_to_judge.md` more valuable than
originally framed. It measures two things, not one:

- **Comparator accuracy** — given candidates whose meaning is known, does it rank a
  correct one first?
- **The ceiling of the method** — how often do *several* candidates mean the same
  thing as the original while returning *different* answers? Every such case is
  unwinnable by meaning-comparison alone.

If the ceiling is high, the comparator is the problem and it is fixable. If the
ceiling is low, the architecture needs a different framing, and no amount of
comparator work will help. **We do not currently know which world we are in, and
almost every hour spent so far assumed the first.**

## Established

- **Firewall is clean.** Zero question-id or question-text overlap between WebQSP
  train and test. All 580 generator-corpus anchors come from train graphs.
- **Gold is correctly defined.** Executing every annotated path over 1,155
  questions yields zero cases returning a non-empty answer set with no gold
  overlap — the signature a direction bug would leave.
- **Nothing cheats.** `select_candidate` takes no gold argument.
- **Generator ≈ 88%** faithful; holds under either annotator alone (90% / 86%).
- **Comparator ≈ 53%** on cases where the generator provably worked.
- **The comparator does not beat the proposer's own top pick** — 0.41 against 0.50,
  17 wins to 25 losses, p = 0.28. No demonstrated benefit in either direction.
- **No off-the-shelf comparator beats the trained one:** QQP-large 0.35, STS-B 0.32,
  NLI-base 0.25, NLI-large 0.19, QQP-distil 0.22, against 0.41. Bidirectional
  entailment, which I predicted would be the stronger prior, is the worst.
- **Winner's curse is real.** As the candidate pool grows 1→100, the winning score
  climbs from −2.74 to −0.31 while accuracy falls 0.50→0.43, and the gold path's own
  score does not improve. Scoring candidates independently and taking the maximum
  systematically selects for lucky overestimates.
- **Ensembling two comparators beats both** (0.53→0.58 EM), consistent with the
  variance story, but p = 0.18.

## Assumed, not established

- That the training corpus is the binding constraint. It is small (2,022 rows, 709
  relations, 45% seen once) and 72% of relations met at inference were never seen.
  But a model trained on 400k human pairs (QQP) does *worse*, and unseen-relation
  candidates are barely overrepresented among errors (12% vs 5%). Volume is
  probably not the lever.
- That path-selection accuracy is the right target. 17% of annotated paths do not
  answer their own question, so this metric has a noise floor and a ceiling well
  below 1.0.
- That the method is novel. The search was four queries with noisy results. A
  proper related-work pass has never been done. Query-graph reranking with
  answer-type information is structurally close.

## Mistakes made

1. **Tuned before measuring.** Endpoint filters, type fixes, voting, K sweeps and
   ensembles were all attempts to move a metric that cannot isolate what they
   change. This is the root cause of the project feeling directionless.
2. **Fixed the disjunctive-type bug in the wrong component.** It appeared in 39% of
   live candidates and 1% of the training corpus; both numbers were available and
   never compared. Cost a generator retrain, which then hallucinated types.
3. **Promoted `vote_filtered` to primary on a one-question margin** before any run.
   The full run reversed it.
4. **Claimed constraint coverage came "free."** Selection cannot subset a
   denotation; 61% of constrained questions need an executable filter.
5. **Never checked where the field stands** until asked. WebQSP SOTA is ~88% Hits@1;
   this pipeline is at ~63% in an easier setting.
6. **Ignored multiple comparisons.** ~20 comparisons so far; the endpoint filter's
   p = 0.031 does not survive correction.

## What decides the next step

Label the 500 candidates. Two numbers come out:

**A. Comparator accuracy** — the benchmark everything else is measured against.

**B. Ceiling** — the fraction of questions where several equally-valid meanings map
to different answers.

- **If B is small (<15%)**: the method can work. Improve the comparator — real-data
  training, larger base model, group-wise scoring instead of independent scoring.
  Then demonstrate value by reranking an existing system's candidates (RoG), which
  is a far easier claim than beating SOTA from scratch.
- **If B is large (>30%)**: meaning-comparison alone cannot select paths, and the
  architecture needs a different role. The natural one is as a *filter* rather than
  a *ranker* — reject paths whose question clearly differs, and let another signal
  break the remaining ties.

Either result is publishable. The current state — three requirements measured, the
fourth assumed — is not.
