# Selection Findings

Source run: `runs/inverse_verifier/full_pipeline_selection_ablation_v1`, 100 WebQSP
questions, 98.5 verified candidates per question, frozen generator
(`faithful_inverse_qwen25_3b_lora_v1`) and comparator
(`deberta_comparator_question_generated_v1`).

Every number below is a post-hoc replay of that run's `candidate_log`. The
comparator scores candidates independently at inference, so truncating or
re-ranking the logged pool is exactly equivalent to having verified fewer or ranked
differently. Selection replay reproduces the recorded choice on 100/100 questions.

`sel_gold` is stored per candidate and is exact. Recomputed EM is understated by up
to 0.02 because `candidate_log` caps answer lists at 50 entries (2.4% of
candidates); the affected selections are named where it matters.

## 1. Answer-set voting: rejected

| policy | sel_gold | EM | F1 |
|---|---:|---:|---:|
| comparator (incumbent) | 0.410 | 0.510 | 0.564 |
| comparator + endpoint filter | 0.450 | 0.560 | 0.615 |
| vote | 0.370 | 0.490 | 0.537 |
| vote + endpoint filter | 0.370 | 0.500 | 0.549 |

A replay over the *previous* run's stored top 10 had ranked voting first (0.570).
The full run reversed it. Mechanism, measured: where voting loses, the winning
answer set is backed by a median 12 routes (max 23) against 1 for the argmax
winner, which was individually ahead by a median 1.44 logits. The affected
questions are occupation and role questions ("what did st augustine do?") where
Freebase reaches one broad answer set many near-equivalent ways.

Summing exponentiated scores treats duplicated aliases and correlated routes as
independent evidence. Removed from the code; recorded here.

**A truncated candidate list systematically understates any failure mode that needs
redundancy to appear.** The top-10 replay could not have found this.

## 2. Endpoint filter: retained

Dropping candidates whose endpoints are all unlabeled machine ids: 5 wins, 0 losses
against the unfiltered incumbent (sign test p = 0.031). It never removed a gold
path (0/100), the empty-pool fallback never fired, and no gold answer set is
entirely unlabeled.

This is a structural validity constraint, not a scoring preference: a candidate
whose denotation cannot be named is not a legal answer. Detecting unresolved labels
by id pattern is a property of this preprocessing, not a fact about Freebase.

## 3. Pool size only mattered because the prior was discarded

| K | recall@K | sel_gold (comparator) | P(gold \| in pool) |
|---:|---:|---:|---:|
| 5 | 0.700 | 0.490 | 0.700 |
| 10 | 0.760 | 0.470 | 0.618 |
| 20 | 0.840 | 0.450 | 0.536 |
| 50 | 0.870 | 0.440 | 0.506 |
| 100 | 0.900 | 0.450 | 0.500 |

Going from K=5 to K=100 buys 20 points of recall and loses 4 points of final
accuracy: conditional precision collapses from 70% to a coin flip. Correct
selections come from median retrieval rank 1; wrong selections from median rank 11
(mean 22.6), and 35% of wrong selections come from beyond rank 20.

But the pool-size sensitivity disappears once the proposer score is restored:

| K | comparator only | comparator + retrieval |
|---:|---:|---:|
| 5 | 0.490 | 0.570 |
| 10 | 0.470 | 0.600 |
| 20 | 0.450 | 0.600 |
| 100 | 0.450 | 0.600 |

Shrinking the pool was a workaround for a discarded prior, not a fix.

## 4. The matched baseline the verifier must beat

Selecting the proposer's own top-ranked candidate, running no verifier at all:

| selector | sel_gold | EM | F1 | has_correct |
|---|---:|---:|---:|---:|
| retrieval rank-1 + filter (no verifier) | 0.530 | 0.530 | 0.585 | 0.620 |
| comparator + filter (current pipeline) | 0.450 | 0.560 | 0.612 | 0.680 |
| **comparator + retrieval + filter** | **0.600** | **0.680** | **0.729** | **0.780** |

**The current pipeline does not beat its own proposer on path selection** (0.450 vs
0.530). It is ahead on answer metrics, but 17 wins against 25 losses on gold-path
selection is a sign test at p = 0.280 — no demonstrated benefit either way.

This baseline had never been run. It should have been the first thing measured.

The comparator is not uninformative: gold-path candidates score −1.04 on average
against −5.80 for non-gold, and the median comparator rank of the gold path when
present is 2. It has signal and fails to convert it into top-1 wins, because
selection discards the proposer score entirely.

Adding the two log-scale scores treats them as independent evidence about the same
candidate. Against the current pipeline: 20 wins, 6 losses, p = 0.0094. Against the
no-verifier baseline: 16 wins, 1 loss, p = 0.0003. Only one combined selection hits
the log cap, so EM is understated by at most 0.01.

λ was swept as a diagnostic (0.25→0.60 over λ ∈ [0.25, 2], broad and smooth) and
λ=1 is the untuned sum, not a fitted value. A fitted λ on this data would be
overfitting.

## 5. Status

`comparator` remains the primary reported policy.
`comparator_retrieval_filtered` leads by a wide, significant margin on one run and
is **not adopted** until a second confirms it — the top-10 replay ranked voting
first and the full run reversed it, which is precisely the failure mode this
discipline exists to catch.

Confirmation requires a run on a disjoint question slice, not a replay of this one.

## 6. What this implies for the research

The hypothesis is that a reconstructed question verifies a path. This run says the
verifier carries real signal but is currently worth less than the proposer's own
ranking when used alone, and clearly positive only in combination. Two readings,
not yet distinguished:

1. The comparator is undertrained for this candidate distribution — it was trained
   on synthetic negative categories and deployed on real proposals labeled
   `unlabeled`. This is the E2 hypothesis and is independently motivated.
2. Question reconstruction is intrinsically a weaker signal than
   question-conditioned relation scoring, and belongs as a complement rather than a
   replacement.

Distinguishing these is the next experiment. Retraining the comparator on executed
candidates tests reading 1 directly: if the retrained comparator alone beats
retrieval rank-1, reading 1 holds. If it still needs the proposer score to win,
reading 2 does, and the contribution claim has to be framed as complementary
evidence rather than verification-as-selection.
