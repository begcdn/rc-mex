# Research Protocol

## 1. State the Hypothesis Before the Architecture

Every method experiment starts with:

- the general failure being addressed;
- the representation or computation believed to cause it;
- the closest prior methods and why they remain insufficient;
- one testable mechanism;
- observations that would support, weaken, or reject the mechanism.

Do not infer a full architecture or novelty claim before this reasoning is
explicit.

## 2. Declare the Experimental Setting

Every run must record:

- dataset and knowledge graph;
- entity-linking assumptions;
- full-KG or supplied-subgraph access;
- supported query structures and operators;
- search, model-call, token, and time budgets;
- whether text or parametric model knowledge is allowed;
- any gold information used for selection or diagnostics.

Controlled oracle experiments are allowed only when their conclusion is
limited to the isolated component.

## 3. Maintain an Evaluation Firewall

- Use inspected questions only for development and diagnosis.
- Reserve a hidden local holdout for aggregate go/no-go decisions.
- Do not tune after inspecting holdout examples.
- Use official test sets only after the method and decision criteria are
  frozen.
- Record question identifiers for every run.

## 4. Use Matched Comparisons

Compare the new mechanism with the closest strong baseline while holding
available evidence, graph substrate, candidate budget, model budget, and
evaluation constant wherever possible.

Change one causal mechanism at a time. If several components change together,
the experiment cannot identify why performance moved.

## 5. Measure the Claimed Mechanism

Report final-answer quality using official dataset metrics and entity-aware
exact match, precision, recall, and F1 where applicable.

Also measure the intermediate quantity the hypothesis predicts. Depending on
the method, this may include candidate coverage, survival after pruning,
logical-form validity, proof correctness, calibration, or answer completeness.

Track compute, latency, model calls, and token usage. Never silently use a
fallback model, missing checkpoint, degraded encoder, or hidden gold signal.

## 6. Make a Research Decision

After each experiment, choose one:

- **Reject:** the predicted intermediate effect did not occur.
- **Revise:** the hypothesis remains plausible, but the experiment exposed a
  specific mismatch between theory and implementation.
- **Retain:** the mechanism changed its predicted quantity and improved or
  credibly raised the ceiling for final QA under matched conditions.

Do not retain compensating modules merely because they repair visible benchmark
errors. Git history is the archive for rejected ideas.
