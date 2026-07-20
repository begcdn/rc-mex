# Research Operating Instructions

## Objective

This repository is a research project whose goal is to discover a novel, reliable method for multi-hop question answering over knowledge graphs and develop it to ICLR standards.

The objective is not to preserve the current architecture, maximize one benchmark by patching observed errors, or continuously add components. Code is an instrument for testing research ideas.

The unit of progress is **research uncertainty reduced**, not code written.

## Research Stance

Before implementing a meaningful method change, state:

1. The general failure being addressed.
2. Why the current representation or computation causes that failure.
3. What the closest prior methods do and why that is insufficient.
4. The proposed hypothesis.
5. An experiment that can falsify the hypothesis.
6. The observations that would support, weaken, or reject it.

Do not present a new architecture or novelty claim before this reasoning is explicit.

Treat benchmark examples as measurements of a general mechanism. Never add a rule merely because it fixes observed questions. A change must be expressible without mentioning a benchmark, relation name, question template, or individual failure case.

Large architectural changes are appropriate when evidence identifies the representation or computation as the problem. Small changes are appropriate only when the underlying hypothesis remains valid and the implementation is demonstrably preventing a fair test of it.

## Literature Practice

Inspect both papers and official code when possible. Analyze methods by:

- what object represents uncertain reasoning;
- what information that representation preserves and destroys;
- where commitment and pruning occur;
- what is learned, searched, executed, or hard-constrained;
- what supervision is required;
- what assumptions are specific to a graph, schema, or benchmark;
- which failure the architecture can and cannot recover from.

Do not copy a paper's module list. Extract the computational principle and compare it with the present hypothesis.

Prefer recent work from ICLR, NeurIPS, ICML, ACL, EMNLP, and related primary sources, while including older work when it established the relevant architecture.

## Experimental Loop

Use this loop:

1. Form one mechanistic hypothesis.
2. Establish a matched baseline and an evaluation firewall.
3. Implement the smallest experiment that distinguishes the hypothesis from alternatives.
4. Measure end-to-end quality and the intermediate quantity the hypothesis claims to improve.
5. Inspect wins, losses, overlap, ceilings, and information lost at each stage.
6. Decide: reject, revise, or retain the hypothesis.
7. Only then design the next experiment.

Do not tune a method repeatedly when the experiment shows that its core representation is wrong. Do not declare success from an intermediate metric when final QA does not improve.

Use gold programs, paths, entities, or answers only for dataset selection and offline diagnostics unless the evaluated setting explicitly supplies them. Clearly label controlled and oracle experiments.

## Architecture Discipline

- Every component must have one necessary role in the hypothesis.
- Do not keep an old flawed method as a production fallback. It may remain as an explicit baseline or ablation.
- Do not combine independent scores with arbitrary fixed weights and call the result a new architecture without a probabilistic, optimization, or empirical justification.
- Do not use a second proposer as a fallback when it encodes a different model of the problem. Define one coherent legal action space.
- Preserve uncertainty only when the retained state contains information needed by later reasoning. More beams or candidates alone are not a contribution.
- Enforce graph validity structurally where possible instead of generating invalid objects and repairing them afterward.
- Separate retrieval failures, representation failures, search failures, and final-ranking failures in evaluation.
- Prefer methods whose logic can transfer across schemas and graph datasets. Dataset adapters may translate storage and query languages, but must not contain benchmark reasoning rules.
- Discarded methods in Git history are not architectural constraints and should not be revived without new evidence.

## ICLR Standard

A plausible ICLR contribution should eventually provide:

- a clear and nontrivial problem diagnosis;
- a computational idea, not merely an assembled pipeline;
- a principled explanation of why it improves over prior architectures;
- evidence across relevant datasets and strong matched baselines;
- mechanism-focused ablations and failure analysis;
- clarity about assumptions, limits, and generalization;
- reproducible implementation.

Universality, engineering scale, or the number of supported datasets is not itself the novelty claim.

## Coding Discipline

Read the existing implementation before changing it. Keep research code comprehensible enough to audit the hypothesis.

- Prefer one canonical runner per active experiment family.
- Do not create a new runner, mode, report, or output file unless it is necessary to distinguish the current hypothesis.
- Reuse dataset loaders and evaluators, but do not let legacy abstractions dictate a new architecture.
- Keep outputs minimal: predictions, aggregate metrics, and one diagnostic artifact unless more is explicitly required.
- Add tests for correctness and leakage boundaries, not tests that freeze accidental experimental behavior.
- Never silently use degraded models, missing checkpoints, gold information, or alternate fallbacks.
- Do not push or commit unless the user asks.

## Communication

Explain the research logic in plain language before implementation details. Define technical terms when they first matter.

Clearly separate:

- what the code currently does;
- what experiments observed;
- what is inferred from those observations;
- what remains a hypothesis.

Be candid when evidence rejects an idea. Do not protect previous work, manufacture novelty, or describe an engineering patch as a research contribution.
