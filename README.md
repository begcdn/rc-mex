# RC-MEX

Relation-Card Marginalized Execution for KGQA.

This repository contains the current MVP code for RC-MEX and the earlier CIGR-D baseline utilities.

Start with `README_RC_MEX_MVP1.md`.

## Current Framing

RC-MEX is focused on relation-card grounding, not on claiming top-k graph search as novel.

- **MVP1:** Relation cards beat relation-name-only baselines and remain useful under hidden or misleading relation labels.
- **MVP2:** Question slots can retrieve/rank the correct relation card from a local candidate frontier.
- **MVP3:** Top-k relation-card execution improves recall but adds noise. This is a diagnostic showing that MVP2's top-k uncertainty contains useful recoverable graph evidence. It is not the novelty claim.
- **MVP3.5:** Card-vs-name retrieval/execution compares relation cards against raw relation names, anonymized IDs, and misleading labels under the same controlled gold-prefix slot setup.

The main RC-MEX claim is:

> Relation cards provide a more robust semantic interface for KG relations than raw relation names or arbitrary IDs, especially when schema labels are hidden, noisy, or misleading.

See `docs/rc_mex_prior_work_lessons.md` for the prior-work lesson motivating this framing.

# Evaluation firewall and architecture ceilings

Before changing the reasoning architecture, freeze a deterministic development
slice and measure where answers disappear. This audit is offline: it makes no
LLM calls and uses gold answers only for diagnostics.

```bash
python3 -m rc_mex.run_architecture_ceiling \
  --data data/cwq/train.jsonl \
  --output runs/firewall_cwq500 \
  --sample-size 500 \
  --seed 20260711
```

The command writes `eval_questions.jsonl`, which is the only data file that
should be used by the method run:

```bash
python3 -m rc_mex.run_query_selection \
  --data runs/firewall_cwq500/eval_questions.jsonl \
  --output runs/firewall_cwq500_qsel \
  --max-hops 2 \
  --model qwen3:8b
```

Then reconstruct the generated menus and attribute the observed failures:

```bash
python3 -m rc_mex.run_architecture_ceiling \
  --data runs/firewall_cwq500/eval_questions.jsonl \
  --predictions runs/firewall_cwq500_qsel/predictions.jsonl \
  --output runs/firewall_cwq500_audit \
  --max-hops 2
```

To prevent reuse of previously evaluated questions, repeat
`--exclude-predictions path/to/old/predictions.jsonl` on the first command.
The output separates supplied-subgraph coverage, exhaustive one/two-hop
reachability, generated-menu recall, oracle menu F1, conditional selector
accuracy, final answer metrics, and earliest failure stage. Full-Freebase
coverage and formal operator coverage are explicitly left unmeasured because
the converted RoG JSONL contains neither full Freebase nor gold SPARQL.
