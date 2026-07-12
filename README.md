# Executable KGQA Research

The active direction is reliable multi-hop KGQA through uncertain semantic
structure and executable evidence. The old RC-MEX relation-card experiments
remain in the repository as historical ablations; they are not the current
architecture or research claim.

## Current Hypothesis

A question should not commit immediately to one path, one relation sequence,
or one deterministic constraint parse. The current experiment represents the
question with a small ensemble of schema-independent semantic sketches and
jointly evaluates those sketches against executable query hypotheses and their
denotations.

The generic sketch vocabulary includes relation roles, answer type, set
operations, filters, temporal/numeric constraints, comparison, aggregation,
and ordering. A question activates only the relevant subset. KG-specific
adapters are responsible for storage details such as Freebase CVTs or Wikidata
qualifiers; those details do not belong in the question semantics.

This first experiment changes the semantic representation used for relation
proposal and final query selection. Retrieval mechanics and budgets, graph
execution, and answer post-processing stay fixed. The evaluation firewall
therefore reports separately whether gains come from better menu recall or
better conditional selection. This is not yet the full proposed architecture.

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
