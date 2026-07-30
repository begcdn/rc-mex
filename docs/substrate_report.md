# Query Contract Search: Stage A Substrate Report

## Decision

Phase 0 uses the MID-keyed development subgraphs released with
[GNN-RAG](https://github.com/cmavro/GNN-RAG). Entity integers in each JSON row
are decoded through `entities.txt` to Freebase MIDs; relation integers are
decoded through `relations.txt`. Names are never used as graph-node identity.

This avoids the entity-name collapse already measured in the RoG serialization.
SubgraphRAG was inspected first, but its preprocessing assigns local integer IDs
after loading name-keyed triples, so it cannot recover identities that were
already merged. The local RoG graphs remain a fallback only.

| dataset | split | questions | substrate |
|---|---|---:|---|
| WebQSP | GNN-RAG dev | 250 | MID-keyed question subgraphs |
| CWQ | GNN-RAG dev | 3,519 | MID-keyed question subgraphs |

## Reference Execution Ceiling

The ceiling asks a narrow question: if the official reference query is run on
the supplied question subgraph, can it reproduce the released gold answer set?
It does not measure a learned model.

| dataset | n | compiled and executed | exact | exact / all | exact / compiled |
|---|---:|---:|---:|---:|---:|
| WebQSP dev | 250 | 240 | 198 | 79.2% | 82.5% |
| CWQ dev | 3,519 | 3,472 | 1,719 | 48.8% | 49.5% |

WebQSP additionally has six unsupported parses and four rows with empty released
gold sets. Empty-gold rows are not treated as successful executions.

### WebQSP constraints

The compiler preserves entity constraints and ordering instead of filtering
those questions out.

| subset | n | exact |
|---|---:|---:|
| no explicit constraint | 176 | 159 (90.3%) |
| constraint-bearing | 64 | 39 (60.9%) |
| ordering | 4 | 3 (75.0%) |

### CWQ composition types

| type | n | exact | not exact | uncompiled |
|---|---:|---:|---:|---:|
| composition | 1,575 | 742 | 831 | 2 |
| conjunction | 1,534 | 977 | 515 | 42 |
| comparative | 219 | 0 | 218 | 1 |
| superlative | 191 | 0 | 189 | 2 |

The zero ceiling for comparative and superlative questions is a property of the
released subgraphs, not the query compiler. Direct inspection of official
queries and graph rows found that the required literal-bearing relations, such
as numeric values and dates used by filters or ordering, are absent. The
name-keyed RoG copy and GNN-RAG's alternate CWQ retrieval files omit the same
facts.

The failure side of compilation was checked separately. Among the 831 compiled
but non-exact CWQ composition programs, 754 (90.7%) execute to an empty set, 74
(8.9%) partially overlap the gold set, and only 3 (0.4%) return a nonempty
disjoint set. A compiler that systematically reversed or mistranslated
relations would produce disjoint entities; the observed pattern instead points
to missing graph evidence. Two additional composition rows are uncompiled and
are not included in this diagnostic.

Therefore Gate 2 may be computed only on reference-executable questions. With
this substrate, that population contains composition and conjunction but
effectively no comparative or superlative questions. Any result must say this
explicitly. A claim about completeness operators across all CWQ types requires
a literal-preserving Freebase store or released subgraphs containing those
attributes.

The 1,719-question Gate-2 population is 977 conjunction questions (56.8%) and
742 composition questions (43.2%). Results must report both the aggregate and
the two composition types separately.

## Manual Check

Twenty deterministic exact cases were inspected against their questions,
compiled programs, and answer sets:

- WebQSP simple: `WebQTrn-11`, `WebQTrn-15`, `WebQTrn-73`,
  `WebQTrn-104`, `WebQTrn-126`
- WebQSP constrained: `WebQTrn-47`, `WebQTrn-196`, `WebQTrn-297`,
  `WebQTrn-428`, `WebQTrn-557`
- CWQ composition:
  `WebQTrn-2181_8d86dc5e03446f0e50fd69bc06ae0658`,
  `WebQTrn-453_8c50e30ac5163e6dabfc999a7129a4ea`,
  `WebQTrn-3287_ebfe3c418f7914f9babf21caade27b05`,
  `WebQTest-811_22a085e4a873315a9c2e63361cbd9248`,
  `WebQTest-1638_85f381b8012b4def553954906bc9fafb`
- CWQ conjunction:
  `WebQTest-823_ed31f9dd431831dbd32a06b958c7c97c`,
  `WebQTrn-2873_143c89d70679c3e5257c93d8e2bc4c67`,
  `WebQTest-1508_872253e47dd6ddaa213ff31eeda8783b`,
  `WebQTrn-2674_831fb3325644a924d433e2b267f6d238`,
  `WebQTrn-2631_770d8b533fd2f564cf8401e6727b03f6`

All twenty compiled structures matched the requested relation or join and
reproduced the complete answer set.

No eligible WebQSP reference program and no CWQ reference program traversed an
identity self-loop. Tests also verify that distinct Freebase MIDs remain
distinct even when display names could be identical.

## Baseline Status

A minimal executable search is implemented in `contract_search/search.py`:

- graph-valid forward and backward relation expansion;
- executable joins against additional topic entities;
- explicit `Finish` actions;
- beam width 10, maximum four atoms, and a hard cap of 500 scored expansions;
- one frozen pair scorer shared by future arms;
- identity-triple and exact duplicate-atom rejection.

Twenty-five-question lexical smoke runs on each dataset confirmed:

- no expansion budget exceeded 500;
- no selected program repeated an identical atom;
- no identity triple survived;
- outputs and full beam traces were written.

Those smoke scores are marked `reported_result: false`; the lexical scorer is
not an experimental baseline. The reportable baseline must use the frozen
fine-tuned BGE reranker on the server.

That reranker was previously trained on linear path serializations, while this
search also serializes joins and constraints. This format mismatch is a stated
baseline limitation. It can increase pruning and ranking errors, so it should
make the Gate-1 silent-incompleteness proportion conservative rather than
artificially favorable. The scorer remains frozen and identical across every
Phase-1 arm.

## Reproduction

```bash
python3 -m contract_search webqsp-ceiling \
  --official data/pattern_alignment/webqsp/WebQSP.train.json \
  --substrate data/contract_search/gnnrag/webqsp \
  --output runs/contract_search/stage_a_webqsp

python3 -m contract_search cwq-ceiling \
  --official data/contract_search/ComplexWebQuestions_dev.json \
  --substrate data/contract_search/gnnrag/CWQ \
  --output runs/contract_search/stage_a_cwq

python3 -m pytest tests/test_contract_search.py -q
```

## Stage A Outcome

The ID-preserving substrate requirement passes. Reference execution is reliable
enough for a ceiling-filtered study of entity composition and conjunction.
The available substrate does not support the full intended CWQ operator
population. Phase 0 can proceed, but its scope is narrower than the original
plan unless a literal-preserving Freebase source is supplied.
