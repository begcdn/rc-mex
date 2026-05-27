# RC-MEX Prior-Work Lessons

MVP3 is frozen as a diagnostic. It showed that carrying the top-k relation-card uncertainty from MVP2 into execution can improve gold-denotation recall, but naive union-style top-k execution also adds noisy entities and lowers precision/F1. This is useful evidence, but it is not the novelty claim.

## Lesson From KGQA Graph Search

High-recall graph retrieval is already a standard pattern in KGQA. The recurring hard part is not merely including more candidate paths or relations. The hard part is pruning and reranking those candidates without losing the correct evidence.

- **ToG** uses iterative beam/path exploration over knowledge graphs. Its core behavior is to keep multiple graph options alive during multi-hop traversal.
- **PoG** explores multi-hop paths and then uses pruning/reranking to reduce noisy paths. The important lesson is that path recall alone is not enough.
- **DAMR** uses LLM-guided top-k relation selection plus lightweight path/path-sequence scoring. This is close to the MVP2/MVP3 setting: top-k relation choice helps recall, but scoring is needed to control noise.
- **GNN-RAG/SubgraphRAG-style methods** retrieve a broader graph candidate set first, then use another scorer or reasoner to filter the graph evidence.

## RC-MEX Framing

RC-MEX should not claim top-k graph search as novelty. The correct claim is narrower and more specific:

> Relation-card uncertainty from MVP2 is carried into execution and improves denotation/proof reachability over top-1 commitment, but future work should use known KGQA search/pruning ideas rather than re-testing top-k.

The RC-MEX-specific contribution remains the semantic interface:

> Relation cards provide a more robust semantic representation of KG predicates than raw relation names or arbitrary relation IDs, especially when schema labels are hidden, noisy, or misleading.

## Current MVP Status

- **MVP1:** Relation cards beat relation-name-only baselines and remain useful under hidden or misleading relation labels.
- **MVP2:** Question slots can retrieve/rank the correct relation card from a local candidate frontier.
- **MVP3:** Top-k relation-card execution improves recall but adds noise. This validates that relation-card uncertainty contains recoverable graph evidence, but naive top-k union is not the final answer method.
- **MVP3.5:** The next experiment should compare relation-card retrieval/execution against raw relation names, anonymized IDs, and misleading labels under the same controlled slot setup.

