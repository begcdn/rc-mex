# CoG Blueprint Code Notes

Paper: **CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs**.

Official code: `https://github.com/zjukg/CoG`.

## What The Code Does

The implementation has two relevant pieces.

### Offline Blueprint Construction

File: `CoG/build_skeleton.py`

The code extracts relation-only skeletons from training SPARQL:

```text
relation_1 -> relation_2 -> relation_3
```

It deduplicates skeletons and keeps a long question as the semantic anchor for each skeleton. Then it encodes anchor questions with a SentenceTransformer and saves an index.

Important detail: the blueprint is not an executable program. It is a soft relation-path prior.

### Online Blueprint Use

Files: `CoG/utils.py`, `CoG/freebase_func.py`, `CoG/main_freebase.py`

At test time, CoG:

1. Masks topic entities in the question.
2. Retrieves similar training questions from the skeleton index.
3. If similarity is very high, directly copies the exemplar skeleton.
4. Otherwise, asks the LLM to predict a skeleton from retrieved exemplars.
5. During graph expansion, uses the predicted relation for the current depth as a soft guide.

The core ranking/pruning logic is in `relation_search_prune`:

```text
final_score =
  0.60 * similarity(question, candidate_relation)
+ 0.25 * similarity(predicted_blueprint_relation, candidate_relation)
+ 0.15 * similarity(global_blueprint, candidate_relation)
```

If the predicted relation matches a candidate very confidently, it fast-tracks that candidate. Otherwise, it sorts and prunes candidates before the LLM sees them.

## RC-MEX Adaptation

RC-MEX should not copy the blueprint object. The useful method is the **soft-prior reranking layer before LLM selection**.

Instead of:

```text
question -> retrieved blueprint relation -> rerank local relation labels
```

RC-MEX uses:

```text
question -> relation-card semantic text -> rerank local relation cards
```

So the soft prior is no longer a training-set relation skeleton. It is the reusable relation card:

```text
description
positive rule
negative rule
argument roles
domain/range types
contrastive examples
```

This creates a CoG-style card-guided candidate pruning step:

```text
score(card) =
  question/card semantic match
+ current entity type compatibility
+ output compatibility
```

This is still not full KGQA. It tests whether relation cards are a better soft structural prior than raw relation names or anonymized IDs.

