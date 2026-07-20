# Inverse Path Verification: First Experiment

## General failure

Graph search often retains paths because each next relation looks locally relevant. A path can therefore be composed of individually plausible relations while expressing the wrong question globally. Direct question-path similarity has the same weakness when lexical overlap rewards the right words in the wrong direction or order.

## Computational cause

The selector estimates whether a path resembles the question, but it is not required to account for the complete question as an output of that path. Local relevance can ignore relation direction, composition order, missing constraints, and answer type.

## Hypothesis

A compact model can learn the inverse mapping from an executable relation path
to the natural-language question that the path answers. A candidate is credible
when its generated question has the same meaning as the user's question and its
endpoint has a compatible semantic type.

The active model is a pure sequence-to-sequence generator. It is fine-tuned on
two tasks with shared parameters:

\[
L = L_{\text{path-to-question}} + L_{\text{answer-type compatibility}}
\]

The second task receives a question and a free-text candidate endpoint type and
answers yes or no. It uses gold endpoint types as positives and synthetic wrong
types as negatives. It does not create a path-ranking head, and the generated
path question contains no required structured type field.

## Closest alternatives

- Direct question-path encoders score compatibility without reconstructing the question.
- Path-to-question generation alone can describe the right prefix while ignoring
  that a continuation changed the endpoint and therefore the requested answer.
- Path rankers trained on hard negatives separate paths but need not learn an
  interpretable inverse mapping.

The experiment compares against a frozen direct MiniLM scorer and the pretrained generator before fine-tuning. The joint loss is not itself a novelty claim; the experiment asks whether inverse conditional verification supplies a useful path-selection signal.

## Falsifying experiment

Train `FLAN-T5-small` on 1--2 hop KQA Pro chains while hiding intermediate and answer entity names. Test on:

1. unseen KQA questions;
2. KQA relations absent from training;
3. KQA relation compositions absent from training;
4. official WebQSP inferential chains, whose Freebase schema is wholly absent from training.

First test gold paths only. Report question-generation token F1, ROUGE-L, and
semantic similarity. Separately report compatibility with the gold endpoint
type and rejection of synthetic wrong endpoint types. The mechanism is not
ready for path selection if it accepts both correct and wrong types.

The hypothesis is weakened if gains occur only on ordinary KQA validation questions. It is rejected for schema-general verification if the fine-tuned model does not improve hard-negative ranking on unseen KQA relations and WebQSP.

Synthetic wrong endpoint types isolate type discrimination but can contain
label noise when two dataset type strings are semantically equivalent. A second
WebQSP split enumerates relation paths that are actually executable in each
supplied local graph. Candidate generation is reported separately.

At candidate verification time, semantic similarity and type compatibility are
kept as separate observations. A candidate is not accepted merely because a
weighted sum hides a failure in one of them.

Two training regimes distinguish architectural transfer from data coverage.
`kqa_only` is a leave-KG-out test against WebQSP. `multi_kg` trains one shared
model on the KQA and WebQSP training partitions, without KG-specific adapters,
and tests unseen questions plus held-out KQA relations/compositions. The second
regime does not prove unseen-KG transfer; it tests whether one semantic model can
learn multiple schemas without sacrificing structural generalization.

## Leakage boundary

- Gold paths are supervision for this controlled verifier experiment, not test-time search input.
- Answer and intermediate entity names are absent from model inputs.
- Entire held-out KQA relations and compositions are excluded from training.
- WebQSP is test-only.
- The model sees readable relation descriptions and type metadata. Inferring semantics from anonymous IDs with no evidence is not claimed.
