# ComplexWebQuestions structure metadata

`cwq_test_structure.json` is a compact projection of the public
ComplexWebQuestions 1.1 test split. It retains only the fields needed for the
controlled SubgraphRAG reader experiment:

- `ID`
- `compositionality_type`
- `sparql`

The source dataset is available from the
[ComplexWebQuestions project page](https://www.tau-nlp.sites.tau.ac.il/compwebq).
Gold SPARQL is used only for offline sample selection and evaluation slicing;
it is never included in model prompts.
