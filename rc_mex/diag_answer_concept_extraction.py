"""Diagnostic: can the final FilterConcept be recovered from question text alone?

The smoke-test questions are Find -> Relate -> FilterConcept -> Relate ->
FilterConcept -> What chains. The final FilterConcept constrains the answer
entity's concept. This script measures, over the same selected examples the
smoke test uses:

1. extraction accuracy: does a question-text-only extractor (KB concept names
   matched as whole-word substrings, start entity name masked out, earliest
   mention preferred, longest name on ties) recover the gold final
   FilterConcept? The extractor never reads the program; the program is used
   only to score the extractor.
2. separation power: among answer-pool siblings, how often is the gold answer
   an instance of the extracted concept (with ancestor closure)?

Usage:
  python3 -m rc_mex.diag_answer_concept_extraction \
      --kb data/kqa_pro/kb.json --questions data/kqa_pro/val.json \
      --max-examples 50
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from cigr_d_mvp1.io_utils import load_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text

from rc_mex.run_proof_state_search_smoke import extract_answer_concept, select_examples

def gold_final_filter_concept(sample: dict) -> str:
    """Name passed to the last FilterConcept before What, or '' if absent."""
    program = sample.get("program", []) or []
    last = ""
    for step in program:
        if step.get("function") == "FilterConcept":
            inputs = step.get("inputs", []) or []
            if inputs:
                last = str(inputs[0])
        if step.get("function") == "What":
            break
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    graph = KnowledgeGraph(load_json(args.kb))
    samples = load_json(args.questions)
    examples, _ = select_examples(graph, samples, args.max_examples, None)
    sample_by_index = {index: sample for index, sample in enumerate(samples)}

    rows = []
    outcome_counts: Counter[str] = Counter()
    for example in examples:
        sample = sample_by_index[example.program_index]
        gold_concept_name = normalize_text(gold_final_filter_concept(sample))
        extracted = extract_answer_concept(graph, example.question, [example.start_entity_name])
        extracted_name = str(extracted["concept_name"]) if extracted else ""
        extracted_ids = set(extracted["concept_ids"]) if extracted else set()

        if not gold_concept_name:
            outcome = "no_gold_filter_concept"
        elif not extracted:
            outcome = "no_concept_extracted"
        elif extracted_name == gold_concept_name or (
            extracted_ids & graph.find_concepts(gold_concept_name)
        ):
            outcome = "extraction_correct"
        else:
            outcome = "extraction_wrong"
        outcome_counts[outcome] += 1

        gold_is_instance = any(
            graph.is_instance_of_any(gold_id, extracted_ids) for gold_id in example.gold_answer_ids
        ) if extracted_ids else False

        rows.append(
            {
                "question_id": example.question_id,
                "question": example.question,
                "start_entity": example.start_entity_name,
                "gold_answers": example.gold_answer_labels[:5],
                "gold_final_filter_concept": gold_concept_name,
                "extracted_concept": extracted_name,
                "outcome": outcome,
                "gold_answer_is_instance_of_extracted": gold_is_instance,
            }
        )

    correct = outcome_counts["extraction_correct"]
    total_with_gold = sum(
        count for outcome, count in outcome_counts.items() if outcome != "no_gold_filter_concept"
    )
    print(json.dumps(dict(outcome_counts), indent=2))
    print(f"extraction accuracy where gold FC exists: {correct}/{total_with_gold}")
    gold_instance_hits = sum(1 for row in rows if row["gold_answer_is_instance_of_extracted"])
    print(f"gold answer is instance of extracted concept: {gold_instance_hits}/{len(rows)}")
    print()
    for row in rows:
        flag = "OK " if row["outcome"] == "extraction_correct" else row["outcome"][:14]
        print(
            f"[{flag}] gold_fc={row['gold_final_filter_concept']!r:30} "
            f"extracted={row['extracted_concept']!r:30} q={row['question'][:80]}"
        )


if __name__ == "__main__":
    main()
