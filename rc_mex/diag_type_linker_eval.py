"""Standalone eval of the answer-type-linker micro-agent vs the string extractor.

For each selected two-hop chain, compares against the gold program's final
FilterConcept (gold is used to SCORE only, never as input):

- string extractor (extract_answer_concept) accuracy
- shortlist recall (is the gold concept in the LLM's candidate list at all —
  a symbolic failure if not, attributable to us, not the model)
- LLM linker accuracy + abstention rate + selection accuracy given recall

Usage:
  python3 -m rc_mex.diag_type_linker_eval \
      --kb data/kqa_pro/kb.json --questions data/kqa_pro/val.json \
      --max-examples 50 [--model llama3.2:3b] [--show-errors]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from cigr_d_mvp1.io_utils import load_json
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text

from rc_mex.diag_answer_concept_extraction import gold_final_filter_concept
from rc_mex.micro_agents import DEFAULT_MODEL, link_answer_concept, link_answer_concept_cascade
from rc_mex.run_proof_state_search_smoke import extract_answer_concept, select_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default="type_linker_v1")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    graph = KnowledgeGraph(load_json(args.kb))
    samples = load_json(args.questions)
    examples, _ = select_examples(graph, samples, args.max_examples, None)

    counts: Counter[str] = Counter()
    errors = []
    started = time.time()
    for index, example in enumerate(examples, start=1):
        sample = samples[example.program_index]
        gold_name = normalize_text(gold_final_filter_concept(sample))
        gold_ids = graph.find_concepts(gold_name) if gold_name else set()
        mask = [example.start_entity_name]

        extracted = extract_answer_concept(graph, example.question, mask)
        string_name = str(extracted["concept_name"]) if extracted else ""
        string_ok = bool(gold_name) and (
            string_name == gold_name or bool(set(extracted["concept_ids"]) & gold_ids) if extracted else False
        )

        if args.prompt_version == "cascade":
            linked = link_answer_concept_cascade(graph, example.question, mask, model=args.model)
            linked["shortlist"] = []
            shortlist_hit = True
            counts[f"source_{linked['source']}"] += 1
            counts["llm_consulted"] += bool(linked["llm_consulted"])
        else:
            linked = link_answer_concept(graph, example.question, mask, model=args.model, prompt_version=args.prompt_version)
            shortlist_hit = gold_name in {normalize_text(name) for name in linked["shortlist"]}
        llm_name = linked["concept_name"] or ""
        llm_ok = bool(gold_name) and (
            normalize_text(llm_name) == gold_name or bool(set(linked["concept_ids"]) & gold_ids)
        )
        if args.prompt_version == "cascade" and llm_ok:
            counts[f"correct_{linked['source']}"] += 1

        counts["total"] += 1
        counts["string_correct"] += string_ok
        counts["llm_correct"] += llm_ok
        counts["shortlist_recall"] += shortlist_hit
        counts["llm_abstained"] += bool(linked.get("abstained", False))
        counts["llm_correct_given_recall"] += llm_ok and shortlist_hit
        counts["llm_server_error"] += bool(linked.get("error", ""))
        if not llm_ok or not string_ok:
            errors.append(
                {
                    "question": example.question,
                    "gold": gold_name,
                    "string": string_name,
                    "llm": llm_name,
                    "llm_raw": linked["raw_response"][:60],
                    "shortlist_hit": shortlist_hit,
                    "string_ok": string_ok,
                    "llm_ok": llm_ok,
                }
            )
        if index % 25 == 0:
            print(f"  ... {index}/{len(examples)} ({time.time()-started:.0f}s)", flush=True)

    total = counts["total"]
    recall = counts["shortlist_recall"]
    print(json.dumps(dict(counts), indent=2))
    print(f"string extractor accuracy: {counts['string_correct']}/{total}")
    print(f"shortlist recall (symbolic candidate gen): {recall}/{total}")
    print(f"LLM linker accuracy: {counts['llm_correct']}/{total}")
    print(f"LLM selection accuracy given shortlist hit: {counts['llm_correct_given_recall']}/{recall}")
    print(f"LLM abstentions: {counts['llm_abstained']}/{total}  server errors: {counts['llm_server_error']}")
    if args.show_errors:
        for row in errors:
            tag = f"str={'Y' if row['string_ok'] else 'n'} llm={'Y' if row['llm_ok'] else 'n'} short={'Y' if row['shortlist_hit'] else 'n'}"
            print(f"[{tag}] gold={row['gold']!r:28} llm={row['llm']!r:24} str={row['string']!r:20} q={row['question'][:70]}")


if __name__ == "__main__":
    main()
