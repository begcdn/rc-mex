"""LLM-navigator baseline: the over-reliance paradigm under our exact conditions.

A faithful in-house implementation of ToG-style KGQA, where the LLM makes
every decision — which relations to follow per hop per path, which target
entities to keep when a branch fans out, which paths survive the beam, when
the evidence suffices, and which entity is the answer. Symbolic code only
executes the graph operations the LLM requests.

Fairness contract (same conditions as the path-family methods):
- same question selection, same KB, same topic entities, same model;
- same candidate-relation lists with the same caps (relation_cap=30,
  sample_entities=25, max_branch_entities=40);
- same beam capacity (3 paths), same max depth (2), answer allowed at any hop;
- same determinism: temperature 0, fixed seed, persistent cache;
- gold never appears in any prompt; the answer must resolve to an entity the
  navigator actually visited (verifiable symbol, like our final ranking).

Per-question LLM call and token counts are recorded for the cost comparison.

Usage:
  python3 -m rc_mex.run_llm_navigator \
      --kb data/kqa_pro/kb.json --questions data/kqa_pro/val.json \
      --output runs/navigator_val --max-examples 250
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any

from cigr_d_mvp1.io_utils import ensure_dir, load_json, write_json, write_jsonl
from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text

from rc_mex.micro_agents import DEFAULT_MODEL, call_local_llm, parse_pick_numbers, probe_llm_endpoint
from rc_mex.run_proof_state_search_smoke import relation_targets, select_examples

BEAM_WIDTH = 3
MAX_DEPTH = 2
RELATIONS_PER_PATH = 2
ENTITIES_PER_BRANCH = 3
ENTITY_PICK_THRESHOLD = 8

NAV_RELATION_PROMPT_VERSION = "navigator_relation_v1"
NAV_RELATION_PROMPT = """You are answering a question by walking a knowledge graph step by step.

Question: "{question}"
Current path: {path}

Possible relations to follow next from "{tail}" ([forward] follows the relation, [backward] follows it in reverse):
{candidates}

Pick the {k} relations most likely to lead toward the answer. Reply with only the numbers separated by commas, best first."""

NAV_ENTITY_PROMPT_VERSION = "navigator_entity_v1"
NAV_ENTITY_PROMPT = """You are answering a question by walking a knowledge graph.

Question: "{question}"
Path so far: {path}
Following the relation "{relation}" leads to these entities:
{candidates}

Pick the {k} entities most likely to lead toward the answer. Reply with only the numbers separated by commas, best first."""

NAV_PRUNE_PROMPT_VERSION = "navigator_prune_v1"
NAV_PRUNE_PROMPT = """You are answering a question by walking a knowledge graph.

Question: "{question}"

Current candidate paths:
{candidates}

Keep the {k} paths most likely to lead to the answer. Reply with only the numbers separated by commas, best first."""

NAV_DECIDE_PROMPT_VERSION = "navigator_decide_v1"
NAV_DECIDE_PROMPT = """You are answering a question using paths walked on a knowledge graph.

Question: "{question}"

Paths found so far:
{candidates}

If one of the entities at the end of a path answers the question, reply exactly:
ANSWER: <entity name copied from a path end>
Otherwise reply exactly:
CONTINUE"""

NAV_FORCE_PROMPT_VERSION = "navigator_force_answer_v1"
NAV_FORCE_PROMPT = """You are answering a question using paths walked on a knowledge graph.

Question: "{question}"

Paths found:
{candidates}

Reply exactly:
ANSWER: <the entity name from a path end that best answers the question>"""


class CallMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = 0

    def ask(self, prompt: str, version: str, model: str) -> str:
        self.calls += 1
        result = call_local_llm(prompt, version, model=model, timeout=180.0)
        if result["error"]:
            self.errors += 1
            return ""
        self.prompt_tokens += int(result.get("prompt_tokens", 0))
        self.completion_tokens += int(result.get("completion_tokens", 0))
        return result["text"]


def path_readable(graph: KnowledgeGraph, path: dict[str, Any]) -> str:
    parts = [graph.entity_name(path["entities"][0])]
    for step, entity_id in zip(path["steps"], path["entities"][1:]):
        parts.append(f"--{step}--> {graph.entity_name(entity_id)}")
    return " ".join(parts)


def navigate(
    graph: KnowledgeGraph,
    question: str,
    start_ids: set[str],
    model: str,
    relation_cap: int = 30,
    sample_entities: int = 25,
    max_branch_entities: int = 40,
) -> dict[str, Any]:
    meter = CallMeter()
    paths = [{"entities": [sid], "steps": []} for sid in sorted(start_ids)[:BEAM_WIDTH]]
    visited: set[str] = set(p["entities"][0] for p in paths)
    answer_id, answer_hop = "", 0

    for hop in range(1, MAX_DEPTH + 1):
        new_paths = []
        for path in paths:
            tail = path["entities"][-1]
            frontier = graph.candidate_relations([tail], cap=relation_cap, sample_entities=sample_entities)
            if not frontier:
                continue
            labels = [f"{c.predicate.replace('_', ' ')} [{c.direction}]" for c in frontier]
            prompt = NAV_RELATION_PROMPT.format(
                question=question,
                path=path_readable(graph, path),
                tail=graph.entity_name(tail),
                candidates="\n".join(f"{i}. {l}" for i, l in enumerate(labels, 1)),
                k=RELATIONS_PER_PATH,
            )
            reply = meter.ask(prompt, NAV_RELATION_PROMPT_VERSION, model)
            picks = parse_pick_numbers(reply, len(labels), top_k=RELATIONS_PER_PATH)
            for pick in picks:
                candidate = frontier[pick - 1]
                step_label = f"{candidate.predicate.replace('_', ' ')}[{candidate.direction}]"
                targets = relation_targets(
                    graph, tail,
                    {"relation_id": candidate.predicate, "direction": candidate.direction},
                    max_branch_entities,
                )
                targets = [t for t in targets if t not in set(path["entities"])]
                if not targets:
                    continue
                if len(targets) > ENTITY_PICK_THRESHOLD:
                    names = [graph.entity_name(t) for t in targets[:20]]
                    eprompt = NAV_ENTITY_PROMPT.format(
                        question=question,
                        path=path_readable(graph, path),
                        relation=step_label,
                        candidates="\n".join(f"{i}. {n}" for i, n in enumerate(names, 1)),
                        k=ENTITIES_PER_BRANCH,
                    )
                    ereply = meter.ask(eprompt, NAV_ENTITY_PROMPT_VERSION, model)
                    epicks = parse_pick_numbers(ereply, len(names), top_k=ENTITIES_PER_BRANCH)
                    chosen = [targets[i - 1] for i in epicks] or targets[:ENTITIES_PER_BRANCH]
                else:
                    chosen = targets[:ENTITIES_PER_BRANCH]
                for target in chosen:
                    new_paths.append({"entities": path["entities"] + [target], "steps": path["steps"] + [step_label]})
                    visited.add(target)
        if not new_paths:
            break
        if len(new_paths) > BEAM_WIDTH:
            readable = [path_readable(graph, p) for p in new_paths[:20]]
            pprompt = NAV_PRUNE_PROMPT.format(
                question=question,
                candidates="\n".join(f"{i}. {r}" for i, r in enumerate(readable, 1)),
                k=BEAM_WIDTH,
            )
            preply = meter.ask(pprompt, NAV_PRUNE_PROMPT_VERSION, model)
            ppicks = parse_pick_numbers(preply, len(readable), top_k=BEAM_WIDTH)
            paths = [new_paths[i - 1] for i in ppicks] or new_paths[:BEAM_WIDTH]
        else:
            paths = new_paths

        readable = [path_readable(graph, p) for p in paths]
        if hop < MAX_DEPTH:
            dprompt = NAV_DECIDE_PROMPT.format(
                question=question,
                candidates="\n".join(f"{i}. {r}" for i, r in enumerate(readable, 1)),
            )
            decision = meter.ask(dprompt, NAV_DECIDE_PROMPT_VERSION, model)
        else:
            decision = meter.ask(
                NAV_FORCE_PROMPT.format(
                    question=question,
                    candidates="\n".join(f"{i}. {r}" for i, r in enumerate(readable, 1)),
                ),
                NAV_FORCE_PROMPT_VERSION,
                model,
            )
        if "ANSWER:" in decision:
            name = normalize_text(decision.split("ANSWER:", 1)[1].strip().splitlines()[0].strip().strip('"'))
            endpoints = {p["entities"][-1]: normalize_text(graph.entity_name(p["entities"][-1])) for p in paths}
            matched = [eid for eid, ename in endpoints.items() if ename == name]
            if not matched:
                matched = [eid for eid in visited if normalize_text(graph.entity_name(eid)) == name]
            if matched:
                answer_id, answer_hop = matched[0], hop
                break

    return {
        "answer_id": answer_id,
        "answer_hop": answer_hop,
        "visited": visited,
        "final_paths": [path_readable(graph, p) for p in paths],
        "llm_calls": meter.calls,
        "prompt_tokens": meter.prompt_tokens,
        "completion_tokens": meter.completion_tokens,
        "llm_errors": meter.errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="runs/llm_navigator")
    parser.add_argument("--max-examples", type=int, default=250)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    probe = probe_llm_endpoint(model=args.model)
    if not probe["ok"]:
        raise SystemExit(f"LLM endpoint unreachable: {probe['error']} (url={probe['url']}, model={probe['model']})")
    print(f"LLM endpoint OK: {probe['url']} (model={args.model})", flush=True)

    graph = KnowledgeGraph(load_json(args.kb))
    samples = load_json(args.questions)
    examples, selection_stats = select_examples(graph, samples, args.max_examples, None)
    print(f"Selected {len(examples)} questions", flush=True)

    rows = []
    counts: Counter[str] = Counter()
    started = time.time()
    for index, example in enumerate(examples, 1):
        result = navigate(graph, example.question, example.start_entity_ids, args.model)
        hit = result["answer_id"] in example.gold_answer_ids if result["answer_id"] else False
        gold_visited = bool(result["visited"] & example.gold_answer_ids)
        counts["total"] += 1
        counts["hits_at_1"] += hit
        counts[f"hits_at_1_{example.hop_count}hop"] += hit
        counts[f"total_{example.hop_count}hop"] += 1
        counts["gold_visited"] += gold_visited
        counts["no_answer"] += not result["answer_id"]
        counts["llm_calls"] += result["llm_calls"]
        counts["prompt_tokens"] += result["prompt_tokens"]
        counts["completion_tokens"] += result["completion_tokens"]
        counts["llm_errors"] += result["llm_errors"]
        rows.append(
            {
                "question_id": example.question_id,
                "question": example.question,
                "gold_hop_count": example.hop_count,
                "gold_answers": example.gold_answer_labels,
                "predicted": graph.entity_name(result["answer_id"]) if result["answer_id"] else "",
                "answer_hop": result["answer_hop"],
                "hits_at_1": hit,
                "gold_visited": gold_visited,
                "llm_calls": result["llm_calls"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "final_paths": result["final_paths"],
            }
        )
        if index % 20 == 0:
            print(f"  ... {index}/{len(examples)} ({time.time()-started:.0f}s, {counts['llm_calls']} calls)", flush=True)

    total = max(1, counts["total"])
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "selection_stats": selection_stats,
        "metrics": {
            "questions": counts["total"],
            "hits_at_1": counts["hits_at_1"] / total,
            "hits_at_1_by_hop": {
                f"{h}hop": {
                    "questions": counts[f"total_{h}hop"],
                    "hits_at_1": counts[f"hits_at_1_{h}hop"] / max(1, counts[f"total_{h}hop"]),
                }
                for h in [1, 2, 3]
                if counts[f"total_{h}hop"]
            },
            "gold_visited_rate": counts["gold_visited"] / total,
            "no_answer_rate": counts["no_answer"] / total,
            "avg_llm_calls_per_question": counts["llm_calls"] / total,
            "avg_prompt_tokens_per_question": counts["prompt_tokens"] / total,
            "avg_completion_tokens_per_question": counts["completion_tokens"] / total,
            "llm_errors": counts["llm_errors"],
        },
    }
    write_jsonl(output_dir / "navigator_predictions.jsonl", rows)
    write_json(output_dir / "navigator_metrics.json", summary)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote navigator outputs to {output_dir}")


if __name__ == "__main__":
    main()
