"""LLM micro-agents: small, verifiable subtasks inside the symbolic KGQA pipeline.

Design contract for every micro-agent:
- The LLM output is either a verifiable symbol (checked against the KB before
  use) or a bounded signal combined into our own scoring; it never controls
  search directly.
- Abstention (NONE / parse failure / server down) degrades to the symbolic
  fallback — the pipeline must behave identically to the non-LLM method.
- temperature 0, fixed seed, and a persistent cache keyed by
  (prompt_version, model, prompt) so runs replay deterministically.

Micro-agent 1: answer-type linker.
Task: given the question and a KB-derived shortlist of concept names, name the
concept the final answer must be an instance of, or NONE. Candidate generation
(the shortlist) stays symbolic so failures are attributable: shortlist recall
is ours, selection accuracy is the LLM's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from cigr_d_mvp1.kg import KnowledgeGraph, normalize_text

# Point these at the serving box, e.g.
#   ollama:           export RC_MEX_LLM_URL=http://localhost:11434/api/generate
#   vLLM / llama.cpp: export RC_MEX_LLM_URL=http://localhost:8000/v1/chat/completions
#   export RC_MEX_LLM_MODEL=qwen3:8b    (vLLM: the served name, e.g. Qwen/Qwen3-8B)
# API style is inferred from the URL; force it with RC_MEX_LLM_API=ollama|openai.
OLLAMA_URL = os.environ.get("RC_MEX_LLM_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("RC_MEX_LLM_MODEL", "llama3.2:3b")
LLM_API_STYLE = os.environ.get(
    "RC_MEX_LLM_API",
    "openai" if ("/v1/" in OLLAMA_URL or OLLAMA_URL.endswith("/chat/completions")) else "ollama",
)
LLM_API_KEY = os.environ.get("RC_MEX_LLM_API_KEY", "")
CACHE_PATH = Path("cache/micro_agents.json")

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

TYPE_LINKER_PROMPT_VERSION = "type_linker_v1"
TYPE_LINKER_PROMPT_TEMPLATE = """Task: identify the answer type of a question over a knowledge base.

Question: "{question}"

Candidate types:
{candidates}

Rules:
- Pick the single candidate type that the FINAL ANSWER to the question must be an instance of.
- The answer type is what the question asks for, not other things merely mentioned in the question.
- If the question asks "who", "whom", "whose", or "which person", the answer type is human.
- If no candidate fits the final answer, answer NONE.

Answer with exactly one type name copied from the candidate list, or NONE. Do not explain."""

TYPE_LINKER_MC_PROMPT_VERSION = "type_linker_v2_mc"
TYPE_LINKER_MC_PROMPT_TEMPLATE = """You classify what type of thing a question is asking for.

Example 1:
Question: "Which town is the birthplace of the author of Carrie?"
A. town
B. author
C. human
Answer: A

Example 2:
Question: "Who directed the movie produced by Kathleen Kennedy?"
A. film
B. human
C. film director
Answer: B

Example 3:
Question: "What is the Academy Awards ceremony that came before the 60th one?"
A. award ceremony
B. academy awards ceremony
C. award
Answer: B

Rules: choose the type of the FINAL answer, not other things mentioned in the question. If the question names the type explicitly, choose the most specific candidate that matches those exact words. "Who" or "whose" means the answer is human. Answer Z if no candidate fits.

Question: "{question}"
{candidates}
Z. none of these

Answer with one letter only.
Answer:"""

TYPE_LINKER_SPAN_PROMPT_VERSION = "type_linker_v3_span"
TYPE_LINKER_SPAN_PROMPT_TEMPLATE = """A question asks for something. Quote the words in the question that name the TYPE of thing the final answer is.

Rules:
- Copy the exact words from the question. Do not paraphrase.
- Give the type of the FINAL answer, not types of other things mentioned along the way.
- If the question asks "who" or "whose", answer: human
- If the type is never named, answer: NONE

Example 1:
Question: "Which town is the birthplace of the author of Carrie?"
Type words: town

Example 2:
Question: "Who directed the movie produced by Kathleen Kennedy?"
Type words: human

Example 3:
Question: "For the record label that signed Nirvana, what album has it released?"
Type words: album

Question: "{question}"
Type words:"""

PATH_RANKER_PROMPT_VERSION = "path_plausibility_v2"
PATH_RANKER_PROMPT_TEMPLATE = """A question is answered by following a path of relations in a knowledge base, starting from the starting entity. [forward] follows the relation, [backward] follows it in reverse.

Starting entity: "{start}"
Question: "{question}"

Candidate relation paths:
{paths}

Pick the {top_k} paths most likely to reach the answer to the question. Reply with only the {top_k} numbers separated by commas, best first."""

_CACHE: dict[str, str] | None = None


def _load_cache() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        if CACHE_PATH.exists():
            _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    if _CACHE is not None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_CACHE, indent=0, sort_keys=True), encoding="utf-8")


def _cache_key(prompt_version: str, model: str, prompt: str) -> str:
    return hashlib.sha256(f"{prompt_version}|{model}|{prompt}".encode("utf-8")).hexdigest()


def call_local_llm(
    prompt: str,
    prompt_version: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 90.0,
    num_predict: int = 256,
) -> dict[str, Any]:
    """Cached, deterministic single-shot generation.

    Returns {text, from_cache, error, prompt_tokens, completion_tokens}.
    Token counts come from the serving backend and are cached alongside the
    text (older cache entries are plain strings with unknown token counts)."""
    cache = _load_cache()
    key = _cache_key(prompt_version, model, prompt)
    if key in cache:
        cached = cache[key]
        if isinstance(cached, dict):
            return {
                "text": cached.get("text", ""),
                "from_cache": True,
                "error": "",
                "prompt_tokens": int(cached.get("prompt_tokens", 0)),
                "completion_tokens": int(cached.get("completion_tokens", 0)),
            }
        return {"text": cached, "from_cache": True, "error": "", "prompt_tokens": 0, "completion_tokens": 0}
    if LLM_API_STYLE == "openai":
        request_body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 7,
            "max_tokens": num_predict,
        }
        if "qwen" in model.lower():
            # vLLM-style switch to disable qwen3 thinking mode.
            request_body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 7, "num_predict": num_predict},
        }
        if "qwen" in model.lower():
            # Disable qwen3 thinking mode; strict verifiers expect the bare answer.
            request_body["think"] = False
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    def _request(body_dict: dict[str, Any]) -> tuple[str, int, int]:
        payload = json.dumps(body_dict).encode("utf-8")
        request = urllib.request.Request(OLLAMA_URL, data=payload, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        if LLM_API_STYLE == "openai":
            raw_text = str((body.get("choices") or [{}])[0].get("message", {}).get("content", ""))
            usage = body.get("usage", {}) or {}
            return THINK_BLOCK_PATTERN.sub("", raw_text).strip(), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        raw_text = str(body.get("response", ""))
        return THINK_BLOCK_PATTERN.sub("", raw_text).strip(), int(body.get("prompt_eval_count", 0)), int(body.get("eval_count", 0))

    try:
        text, prompt_tokens, completion_tokens = _request(request_body)
        # qwen3 + ollama occasionally returns an empty completion when thinking
        # is disabled on a long prompt (observed on 25-candidate adjudication
        # prompts). Salvage once by letting it think, then strip the block.
        if not text and "qwen" in model.lower():
            retry_body = dict(request_body)
            retry_body.pop("think", None)
            retry_body.pop("chat_template_kwargs", None)
            if LLM_API_STYLE != "openai":
                retry_body["options"] = {**retry_body.get("options", {}), "num_predict": max(num_predict, 1024)}
            else:
                retry_body["max_tokens"] = max(num_predict, 1024)
            retry_text, retry_pt, retry_ct = _request(retry_body)
            if retry_text:
                text, prompt_tokens, completion_tokens = retry_text, retry_pt, retry_ct
    except Exception as exc:
        return {"text": "", "from_cache": False, "error": f"{type(exc).__name__}: {exc}", "prompt_tokens": 0, "completion_tokens": 0}
    cache[key] = {"text": text, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    _save_cache()
    return {"text": text, "from_cache": False, "error": "", "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def build_concept_shortlist(
    graph: KnowledgeGraph,
    question: str,
    mask_names: list[str],
    limit: int = 25,
) -> list[str]:
    """KB-side candidate generation: exact concept mentions first, then the
    concepts most lexically similar to the question, plus 'human'. Symbolic on
    purpose — shortlist recall is measured separately from LLM selection."""
    from rc_mex.run_proof_state_search_smoke import char_ngram_similarity, question_concept_mentions

    shortlist: list[str] = []
    for mention in question_concept_mentions(graph, question, mask_names):
        name = str(mention["concept_name"])
        if name not in shortlist:
            shortlist.append(name)
    if "human" not in shortlist and graph.find_concepts("human"):
        shortlist.append("human")
    scored = sorted(
        ((char_ngram_similarity(question, name), name) for name in graph.name_to_concept_ids if name),
        key=lambda item: (-item[0], item[1]),
    )
    for score, name in scored:
        if len(shortlist) >= limit:
            break
        if score <= 0.0:
            break
        if name not in shortlist:
            shortlist.append(name)
    return shortlist[:limit]


def link_answer_concept(
    graph: KnowledgeGraph,
    question: str,
    mask_names: list[str],
    model: str = DEFAULT_MODEL,
    prompt_version: str = TYPE_LINKER_PROMPT_VERSION,
) -> dict[str, Any]:
    """Mode-1 micro-agent: returns a verified concept name from the shortlist or None.

    Output is only used if it names a real KB concept from the shortlist;
    anything else (NONE, hallucination, parse noise, server error) becomes
    abstention, and the caller falls back to the symbolic extractor or no-op.
    """
    shortlist = build_concept_shortlist(graph, question, mask_names)
    if not shortlist:
        return {"concept_name": None, "concept_ids": set(), "shortlist": [], "raw_response": "", "abstained": True, "error": ""}
    if prompt_version == TYPE_LINKER_SPAN_PROMPT_VERSION:
        prompt = TYPE_LINKER_SPAN_PROMPT_TEMPLATE.format(question=question)
        result = call_local_llm(prompt, prompt_version, model=model)
        raw = result["text"]
        chosen = _resolve_span_to_concept(graph, raw, mask_names, question)
    elif prompt_version == TYPE_LINKER_MC_PROMPT_VERSION:
        letters = [chr(ord("A") + index) for index in range(len(shortlist))]
        candidates_block = "\n".join(f"{letter}. {name}" for letter, name in zip(letters, shortlist))
        prompt = TYPE_LINKER_MC_PROMPT_TEMPLATE.format(question=question, candidates=candidates_block)
        result = call_local_llm(prompt, prompt_version, model=model)
        raw = result["text"]
        chosen = _verify_letter_choice(raw, shortlist)
    else:
        candidates_block = "\n".join(f"- {name}" for name in shortlist)
        prompt = TYPE_LINKER_PROMPT_TEMPLATE.format(question=question, candidates=candidates_block)
        result = call_local_llm(prompt, prompt_version, model=model)
        raw = result["text"]
        chosen = _verify_choice(raw, shortlist)
    concept_ids = graph.find_concepts(chosen) if chosen else set()
    if chosen and not concept_ids:
        chosen = None
        concept_ids = set()
    return {
        "concept_name": chosen,
        "concept_ids": concept_ids,
        "shortlist": shortlist,
        "raw_response": raw,
        "abstained": chosen is None,
        "error": result["error"],
    }


def link_answer_concept_cascade(
    graph: KnowledgeGraph,
    question: str,
    mask_names: list[str],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Selective cascade: symbolic extractor when confident, LLM on the residual.

    Confidence is defined symbolically (no LLM involved): the wh-person rule
    fired, or the chosen concept mention directly follows a wh-word. Those
    signals were the high-precision core of the string extractor. The LLM is
    consulted only on the low-confidence residual (earliest-mention fallback
    or no mention at all); if it abstains, fall back to the string result.
    """
    from rc_mex.run_proof_state_search_smoke import extract_answer_concept

    extracted = extract_answer_concept(graph, question, mask_names)
    string_confident = bool(extracted) and (
        int(extracted["position"]) == -1 or bool(extracted["wh_adjacent"])
    )
    if string_confident:
        return {
            "concept_name": str(extracted["concept_name"]),
            "concept_ids": set(str(cid) for cid in extracted["concept_ids"]),
            "source": "string_confident",
            "raw_response": "",
            "llm_consulted": False,
        }
    linked = link_answer_concept(
        graph, question, mask_names, model=model, prompt_version=TYPE_LINKER_SPAN_PROMPT_VERSION
    )
    if linked["concept_name"]:
        return {
            "concept_name": linked["concept_name"],
            "concept_ids": set(str(cid) for cid in linked["concept_ids"]),
            "source": "llm_span",
            "raw_response": linked["raw_response"],
            "llm_consulted": True,
        }
    if extracted:
        return {
            "concept_name": str(extracted["concept_name"]),
            "concept_ids": set(str(cid) for cid in extracted["concept_ids"]),
            "source": "string_fallback",
            "raw_response": linked["raw_response"],
            "llm_consulted": True,
        }
    return {"concept_name": None, "concept_ids": set(), "source": "none", "raw_response": linked["raw_response"], "llm_consulted": True}


FINAL_ADJUDICATOR_PROMPT_VERSION = "final_adjudicator_v1"
FINAL_ADJUDICATOR_PROMPT_TEMPLATE = """A question was answered by following evidence paths in a knowledge base, starting from "{start}".

Question: "{question}"

Candidate answers with their evidence paths:
{candidates}

Pick the candidate whose evidence path correctly and completely answers the question. A path that stops at an intermediate step does not answer it. Reply with only the number."""

# v2: conservative prior. The break anatomy on train (36 breaks) showed the
# model overriding a candidate-1 whose relation names literally matched the
# question wording (12 same-depth swaps), or detouring onto longer paths when
# a direct edge answered the question (11 cases). Candidate 1 is always the
# current symbolic top, so anchoring on it targets both classes.
FINAL_ADJUDICATOR_PROMPT_TEMPLATES = {
    "final_adjudicator_v1": FINAL_ADJUDICATOR_PROMPT_TEMPLATE,
    "final_adjudicator_v2": """A question was answered by following evidence paths in a knowledge base, starting from "{start}".

Question: "{question}"

Candidate answers with their evidence paths:
{candidates}

Candidate 1 is currently ranked first. Keep candidate 1 unless its evidence path clearly fails to answer the question — for example it stops at an intermediate step, or its relations do not match what the question asks. Prefer the candidate whose relation names match the question wording; do not switch just because another answer sounds more familiar. Reply with only the number.""",
}


def adjudicate_answer_candidates(
    question: str,
    start_entity_name: str,
    candidate_blocks: list[str],
    model: str = DEFAULT_MODEL,
    prompt_version: str = FINAL_ADJUDICATOR_PROMPT_VERSION,
) -> dict[str, Any]:
    """Micro-agent 3 (entity-aware form): one listwise call over the top final
    candidates, each shown as answer name + evidence path. Unlike the
    retention ranker this deliberately exposes entity names, importing the
    model's world knowledge at exactly one bounded decision. Abstention or
    error returns pick=None (caller keeps the symbolic ranking)."""
    if not candidate_blocks:
        return {"pick": None, "raw_response": "", "error": "", "prompt_tokens": 0, "completion_tokens": 0}
    numbered = "\n".join(f"{i}. {block}" for i, block in enumerate(candidate_blocks, start=1))
    prompt = FINAL_ADJUDICATOR_PROMPT_TEMPLATES[prompt_version].format(
        start=start_entity_name, question=question, candidates=numbered
    )
    result = call_local_llm(prompt, prompt_version, model=model, timeout=180.0)
    if result["error"]:
        return {"pick": None, "raw_response": "", "error": result["error"], "prompt_tokens": 0, "completion_tokens": 0}
    picks = parse_pick_numbers(result["text"], len(candidate_blocks), top_k=1)
    return {
        "pick": picks[0] - 1 if picks else None,
        "raw_response": result["text"][:120],
        "error": "",
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


RELATION_PROPOSAL_PROMPT_VERSION = "relation_proposal_v1"
RELATION_PROPOSAL_PROMPT_TEMPLATE = """A question will be answered by following relations in a knowledge base, starting from the entity "{start}".

Question: "{question}"

Candidate relations out of "{start}":
{relations}

Reply with the numbers of up to {k} relations that could lead toward the answer, comma-separated, most promising first."""


def propose_relations(
    question: str,
    start_entity_name: str,
    relation_labels: list[str],
    model: str = DEFAULT_MODEL,
    top_k: int = 10,
) -> dict[str, Any]:
    """Micro-agent 4: pick promising frontier relations from a shortlist.

    Integration contract is union with the symbolic top-K (recall monotone by
    construction); abstention or error returns picks=[] (caller keeps the
    symbolic proposal unchanged)."""
    if not relation_labels:
        return {"picks": [], "raw_response": "", "error": "", "prompt_tokens": 0, "completion_tokens": 0}
    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(relation_labels, start=1))
    prompt = RELATION_PROPOSAL_PROMPT_TEMPLATE.format(
        start=start_entity_name, question=question, relations=numbered, k=top_k
    )
    result = call_local_llm(prompt, RELATION_PROPOSAL_PROMPT_VERSION, model=model, timeout=180.0)
    if result["error"]:
        return {"picks": [], "raw_response": "", "error": result["error"], "prompt_tokens": 0, "completion_tokens": 0}
    picks = parse_pick_numbers(result["text"], len(relation_labels), top_k=top_k)
    return {
        "picks": [pick - 1 for pick in picks],
        "raw_response": result["text"][:120],
        "error": "",
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


TYPED_SELECTOR_PROMPT_VERSION = "typed_selector_v1"
TYPED_SELECTOR_PROMPT_TEMPLATE = """You are identifying the answer to a question by selecting one candidate entity from a knowledge base, starting from "{start}".

Question: "{question}"
{type_line}
Candidates (name [type] — reached by relation):
{candidates}

Pick the candidate whose type and relation match what the question asks for. Reply with only the number."""


def select_answer_typed(
    question: str,
    start_entity_name: str,
    answer_type_hint: str,
    candidate_blocks: list[str],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Re-posed selection: each candidate carries its type and a clean relation,
    and the question's expected answer-type is stated explicitly. Tests whether
    a well-posed selection task lifts the selector's accuracy. Abstain/error
    returns pick=None."""
    if not candidate_blocks:
        return {"pick": None, "raw_response": "", "error": "", "prompt_tokens": 0, "completion_tokens": 0}
    numbered = "\n".join(f"{i}. {block}" for i, block in enumerate(candidate_blocks, start=1))
    type_line = f"The answer should be a: {answer_type_hint}\n" if answer_type_hint else ""
    prompt = TYPED_SELECTOR_PROMPT_TEMPLATE.format(
        start=start_entity_name, question=question, type_line=type_line, candidates=numbered
    )
    result = call_local_llm(prompt, TYPED_SELECTOR_PROMPT_VERSION, model=model, timeout=180.0)
    if result["error"]:
        return {"pick": None, "raw_response": "", "error": result["error"], "prompt_tokens": 0, "completion_tokens": 0}
    picks = parse_pick_numbers(result["text"], len(candidate_blocks), top_k=1)
    return {
        "pick": picks[0] - 1 if picks else None,
        "raw_response": result["text"][:120],
        "error": "",
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


RELATION_DESCRIPTION_PROMPT_VERSION = "relation_description_v2"
RELATION_DESCRIPTION_PROMPT_TEMPLATE = """A question is answered by looking up one property of the entity "{start}" in a knowledge base.

Question: "{question}"

Name the property of "{start}" that gives the answer. Name the property itself, not the answer. Reply with 2 or 3 alternative short property names, comma-separated, most likely first. Example — for "where was he born?": place of birth, birthplace, hometown."""


def describe_target_relation(
    question: str,
    start_entity_name: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Micro-agent for HyDE-style schema linking: the LLM names the property
    that answers the question (semantic intent), WITHOUT seeing the schema.
    The caller embeds the name(s) and matches them against the real relations,
    so the LLM supplies meaning and the embedding supplies scale. v2 asks for
    2-3 alternative names (max-pooled by the caller) to kill single-sample
    phrasing variance on vague questions ("what did X do")."""
    prompt = RELATION_DESCRIPTION_PROMPT_TEMPLATE.format(start=start_entity_name, question=question)
    result = call_local_llm(prompt, RELATION_DESCRIPTION_PROMPT_VERSION, model=model, timeout=120.0)
    first_line = result["text"].strip().splitlines()[0] if result["text"].strip() else ""
    # Small models wrap the answer in prose ('The property ... is "X"'); keep
    # only what follows the final ' is '.
    boilerplate = re.search(r"\bis[:\s]+(.+)$", first_line)
    if boilerplate and len(first_line.split()) > 6:
        first_line = boilerplate.group(1)
    names = [n.strip().strip('".') for n in first_line.split(",")]
    names = [n for n in names if n][:3]
    return {
        "names": names,
        "text": names[0] if names else "",
        "error": result["error"],
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


ANSWER_TYPE_DESCRIPTION_PROMPT_VERSION = "answer_type_description_v2"
ANSWER_TYPE_DESCRIPTION_PROMPT_TEMPLATE = """A question will be answered with an entity from a knowledge base.

Question: "{question}"

What type of entity is the ANSWER (not the person or thing the question is about)? Example — for "where was Barack Obama born?" the answer is a city, so reply "city", not "person". Reply with only a short type name of 1 to 4 words."""


def describe_answer_type(question: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """HyDE-style type linking: the LLM names the expected answer type without
    seeing the KB's type inventory; the caller embeds it and matches against
    the actual types present."""
    prompt = ANSWER_TYPE_DESCRIPTION_PROMPT_TEMPLATE.format(question=question)
    result = call_local_llm(prompt, ANSWER_TYPE_DESCRIPTION_PROMPT_VERSION, model=model, timeout=120.0)
    text = result["text"].strip().strip('".').splitlines()[0] if result["text"].strip() else ""
    return {
        "text": text,
        "error": result["error"],
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


def probe_llm_endpoint(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """One cheap generation to verify the endpoint before a long run.

    Cached like everything else, so a healthy endpoint is probed at most once
    per (model, prompt version). Returns {ok, error, url, api_style, model}."""
    result = call_local_llm("Reply with the single word: ok", "endpoint_probe_v1", model=model, timeout=30.0)
    return {
        "ok": not result["error"],
        "error": result["error"],
        "url": OLLAMA_URL,
        "api_style": LLM_API_STYLE,
        "model": model,
    }


def relation_path_label(evidence_steps: list[dict[str, Any]]) -> str:
    """Relation-only label for a path family, grouped by hop.

    Uses only relation ids and directions — no entities, no benchmark-specific
    vocabulary — so the micro-agent task transfers to any KB."""
    by_hop: dict[int, list[str]] = {}
    for step in evidence_steps or []:
        hop = int(step.get("hop", 0))
        label = f"{str(step.get('relation_id', '')).replace('_', ' ')} [{step.get('direction', '')}]"
        bucket = by_hop.setdefault(hop, [])
        if label not in bucket:
            bucket.append(label)
    return " -> ".join(" & ".join(by_hop[hop]) for hop in sorted(by_hop))


def parse_pick_numbers(raw: str, count: int, top_k: int = 3) -> list[int]:
    """First top_k distinct in-range numbers from the reply (1-based)."""
    picks: list[int] = []
    for token in re.findall(r"\d+", raw):
        value = int(token)
        if 1 <= value <= count and value not in picks:
            picks.append(value)
        if len(picks) == top_k:
            break
    return picks


def rank_relation_paths(
    question: str,
    start_entity_name: str,
    path_labels: list[str],
    model: str = DEFAULT_MODEL,
    top_k: int = 3,
) -> dict[str, Any]:
    """Mode-2 micro-agent: listwise plausibility ranking of relation paths.

    One call per (question, candidate list). Returns 0-based indices of the
    LLM's top_k picks; empty picks on any error or parse failure, which the
    caller must treat as abstention (fall back to symbolic selection)."""
    if not path_labels:
        return {"picks": [], "raw_response": "", "error": "", "consulted": False}
    numbered = "\n".join(f"{i}. {label}" for i, label in enumerate(path_labels, start=1))
    prompt = PATH_RANKER_PROMPT_TEMPLATE.format(
        start=start_entity_name, question=question, paths=numbered, top_k=top_k
    )
    result = call_local_llm(prompt, PATH_RANKER_PROMPT_VERSION, model=model, timeout=180.0)
    if result["error"]:
        return {"picks": [], "raw_response": "", "error": result["error"], "consulted": True, "prompt_tokens": 0, "completion_tokens": 0}
    picks = parse_pick_numbers(result["text"], len(path_labels), top_k=top_k)
    return {
        "picks": [index - 1 for index in picks],
        "raw_response": result["text"][:120],
        "error": "",
        "consulted": True,
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
    }


def _verify_letter_choice(raw_response: str, shortlist: list[str]) -> str | None:
    text = raw_response.strip()
    if not text:
        return None
    first = text[0].upper()
    if first == "Z":
        return None
    index = ord(first) - ord("A")
    if 0 <= index < len(shortlist) and (len(text) == 1 or not text[1].isalpha()):
        return shortlist[index]
    return None


def _resolve_span_to_concept(
    graph: KnowledgeGraph,
    raw_response: str,
    mask_names: list[str],
    question: str = "",
) -> str | None:
    """Map an LLM-quoted span to a KB concept with the proven symbolic machinery.

    The LLM only locates the wh-focus words; concept resolution stays symbolic
    and verifiable. Small models reliably name the focus head noun but drop
    modifiers ("institution" for "higher education institution"), so the span
    is first expanded to the most specific concept mention in the question
    that contains all of the span's tokens."""
    from rc_mex.run_proof_state_search_smoke import question_concept_mentions

    span = raw_response.strip().splitlines()[0].strip().strip('"').strip("'").strip(".") if raw_response.strip() else ""
    normalized = normalize_text(span)
    if normalized.startswith("type words"):
        normalized = normalize_text(normalized[len("type words"):].lstrip(" :"))
    if not normalized or normalized == "none":
        return None
    if len(normalized.split()) > 6:
        return None
    span_tokens = set(normalized.split())
    if question:
        containing = [
            mention for mention in question_concept_mentions(graph, question, mask_names)
            if span_tokens <= set(str(mention["concept_name"]).split())
        ]
        if containing:
            containing.sort(key=lambda m: (-int(m["length"]), int(m["position"])))
            return str(containing[0]["concept_name"])
    direct = graph.find_concepts(normalized)
    if direct:
        return normalized
    mentions = question_concept_mentions(graph, span, mask_names)
    if mentions:
        return str(mentions[0]["concept_name"])
    return None


def _verify_choice(raw_response: str, shortlist: list[str]) -> str | None:
    """Strict verification: the reply must resolve to exactly one shortlist name."""
    text = normalize_text(raw_response.strip().strip('"').strip("'").strip("."))
    if not text or text == "none":
        return None
    by_normalized = {normalize_text(name): name for name in shortlist}
    if text in by_normalized:
        return by_normalized[text]
    first_line = normalize_text(raw_response.strip().splitlines()[0].strip().strip('"').strip("'").strip("."))
    if first_line in by_normalized:
        return by_normalized[first_line]
    # Last resort: exactly one shortlist name appearing verbatim in the reply.
    contained = [name for normalized, name in by_normalized.items() if f" {normalized} " in f" {text} "]
    if len(contained) == 1:
        return contained[0]
    return None
