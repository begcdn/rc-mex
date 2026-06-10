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
#   export RC_MEX_LLM_URL=http://<server>:11434/api/generate
#   export RC_MEX_LLM_MODEL=qwen3:8b
OLLAMA_URL = os.environ.get("RC_MEX_LLM_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("RC_MEX_LLM_MODEL", "llama3.2:3b")
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
) -> dict[str, Any]:
    """Cached, deterministic single-shot generation. Returns {text, from_cache, error}."""
    cache = _load_cache()
    key = _cache_key(prompt_version, model, prompt)
    if key in cache:
        return {"text": cache[key], "from_cache": True, "error": ""}
    request_body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 7, "num_predict": 64},
    }
    if "qwen" in model.lower():
        # Disable qwen3 thinking mode; strict verifiers expect the bare answer.
        request_body["think"] = False
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = THINK_BLOCK_PATTERN.sub("", str(body.get("response", ""))).strip()
    except Exception as exc:
        return {"text": "", "from_cache": False, "error": f"{type(exc).__name__}: {exc}"}
    cache[key] = text
    _save_cache()
    return {"text": text, "from_cache": False, "error": ""}


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
        chosen = _resolve_span_to_concept(graph, raw, mask_names)
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


def _resolve_span_to_concept(graph: KnowledgeGraph, raw_response: str, mask_names: list[str]) -> str | None:
    """Map an LLM-quoted span to a KB concept with the proven symbolic machinery.

    The LLM only locates the wh-focus words; concept resolution (exact lookup,
    aliases, county rewrite) stays symbolic and verifiable."""
    from rc_mex.run_proof_state_search_smoke import question_concept_mentions

    span = raw_response.strip().splitlines()[0].strip().strip('"').strip("'").strip(".") if raw_response.strip() else ""
    normalized = normalize_text(span)
    if not normalized or normalized == "none":
        return None
    if len(normalized.split()) > 6:
        return None
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
