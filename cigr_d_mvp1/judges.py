from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .witness import WitnessCard


@dataclass
class RankingResult:
    ranked_candidate_ids: list[str]
    raw_output: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    latency_seconds: float


class Ranker(Protocol):
    name: str

    def rank(self, question: str, cards: list[WitnessCard], evidence: str) -> RankingResult:
        ...


class RandomRanker:
    name = "random"

    def __init__(self, seed: int):
        self.seed = seed

    def rank(self, question: str, cards: list[WitnessCard], evidence: str) -> RankingResult:
        rng = random.Random(f"{self.seed}:{question}:{len(cards)}:{evidence}")
        ids = [card.candidate_id for card in cards]
        rng.shuffle(ids)
        return RankingResult(ids, "", 0, 0, 0.0)


class HashEmbeddingRanker:
    """Dependency-free char n-gram cosine baseline.

    This is a small local stand-in for an embedding baseline when no embedding
    packages are installed. It is deterministic and intentionally simple.
    """

    name = "embedding_schema"

    def rank(self, question: str, cards: list[WitnessCard], evidence: str) -> RankingResult:
        q_vec = char_ngram_vector(question)
        scored = []
        for card in cards:
            text = card.schema_text() if evidence == "schema" else card.witness_text()
            scored.append((cosine(q_vec, char_ngram_vector(text)), card.candidate_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return RankingResult([cid for _, cid in scored], "", 0, 0, 0.0)


class LLMListwiseRanker:
    def __init__(self, name: str, client: "LLMClient", max_retries: int = 2):
        self.name = name
        self.client = client
        self.max_retries = max_retries

    def rank(self, question: str, cards: list[WitnessCard], evidence: str) -> RankingResult:
        prompt = build_listwise_prompt(question, cards, evidence)
        start = time.time()
        last_output = ""
        for _ in range(self.max_retries + 1):
            last_output = self.client.complete(prompt)
            ranked = parse_ranked_ids(last_output, {card.candidate_id for card in cards})
            if ranked:
                ranked = ranked + [card.candidate_id for card in cards if card.candidate_id not in ranked]
                return RankingResult(
                    ranked_candidate_ids=ranked,
                    raw_output=last_output,
                    prompt_tokens_estimate=estimate_tokens(prompt),
                    completion_tokens_estimate=estimate_tokens(last_output),
                    latency_seconds=time.time() - start,
                )
        return RankingResult(
            ranked_candidate_ids=[card.candidate_id for card in cards],
            raw_output=last_output,
            prompt_tokens_estimate=estimate_tokens(prompt),
            completion_tokens_estimate=estimate_tokens(last_output),
            latency_seconds=time.time() - start,
        )


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class MockLLMClient:
    """Deterministic smoke-test client; not valid for real experiments."""

    def complete(self, prompt: str) -> str:
        question_match = re.search(r"QUESTION:\n(.+?)\n\nCANDIDATES:", prompt, re.S)
        question = question_match.group(1) if question_match else prompt
        cards = re.findall(r"^(C\d{3})\.\s*(.+)$", prompt, flags=re.M)
        q_vec = char_ngram_vector(question)
        scored = []
        for cid, text in cards:
            scored.append((cosine(q_vec, char_ngram_vector(text)), cid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return json.dumps({"ranking": [cid for _, cid in scored]})


class OllamaClient:
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        data = http_json(f"{self.host}/api/chat", payload)
        return data.get("message", {}).get("content", "")


class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LOCAL_LLM_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_LLM_API_KEY")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = http_json(f"{self.base_url}/chat/completions", payload, headers=headers)
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


class CommandClient:
    def __init__(self, command: str):
        self.command = command

    def complete(self, prompt: str) -> str:
        completed = subprocess.run(
            self.command,
            input=prompt,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"command failed: {self.command}")
        return completed.stdout


def build_listwise_prompt(question: str, cards: list[WitnessCard], evidence: str) -> str:
    if evidence not in {"schema", "witness"}:
        raise ValueError(f"unknown evidence mode: {evidence}")
    candidate_text = "\n".join(
        card.schema_text() if evidence == "schema" else card.witness_text()
        for card in cards
    )
    return (
        "You are ranking candidate knowledge-graph relation/direction choices.\n"
        "Choose the candidates that best match the question's intended next graph step.\n"
        "Use only the evidence shown. Return JSON exactly as: {\"ranking\": [\"C001\", \"C002\"]}.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATES:\n{candidate_text}\n\n"
        "Rank all candidate IDs from best to worst."
    )


def parse_ranked_ids(output: str, valid_ids: set[str]) -> list[str]:
    try:
        data = json.loads(output)
        ranking = data.get("ranking", [])
        if isinstance(ranking, list):
            return [str(cid) for cid in ranking if str(cid) in valid_ids]
    except json.JSONDecodeError:
        pass
    ids = re.findall(r"C\d{3}", output)
    out = []
    for cid in ids:
        if cid in valid_ids and cid not in out:
            out.append(cid)
    return out


def http_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def char_ngram_vector(text: str, n: int = 3) -> dict[str, int]:
    normalized = f"  {text.casefold()}  "
    if len(normalized) < n:
        return {normalized: 1}
    counts: dict[str, int] = {}
    for idx in range(len(normalized) - n + 1):
        gram = normalized[idx : idx + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def cosine(left: dict[str, int], right: dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
