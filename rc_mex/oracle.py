from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from cigr_d_mvp1.judges import CommandClient, OllamaClient, OpenAICompatibleClient, estimate_tokens

from .cards import GeneratedCardText, RelationCard


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class PairPrediction:
    satisfies: bool
    direction_correct: bool
    confidence: float
    raw_output: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    latency_seconds: float


class MockRCMexClient:
    """Deterministic smoke-test client; not valid for real experiments."""

    def complete(self, prompt: str) -> str:
        if "CARD GENERATION TASK" in prompt:
            relation = extract_field(prompt, "Visible relation")
            domain = extract_field(prompt, "Domain types") or "entity"
            range_ = extract_field(prompt, "Range types") or "entity"
            opaque = relation.startswith("R_") and "ENTITY_" in prompt and "Domain types:\n<hidden>" in prompt
            description = f"{relation} relation from {domain} to {range_}"
            if opaque:
                description = "opaque relation between anonymous entities"
            return json.dumps(
                {
                    "predicate_description": description,
                    "argument_1_role": domain,
                    "argument_2_role": range_,
                    "domain": domain,
                    "range": range_,
                    "direction": extract_field(prompt, "Direction") or "unknown",
                    "confidence": 0.25 if opaque else 0.75,
                    "opaque": opaque,
                    "opaque_reason": "insufficient semantic evidence" if opaque else "",
                }
            )
        if "PAIR CLASSIFICATION TASK" in prompt:
            expected = extract_field(prompt, "Expected label for mock")
            is_positive = expected == "true"
            return json.dumps(
                {
                    "satisfies": is_positive,
                    "direction_correct": is_positive,
                    "confidence": 0.95,
                }
            )
        return "{}"


class CardGenerator:
    def __init__(self, client: LLMClient, max_retries: int = 2):
        self.client = client
        self.max_retries = max_retries

    def generate(self, prompt_payload: dict[str, Any]) -> GeneratedCardText:
        prompt = build_card_generation_prompt(prompt_payload)
        start = time.time()
        last_output = ""
        for _ in range(self.max_retries + 1):
            last_output = self.client.complete(prompt)
            data = parse_json_object(last_output)
            if data:
                return generated_card_from_json(data, last_output)
        return GeneratedCardText(
            predicate_description="",
            argument_1_role="",
            argument_2_role="",
            domain="",
            range="",
            direction="",
            confidence=0.0,
            opaque=True,
            opaque_reason="card generator did not return parseable JSON",
            raw_output=last_output,
        )


class PairClassifier:
    def __init__(self, client: LLMClient, include_mock_label: bool = False, max_retries: int = 2):
        self.client = client
        self.include_mock_label = include_mock_label
        self.max_retries = max_retries

    def classify(self, card: RelationCard, pair: dict[str, Any], expected_label: bool | None = None) -> PairPrediction:
        prompt = build_pair_classification_prompt(
            card=card,
            pair=pair,
            expected_label=expected_label if self.include_mock_label else None,
        )
        start = time.time()
        last_output = ""
        for _ in range(self.max_retries + 1):
            last_output = self.client.complete(prompt)
            data = parse_json_object(last_output)
            if data and "satisfies" in data:
                return PairPrediction(
                    satisfies=parse_bool(data.get("satisfies")),
                    direction_correct=parse_bool(data.get("direction_correct", data.get("satisfies"))),
                    confidence=parse_confidence(data.get("confidence")),
                    raw_output=last_output,
                    prompt_tokens_estimate=estimate_tokens(prompt),
                    completion_tokens_estimate=estimate_tokens(last_output),
                    latency_seconds=time.time() - start,
                )
        return PairPrediction(
            satisfies=False,
            direction_correct=False,
            confidence=0.0,
            raw_output=last_output,
            prompt_tokens_estimate=estimate_tokens(prompt),
            completion_tokens_estimate=estimate_tokens(last_output),
            latency_seconds=time.time() - start,
        )


OPENAI_API_BASE_URL = "https://api.openai.com/v1"


def make_client(backend: str, model: str, ollama_host: str | None, openai_base_url: str | None, openai_api_key: str | None, command: str | None) -> LLMClient:
    if backend == "mock":
        return MockRCMexClient()
    if backend == "ollama":
        return OllamaClient(model=model, host=ollama_host)
    if backend == "openai":
        return OpenAICompatibleClient(
            model=model,
            base_url=OPENAI_API_BASE_URL,
            api_key=openai_api_key,
        )
    if backend == "openai-compatible":
        return OpenAICompatibleClient(model=model, base_url=openai_base_url, api_key=openai_api_key)
    if backend == "command":
        if not command:
            raise SystemExit("--oracle-command is required for command backend")
        return CommandClient(command)
    raise SystemExit(f"unknown oracle backend: {backend}")


def build_card_generation_prompt(payload: dict[str, Any]) -> str:
    return (
        "CARD GENERATION TASK\n"
        "You are creating a reusable semantic card for one knowledge-graph primitive.\n"
        "Infer the shortest binary predicate p(x,y) that is true for positives and false for hard negatives.\n"
        "Use only the relation display, examples, and type evidence shown. If evidence is incoherent, mark opaque=true.\n"
        "Return only JSON with keys: predicate_description, argument_1_role, argument_2_role, domain, range, direction, confidence, opaque, opaque_reason.\n"
        "Use a numeric confidence from 0.0 to 1.0.\n\n"
        f"Card variant:\n{payload['card_variant']}\n\n"
        f"Visible relation:\n{payload['visible_relation']}\n\n"
        f"Direction:\n{payload['direction']}\n\n"
        f"Domain types:\n{format_list(payload['domain_types'])}\n\n"
        f"Range types:\n{format_list(payload['range_types'])}\n\n"
        f"Positive examples:\n{json.dumps(payload['positive_examples'], ensure_ascii=False, indent=2)}\n\n"
        f"Negative examples:\n{json.dumps(payload['negative_examples'], ensure_ascii=False, indent=2)}\n"
    )


def build_pair_classification_prompt(
    card: RelationCard,
    pair: dict[str, Any],
    expected_label: bool | None,
) -> str:
    mock_line = ""
    if expected_label is not None:
        mock_line = f"\nExpected label for mock:\n{str(expected_label).lower()}\n"
    return (
        "PAIR CLASSIFICATION TASK\n"
        "Given a frozen relation card and one ordered pair (x,y), decide whether the pair satisfies the card predicate.\n"
        "The order matters. Return only JSON with keys: satisfies, direction_correct, confidence.\n"
        "Use booleans for satisfies/direction_correct and a numeric confidence from 0.0 to 1.0.\n\n"
        f"Card predicate:\n{card.description}\n\n"
        f"Argument 1 role:\n{card.generated.get('argument_1_role', '')}\n\n"
        f"Argument 2 role:\n{card.generated.get('argument_2_role', '')}\n\n"
        f"Domain types:\n{format_list(card.domain_types)}\n\n"
        f"Range types:\n{format_list(card.range_types)}\n\n"
        f"Pair:\n{json.dumps(pair, ensure_ascii=False, indent=2)}\n"
        f"{mock_line}"
    )


def generated_card_from_json(data: dict[str, Any], raw_output: str) -> GeneratedCardText:
    opaque = parse_bool(data.get("opaque", False))
    opaque_reason = str(data.get("opaque_reason", "") or "")
    if opaque and not opaque_reason:
        opaque_reason = "generator marked card opaque"
    return GeneratedCardText(
        predicate_description=str(data.get("predicate_description", "") or ""),
        argument_1_role=str(data.get("argument_1_role", "") or ""),
        argument_2_role=str(data.get("argument_2_role", "") or ""),
        domain=str(data.get("domain", "") or ""),
        range=str(data.get("range", "") or ""),
        direction=str(data.get("direction", "") or ""),
        confidence=parse_confidence(data.get("confidence")),
        opaque=opaque,
        opaque_reason=opaque_reason,
        raw_output=raw_output,
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0", ""}:
            return False
    return bool(value)


def parse_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return clamp_confidence(float(value))
    if isinstance(value, str):
        normalized = value.strip().casefold()
        labels = {
            "none": 0.0,
            "very low": 0.1,
            "low": 0.25,
            "medium-low": 0.4,
            "medium": 0.5,
            "moderate": 0.5,
            "medium-high": 0.7,
            "high": 0.85,
            "very high": 0.95,
        }
        if normalized in labels:
            return labels[normalized]
        match = re.search(r"-?\d+(?:\.\d+)?", normalized)
        if match:
            number = float(match.group(0))
            if "%" in normalized:
                number /= 100.0
            return clamp_confidence(number)
    return 0.0


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def extract_field(prompt: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\n(.+?)(?:\n\n|$)", prompt, flags=re.S)
    if not match:
        return ""
    return match.group(1).strip()


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "<hidden>"
