"""Minimal, provider-agnostic text completion over plain HTTP (no vendor SDKs).

Providers:
  gemini  - Google AI Studio free tier   (GEMINI_API_KEY)
  groq    - Groq free tier, OpenAI-style (GROQ_API_KEY)
  ollama  - local Ollama server           (no key)

Only used for offline jobs (translating public datasets) and local inference.
User conversations must only ever go to the local `ollama` provider.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import Any

import httpx

DEFAULT_MODELS = {
    # "gemini-flash-latest" and gemini-3.8-flash hung on the free tier (tested 2026-11-10).
    "gemini": "gemini-3.5-flash",
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen3.5:4b",
}


class RateLimited(Exception):
    """Per-minute limit hit; safe to wait and retry."""

    def __init__(self, retry_after: float, message: str = "") -> None:
        super().__init__(message or f"rate limited, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


class EmptyResponse(Exception):
    """The provider answered but returned no text (e.g. blocked by a safety filter or cut off)."""


class QuotaExhausted(Exception):
    """Daily / hard quota hit; retrying today will not help. Switch provider."""


def _parse_seconds(value: str | None, default: float) -> float:
    if not value:
        return default
    m = re.match(r"([\d.]+)", value)
    return float(m.group(1)) if m else default


def _gemini(prompt: str, model: str, json_mode: bool, temperature: float, timeout: float) -> str:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    config: dict[str, Any] = {"temperature": temperature}
    if json_mode and not model.startswith("gemma"):  # Gemma returns HTTP 500 with JSON mode
        config["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    r = httpx.post(url, json=body, headers={"x-goog-api-key": key}, timeout=timeout)
    if r.status_code == 429:
        err = r.json().get("error", {})
        details = err.get("details", [])
        text = str(details) + err.get("message", "")
        if "PerDay" in text or "per day" in text.lower():
            raise QuotaExhausted(f"Gemini daily quota reached for {model}")
        delay = next(
            (_parse_seconds(d.get("retryDelay"), 30) for d in details if "retryDelay" in d), 30.0
        )
        raise RateLimited(delay)
    r.raise_for_status()
    data = r.json()
    candidate = (data.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts")
    if not parts:
        reason = candidate.get("finishReason") or data.get("promptFeedback", {}).get("blockReason")
        raise EmptyResponse(f"{model}: empty response (reason: {reason})")
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _groq(prompt: str, model: str, json_mode: bool, temperature: float, timeout: float) -> str:
    key = os.environ["GROQ_API_KEY"]
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    if r.status_code == 429:
        msg = r.text.lower()
        if "per day" in msg:
            raise QuotaExhausted(f"Groq daily quota reached for {model}")
        raise RateLimited(_parse_seconds(r.headers.get("retry-after"), 20))
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"]["content"])


def _ollama(prompt: str, model: str, json_mode: bool, temperature: float, timeout: float) -> str:
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        body["format"] = "json"
    r = httpx.post(f"{base}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return str(r.json()["message"]["content"])


_PROVIDERS = {"gemini": _gemini, "groq": _groq, "ollama": _ollama}


class ProviderFailed(RuntimeError):
    """A provider kept failing (timeouts / 5xx) after all retries."""


def make_llm(
    provider: str,
    model: str | None = None,
    json_mode: bool = False,
    temperature: float = 0.3,
    timeout: float = 120.0,
    max_retries: int = 6,
    on_wait: Callable[[str], None] | None = None,
) -> Callable[[str], str]:
    """Return `prompt -> text`. Waits and retries on rate limits, timeouts and 5xx;
    raises QuotaExhausted (daily quota gone) or ProviderFailed (still failing after retries)."""
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(_PROVIDERS)}")
    fn = _PROVIDERS[provider]
    name = model or os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    def call(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                return fn(prompt, name, json_mode, temperature, timeout)
            except RateLimited as e:
                wait, why = min(e.retry_after + 1, 120), "rate limited"
            except (httpx.TimeoutException, httpx.TransportError) as e:
                wait, why = 10.0 * (attempt + 1), f"network/timeout ({type(e).__name__})"
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                wait, why = 10.0 * (attempt + 1), f"server busy (HTTP {e.response.status_code})"
            if on_wait:
                on_wait(f"{provider}: {why}, waiting {wait:.0f}s")
            time.sleep(wait)
        raise ProviderFailed(f"{provider}/{name}: still failing after {max_retries} retries")

    return call


class LLMChain:
    """Try `provider:model` specs in order. Once one hits its daily quota or keeps failing,
    switch to the next for the rest of the run (good for unattended jobs on free tiers).
    `current` names the model in use, so outputs can record which model produced them."""

    def __init__(self, specs: list[str], log: Callable[[str], None] = print, **kwargs: Any) -> None:
        self.specs = specs
        self.log = log
        self.i = 0
        self._calls = []
        for spec in specs:
            provider, _, model = spec.partition(":")
            self._calls.append(make_llm(provider, model or None, on_wait=log, **kwargs))

    @property
    def current(self) -> str:
        return self.specs[min(self.i, len(self.specs) - 1)]

    def __call__(self, prompt: str) -> str:
        while self.i < len(self._calls):
            try:
                return self._calls[self.i](prompt)
            except (QuotaExhausted, ProviderFailed) as e:
                self.log(f"{e} -> switching to the next model")
                self.i += 1
        raise QuotaExhausted(f"all models exhausted for today: {self.specs}")


def make_llm_chain(specs: list[str], log: Callable[[str], None] = print, **kwargs: Any) -> LLMChain:
    return LLMChain(specs, log=log, **kwargs)


# A sentence end: . ! ? or the Devanagari danda, optionally followed by a closing quote or
# bracket. A "." right after a digit is skipped, so a list marker ("4.") never ends the text.
_SENTENCE_END = re.compile(r"(?:(?<!\d)\.|[!?।॥])[\"')\]]*(?=\s|$)")


def trim_to_sentence(text: str) -> str:
    """Drop the unfinished tail of a reply that was cut off by the token limit. If no complete
    sentence exists, the text is returned unchanged rather than emptied."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()].rstrip() if ends else text


ChatMessage = dict[str, str]
ChatLLM = Callable[[list[ChatMessage]], str]


def make_chat_llm(
    model: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 320,
    timeout: float = 180.0,
) -> ChatLLM:
    """Chat completion on the LOCAL Ollama server (the only provider allowed to see user text).
    `messages` are {"role": "system" | "user" | "assistant", "content": ...}."""
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    name = model or os.getenv("CHAT_MODEL") or DEFAULT_MODELS["ollama"]

    def chat(messages: list[ChatMessage]) -> str:
        body: dict[str, Any] = {
            "model": name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        r = httpx.post(f"{base}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = str(data["message"]["content"]).strip()
        if data.get("done_reason") == "length":  # hit the token cap mid-sentence
            text = trim_to_sentence(text)
        return text

    return chat
