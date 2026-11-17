"""Provider adapters and retry logic, with httpx.post replaced by canned responses."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tai_bot import llm_clients
from tai_bot.llm_clients import (
    EmptyResponse,
    ProviderFailed,
    QuotaExhausted,
    RateLimited,
    make_llm,
)


def _response(
    status: int, body: Any = None, headers: dict[str, str] | None = None
) -> httpx.Response:
    req = httpx.Request("POST", "https://example.test")
    if isinstance(body, str):
        return httpx.Response(status, text=body, headers=headers, request=req)
    return httpx.Response(status, json=body or {}, headers=headers, request=req)


class FakeHTTP:
    def __init__(self) -> None:
        self.queue: list[httpx.Response] = []  # responses to return, in order
        self.sent: list[dict[str, Any]] = []  # request bodies, in order


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> FakeHTTP:
    fake = FakeHTTP()

    def fake_post(url: str, json: dict[str, Any], **_: Any) -> httpx.Response:
        fake.sent.append(json)
        return fake.queue.pop(0)

    monkeypatch.setattr(llm_clients.httpx, "post", fake_post)
    monkeypatch.setattr(llm_clients.time, "sleep", lambda _: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    return fake


def test_parse_seconds() -> None:
    assert llm_clients._parse_seconds("12.5s", 30) == 12.5
    assert llm_clients._parse_seconds(None, 30) == 30
    assert llm_clients._parse_seconds("soon", 7) == 7


def test_gemini_joins_text_and_skips_thoughts(http: FakeHTTP) -> None:
    parts = [{"text": "plan", "thought": True}, {"text": "Hello "}, {"text": "there"}]
    http.queue.append(_response(200, {"candidates": [{"content": {"parts": parts}}]}))
    assert llm_clients._gemini("hi", "gemini-x", True, 0.1, 5) == "Hello there"
    assert http.sent[0]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_no_json_mode_for_gemma(http: FakeHTTP) -> None:
    http.queue.append(_response(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}))
    llm_clients._gemini("hi", "gemma-4", True, 0.1, 5)
    assert "responseMimeType" not in http.sent[0]["generationConfig"]


def test_gemini_errors(http: FakeHTTP) -> None:
    q = http.queue
    q.append(_response(429, {"error": {"message": "Quota exceeded: PerDay limit"}}))
    with pytest.raises(QuotaExhausted):
        llm_clients._gemini("hi", "g", False, 0.1, 5)
    q.append(_response(429, {"error": {"details": [{"retryDelay": "12s"}], "message": "slow"}}))
    with pytest.raises(RateLimited) as e:
        llm_clients._gemini("hi", "g", False, 0.1, 5)
    assert e.value.retry_after == 12
    q.append(_response(200, {"candidates": [{"finishReason": "SAFETY"}]}))
    with pytest.raises(EmptyResponse, match="SAFETY"):
        llm_clients._gemini("hi", "g", False, 0.1, 5)


def test_groq_success_and_limits(http: FakeHTTP) -> None:
    q = http.queue
    q.append(_response(200, {"choices": [{"message": {"content": "namaste"}}]}))
    assert llm_clients._groq("hi", "m", True, 0.1, 5) == "namaste"
    assert http.sent[0]["response_format"] == {"type": "json_object"}
    q.append(_response(429, "Rate limit reached: tokens per day"))
    with pytest.raises(QuotaExhausted):
        llm_clients._groq("hi", "m", False, 0.1, 5)
    q.append(_response(429, "slow down", headers={"retry-after": "3"}))
    with pytest.raises(RateLimited) as e:
        llm_clients._groq("hi", "m", False, 0.1, 5)
    assert e.value.retry_after == 3


def test_ollama_json_mode(http: FakeHTTP) -> None:
    http.queue.append(_response(200, {"message": {"content": '{"risk": "none"}'}}))
    assert llm_clients._ollama("hi", "qwen", True, 0.0, 5) == '{"risk": "none"}'
    assert http.sent[0]["format"] == "json" and http.sent[0]["think"] is False


def test_make_llm_retries_then_succeeds(http: FakeHTTP) -> None:
    q = http.queue
    q += [_response(503), _response(200, {"message": {"content": "ok"}})]
    waits: list[str] = []
    assert make_llm("ollama", "qwen", on_wait=waits.append)("hi") == "ok"
    assert "HTTP 503" in waits[0]


def test_make_llm_raises_client_errors_and_gives_up(http: FakeHTTP) -> None:
    q = http.queue
    q.append(_response(400))
    with pytest.raises(httpx.HTTPStatusError):
        make_llm("ollama", "qwen")("hi")
    q += [_response(500), _response(500)]
    with pytest.raises(ProviderFailed):
        make_llm("ollama", "qwen", max_retries=2)("hi")


def test_make_llm_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm("openai")
