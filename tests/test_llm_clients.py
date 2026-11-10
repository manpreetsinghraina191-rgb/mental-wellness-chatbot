from collections.abc import Callable
from typing import Any

import pytest

from tai_bot import llm_clients
from tai_bot.llm_clients import LLMChain, QuotaExhausted


def test_chain_switches_model_when_quota_is_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_make_llm(provider: str, model: str | None = None, **_: Any) -> Callable[[str], str]:
        def call(prompt: str) -> str:
            if model == "first":
                raise QuotaExhausted("daily quota")
            return f"{provider}:{model} says {prompt}"

        return call

    monkeypatch.setattr(llm_clients, "make_llm", fake_make_llm)
    logs: list[str] = []
    chain = LLMChain(["gemini:first", "groq:second"], log=logs.append)
    assert chain.current == "gemini:first"
    assert chain("hi") == "groq:second says hi"
    assert chain.current == "groq:second"
    assert any("switching" in m for m in logs)


def test_chain_raises_when_all_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_exhausted(*_: Any, **__: Any) -> Callable[[str], str]:
        def call(prompt: str) -> str:
            raise QuotaExhausted("gone")

        return call

    monkeypatch.setattr(llm_clients, "make_llm", always_exhausted)
    with pytest.raises(QuotaExhausted):
        LLMChain(["gemini:a", "groq:b"], log=lambda _: None)("hi")
