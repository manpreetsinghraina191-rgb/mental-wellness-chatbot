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


def test_trim_to_sentence_drops_cut_off_tail() -> None:
    from tai_bot.llm_clients import trim_to_sentence

    assert trim_to_sentence("Aap try karein. Kya aapko lagta hai ki yeh") == "Aap try karein."
    assert trim_to_sentence("यह सामान्य है। क्या आप") == "यह सामान्य है।"
    assert trim_to_sentence("Are you okay? I was") == "Are you okay?"
    # a list marker is not a sentence end
    assert trim_to_sentence("Try this.\n1. Sit down.\n2.") == "Try this.\n1. Sit down."
    # nothing complete: keep the text rather than return an empty reply
    assert trim_to_sentence("no full stop here") == "no full stop here"


def test_chat_trims_only_when_cut_off(monkeypatch: Any) -> None:
    from tai_bot import llm_clients

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> Any:
        class R:
            def raise_for_status(self) -> None: ...

            def json(self) -> dict[str, Any]:
                return {"message": {"content": reply}, "done_reason": reason}

        return R()

    monkeypatch.setattr(llm_clients.httpx, "post", fake_post)
    chat = llm_clients.make_chat_llm("x")
    reply, reason = "One. Two thr", "length"
    assert chat([]) == "One."
    reply, reason = "One. Two thr", "stop"
    assert chat([]) == "One. Two thr"
