"""Retrieval logic with a tiny fake embedder (no model download; runs in CI if faiss exists)."""

import json
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from tai_bot.retrieval import Retriever, reply_lang_for  # noqa: E402

CARDS = [
    {"id": "sleep", "en": "sleep tips bed", "hi": "नींद के उपाय", "source": "NHS", "url": "u1"},
    {"id": "breath", "en": "breathing calm", "hi": "साँस शांत", "source": "NHS", "url": "u2"},
    {"id": "tele", "en": "helpline tele-manas", "hi": "हेल्पलाइन", "source": "MoHFW", "url": "u3"},
]
VOCAB = {
    "sleep": 0, "bed": 0, "tips": 0, "नींद": 0, "neend": 0, "के": 0, "उपाय": 0,
    "breathing": 1, "calm": 1, "साँस": 1,
    "helpline": 2, "tele-manas": 2, "हेल्पलाइन": 2,
}  # fmt: skip


def fake_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), 3), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            if w in VOCAB:  # unknown words contribute nothing
                out[i, VOCAB[w]] += 1
        out[i] /= np.linalg.norm(out[i]) or 1
    return out


def test_search_finds_card_and_returns_reply_language() -> None:
    r = Retriever(CARDS, fake_embed)
    hits = r.search(["how to sleep in bed"], reply_lang="hi", use_reranker=False)
    assert hits[0].card_id == "sleep"
    assert hits[0].text == "नींद के उपाय"
    assert hits[0].source == "NHS"


def test_threshold_returns_nothing_for_unrelated_query() -> None:
    r = Retriever(CARDS, fake_embed)
    assert r.search(["who won the cricket match"], use_reranker=False) == []


def test_multiple_query_variants_are_merged() -> None:
    r = Retriever(CARDS, fake_embed)
    # the Roman query alone is unknown to the embedder; the transliterated copy finds the card
    hits = r.search(["xyz", "नींद"], use_reranker=False)
    assert [h.card_id for h in hits][:1] == ["sleep"]


def test_reranker_reorders_and_filters() -> None:
    def rerank(query: str, passages: list[str]) -> list[float]:
        return [0.9 if "helpline" in p else 0.05 for p in passages]

    r = Retriever(CARDS, fake_embed, reranker=rerank)
    hits = r.search(["helpline sleep"], min_score=0.1, min_rerank=0.2)
    assert [h.card_id for h in hits] == ["tele"]  # sleep filtered out by the reranker


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "knowledge_base.json").write_text(json.dumps(CARDS), encoding="utf-8")
    Retriever(CARDS, fake_embed).save(tmp_path)
    loaded = Retriever.load(fake_embed, directory=tmp_path)
    assert loaded.search(["breathing"], use_reranker=False)[0].card_id == "breath"


def test_reply_lang() -> None:
    assert reply_lang_for("devanagari") == "hi"
    assert reply_lang_for("mixed") == "hi"
    assert reply_lang_for("roman_hindi") == "en"
