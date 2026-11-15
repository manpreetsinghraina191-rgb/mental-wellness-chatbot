"""Retrieval over the bilingual knowledge base (step 5).

  * Each knowledge card is stored twice, in English and Hindi, as separate vectors.
  * Embeddings: BGE-M3 (dense), normalized, in a FAISS IndexFlatIP (= cosine similarity).
  * Every query variant (original text + Devanagari transliteration for Hinglish) is searched,
    hits are merged per card (best score wins), optionally reranked with bge-reranker-v2-m3,
    and only cards above a similarity threshold are returned (no context beats wrong context).
  * The card text is returned in the language closest to the reply: Hindi for Devanagari or
    mixed replies, English otherwise (the generator writes Hinglish from English facts).

The heavy libraries (torch, sentence-transformers, faiss) are imported lazily, so the rest of
the app and the tests run without them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def __call__(self, query: str, passages: list[str]) -> list[float]: ...


def load_bge_m3(device: str = "cpu") -> Embedder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device=device)

    def embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=16)
        return np.asarray(vecs, dtype="float32")

    return embed


def load_reranker(device: str = "cpu") -> Reranker:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANK_MODEL, device=device)

    def rerank(query: str, passages: list[str]) -> list[float]:
        scores = model.predict([(query, p) for p in passages], activation_fn=_sigmoid)
        return [float(s) for s in scores]

    return rerank


def _sigmoid(x: Any) -> Any:
    import torch

    return torch.sigmoid(x)


@dataclass
class Hit:
    card_id: str
    score: float  # cosine similarity (best over query variants and languages)
    rerank: float | None
    text: str
    source: str
    url: str


class Retriever:
    def __init__(
        self,
        cards: list[dict[str, Any]],
        embed: Embedder,
        reranker: Reranker | None = None,
        index: Any = None,
        rows: list[dict[str, str]] | None = None,
    ) -> None:
        import faiss

        self.cards = {c["id"]: c for c in cards}
        self.embed = embed
        self.reranker = reranker
        if index is None or rows is None:
            rows = [
                {"card_id": c["id"], "lang": lang, "text": c[lang]}
                for c in cards
                for lang in ("en", "hi")
                if c.get(lang)
            ]
            vecs = embed([r["text"] for r in rows])
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
        self.index = index
        self.rows = rows

    # -- persistence -------------------------------------------------------------------------

    def save(self, directory: Path = KB_DIR) -> None:
        import faiss

        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(
        cls, embed: Embedder, reranker: Reranker | None = None, directory: Path = KB_DIR
    ) -> Retriever:
        import faiss

        cards = json.loads((directory / "knowledge_base.json").read_text(encoding="utf-8"))
        rows = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(directory / "index.faiss"))
        return cls(cards, embed, reranker, index, rows)

    # -- search ------------------------------------------------------------------------------

    def candidates(self, queries: list[str], k: int = 8) -> list[tuple[str, float]]:
        """Merge FAISS hits for all query variants; one score per card (the best)."""
        qv = self.embed(queries)
        scores, ids = self.index.search(qv, min(k, self.index.ntotal))
        best: dict[str, float] = {}
        for row_scores, row_ids in zip(scores, ids, strict=True):
            for s, i in zip(row_scores, row_ids, strict=True):
                if i < 0:
                    continue
                cid = self.rows[i]["card_id"]
                best[cid] = max(best.get(cid, -1.0), float(s))
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def search(
        self,
        queries: list[str],
        reply_lang: str = "en",
        top_k: int = 3,
        min_score: float = 0.45,
        min_rerank: float = 0.0,
        use_reranker: bool = True,
    ) -> list[Hit]:
        """`min_score` (dense cosine) decides relevance. The reranker only reorders by default:
        measured on the eval set, its sigmoid scores on emotional, statement-style queries are
        tiny (relevant median 0.048 vs irrelevant 0.000), so a 0.2 cut-off removed almost every
        correct card (Recall@3 84% -> 42%)."""
        cands = [(cid, s) for cid, s in self.candidates(queries) if s >= min_score]
        rerank_scores: dict[str, float] = {}
        if use_reranker and self.reranker is not None and cands:
            passages = [self.cards[cid]["en"] for cid, _ in cands]
            scores = self.reranker(queries[0], passages)
            rerank_scores = dict(zip([cid for cid, _ in cands], scores, strict=True))
            cands = [(cid, s) for cid, s in cands if rerank_scores[cid] >= min_rerank]
            cands.sort(key=lambda kv: rerank_scores[kv[0]], reverse=True)
        hits = []
        for cid, s in cands[:top_k]:
            card = self.cards[cid]
            text = card.get(reply_lang) or card["en"]
            hits.append(Hit(cid, s, rerank_scores.get(cid), text, card["source"], card["url"]))
        return hits


def reply_lang_for(script: str) -> str:
    return "hi" if script in ("devanagari", "mixed") else "en"
