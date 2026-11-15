"""Build the bilingual knowledge-base index.

1. Translate any card without a Hindi version (free-tier LLM; public text only).
2. Embed English + Hindi texts with BGE-M3 (CPU) and save a FAISS index.

uv run --extra rag python scripts/build_kb_index.py
-> data/kb/knowledge_base.json (Hindi added), data/kb/index.faiss, data/kb/chunks.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from tai_bot.llm_clients import make_llm_chain
from tai_bot.retrieval import KB_DIR, Retriever, load_bge_m3

ROOT = Path(__file__).resolve().parents[1]
KB = KB_DIR / "knowledge_base.json"

TRANSLATE = """Translate these mental-health information cards into simple, natural, everyday \
Hindi in Devanagari script, the way a caring counsellor in North India would explain it. Keep \
numbers (14416, 1-800-891-4416, 112) exactly as they are. Keep names like Tele-MANAS, WHO, \
NHS, NIMHANS in English letters. Do not add or remove information.

Cards (JSON): {cards}

Return ONLY JSON: {{"items": [{{"id": "...", "hi": "..."}}]}}"""


def translate_missing(cards: list[dict[str, str]]) -> None:
    todo = [c for c in cards if not c.get("hi")]
    if not todo:
        return
    llm = make_llm_chain(
        ["gemini:gemini-2.5-flash", "gemini:gemini-3.1-flash-lite", "groq:openai/gpt-oss-120b"],
        json_mode=True,
    )
    for b in range(0, len(todo), 5):
        batch = todo[b : b + 5]
        payload = json.dumps([{"id": c["id"], "en": c["en"]} for c in batch], ensure_ascii=False)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", llm(TRANSLATE.format(cards=payload)).strip())
        for item in json.loads(raw)["items"]:
            for c in batch:
                if c["id"] == item["id"] and re.search(r"[ऀ-ॿ]", item["hi"]):
                    c["hi"] = item["hi"].strip()
                    c["hi_translated_by"] = llm.current
        print(f"  translated {min(b + 5, len(todo))}/{len(todo)}")
        KB.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    load_dotenv(ROOT / ".env")
    cards = json.loads(KB.read_text(encoding="utf-8"))
    translate_missing(cards)
    missing = [c["id"] for c in cards if not c.get("hi")]
    if missing:
        print(f"WARNING: no Hindi version for {missing}")
    t0 = time.time()
    print("Loading BGE-M3 (first run downloads ~2.3 GB)...")
    embed = load_bge_m3()
    retriever = Retriever(cards, embed)
    retriever.save()
    print(f"Indexed {len(retriever.rows)} texts from {len(cards)} cards in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
