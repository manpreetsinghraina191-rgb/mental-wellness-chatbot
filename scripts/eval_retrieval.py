"""Evaluate retrieval: Recall@3 and MRR per language, with/without the reranker, and
with/without transliterating Hinglish queries to Devanagari (the brief's key hypothesis).

Queries are the technique / info / support items of the eval set (parallel in English,
Hinglish and Hindi), each labelled with the knowledge card(s) that answer it. Off-topic items
are negatives: the right answer is NO context.

  uv run --extra rag python scripts/eval_retrieval.py [--translit-model qwen3.5:4b]
  -> results/retrieval_eval.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from tai_bot.eval_data import load_eval_set
from tai_bot.llm_clients import make_llm
from tai_bot.preprocess import TRANSLIT_PROMPT, normalize, to_devanagari_llm
from tai_bot.retrieval import Retriever, load_bge_m3, load_reranker
from tai_bot.stats import pct_ci


@dataclass(frozen=True)
class Query:
    id: str
    lang: str
    text: str


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / ".translit_cache.json"

# eval-set suffix -> relevant card ids (same for the en / hg / hi twins)
RELEVANT = {
    "tec-01": {"breathing-exercise", "grounding-54321"},
    "tec-02": {"grounding-54321", "breathing-exercise"},
    "tec-03": {"exam-stress"},
    "tec-04": {"sleep-hygiene"},
    "tec-05": {"thought-reframing"},
    "inf-01": {"tele-manas"},
    "inf-02": {"exam-stress", "anxiety-what"},
    "inf-03": {"sadness-vs-depression", "depression-what"},
    "inf-04": {"sleep-hygiene"},
    "inf-05": {"seeking-help"},
    "sup-01": {"sleep-hygiene", "thought-reframing"},
    "sup-07": {"loneliness"},
}
NEGATIVE = {"off-01", "off-02", "off-03", "off-04"}
MIXED = {"mx-tec-01": {"breathing-exercise", "grounding-54321"}, "mx-inf-01": {"tele-manas"}}


def pct(a: float, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    p = argparse.ArgumentParser()
    p.add_argument("--translit-model", default="qwen3.5:4b")
    p.add_argument("--no-reranker", action="store_true", help="skip the slow CPU reranker configs")
    args = p.parse_args()

    items = load_eval_set(ROOT / "data" / "eval" / "eval_set.csv")
    queries = []
    for it in items:
        key = it.id[3:]
        if it.id in MIXED:
            queries.append((it, MIXED[it.id]))
        elif key in RELEVANT and it.lang != "mixed":
            queries.append((it, RELEVANT[key]))
        elif key in NEGATIVE and it.lang != "mixed":
            queries.append((it, set()))
    sets = {"original": queries}
    extra = ROOT / "data" / "eval" / "retrieval_extra.csv"
    if extra.exists():  # questions for the cards added in step 9, written before testing
        with extra.open(encoding="utf-8-sig", newline="") as f:
            sets["new topics"] = [
                (Query(r["id"], r["lang"], r["text"]), set(r["relevant"].split("|")))
                for r in csv.DictReader(f)
            ]
    queries = [q for qs in sets.values() for q in qs]

    # Transliteration of Hinglish queries with the same local model the app would use.
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    llm = make_llm("ollama", args.translit_model, temperature=0.0)
    translit: dict[str, str | None] = {}
    for it, _ in queries:
        if it.lang != "roman_hindi":
            continue
        clean = normalize(it.text, aggressive=True)
        k = hashlib.sha1(f"{args.translit_model}\n{TRANSLIT_PROMPT}\n{clean}".encode()).hexdigest()
        if k not in cache:
            cache[k] = to_devanagari_llm(clean, llm)
        translit[it.id] = cache[k]
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Loading models...")
    retriever = Retriever.load(load_bge_m3(), None if args.no_reranker else load_reranker())

    configs = [
        ("dense only", False, False),
        ("dense + reranker", True, False),
        ("dense + translit (Hinglish)", False, True),
        ("dense + translit + reranker", True, True),
    ]
    if args.no_reranker:
        configs = [c for c in configs if not c[1]]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Query sets: original = {len(sets['original'])} eval-set queries "
        f"({sum(1 for _, r in sets['original'] if not r)} of them off-topic negatives); "
        f"new topics = {len(sets.get('new topics', []))} queries for the cards added later, "
        "written before testing. Knowledge base: "
        f"{len(retriever.cards)} cards x 2 languages. BGE-M3 dense + FAISS, bge-reranker-v2-m3, "
        f"threshold filtering. Hinglish transliteration by local `{args.translit_model}`. "
        "Recall@3 with a 95% Wilson interval.",
        "",
        "| Query set | Config | Language | Recall@3 | MRR | Off-topic queries given context |",
        "|---|---|---|---|---|---|",
    ]
    examples: list[str] = []
    top1: dict[str, list[str]] = {}
    for set_name, set_queries in sets.items():
        for name, use_rr, use_tl in configs:
            for lang in ["english", "roman_hindi", "devanagari", "mixed", "all"]:
                sub = [(it, rel) for it, rel in set_queries if lang == "all" or it.lang == lang]
                if not sub:
                    continue
                pos = [(it, rel) for it, rel in sub if rel]
                neg = [(it, rel) for it, rel in sub if not rel]
                recall = mrr = 0.0
                neg_ctx = 0
                for it, rel in sub:
                    qs = [normalize(it.text)]
                    if use_tl and it.lang == "roman_hindi" and translit.get(it.id):
                        qs.append(translit[it.id] or "")
                    hits = retriever.search(qs, use_reranker=use_rr)
                    ids = [h.card_id for h in hits]
                    if rel:
                        recall += any(i in rel for i in ids)
                        rank = next((n for n, i in enumerate(ids, 1) if i in rel), None)
                        mrr += 1 / rank if rank else 0
                        full = name == "dense + translit (Hinglish)" and lang == "all"
                        if full and not any(i in rel for i in ids):
                            examples.append(f"- miss `{it.id}`: {it.text} -> {ids or 'nothing'}")
                        if full and set_name == "new topics":
                            top1[it.id] = ids[:1]
                    else:
                        neg_ctx += bool(ids)
                lines.append(
                    f"| {set_name} | {name} | {lang} | {pct_ci(int(recall), len(pos))} "
                    f"| {mrr / max(1, len(pos)):.2f} | {neg_ctx}/{len(neg)} |"
                )
    lines += ["", "Transliterations used (Hinglish -> Devanagari):", ""]
    lines += [f"- `{k}`: {v}" for k, v in translit.items()]
    lines += [
        "",
        "Misses in the app's configuration (dense + translit):",
        "",
        *(examples or ["- none"]),
    ]
    out = ROOT / "results" / "retrieval_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
