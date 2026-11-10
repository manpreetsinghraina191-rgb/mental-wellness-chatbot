"""Baselines that need no LLM: script detection and keyword-only crisis detection.

uv run python scripts/eval_baselines.py   ->  results/baseline_preprocess.md
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from tai_bot.eval_data import load_eval_set
from tai_bot.preprocess import detect_script, normalize
from tai_bot.safety_terms import has_crisis_hint

ROOT = Path(__file__).resolve().parents[1]
LANG_ORDER = ["english", "roman_hindi", "devanagari", "mixed"]


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "n/a"


def independent_script_check() -> list[str]:
    """Script detection on real data never used to tune the detector (the eval set was)."""
    lines = ["", "## Script detection on independent real data", ""]
    esconv = ROOT / "data" / "raw" / "ESConv.json"
    erc = ROOT / "data" / "raw" / "erc_hinglish_sample.jsonl"
    if not (esconv.exists() and erc.exists()):
        return [*lines, "_skipped: run `prepare_sft_data.py download` and fetch the erc sample_"]
    dialogs = json.loads(esconv.read_text(encoding="utf-8"))
    en = [
        t["content"].strip()
        for d in dialogs
        for t in d["dialog"]
        if t["speaker"] == "seeker" and len(t["content"].split()) >= 4
    ]
    random.Random(0).shuffle(en)
    en = en[:2000]
    hg: list[str] = []
    for line in erc.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows = json.loads(line)["rows"]
            hg += [r["row"]["utterance"] for r in rows if len(r["row"]["utterance"].split()) >= 4]
    en_ok = sum(detect_script(normalize(s)) == "english" for s in en)
    hg_ok = sum(detect_script(normalize(s)) == "roman_hindi" for s in hg)
    return [
        *lines,
        "| Data | Expected | Accuracy |",
        "|---|---|---|",
        f"| ESConv seeker messages (real English) | english | {pct(en_ok, len(en))} |",
        f"| erc-hinglish sitcom lines (real Hinglish) | roman_hindi | {pct(hg_ok, len(hg))} |",
        "",
        "The eval-set accuracy above is optimistic (word list was extended after a first run on "
        "it); these independent numbers are the ones to report. Some erc-hinglish lines are "
        "actually English, so the Hinglish figure is a lower bound.",
    ]


def main() -> None:
    items = load_eval_set(ROOT / "data" / "eval" / "eval_set.csv")
    lines = ["# Baselines (no LLM)", "", f"Eval items: {len(items)}", ""]

    # 1. Script detection
    lines += ["## Script detection accuracy", "", "| Language | Accuracy |", "|---|---|"]
    misses: list[str] = []
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for it in items:
        got = detect_script(normalize(it.text))
        by_lang[it.lang].append(got == it.lang)
        if got != it.lang:
            misses.append(f"- `{it.id}` expected {it.lang}, got **{got}**: {it.text}")
    for lang in LANG_ORDER:
        r = by_lang[lang]
        lines.append(f"| {lang} | {pct(sum(r), len(r))} |")
    total = [x for r in by_lang.values() for x in r]
    lines += [f"| **all** | **{pct(sum(total), len(total))}** |", "", "Misses:", *misses, ""]

    # 2. Keyword-only crisis detection (the English list used to clean training data)
    lines += [
        "## Crisis detection: keyword list only (current `CRISIS_HINTS`)",
        "",
        "| Language | Recall on high-risk | False alarms on safe idioms |",
        "|---|---|---|",
    ]
    for lang in LANG_ORDER:
        high = [i for i in items if i.lang == lang and i.expected_risk == "high"]
        idioms = [i for i in items if i.lang == lang and i.type == "idiom_safe"]
        tp = sum(has_crisis_hint(i.text) for i in high)
        fp = sum(has_crisis_hint(i.text) for i in idioms)
        lines.append(f"| {lang} | {pct(tp, len(high))} | {pct(fp, len(idioms))} |")
    lines += [
        "",
        "Missed crisis messages are the costly error. This baseline motivates the multilingual "
        "lexicon + LLM classifier in step 3.",
    ]

    lines += independent_script_check()

    out = ROOT / "results" / "baseline_preprocess.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
