"""Evaluate crisis detection on the eval set: lexicon vs LLM vs combined, per language, plus
the multi-turn ablation (classifier with vs without conversation history).

  uv run python scripts/eval_crisis.py                  # uses local Ollama
  uv run python scripts/eval_crisis.py --no-llm         # lexicon only
  -> results/crisis_eval.md, results/crisis_eval_rows.csv

LLM answers are cached in results/.crisis_llm_cache.json so re-runs are instant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from tai_bot.crisis import RISK_ORDER, build_classifier_prompt, lexicon_check, max_risk, parse_risk
from tai_bot.eval_data import load_eval_set, load_multiturn
from tai_bot.llm_clients import make_llm
from tai_bot.preprocess import normalize
from tai_bot.stats import pct_ci

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = RESULTS / ".crisis_llm_cache.json"
LANGS = ["english", "roman_hindi", "devanagari", "mixed"]


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a}/{b})" if b else "n/a"


class CachedClassifier:
    def __init__(self, llm: Callable[[str], str] | None, model: str) -> None:
        self.llm = llm
        self.model = model
        self.cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.latencies: list[float] = []

    def __call__(self, message: str, history: list[str]) -> str | None:
        if self.llm is None:
            return None
        prompt = build_classifier_prompt(message, history)
        key = hashlib.sha1(f"{self.model}\n{prompt}".encode()).hexdigest()
        if key not in self.cache:
            t0 = time.time()
            try:
                self.cache[key] = parse_risk(self.llm(prompt)) or "error"
            except Exception as e:
                print(f"  LLM error: {e}")
                self.cache[key] = "error"
            self.latencies.append(time.time() - t0)
            CACHE.write_text(json.dumps(self.cache))
        value = self.cache[key]
        return None if value == "error" else value


def is_high(r: str | None) -> bool:
    return r == "high"


def single_turn(clf: CachedClassifier) -> tuple[list[str], list[dict[str, str]]]:
    items = load_eval_set(ROOT / "data" / "eval" / "eval_set.csv")
    rows: list[dict[str, str]] = []
    for n, it in enumerate(items, 1):
        clean = normalize(it.text)
        lex = lexicon_check([clean]).risk
        llm = clf(clean, [])
        rows.append(
            {
                "id": it.id,
                "lang": it.lang,
                "type": it.type,
                "text": it.text,
                "expected": it.expected_risk,
                "lexicon": lex,
                "llm": llm or "",
                "combined": max_risk(lex, llm or "none"),
            }
        )
        if n % 25 == 0:
            print(f"  single-turn {n}/{len(items)}")

    lines = ["## Single-turn eval set", ""]
    for system in ("lexicon", "llm", "combined"):
        if system != "lexicon" and clf.llm is None:
            continue
        lines += [
            f"### {system}",
            "",
            "| Language | Crisis recall (high) | Precision (high) | Medium flagged (>= medium) "
            "| False alarm on safe items | Idioms triggering high |",
            "|---|---|---|---|---|---|",
        ]
        for lang in [*LANGS, "all"]:
            sub = [r for r in rows if lang == "all" or r["lang"] == lang]
            exp_high = [r for r in sub if r["expected"] == "high"]
            pred_high = [r for r in sub if is_high(r[system])]
            tp = sum(r["expected"] == "high" for r in pred_high)
            med = [r for r in sub if r["expected"] == "medium"]
            med_ok = sum(RISK_ORDER.get(r[system], 0) >= 2 for r in med)
            safe = [r for r in sub if r["expected"] == "none"]
            fa = sum(is_high(r[system]) for r in safe)
            idioms = [r for r in sub if r["type"] == "idiom_safe"]
            idiom_fa = sum(is_high(r[system]) for r in idioms)
            name = f"**{lang}**" if lang == "all" else lang
            lines.append(
                f"| {name} | {pct(tp, len(exp_high))} | {pct(tp, len(pred_high))} "
                f"| {pct(med_ok, len(med))} | {pct(fa, len(safe))} | {pct(idiom_fa, len(idioms))} |"
            )
        lines.append("")
    missed = [r for r in rows if r["expected"] == "high" and not is_high(r["combined"])]
    false_alarms = [r for r in rows if r["expected"] == "none" and is_high(r["combined"])]
    lines += ["Missed crisis messages (combined):", ""]
    lines += [f"- `{r['id']}` {r['text']}" for r in missed] or ["- none"]
    lines += ["", "False alarms (combined, expected none, predicted high):", ""]
    lines += [f"- `{r['id']}` {r['text']}" for r in false_alarms] or ["- none"]
    return lines, rows


def multi_turn(clf: CachedClassifier) -> list[str]:
    if clf.llm is None:
        return []
    convs = load_multiturn(ROOT / "data" / "eval" / "multiturn.jsonl")
    lines = [
        "",
        "## Multi-turn: does conversation history help? (LLM + lexicon, combined)",
        "",
        "| Conversation | Kind | Expected | Without history | With history (last 4 turns) |",
        "|---|---|---|---|---|",
    ]
    totals = {"no_hist": 0, "hist": 0, "n": 0}
    final_caught = {"no_hist": 0, "hist": 0, "n": 0}
    for c in convs:
        preds: dict[str, list[str]] = {"no_hist": [], "hist": []}
        for i, turn in enumerate(c.turns):
            clean = normalize(turn)
            lex = lexicon_check([clean]).risk
            history = [normalize(t) for t in c.turns[:i]][-4:]
            preds["no_hist"].append(max_risk(lex, clf(clean, []) or "none"))
            preds["hist"].append(max_risk(lex, clf(clean, history) or "none"))
        for k in ("no_hist", "hist"):
            totals[k] += sum(
                is_high(p) == is_high(e) for p, e in zip(preds[k], c.expected_risk, strict=True)
            )
            if c.kind == "escalation":
                final_caught[k] += is_high(preds[k][-1])
        totals["n"] += len(c.turns)
        final_caught["n"] += c.kind == "escalation"
        lines.append(
            f"| {c.id} ({c.lang}) | {c.kind} | {' > '.join(c.expected_risk)} "
            f"| {' > '.join(preds['no_hist'])} | {' > '.join(preds['hist'])} |"
        )
    lines += [
        "",
        "| Metric | Without history | With history |",
        "|---|---|---|",
        f"| Per-turn agreement on 'is it high?' | {pct(totals['no_hist'], totals['n'])} "
        f"| {pct(totals['hist'], totals['n'])} |",
        f"| Escalations caught at the final turn "
        f"| {pct(final_caught['no_hist'], final_caught['n'])} "
        f"| {pct(final_caught['hist'], final_caught['n'])} |",
    ]
    return lines


def heldout(clf: CachedClassifier) -> list[str]:
    """Blind test: generated by a different author, lexicon frozen before scoring."""
    path = ROOT / "data" / "eval" / "heldout_crisis.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        items = [r for r in csv.DictReader(f) if r["expected_risk"] != "excluded"]
    # Batch 2: only rows a human has checked. label_checked = ok (keep the generator's label),
    # high / medium / none (corrected label), or x (exclude: unrealistic, duplicate, unclear).
    path2 = ROOT / "data" / "eval" / "heldout_crisis_v2.csv"
    unchecked = 0
    if path2.exists():
        with path2.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                check = r["label_checked"].strip().lower()
                if check in {"high", "medium", "none"}:
                    items.append({**r, "expected_risk": check})
                elif check == "ok":
                    items.append(r)
                elif check != "x":
                    unchecked += 1
    authors = ", ".join(sorted({r["generator"] for r in items}))
    preds = []
    for r in items:
        clean = normalize(r["text"])
        lex = lexicon_check([clean]).risk
        llm = clf(clean, [])
        preds.append(
            {**r, "lexicon": lex, "llm": llm or "", "combined": max_risk(lex, llm or "none")}
        )
    lines = [
        "",
        "## Blind held-out set (the number to report)",
        "",
        f"{len(items)} messages generated by other authors ({authors}), "
        f"labels checked by hand ({unchecked} batch-2 rows not yet checked are left out), "
        "crisis lexicon frozen before scoring "
        "(see results/crisis_lexicon_frozen.sha1). The eval-set numbers above are optimistic "
        "because the lexicon was written after reading the eval set.",
        "",
    ]
    for system in ("lexicon", "llm", "combined"):
        if system != "lexicon" and clf.llm is None:
            continue
        lines += [
            f"### {system}",
            "",
            "| Language | Crisis recall (high) | Precision (high) | Medium flagged (>= medium) "
            "| Safe items predicted high |",
            "|---|---|---|---|---|",
        ]
        for lang in ["english", "roman_hindi", "devanagari", "all"]:
            sub = [r for r in preds if lang == "all" or r["lang"] == lang]
            exp_high = [r for r in sub if r["expected_risk"] == "high"]
            pred_high = [r for r in sub if is_high(r[system])]
            tp = sum(r["expected_risk"] == "high" for r in pred_high)
            med = [r for r in sub if r["expected_risk"] == "medium"]
            med_ok = sum(RISK_ORDER.get(r[system], 0) >= 2 for r in med)
            safe = [r for r in sub if r["expected_risk"] == "none"]
            fa = sum(is_high(r[system]) for r in safe)
            name = f"**{lang}**" if lang == "all" else lang
            lines.append(
                f"| {name} | {pct_ci(tp, len(exp_high))} | {pct_ci(tp, len(pred_high))} "
                f"| {pct_ci(med_ok, len(med))} | {pct_ci(fa, len(safe))} |"
            )
        lines.append("")
    lines += [
        "CI = 95% Wilson interval: with this many items, the true rate could be anywhere in it.",
        "",
    ]
    missed = [r for r in preds if r["expected_risk"] == "high" and not is_high(r["combined"])]
    fas = [r for r in preds if r["expected_risk"] == "none" and is_high(r["combined"])]
    lines += ["Missed (combined):", ""]
    lines += [f"- `{r['id']}` [{r['lexicon']}/{r['llm']}] {r['text']}" for r in missed] or [
        "- none"
    ]
    lines += ["", "False alarms on safe items (combined):", ""]
    lines += [f"- `{r['id']}` [{r['lexicon']}/{r['llm']}] {r['text']}" for r in fas] or ["- none"]
    return lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # Windows consoles
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--model", default="qwen3.5:4b")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    llm = None if args.no_llm else make_llm("ollama", args.model, json_mode=True, temperature=0.0)
    clf = CachedClassifier(llm, args.model)

    header = [
        "# Crisis detection evaluation",
        "",
        f"Classifier: {'none (lexicon only)' if llm is None else 'Ollama ' + args.model}. "
        "Combined = lexicon OR LLM (higher risk wins). 'Recall' counts only items predicted "
        "exactly `high`.",
        "",
    ]
    single, rows = single_turn(clf)
    multi = multi_turn(clf)
    multi += heldout(clf)
    if clf.latencies:
        avg = sum(clf.latencies) / len(clf.latencies)
        header.append(f"Average classifier latency (uncached calls): {avg:.1f}s\n")
    text = "\n".join(header + single + multi) + "\n"
    (RESULTS / "crisis_eval.md").write_text(text, encoding="utf-8")
    with (RESULTS / "crisis_eval_rows.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(text)


if __name__ == "__main__":
    main()
