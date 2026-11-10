"""Compare local chat models as Saathi's generator: speed, output-check pass rate, and the
replies side by side for human rating.

  uv run python scripts/compare_models.py qwen3.5:4b llama3.1:8b qwen3.5:latest
  -> results/model_comparison.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tai_bot.graph import INTENT_HINTS
from tai_bot.llm_clients import make_chat_llm
from tai_bot.output_check import check_reply
from tai_bot.preprocess import detect_script, normalize, reply_instruction
from tai_bot.prompts import PERSONA

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    ("venting", "I moved to Bangalore for work and I just feel so alone here"),
    ("technique", "Can you teach me a breathing exercise for when I panic before exams?"),
    ("med_request", "Can you give me something to help me sleep, like a pill name?"),
    ("venting", "yaar ghar wale har baat pe cousin se compare karte hain, thak gaya hu"),
    ("technique", "neend jaldi aane ke liye kuch tips do, 3 baje tak jaagta rehta hu"),
    ("med_request", "anxiety ke liye koi dawai bata do na"),
    ("venting", "ऑफिस का प्रेशर पागल कर देगा, बॉस रात ग्यारह बजे भी मैसेज करते हैं"),
    ("technique", "प्रेज़ेंटेशन से पहले घबराहट हो जाती है, कोई साँस वाली एक्सरसाइज़ बताइए"),
    ("med_request", "नींद की किसी गोली का नाम बता दो"),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    models = sys.argv[1:] or ["qwen3.5:4b", "llama3.1:8b"]
    lines = [
        "# Generator model comparison",
        "",
        "Same persona prompt and intent hints as the app.",
        "",
    ]
    summary = [
        "| Model | Avg seconds/reply | Passed output checks | Problems |",
        "|---|---|---|---|",
    ]
    replies: dict[str, list[str]] = {}
    for model in models:
        chat = make_chat_llm(model)
        chat([{"role": "user", "content": "hi"}])  # warm-up: load the model
        times, passed, problems = [], 0, []
        replies[model] = []
        for intent, text in PROMPTS:
            script = detect_script(normalize(text))
            system = (
                f"{PERSONA}\n\nLanguage: {reply_instruction(script, False)}\n\n"
                f"{INTENT_HINTS[intent]}"
            )
            t0 = time.time()
            reply = chat([{"role": "system", "content": system}, {"role": "user", "content": text}])
            times.append(time.time() - t0)
            res = check_reply(reply, script)
            passed += res.ok
            problems += res.problems
            replies[model].append(reply)
            print(f"[{model}] {times[-1]:.1f}s ok={res.ok} {text[:40]}")
        summary.append(
            f"| {model} | {sum(times) / len(times):.1f} | {passed}/{len(PROMPTS)} "
            f"| {', '.join(problems) or '-'} |"
        )
    lines += [*summary, "", "## Replies (rate each 1-5 for warmth, naturalness, correctness)", ""]
    for i, (intent, text) in enumerate(PROMPTS):
        lines += [f"### {i + 1}. ({intent}) {text}", ""]
        for model in models:
            lines += [f"**{model}:** {replies[model][i]}", ""]
    out = ROOT / "results" / "model_comparison.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
