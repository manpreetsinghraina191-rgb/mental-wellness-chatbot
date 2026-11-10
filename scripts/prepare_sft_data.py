"""Build the Hindi / Hinglish / English fine-tuning dataset.

Usage (from the repo root):
  uv run python scripts/prepare_sft_data.py download
  uv run python scripts/prepare_sft_data.py extract
  uv run python scripts/prepare_sft_data.py pilot --n 20             # translate 20 + review sheet
  uv run python scripts/prepare_sft_data.py translate --limit 1000   # resumable
  uv run python scripts/prepare_sft_data.py translate --provider groq  # if Gemini quota runs out
  uv run python scripts/prepare_sft_data.py filter
  uv run python scripts/prepare_sft_data.py format

Only public dataset text is sent to the translation API. Never user conversations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from tai_bot.llm_clients import QuotaExhausted, make_llm_chain
from tai_bot.sft_data import (
    build_sft,
    extract_esconv,
    filter_records,
    read_jsonl,
    review_markdown,
    translate_samples,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ESConv.json"
PROC = ROOT / "data" / "processed"
CANDIDATES = PROC / "esconv_candidates.jsonl"
TRANSLATED = PROC / "translations.jsonl"
PILOT = PROC / "pilot_translations.jsonl"
FILTERED = PROC / "filtered.jsonl"
# Each Gemini/Gemma model has its own free daily quota; Gemma's is the largest.
DEFAULT_CHAIN = ",".join(
    [
        "gemini:gemma-4-26b-a4b-it",
        "gemini:gemini-3.1-flash-lite",
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-2.5-flash",
        "groq:openai/gpt-oss-120b",
    ]
)
ESCONV_URL = "https://huggingface.co/datasets/thu-coai/esconv/resolve/main/ESConv.json"


def show(stats: dict[str, int]) -> None:
    for k, v in sorted(stats.items()):
        print(f"  {k:40s} {v}")


def cmd_download(_: argparse.Namespace) -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(ESCONV_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    RAW.write_bytes(r.content)
    print(f"saved {RAW} ({len(r.content) / 1e6:.1f} MB)")


def cmd_extract(args: argparse.Namespace) -> None:
    dialogs = json.loads(RAW.read_text(encoding="utf-8"))
    samples, stats = extract_esconv(dialogs, max_per_dialog=args.per_dialog, seed=args.seed)
    write_jsonl(CANDIDATES, samples)
    show(stats)
    print(f"-> {CANDIDATES}")


def _translate(args: argparse.Namespace, out: Path, limit: int) -> None:
    samples = list(read_jsonl(CANDIDATES))
    if not samples:
        sys.exit("No candidates. Run `extract` first.")
    random.Random(args.seed).shuffle(samples)
    samples = samples[:limit]
    providers = [p.strip() for p in args.provider.split(",") if p.strip()]
    llm = make_llm_chain(providers, log=lambda m: print(f"  {m}"), json_mode=True)
    print(f"Translating {len(samples)} samples with {' -> '.join(providers)} ...")
    try:
        stats = translate_samples(samples, llm, out, batch_size=args.batch_size, pause=args.pause)
    except QuotaExhausted as e:
        print(f"\n{e}. Progress is saved; re-run with another --provider to continue.")
        sys.exit(2)
    show(stats)


def cmd_translate(args: argparse.Namespace) -> None:
    _translate(args, TRANSLATED, args.limit)


def cmd_pilot(args: argparse.Namespace) -> None:
    _translate(args, PILOT, args.n)
    records, stats = filter_records(read_jsonl(PILOT))
    show(stats)
    sheet = PROC / "pilot_review.md"
    sheet.write_text(review_markdown(records), encoding="utf-8")
    print(f"-> review sheet: {sheet}")


def cmd_filter(_: argparse.Namespace) -> None:
    records, stats = filter_records(read_jsonl(TRANSLATED))
    write_jsonl(FILTERED, records)
    show(stats)
    (PROC / "filter_report.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"-> {FILTERED}")


def cmd_format(args: argparse.Namespace) -> None:
    train, val, stats = build_sft(list(read_jsonl(FILTERED)), seed=args.seed)
    write_jsonl(PROC / "sft_train.jsonl", train)
    write_jsonl(PROC / "sft_val.jsonl", val)
    show(stats)
    print(f"-> {PROC / 'sft_train.jsonl'} ({len(train)}), sft_val.jsonl ({len(val)})")


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    ex = sub.add_parser("extract")
    ex.add_argument("--per-dialog", type=int, default=2)
    ex.set_defaults(fn=cmd_extract)
    for name, fn in (("translate", cmd_translate), ("pilot", cmd_pilot)):
        t = sub.add_parser(name)
        t.add_argument(
            "--provider",
            default=os.getenv("TRANSLATE_MODELS", DEFAULT_CHAIN),
            help="comma-separated provider:model fallback order",
        )
        t.add_argument("--batch-size", type=int, default=5)
        t.add_argument("--pause", type=float, default=6.0, help="seconds between requests")
        if name == "translate":
            t.add_argument("--limit", type=int, default=1000)
        else:
            t.add_argument("--n", type=int, default=20)
        t.set_defaults(fn=fn)
    sub.add_parser("filter").set_defaults(fn=cmd_filter)
    sub.add_parser("format").set_defaults(fn=cmd_format)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
