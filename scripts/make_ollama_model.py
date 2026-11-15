"""Load the fine-tuned GGUF (from notebooks/finetune_saathi.ipynb) into Ollama as `saathi-ft`.

It copies the base model's chat settings (renderer / parser / sampling parameters) from
`qwen3.5:4b` and only swaps the weights, so base and fine-tuned models are prompted identically.
A chat-template mismatch is the most common cause of a fine-tune looking worse than it is.

  uv run python scripts/make_ollama_model.py models/saathi-q4_k_m.gguf [--name saathi-ft]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gguf", type=Path)
    p.add_argument("--base", default="qwen3.5:4b")
    p.add_argument("--name", default="saathi-ft")
    args = p.parse_args()
    gguf = args.gguf.resolve()
    if not gguf.exists():
        sys.exit(f"GGUF not found: {gguf}")

    base = subprocess.run(
        ["ollama", "show", args.base, "--modelfile"], capture_output=True, text=True, check=True
    ).stdout
    keep = [
        line
        for line in base.splitlines()
        if line.split(" ", 1)[0] in {"TEMPLATE", "RENDERER", "PARSER", "PARAMETER", "SYSTEM"}
    ]
    modelfile = ROOT / "models" / f"Modelfile.{args.name}"
    modelfile.parent.mkdir(exist_ok=True)
    modelfile.write_text("\n".join([f"FROM {gguf}", *keep]) + "\n", encoding="utf-8")
    print(modelfile.read_text(encoding="utf-8"))
    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)
    print(f"\nCreated Ollama model '{args.name}'. Try it:  ollama run {args.name}")
    print(f"Use it in the app:  set CHAT_MODEL={args.name} in .env")


if __name__ == "__main__":
    main()
