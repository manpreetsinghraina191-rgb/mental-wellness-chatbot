# TAI — Hindi / Hinglish / English mental-health support chatbot

A RAG-based **peer-support** chatbot that replies in the script the user writes in
(Devanagari, Romanized Hindi, English or mixed), with a safety-first crisis path.

> **Not a therapist, not a diagnostic tool.** In India, if you are in crisis call
> **Tele-MANAS 14416** or emergency **112**.

## Status

- [x] Preprocessing: normalization, script detection, transliteration fallback (`src/tai_bot/preprocess.py`)
- [ ] Eval set
- [ ] Crisis detection + fixed crisis protocol
- [ ] Knowledge base + retrieval
- [ ] LangGraph pipeline + Gradio demo
- [ ] Fine-tuning + ablation results

## Quickstart

```bash
uv sync                 # Python 3.12 env + dev tools
uv run pytest           # tests + coverage
uv run ruff check . && uv run mypy
uv run pre-commit install
```

### Optional: IndicXlit transliteration

`ai4bharat-transliteration` does not install on Python 3.11+. Use a separate Python 3.10
env (`uv venv --python 3.10 .venv-xlit`) or Colab. Without it, the bot falls back to
LLM-based transliteration.

## License

MIT (code). Datasets keep their own licenses.
