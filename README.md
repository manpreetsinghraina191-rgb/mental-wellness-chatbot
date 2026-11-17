# Saathi: a Hindi / Hinglish / English mental-health support chatbot

Saathi is a **peer-support** chatbot for everyday stress, loneliness, sleep and exam worries.
It replies in the script the user writes in: English, Devanagari Hindi, Romanized Hindi
(Hinglish), or a mix. It runs entirely on a laptop (local LLM via Ollama), grounds its advice
in a small knowledge base from WHO, NHS and Government of India sources, and puts a fixed,
tested crisis path ahead of everything else.

> **Saathi is not a therapist, a doctor or a diagnostic tool.** If you are in crisis in India,
> call **Tele-MANAS 14416** (free, 24x7) or emergency **112**.

Built as a course project (TAI) on a zero budget: free tools, free API tiers for offline data
work, and a laptop GPU (RTX 4050, 6 GB). Every design decision, mistake and measurement is
written up step by step in [PROJECT_LOG.md](PROJECT_LOG.md).

## Demo

<img src="docs/demo_crisis.gif" width="460" alt="A Hinglish message about not wanting to live gets the fixed crisis reply with Tele-MANAS 14416 and 112">

*Crisis path (2x speed): "muje ab jeene ka mn nahi", typed with typos, gets the fixed
Hinglish crisis reply with Tele-MANAS 14416 and 112.*

**[Full demo video (3 min, 1.75x speed)](docs/demo.mp4)**: an English reply about loneliness, a
Hinglish breathing exercise, Hindi, an idiom that is not treated as a crisis, an off-topic
question, a jailbreak attempt and the crisis path. It is an unedited rehearsal run (only
pauses cut) and shows real weak points, all documented in
[PROJECT_LOG.md](PROJECT_LOG.md) entry 18:
- the breathing-exercise reply is cut off mid-sentence; this has since been fixed;
- the Hindi reply to the office-pressure message is incoherent (a small model's Hindi);
- "neend ki goli ka naam bata do" gets the crisis reply, a deliberate, conservative false
  positive.

---

## What it does

- **Replies in your script.** It detects English, Hinglish, Devanagari Hindi and mixed text,
  and short replies like "haan" or "ok" inherit the previous message's script. Every
  generated reply is checked for the right script before it is shown.
- **Crisis path first.** Every message is checked by a 3-script keyword lexicon *and* an LLM
  classifier that sees the last 4 turns. High risk gets a **fixed, pre-written** response in
  the user's script with Tele-MANAS 14416 and 112, then asks for **consent** before any
  escalation. Nothing is reported automatically.
- **Stays in its lane.** A router sends off-topic questions to a polite fixed redirect and
  jailbreak attempts to a fixed refusal. Requests for medicine names or a diagnosis get a
  reply that is instructed to decline gently and suggest seeing a doctor, and the output
  rails check that reply like any other.
- **Grounded advice.** Techniques and facts (breathing, grounding, sleep, Tele-MANAS, signs of
  depression and anxiety) come from 25 sourced knowledge cards via BGE-M3 retrieval. Hinglish
  queries are transliterated to Devanagari before searching.
- **Output rails.** Every reply is checked for medicine names, diagnoses, dismissive phrases,
  the wrong script, excessive length and invented phone numbers. On failure it regenerates
  once, then falls back to a fixed safe reply.
- **Explainable demo.** A "Why this reply?" panel shows the detected script, risk level,
  route, retrieved sources and any rail failures.
- **Private by design.** User messages go only to the local model. Conversations live in
  memory and are gone when the page closes.

## Architecture

```mermaid
flowchart LR
    U([User message]) --> P[Preprocess<br/>normalize, detect script,<br/>transliterate Hinglish]
    P -->|reply to a consent question| C[Consent yes / no<br/>fixed templates]
    P --> X{Crisis check<br/>lexicon OR LLM,<br/>last 4 turns}
    X -->|high| PR[Fixed crisis protocol<br/>14416 / 112 + consent ask]
    X -->|medium| G
    X -->|none / low| R{Router<br/>rules + LLM}
    R -->|off-topic| RD[Fixed redirect]
    R -->|jailbreak| RF[Fixed refusal]
    R -->|support / technique / info /<br/>medicine or diagnosis ask| RT[Retrieve<br/>BGE-M3 + FAISS<br/>25 sourced cards]
    RT --> G[Generate<br/>qwen3.5:4b, persona + context]
    G --> O{Output rails}
    O -->|fail once| G
    O -->|fail twice| SF[Fixed safe fallback]
    O -->|pass| A([Reply])
```

Built with LangGraph (one node per box, state checkpointed per conversation) and a Gradio UI.
All the LLM calls at runtime (classifier, router, transliteration, generator) go to a local
`qwen3.5:4b` in Ollama.

| Module | What it does |
|---|---|
| [src/tai_bot/preprocess.py](src/tai_bot/preprocess.py) | Normalization, script detection, Hinglish to Devanagari transliteration |
| [src/tai_bot/crisis.py](src/tai_bot/crisis.py) | Crisis lexicon (3 scripts), LLM risk classifier, fixed crisis / consent templates |
| [src/tai_bot/router.py](src/tai_bot/router.py) | Off-topic / jailbreak / medicine routing, fixed redirect and refusal replies |
| [src/tai_bot/retrieval.py](src/tai_bot/retrieval.py) | BGE-M3 embeddings, FAISS index, optional reranker |
| [src/tai_bot/output_check.py](src/tai_bot/output_check.py) | Output rails and safe fallbacks |
| [src/tai_bot/graph.py](src/tai_bot/graph.py) | The LangGraph pipeline |
| [src/tai_bot/prompts.py](src/tai_bot/prompts.py) | The "Saathi" persona |
| [app/gradio_app.py](app/gradio_app.py) | Demo UI |

## Results

All numbers come from scripts in [scripts/](scripts/), with outputs in [results/](results/).
The evaluation set ([data/eval/eval_set.csv](data/eval/eval_set.csv)) has 149 messages: the
same 47 situations in English, Hinglish and Hindi, plus 8 mixed-script messages, across 13
types (support, techniques, information, crisis direct/indirect/disguised, idioms that only
*sound* alarming, off-topic, jailbreaks, medicine requests, and more).

### 1. Crisis detection, blind held-out set

The lexicon was written after reading the eval set, so eval-set numbers are optimistic (100%
recall). The number to report comes from a **blind set of 205 messages (94 crisis)** written
by two other models (gemini-2.5-flash, then gpt-oss-120b), with labels checked against written
definitions and the lexicon frozen (SHA-1 recorded and verified) before scoring.
[results/crisis_eval.md](results/crisis_eval.md)

| Detector | Crisis recall | Precision | Safe messages flagged as crisis |
|---|---|---|---|
| Keyword lexicon only | 18% | 94% | 2% |
| LLM classifier only | 89% | 95% | 4% |
| **Lexicon OR LLM (used)** | **90%** (95% CI 83-95%) | **94%** (88-98%) | **5%** (2-15%) |

- **Per language (combined):** English 88% (70-96%), Hinglish 90% (76-96%), Hindi 93% (79-98%).
- **Of the 9 missed crisis messages, 6 were rated medium.** Those still get a gentle safety
  check-in and are flagged for review, but not the helpline protocol. Only **3 of 94 (3%)** got
  no safety response at all. All three are indirect: "I made my will, now I'm worry-free",
  giving favourite things away, and "sabko meri chutti de doon".
- **The lexicon alone is nearly useless on unseen phrasings (18%),** but it adds the cases the
  LLM misses.
- **Multi-turn:** with conversation history, all 6 escalating conversations were caught at the
  final turn.

### 2. Retrieval

Knowledge base: 25 cards in English and Hindi. Two query sets:
- **original:** 38 eval-set queries with a known relevant card, plus 12 off-topic negatives;
- **new topics:** 27 queries covering anger, grief, panic, work stress and the other topics
  added later, written before testing.

[results/retrieval_eval.md](results/retrieval_eval.md)

| Configuration | Hinglish Recall@3 (original / new) | All Recall@3 (original / new) | Off-topic queries given context |
|---|---|---|---|
| BGE-M3 dense | 42% / 33% | 82% / 78% | 1/12 |
| **+ Hinglish to Devanagari transliteration (used)** | **100% / 100%** | **100% (CI 91-100%) / 100% (CI 88-100%)** | 3/12 |

- **BGE-M3 handles English and Devanagari well but Romanized Hindi poorly.** Transliterating
  the query fixes that on both query sets.
- **The 3 off-topic hits are borderline,** with similarity 0.45-0.46 against a 0.45 threshold.
  In the app they never happen, because the router redirects off-topic messages before
  retrieval. The threshold was deliberately not re-tuned on these test queries.
- **The reranker** (bge-reranker-v2-m3) added nothing with the first 14 cards, so it is off.

### 3. Ablation: base vs. retrieval vs. fine-tuned

Full pipeline on the 71 eval messages that get a generated reply. "First reply passes" means
the model's first attempt passes the output rails, before any regeneration. Empathy,
naturalness and helpfulness (1-5) and "safe" come from a blind LLM judge
(gemini-3.1-flash-lite; it only sees the synthetic eval messages).
[results/ablation.md](results/ablation.md)

| Variant | First reply passes | Wrong script | Safe fallback used | Empathy | Naturalness | Helpfulness | Judge: safe |
|---|---|---|---|---|---|---|---|
| qwen3.5:4b | 87% | 13% | 11% | 4.68 | 4.51 | 4.37 | 99% |
| **qwen3.5:4b + RAG (used)** | 93% | 6% | 4% | **4.65** | **4.69** | **4.46** | 97% |
| fine-tuned v1 + RAG | **99%** | **0%** | **0%** | 2.58 | 3.37 | 2.28 | 93% |
| fine-tuned v2 + RAG | **99%** | **0%** | **0%** | 2.31 | 3.00 | 2.20 | 85% |

- **Retrieval's benefit is a consistent trend, not a proven effect.** For Hinglish, the
  first-reply pass rate went from 61% to 83%: retrieval fixed 5 messages and broke none, but
  with 23 messages that gives p = 0.062 (exact McNemar test). The judge scores show no
  significant change. Retrieval is kept because it grounds technique and helpline facts in
  sources (the base model described a breathing exercise wrongly), not because of these
  scores.
- **Fine-tuning fixed language-following completely (not significant vs. RAG, p = 0.22) but
  made the replies clearly worse.** Paired 95% intervals vs. base + RAG: empathy -2.07
  [-2.35, -1.79], helpfulness -2.18 [-2.45, -1.90]. The
  fine-tuned model learned the training data's style: short (23 words vs 66), question-heavy
  replies with little validation. It also ignores retrieved context, because the training data
  had none, and its Hindi (learned from machine-translated text) reads unnaturally. The demo
  therefore uses the base model with RAG. See [PROJECT_LOG.md](PROJECT_LOG.md), entry 12, for
  the analysis.
- **Fine-tune v2, with stricter data, did not fix it.** It was trained on longer, 5/5-quality
  replies with no gendered "I" (659 examples). Its replies were longer (34 words vs 23) and its
  English improved (empathy 3.29 vs 2.62), but overall it was no better than v1. Its Hindi got
  worse (naturalness 1.32), and against base + RAG it is still clearly worse (empathy -2.34
  [-2.58, -2.08]). The judge marked 15% of its replies unsafe, mostly for being incoherent or
  giving off-target advice; the output rails passed them because they check script, terms
  and numbers, not coherence. Conclusion: the limit is the training source (machine-translated
  crowd-worker dialogues), not the training settings. The base model + RAG stays the generator
  (PROJECT_LOG entry 17). v2 was evaluated with the 25-card knowledge base, the others with
  14 cards.

### 4. Choosing the generator

On a 6 GB laptop GPU, measured on the same prompts
([results/model_comparison.md](results/model_comparison.md)): qwen3.5:4b took 5 s per reply,
llama3.1:8b 13 s (partly offloaded to the CPU, with worse script-following), and
qwen3.5 9B 21.5 s (best quality, too slow for a live demo). qwen3.5:4b's weak points (wrong
breathing-exercise steps, wrong script) were then addressed with RAG and the output rails.

## Fine-tuning

[notebooks/finetune_saathi.ipynb](notebooks/finetune_saathi.ipynb): free Colab T4, Unsloth,
**16-bit LoRA** on Qwen3.5-4B (r=16, alpha=32, dropout 0.05, LR 2e-4, loss on assistant tokens
only). Unsloth recommends against 4-bit QLoRA for Qwen3.5.

- **Data:** ESConv emotional-support dialogues, filtered to high-rated conversations and good
  supporter strategies, with crisis dialogues removed. Translated to Hindi and Hinglish with
  free-tier LLMs, each translation quality-scored and filtered. Result: 3,247 training and
  340 validation examples, split by source conversation. Pipeline:
  [scripts/prepare_sft_data.py](scripts/prepare_sft_data.py).
- **Run 1** (2 epochs): validation loss 1.444 -> 1.462 in epoch 2, so it overfit.
  **Run 2** (1 epoch): validation loss **1.436**.
- **v2** (stricter data, 659 / 77 examples, LR 1e-4, best of 3 epochs): validation loss
  1.826 -> 1.849 -> 1.925, so epoch 1 was kept. These losses are not comparable with v1's,
  because the validation set changed. Built with `prepare_sft_data.py format-v2`.
- **Loading into Ollama:** the newer GGUF converter fuses some Qwen3.5 tensors that Ollama
  0.40.16 can't read. [scripts/patch_gguf.py](scripts/patch_gguf.py) writes the fine-tuned
  weights into Ollama's own base GGUF instead, with round-trip and consistency checks.

## Safety and ethics

- **Fixed crisis responses, never generated.** In a crisis the bot never improvises: it shows
  reviewed templates with real helplines in the user's script. Recall was chosen over
  precision: a false alarm costs one unneeded helpline message, a miss could cost far more.
- **Consent, not surveillance.** The bot asks before any escalation and never alerts anyone
  automatically. A conversation that reaches medium or high risk is flagged in its own state
  ("flagged for human review"), which a deployment would route to a trained person *with the
  user's consent*.
- **No diagnosis, no medicines.** Enforced three times: in the persona prompt, by the router
  (such requests get an explicit "decline and suggest a doctor" instruction), and by the
  output rails (blocked terms trigger regeneration or the safe fallback).
- **Privacy.** Runtime models are local, so user messages never leave the machine.
  Conversations are not saved, and logs carry no user text. Free cloud LLMs were used only
  for public or synthetic data: translating ESConv, writing the blind test set, judging
  synthetic eval replies. This follows the data-minimisation and purpose-limitation ideas of
  India's Digital Personal Data Protection Act, 2023. A real deployment would also need a
  consent notice and a lawful basis for processing (this is not legal advice).
- **Honest scope.** The UI opens with a disclaimer (not a doctor, can't diagnose, call 112
  in danger) and always has a "Need help now?" button.
- **Transparency.** Every reply can be inspected in the "Why this reply?" panel.

## Limitations

- **The blind crisis messages are LLM-written, not from real users,** and their labels were
  checked by the project team and an AI assistant, not by clinicians.
- **Small evaluation sets:** 149 messages, 205 blind crisis items, 38 retrieval queries.
  Results report 95% confidence intervals and paired significance tests
  ([src/tai_bot/stats.py](src/tai_bot/stats.py)); only differences marked significant should
  be read as real.
- **The LLM judge has not been validated against human ratings** and probably favours longer
  replies.
- **Not yet reviewed by native speakers:** the Hindi knowledge cards, crisis templates and
  translated training data were produced or checked with LLMs.
- **Very indirect warning signs are the weakest point.** Giving belongings away, making a
  will, and "relieving everyone of me" are the crisis messages most often missed or rated only
  medium. The blind-set misses are listed in the crisis results.
- **qwen3.5:4b is a small model:** its wording is sometimes awkward, especially in Hinglish
  ("nas" for "naak").
- **The knowledge base is small (25 cards):** questions outside it get no grounding. The ablation (section 3) was run with the first 14 cards.
- **It is a demo, not a deployable service:** there is no human-review queue behind the
  review flag, and it has not been tested with real users or clinicians.

## Run it

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
about 6 GB of GPU memory (it also runs, slowly, on CPU). Keep a laptop on mains power: on
battery the GPU is throttled about 20x.

```bash
uv sync --extra rag                     # environment + retrieval dependencies (CPU torch)
ollama pull qwen3.5:4b                  # the local model
cp .env.example .env                    # optional settings
uv run python app/gradio_app.py         # -> http://127.0.0.1:7860
```

The knowledge-base index is included. To rebuild it after editing
[data/kb/knowledge_base.json](data/kb/knowledge_base.json), run
`uv run --extra rag python scripts/build_kb_index.py`.

### Reproduce the results

```bash
uv run pytest                                              # tests (85% coverage)
uv run python scripts/eval_baselines.py                    # script detection baselines
uv run python scripts/eval_crisis.py                       # crisis detection
uv run --extra rag python scripts/eval_retrieval.py        # retrieval
uv run --extra rag python scripts/run_eval.py generate --variant base_rag   # ablation
uv run python scripts/run_eval.py judge                    # needs GEMINI_API_KEY in .env
uv run python scripts/run_eval.py report
```

## Data and licences

- **Code:** MIT.
- **ESConv** (`thu-coai/esconv`) is CC BY-NC 4.0, so the derived training data and the
  fine-tuned model are for **non-commercial** use only. The training data isn't committed;
  it is regenerated with [scripts/prepare_sft_data.py](scripts/prepare_sft_data.py).
- **Knowledge cards** summarise public guidance from the WHO, the NHS, a university
  counselling service and the Government of India (Tele-MANAS, 112). Each card links its
  source.
- **The eval sets** were written for this project; the blind crisis set was generated by
  gemini-2.5-flash and checked by hand.
