"""Fine-tuning data pipeline: extract -> translate -> filter -> format.

Design (see README / report "Data" section):
  * Source: ESConv (CC BY-NC 4.0). Only conversations the seeker rated >= 4/5 for empathy.
  * Excluded supporter turns: "Self-disclosure" (a bot must not invent a personal life),
    "Information" (often US-specific facts), "Others" (mostly greetings/filler).
  * Whole conversations with crisis language are dropped: crisis replies come from the fixed
    protocol, so the model must not learn a free-text crisis style.
  * Each kept sample is translated into Devanagari Hindi and Romanized Hinglish by an LLM,
    then automatically filtered (script check with our own detector, length ratio, safety terms).
  * Train/val split is done by SOURCE sample, so the Hindi/Hinglish/English versions of one
    conversation never end up on both sides of the split.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

from tai_bot.llm_clients import EmptyResponse
from tai_bot.preprocess import DEVANAGARI, Script, detect_script
from tai_bot.prompts import build_system_prompt
from tai_bot.safety_terms import (
    FALSE_PROMISE,
    JUDGMENTAL,
    OUT_OF_CONTEXT,
    SELF_DISCLOSURE,
    find_unsafe,
    has_crisis_hint,
)

Record = dict[str, Any]

GOOD_STRATEGIES = {
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Affirmation and Reassurance",
    "Providing Suggestions",
}
EXCLUDED_STRATEGIES = {"Self-disclosure", "Information", "Others"}
LANGS: dict[str, Script] = {"en": "english", "hindi": "devanagari", "hinglish": "roman_hindi"}


# --- JSONL helpers ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[Record]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# --- 1. Extract ------------------------------------------------------------------------------


def merge_turns(dialog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ESConv turns to alternating {role, text, strategies}. Handles both the raw
    ESConv.json format (speaker/content/annotation) and the HF viewer format (usr/sys/text)."""
    out: list[dict[str, Any]] = []
    for t in dialog:
        speaker = t.get("speaker")
        role = "user" if speaker in ("seeker", "usr") else "assistant"
        text = re.sub(r"\s+", " ", str(t.get("content", t.get("text", "")))).strip()
        if not text:
            continue
        strategy = t.get("annotation", {}).get("strategy") or t.get("strategy")
        if out and out[-1]["role"] == role:
            out[-1]["text"] += " " + text
            if strategy:
                out[-1]["strategies"].append(strategy)
        else:
            out.append({"role": role, "text": text, "strategies": [strategy] if strategy else []})
    return out


def response_problem(text: str, min_words: int = 6, max_words: int = 90) -> str | None:
    words = len(text.split())
    if words < min_words:
        return "too_short"
    if words > max_words:
        return "too_long"
    if re.search(r"https?://|www\.", text):
        return "url"
    term = find_unsafe(text)
    if term:
        return f"unsafe:{term}"
    for name, pattern in (
        ("self_disclosure", SELF_DISCLOSURE),
        ("false_promise", FALSE_PROMISE),
        ("out_of_context", OUT_OF_CONTEXT),
        ("judgmental", JUDGMENTAL),
    ):
        if pattern.search(text):
            return name
    return None


def extract_esconv(
    dialogs: list[dict[str, Any]],
    max_per_dialog: int = 2,
    context_turns: int = 4,
    min_empathy: int = 4,
    seed: int = 42,
) -> tuple[list[Record], Counter[str]]:
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    samples: list[Record] = []
    for d_idx, d in enumerate(dialogs):
        stats["dialogs"] += 1
        try:
            empathy = int(d["survey_score"]["seeker"]["empathy"])
        except (KeyError, TypeError, ValueError):
            empathy = 0
        if empathy < min_empathy:
            stats["drop_dialog:low_or_missing_empathy"] += 1
            continue
        turns = merge_turns(d["dialog"])
        if any(has_crisis_hint(t["text"]) for t in turns):
            stats["drop_dialog:crisis_language"] += 1
            continue
        candidates: list[Record] = []
        for i, t in enumerate(turns):
            if t["role"] != "assistant" or i < 3:
                continue
            strategies = set(t["strategies"])
            if not strategies & GOOD_STRATEGIES or strategies & EXCLUDED_STRATEGIES:
                stats["drop_turn:strategy"] += 1
                continue
            problem = response_problem(t["text"])
            if problem:
                stats[f"drop_turn:{problem.split(':')[0]}"] += 1
                continue
            start = max(0, i - context_turns)
            # Every assistant turn in the window is trained on, so earlier supporter turns must
            # be clean too: start the window after the last bad one.
            for j in range(start, i):
                if turns[j]["role"] == "assistant" and (
                    response_problem(turns[j]["text"], min_words=1)
                    or set(turns[j]["strategies"]) & EXCLUDED_STRATEGIES - {"Others"}
                ):
                    start = j + 1
            if turns[start]["role"] != "user":
                start += 1
            window = turns[start : i + 1]
            candidates.append(
                {
                    "id": f"esconv-{d_idx}-{i}",
                    "source": "esconv",
                    "meta": {
                        "emotion": d.get("emotion_type"),
                        "problem": d.get("problem_type"),
                        "strategies": sorted(strategies),
                    },
                    "en": [{"role": w["role"], "text": w["text"]} for w in window],
                }
            )
        rng.shuffle(candidates)
        samples.extend(candidates[:max_per_dialog])
    stats["kept_samples"] = len(samples)
    return samples, stats


# --- 2. Translate ----------------------------------------------------------------------------

# Real Hinglish lines from public Hinglish corpora, shown to the translator as a style anchor.
STYLE_LINES = [
    "kya yaar, traffic mein stuck ho gaya.",
    "Aaj mood off hai bhai",
    "Pata nahi kya ho raha hai",
    "Mujhe darr lag raha hai",
    "Usne meri madad ki bina kuch kahe",
]

TRANSLATE_PROMPT = """You translate short emotional-support conversations from English for an \
Indian mental-health support chatbot. For EVERY item, produce two versions of EVERY turn:

1. "hindi": natural, everyday spoken Hindi in Devanagari script, the way people in North India \
actually talk. Avoid heavy Sanskritized/textbook words when a common word exists. Write the \
whole turn in Devanagari (English loanwords like "ऑफिस", "स्ट्रेस" are fine, in Devanagari).
2. "hinglish": Romanized Hindi the way young Indians text on WhatsApp. Hindi grammar, Roman \
script only, no Devanagari, no diacritics. Keep English words people normally say in English \
(stress, anxiety, exam, job, office, relationship, feel, boss). Common spellings: nahi, bahut, \
hai, mujhe, kya, accha, aap, kyunki.

Style reference (real Hinglish messages): {style}

Rules:
- Keep the meaning, emotional tone and roughly the same length. Do not add advice, do not drop \
content, do not make it more formal.
- The supporter (assistant) addresses the user respectfully with "aap" / "आप".
- The supporter has no gender: prefer gender-neutral first-person phrasing such as \
"मुझे लगता है", "मुझे समझ आ रहा है", "mujhe lagta hai", "samajh aa raha hai" instead of \
"मैं समझ सकता/सकती हूँ".
- If a turn mentions something US-specific (dollars, states, 911, holidays), adapt it neutrally \
(e.g. "paise", "emergency number"). Never invent phone numbers or organisations.
- Return the same number of turns, in the same order, for every item.

Example:
English turns: ["I've been feeling really lonely since I moved for my job.", "That sounds \
really hard. Moving to a new place can leave you without your usual people around. What has \
been the toughest part?"]
hindi: ["नौकरी के लिए शिफ्ट होने के बाद से मुझे बहुत अकेलापन महसूस हो रहा है।", "यह सच में \
मुश्किल लग रहा है। नई जगह जाने पर अक्सर अपने लोग आस-पास नहीं होते। आपके लिए सबसे मुश्किल क्या \
रहा है?"]
hinglish: ["Job ke liye shift hone ke baad se mujhe bahut lonely feel ho raha hai.", "Yeh sach \
mein mushkil lag raha hai. Nayi jagah jaane par aksar apne log aas-paas nahi hote. Aapke liye \
sabse mushkil kya raha hai?"]

ALSO rate every SUPPORTER (assistant) turn of the ORIGINAL English from 1 to 5 as a reply \
from a peer-support chatbot. 5 = warm, validates feelings, non-judgmental, respectful, realistic. \
Lower the score for: judging or blaming the user, false promises ("you will be fine"), \
talking about the supporter's own life or experiences, medical or diagnostic claims, pushy \
orders, dismissiveness, generic filler, or content that only makes sense in the US.
"scores" has one integer per supporter turn, in order; "issues" briefly names any problems.

Return ONLY JSON of the form:
{{"items": [{{"id": "...", "hindi": ["turn 1", ...], "hinglish": ["turn 1", ...], \
"scores": [5, ...], "issues": "..."}}]}}

Items to translate:
{items}"""


def build_translate_prompt(batch: list[Record]) -> str:
    items = [{"id": s["id"], "turns": [t["text"] for t in s["en"]]} for s in batch]
    return TRANSLATE_PROMPT.format(
        style=" | ".join(STYLE_LINES), items=json.dumps(items, ensure_ascii=False, indent=1)
    )


def parse_translation(raw: str, batch: list[Record]) -> tuple[list[Record], list[str]]:
    """Return (translated records, error ids). Malformed items are reported, not raised."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], [s["id"] for s in batch]
    items = data.get("items", data) if isinstance(data, dict) else data
    by_id = {str(it.get("id")): it for it in items if isinstance(it, dict)}
    out: list[Record] = []
    errors: list[str] = []
    for s in batch:
        it = by_id.get(s["id"])
        n = len(s["en"])
        ok = (
            it is not None
            and isinstance(it.get("hindi"), list)
            and isinstance(it.get("hinglish"), list)
            and len(it["hindi"]) == n
            and len(it["hinglish"]) == n
        )
        if not ok or it is None:
            errors.append(s["id"])
            continue
        n_assistant = sum(t["role"] == "assistant" for t in s["en"])
        scores = it.get("scores")
        valid_scores = (
            isinstance(scores, list)
            and len(scores) == n_assistant
            and all(isinstance(x, int | float) for x in scores)
        )
        out.append(
            {
                **s,
                "hindi": [str(x).strip() for x in it["hindi"]],
                "hinglish": [str(x).strip() for x in it["hinglish"]],
                "scores": [int(x) for x in scores] if valid_scores and scores else None,
                "issues": str(it.get("issues", "")),
            }
        )
    return out, errors


def translate_samples(
    samples: list[Record],
    llm: Callable[[str], str],
    out_path: Path,
    batch_size: int = 5,
    pause: float = 0.0,
    log: Callable[[str], None] = print,
) -> Counter[str]:
    """Translate and APPEND to out_path. Resumable: ids already in out_path are skipped."""
    import time

    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [s for s in samples if s["id"] not in done]
    stats: Counter[str] = Counter(already_done=len(done))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def ask(items: list[Record]) -> tuple[list[Record], list[str]]:
        try:
            return parse_translation(llm(build_translate_prompt(items)), items)
        except httpx.HTTPStatusError as e:  # e.g. provider rejected malformed JSON output
            log(f"  batch failed ({e.response.status_code}); will retry on the next run")
            return [], [s["id"] for s in items]
        except EmptyResponse as e:  # e.g. safety filter on crisis-adjacent text
            log(f"  batch failed ({e}); will retry on the next run")
            return [], [s["id"] for s in items]

    for b in range(0, len(todo), batch_size):
        batch = todo[b : b + batch_size]
        translated, errors = ask(batch)
        if errors and len(batch) > 1:  # retry failed ones individually once
            for s in [s for s in batch if s["id"] in errors]:
                translated += ask([s])[0]
        model_name = getattr(llm, "current", None)
        if model_name:
            translated = [{**r, "translator": model_name} for r in translated]
        with out_path.open("a", encoding="utf-8") as f:
            for r in translated:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["translated"] += len(translated)
        stats["failed"] += len(batch) - len(translated)
        log(f"  {stats['translated'] + len(done)}/{len(samples)} translated")
        if pause:
            time.sleep(pause)
    return stats


# --- 3. Filter -------------------------------------------------------------------------------


def check_language(rec: Record, lang: str) -> list[str]:
    texts: list[str] = rec.get(lang) or []
    en = [t["text"] for t in rec["en"]]
    if len(texts) != len(en) or not all(texts):
        return ["turn_count"]
    problems: list[str] = []
    resp, en_resp = texts[-1], en[-1]
    if lang == "hindi" and detect_script(resp) != "devanagari":
        problems.append("script")
    if lang == "hinglish":
        if any(DEVANAGARI.search(t) for t in texts):
            problems.append("devanagari_in_hinglish")
        elif detect_script(resp) != "roman_hindi":
            problems.append("script")
    ratio = len(resp.split()) / max(1, len(en_resp.split()))
    if not 0.5 <= ratio <= 2.2:
        problems.append("length_ratio")
    if find_unsafe(" ".join(texts)):
        problems.append("unsafe_term")
    return problems


def filter_records(
    records: Iterable[Record], min_score: int = 4
) -> tuple[list[Record], Counter[str]]:
    """Per-language keep flags. A low LLM quality score drops ALL languages of a sample
    (the problem is the source reply itself, not the translation)."""
    out: list[Record] = []
    stats: Counter[str] = Counter()
    for r in records:
        stats["records"] += 1
        scores = r.get("scores")
        if scores is not None and min(scores, default=5) < min_score:
            stats["dropped_low_quality_source"] += 1
            reason = [f"quality<{min_score}: {r.get('issues', '')}"]
            keep_none = {"en": False, "hindi": False, "hinglish": False}
            out.append({**r, "keep": keep_none, "problems": {"source": reason}})
            continue
        keep: dict[str, bool] = {"en": True}
        problems: dict[str, list[str]] = {}
        for lang in ("hindi", "hinglish"):
            p = check_language(r, lang)
            keep[lang] = not p
            problems[lang] = p
            stats[f"{lang}:kept" if not p else f"{lang}:dropped"] += 1
            for why in p:
                stats[f"{lang}:reason:{why}"] += 1
        out.append({**r, "keep": keep, "problems": problems})
    return out, stats


# --- 4. Format -------------------------------------------------------------------------------


def to_chat(rec: Record, lang: str) -> Record:
    script = LANGS[lang]
    texts = [t["text"] for t in rec["en"]] if lang == "en" else rec[lang]
    roles = [t["role"] for t in rec["en"]]
    messages = [{"role": "system", "content": build_system_prompt(script)}]
    messages += [{"role": role, "content": text} for role, text in zip(roles, texts, strict=True)]
    return {"id": f"{rec['id']}:{lang}", "source_id": rec["id"], "lang": lang, "messages": messages}


def build_sft(
    filtered: list[Record], val_frac: float = 0.1, seed: int = 42
) -> tuple[list[Record], list[Record], Counter[str]]:
    ids = sorted({r["id"] for r in filtered})
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[: max(1, round(len(ids) * val_frac))]) if len(ids) > 1 else set()
    train: list[Record] = []
    val: list[Record] = []
    stats: Counter[str] = Counter()
    for r in filtered:
        for lang in LANGS:
            if not r.get("keep", {}).get(lang, lang == "en"):
                continue
            ex = to_chat(r, lang)
            split = "val" if r["id"] in val_ids else "train"
            (val if split == "val" else train).append(ex)
            stats[f"{split}:{lang}"] += 1
    return train, val, stats


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def is_question_only(text: str) -> bool:
    """True when every sentence is a question: no validation, no idea, just asking."""
    return all(s.rstrip().endswith("?") for s in _sentences(text))


def v2_problem(
    rec: Record, min_last_words: int = 20, min_words: int = 12, min_score: int = 5
) -> str | None:
    """Stricter selection for fine-tune v2. v1 taught the model short, question-heavy replies
    (median 22 words, 43% ending in a question) and stiff Hindi; v2 keeps only windows whose
    supporter replies are fuller, never just a question, and translated at quality 5."""
    if min(rec.get("scores") or [5]) < min_score:
        return "quality"
    replies = [t["text"] for t in rec["en"] if t["role"] == "assistant"]
    if not replies or len(replies[-1].split()) < min_last_words:
        return "short_final_reply"
    if any(len(t.split()) < min_words for t in replies):
        return "short_reply"
    if any(is_question_only(t) for t in replies):
        return "question_only"
    return None


# The persona speaks gender-neutrally ("mujhe samajh aa raha hai"), but ~1 in 4 translated
# supporter turns use a gendered "I" ("main samajh sakta hoon", "मैं समझ गया").
GENDERED_SELF = re.compile(
    r"(सकता|सकती|गया|गई|रहा|रही)\s*(हूँ|हूं)|मैं\s+समझ\s+(गया|गई)"
    r"|\b(sakta|sakti|gaya|gayi|raha|rahi)\s+(hoon|hun|hu)\b|\bsamajh\s+(gaya|gayi)\b",
    re.IGNORECASE,
)


def select_v2(filtered: list[Record]) -> tuple[list[Record], Counter[str]]:
    stats: Counter[str] = Counter()
    out = []
    for r in filtered:
        if not any(r.get("keep", {}).values()):
            continue
        why = v2_problem(r)
        stats[why or "kept"] += 1
        if why is not None:
            continue
        keep = dict(r["keep"])
        roles = [t["role"] for t in r["en"]]
        for lang in ("hindi", "hinglish"):
            replies = [t for t, role in zip(r[lang], roles, strict=True) if role == "assistant"]
            if keep.get(lang) and any(GENDERED_SELF.search(t) for t in replies):
                keep[lang] = False
                stats[f"{lang}:dropped_gendered_supporter"] += 1
        out.append({**r, "keep": keep})
    return out, stats


# --- Review sheet for native speakers --------------------------------------------------------


def review_markdown(records: list[Record]) -> str:
    lines = [
        "# Translation review sheet",
        "",
        "For each item mark: OK / fix (write the correction) / bad. Check that the Hinglish "
        "sounds like real texting and the Hindi is natural, not textbook.",
        "",
    ]
    for r in records:
        keep = r.get("keep", {})
        verdict = "KEPT" if any(keep.values()) else "DROPPED"
        lines.append(
            f"## {r['id']}  ({r['meta'].get('problem')}, {r['meta'].get('emotion')})"
            f" - {verdict}, scores={r.get('scores')}"
        )
        probs = r.get("problems", {})
        if any(probs.values()):
            lines.append(f"_auto-filter flags: {probs}_")
        for i, t in enumerate(r["en"]):
            who = "User" if t["role"] == "user" else "Supporter"
            lines += [
                f"**{who}**",
                f"- EN: {t['text']}",
                f"- HI: {r['hindi'][i]}",
                f"- HG: {r['hinglish'][i]}",
            ]
        lines.append("")
    return "\n".join(lines)
