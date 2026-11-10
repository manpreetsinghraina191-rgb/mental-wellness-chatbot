"""
Preprocessing node for the Hindi / Hinglish / English mental-health chatbot.

Responsibilities:
  1. Clean and normalize the raw message
  2. Detect script: devanagari | roman_hindi | english | mixed
  3. Build a Devanagari copy of Romanized Hindi for retrieval and crisis checks
  4. Decide which script the reply should use

Changes vs. the original prototype (each one has a regression test):
  - English words that are also Hinglish shorthand ("he", "main", "par", "see") no longer
    flip an English message to Hinglish.
  - Stretched words ("nahiiii", "bahuuut") are recognised after collapsing.
  - Punctuation is preserved when spelling variants are replaced.
  - Short acknowledgements ("hmm", "ok") that inherit the previous script get a softer,
    explicit reply instruction and a `script_inherited` flag.
  - IndicXlit loads lazily (no slow model load at import) and an LLM fallback can be injected.

IndicXlit is optional (needs Python 3.10, see README). Without it, `to_devanagari` returns
None and the LLM fallback (if provided) is used.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal

Script = Literal["devanagari", "roman_hindi", "english", "mixed"]

# --- Normalization ---------------------------------------------------------------------------

# Safe to apply to every Latin message: these spellings are not real English words.
HINGLISH_VARIANTS: dict[str, str] = {
    "nahin": "nahi", "nai": "nahi", "nhi": "nahi", "ni": "nahi",
    "bht": "bahut", "boht": "bahut", "bohot": "bahut", "bhut": "bahut",
    "kyu": "kyun", "kyon": "kyun", "q": "kyun",
    "h": "hai", "hy": "hai",
    "mje": "mujhe", "mujhko": "mujhe",
    "acha": "accha", "achha": "accha",
    "yr": "yaar", "yar": "yaar",
    "kch": "kuch", "kuchh": "kuch",
    "pta": "pata",
    "hu": "hoon", "hun": "hoon", "hoo": "hoon",
}  # fmt: skip

# Only applied once we already know the message is Hinglish ("he" is an English pronoun).
AMBIGUOUS_VARIANTS: dict[str, str] = {"he": "hai"}

_LATIN_WORD = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
_STRETCH3 = re.compile(r"(.)\1{2,}")
_ANY_REPEAT = re.compile(r"(.)\1+")


def _normalize_word(token: str, aggressive: bool) -> str:
    m = _LATIN_WORD.match(token)
    if not m:
        return _STRETCH3.sub(r"\1\1", token)
    pre, core, post = m.groups()
    lower = core.lower()
    table = {**HINGLISH_VARIANTS, **AMBIGUOUS_VARIANTS} if aggressive else HINGLISH_VARIANTS
    if lower in table:
        return pre + table[lower] + post
    if _STRETCH3.search(lower):  # user stretched the word: "nahiiii", "bahuuut"
        single = _ANY_REPEAT.sub(r"\1", lower)
        if single in table:
            return pre + table[single] + post
        if single in HINDI_MARKERS:
            return pre + single + post
        return pre + _STRETCH3.sub(r"\1\1", core) + post
    return token


def normalize(text: str, aggressive: bool = False) -> str:
    """NFC, collapse whitespace and stretched letters, map Hinglish spelling variants."""
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", " ", text)
    if not text:
        return text
    return " ".join(_normalize_word(w, aggressive) for w in text.split(" "))


# --- Script detection ------------------------------------------------------------------------

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# Romanized Hindi words that are NOT common English words.
STRONG_HINDI_MARKERS = {
    "hai", "hain", "nahi", "mujhe", "mera", "meri", "mere", "kya", "kyun", "kaise",
    "bahut", "kuch", "bhi", "toh", "aur", "lekin", "raha", "rahi", "rahe", "tha", "thi",
    "kar", "karna", "lag", "yaar", "accha", "sab", "koi", "kab", "abhi", "pata", "mann",
    "wala", "wali", "hoon", "mein", "haan", "theek", "kyunki", "lagta", "lagti", "neend",
    "akela", "akeli", "dimag", "pareshan", "dar", "ghar", "kal", "aaj", "zindagi", "jeena",
    "ji", "aap", "tum", "apna", "apni", "chahiye", "sakta", "sakti", "hota", "hoti",
    # common Hinglish function / everyday words (added after the first eval baseline)
    "bas", "kabhi", "pe", "likh", "kisne", "kaun", "kaunsa", "kahan", "yahan", "wahan",
    "saal", "pichle", "agle", "jaun", "jau", "karu", "karun", "batao", "bata", "chahta",
    "chahti", "unka", "uska", "uski", "mujhse", "tujhe", "tera", "teri", "kuchh", "zyada",
    "thoda", "jaldi", "phir", "fir", "waise", "matlab", "sach", "bilkul", "baat", "logon",
    "dost", "padhai", "naukri",
    # deliberately NOT here (also common English words): hum, mat, log, do, to, us, bat
}  # fmt: skip
# Hinglish function words that are also English words or very short/ambiguous.
WEAK_HINDI_MARKERS = {"main", "hum", "ho", "ke", "ki", "ka", "ko", "se", "par", "ab", "jab", "dil"}
HINDI_MARKERS = STRONG_HINDI_MARKERS | WEAK_HINDI_MARKERS

SHORT_MESSAGES = {
    "ok", "okay", "k", "hmm", "hm", "haan", "ha", "nahi", "thanks", "thank you", "acha",
    "accha", "ji", "theek hai", "yes", "no",
}  # fmt: skip


def is_short_message(text: str) -> bool:
    return text.lower().strip(" .!?") in SHORT_MESSAGES


def detect_script(text: str, previous_script: str | None = None) -> Script:
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev == 0 and lat == 0:  # emoji / punctuation only
        return _as_script(previous_script)
    if previous_script and is_short_message(text):
        return _as_script(previous_script)
    if dev and lat and min(dev, lat) / (dev + lat) > 0.2:
        return "mixed"
    if dev > lat:
        return "devanagari"
    tokens = re.findall(r"[a-z]+", text.lower())
    strong = sum(t in STRONG_HINDI_MARKERS for t in tokens)
    weak = sum(t in WEAK_HINDI_MARKERS for t in tokens)
    ratio = (strong + weak) / len(tokens)
    if strong >= 1 and ratio >= 0.15:
        return "roman_hindi"
    if strong >= 1 and len(tokens) <= 4:  # "kaise ho", "neend nahi"
        return "roman_hindi"
    return "english"


def _as_script(value: str | None) -> Script:
    if value in ("devanagari", "roman_hindi", "english", "mixed"):
        return value  # type: ignore[return-value]
    return "english"


# --- Transliteration -------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_xlit() -> Any | None:
    if os.getenv("TAI_DISABLE_XLIT") == "1":
        return None
    try:
        from ai4bharat.transliteration import XlitEngine

        return XlitEngine("hi", beam_width=4, rescore=True)
    except Exception:
        return None


def _looks_hindi(word: str) -> bool:
    w = word.lower()
    return w.endswith(("na", "ne", "ta", "ti", "te", "kar", "wala", "wali", "iya", "aa"))


def to_devanagari(text: str) -> str | None:
    """Transliterate Hindi-looking words only; keep English words like 'anxiety' in Latin."""
    xlit = _get_xlit()
    if xlit is None:
        return None
    out = []
    for w in text.split():
        core = w.lower().strip(".,!?")
        if core in HINDI_MARKERS or _looks_hindi(core):
            out.append(xlit.translit_word(core, topk=1)["hi"][0])
        else:
            out.append(w)
    return " ".join(out)


TRANSLIT_PROMPT = (
    "Convert the Hindi words in this Romanized Hindi text to Devanagari. "
    "Keep English words (like anxiety, exam, stress) in Latin script. "
    "Return only the converted text.\n\nText: {text}"
)

# Any callable prompt -> completion string (wraps a LangChain LLM, an API client, or a mock).
TextLLM = Callable[[str], str]


def to_devanagari_llm(text: str, llm: TextLLM) -> str | None:
    try:
        out = llm(TRANSLIT_PROMPT.format(text=text)).strip()
    except Exception:
        return None
    return out if DEVANAGARI.search(out) else None


# --- Reply instructions ----------------------------------------------------------------------

REPLY_INSTRUCTIONS: dict[Script, str] = {
    "devanagari": "Reply in Hindi using Devanagari script.",
    "roman_hindi": "Reply in Hinglish using Roman script, the way the user writes. "
    "Mix English words in naturally, as they do.",
    "english": "Reply in English.",
    "mixed": "Reply in Hindi (Devanagari), keeping the English words the user used.",
}

INHERITED_NOTE = (
    " The user's latest message is a short acknowledgement ({msg!r}), so keep using the same "
    "language and script as the conversation so far. Keep the reply short and gently invite "
    "them to share more if they want to."
)


def reply_instruction(script: Script, inherited: bool, message: str = "") -> str:
    base = REPLY_INSTRUCTIONS[script]
    return base + INHERITED_NOTE.format(msg=message) if inherited else base


# --- LangGraph node --------------------------------------------------------------------------


def make_preprocess_node(llm: TextLLM | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the preprocess node. `llm` is the transliteration fallback when IndicXlit is absent."""

    def preprocess_node(state: dict[str, Any]) -> dict[str, Any]:
        raw: str = state["raw_input"]
        previous: str | None = state.get("script")
        clean = normalize(raw)
        script = detect_script(clean, previous)
        inherited = (
            previous is not None
            and script == previous
            and (is_short_message(clean) or not (DEVANAGARI.search(clean) or LATIN.search(clean)))
        )
        if script == "roman_hindi":
            clean = normalize(raw, aggressive=True)
        queries = [clean]
        if script == "roman_hindi" and not inherited:
            dev = to_devanagari(clean)
            if dev is None and llm is not None:
                dev = to_devanagari_llm(clean, llm)
            if dev and dev != clean:
                queries.append(dev)
        return {
            "clean_input": clean,
            "script": script,
            "script_inherited": inherited,
            "retrieval_queries": queries,
            "crisis_check_texts": list(queries),
            "reply_instruction": reply_instruction(script, inherited, clean),
        }

    return preprocess_node


preprocess_node = make_preprocess_node()
