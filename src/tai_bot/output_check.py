"""Output rails: checks on every generated reply before the user sees it.

  1. No medicine names, diagnoses or dismissive phrases (shared keyword lists).
  2. The reply is in the script the user wrote in.
  3. Length (the persona asks for 2-5 sentences; we allow some slack).
  4. No invented phone numbers: the only numbers allowed are the real helplines.

On failure the graph regenerates once with a correction note, then falls back to a fixed,
safe reply in the user's script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tai_bot.preprocess import Script, detect_script
from tai_bot.safety_terms import find_unsafe

MAX_WORDS = 130
ALLOWED_NUMBERS = {"14416", "112", "1800-891-4416", "18008914416"}
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d -]{5,}\d)(?!\d)")

# Which detected scripts are acceptable for each expected reply script.
ACCEPTABLE: dict[Script, set[str]] = {
    "english": {"english"},
    "roman_hindi": {"roman_hindi"},
    "devanagari": {"devanagari", "mixed"},
    "mixed": {"devanagari", "mixed"},
}


@dataclass
class CheckResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check_reply(reply: str, script: Script) -> CheckResult:
    problems: list[str] = []
    if not reply.strip():
        return CheckResult(False, ["empty"])
    term = find_unsafe(reply)
    if term:
        problems.append(f"unsafe_term:{term}")
    got = detect_script(reply)
    if got not in ACCEPTABLE[script]:
        problems.append(f"wrong_script:{got}")
    if len(reply.split()) > MAX_WORDS:
        problems.append("too_long")
    for m in _PHONE.finditer(reply):
        number = re.sub(r"[ +]", "", m.group(1))
        if number not in ALLOWED_NUMBERS:
            problems.append(f"unknown_number:{m.group(1).strip()}")
    return CheckResult(not problems, problems)


CORRECTION_NOTES = {
    "unsafe_term": "Do not mention any medicine, diagnosis, or dismissive phrase.",
    "wrong_script": "You replied in the wrong language/script. {instruction}",
    "too_long": "Keep it to 2 to 5 short sentences.",
    "unknown_number": "Do not give any phone number other than Tele-MANAS 14416 or 112.",
    "empty": "Write a short, warm reply.",
}


def correction_note(problems: list[str], instruction: str) -> str:
    kinds = {p.split(":")[0] for p in problems}
    notes = [CORRECTION_NOTES[k].format(instruction=instruction) for k in sorted(kinds)]
    return "Your previous draft had problems. " + " ".join(notes)


# Fixed, safe fallback when the model fails twice.
SAFE_FALLBACK: dict[Script, str] = {
    "english": (
        "Thank you for sharing this with me. It sounds like a lot to carry, and your feelings "
        "make sense. I'm here to listen. Would you like to tell me a little more about what's "
        "been going on?"
    ),
    "roman_hindi": (
        "Mujhse share karne ke liye shukriya. Lag raha hai aap bahut kuch jhel rahe hain, aur "
        "aapki feelings bilkul samajh mein aati hain. Main sunne ke liye yahin hoon. Kya aap "
        "thoda aur bataana chahenge ki kya chal raha hai?"
    ),
    "devanagari": (
        "मुझसे साझा करने के लिए शुक्रिया। लग रहा है आप बहुत कुछ झेल रहे हैं, और आपकी भावनाएँ "
        "बिल्कुल समझ में आती हैं। मैं सुनने के लिए यहीं हूँ। क्या आप थोड़ा और बताना चाहेंगे कि "
        "क्या चल रहा है?"
    ),
    "mixed": "",
}
SAFE_FALLBACK["mixed"] = SAFE_FALLBACK["devanagari"]
