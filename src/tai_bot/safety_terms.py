"""Shared keyword lists: used to clean training data and (later) by the output rails.

These are deliberately simple word lists. They are NOT the crisis detector (see crisis.py);
`CRISIS_HINTS` here only keeps crisis conversations out of the tone fine-tuning data,
because crisis replies must come from the fixed protocol, not from learned text.
"""

from __future__ import annotations

import re

MEDICATION_TERMS = [
    "sertraline", "zoloft", "fluoxetine", "prozac", "escitalopram", "lexapro", "citalopram",
    "paroxetine", "paxil", "venlafaxine", "effexor", "duloxetine", "cymbalta", "bupropion",
    "wellbutrin", "mirtazapine", "trazodone", "amitriptyline", "alprazolam", "xanax",
    "clonazepam", "klonopin", "lorazepam", "ativan", "diazepam", "valium", "lithium",
    "quetiapine", "seroquel", "olanzapine", "risperidone", "aripiprazole", "abilify",
    "adderall", "ritalin", "methylphenidate", "zolpidem", "ambien", "melatonin",
    "antidepressant", "antidepressants", "anti-depressant", "benzodiazepine", "ssri", "ssris",
    "sleeping pills", "sleeping pill", "medication", "medications", "meds", "prescription",
    "dosage", "mg",
]  # fmt: skip

DIAGNOSIS_PHRASES = [
    "you have depression", "you are depressed", "you're depressed", "you have anxiety disorder",
    "you have an anxiety disorder", "you have bipolar", "you are bipolar", "you have ptsd",
    "you have ocd", "you have adhd", "sounds like you have", "you might have", "you may have",
    "you suffer from", "diagnose", "diagnosis", "diagnosed",
]  # fmt: skip

DISMISSIVE_PHRASES = [
    "get over it", "just relax", "calm down", "stop overthinking", "not a big deal",
    "others have it worse", "man up", "stop being", "you're overreacting", "just be positive",
]  # fmt: skip

CRISIS_HINTS = [
    "suicide", "suicidal", "kill myself", "killing myself", "end my life", "ending my life",
    "self harm", "self-harm", "cutting myself", "cut myself", "want to die", "wanna die",
    "overdose", "take my own life", "no reason to live", "better off dead",
    # indirect phrasings (one slipped through the first pilot: "easier to just be dead")
    "be dead", "being dead", "not be here", "not wake up", "end it all", "no point living",
    "no point in living", "disappear forever", "can't go on", "cannot go on", "hurt myself",
]  # fmt: skip

# Supporter-turn quality rules used only for cleaning training data.
SELF_DISCLOSURE = re.compile(
    r"\b(my (school|job|wife|husband|mom|mother|dad|father|kids?|son|daughter|friend|boss|"
    r"family|experience|own background|sister|brother)|when i was|i was in a similar|"
    r"i (once|used to|went through|lost my|had a)|happened to me|i've been through|"
    r"same thing happened|for myself)\b",
    re.IGNORECASE,
)
FALSE_PROMISE = re.compile(
    r"\b(you will be fine|you'll be fine|everything will be (fine|ok|okay)|i assure|"
    r"i am assuring|i promise|i guarantee|will definitely)\b",
    re.IGNORECASE,
)
OUT_OF_CONTEXT = re.compile(
    r"\b(pandemic|covid|coronavirus|quarantine|lockdown|uber|lyft|dollars?|thanksgiving|"
    r"lol|lmao)\b|\$\d",
    re.IGNORECASE,
)
JUDGMENTAL = re.compile(
    r"\b(terrible thing|you should not have|you shouldn't have|your fault|that's wrong of you|"
    r"bad apples|health problems? (can|will)|lead to (so many )?diseases?)\b",
    re.IGNORECASE,
)


def _compile(terms: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


_UNSAFE = _compile(MEDICATION_TERMS + DIAGNOSIS_PHRASES + DISMISSIVE_PHRASES)
_CRISIS = _compile(CRISIS_HINTS)


def find_unsafe(text: str) -> str | None:
    """Return the first medication / diagnosis / dismissive term found, else None."""
    m = _UNSAFE.search(text)
    return m.group(1).lower() if m else None


def has_crisis_hint(text: str) -> bool:
    return _CRISIS.search(text) is not None
