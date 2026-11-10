"""Crisis detection gets the most tests: direct and indirect phrasing in all three scripts,
both verb genders, idioms that must NOT trigger, and the fixed protocol templates."""

import pytest

from tai_bot.crisis import (
    CONSENT_NO,
    CONSENT_YES,
    HELPLINE_REMINDER,
    PROTOCOL,
    build_classifier_prompt,
    check_crisis,
    classify_consent,
    fold,
    lexicon_check,
    max_risk,
    parse_risk,
)
from tai_bot.preprocess import detect_script

# Phrases written independently of the eval set (different wording on purpose).
HIGH = [
    # English
    "honestly I just want to die",
    "I've been having suicidal thoughts",
    "I keep thinking about ending my life",
    "everyone would be better off without me",
    "I don't want to wake up tomorrow",
    "I burned myself last night",
    "how many sleeping tablets would be lethal",
    "I wrote a suicide note",
    # Hinglish, both genders
    "mujhe ab marna hai",
    "main marna chahti hoon",
    "zindagi khatam kar dunga aaj",
    "jeene ka dil nahi karta ab",
    "khud ko hurt karne ka mann karta hai",
    "nas kaat lungi",
    "kal ke baad main nahi rahunga",
    "chhat se kood jaungi",
    "ab aur nahi seh sakti",
    # Devanagari, both genders
    "मैं मरना चाहती हूँ",
    "ज़िंदगी खत्म कर दूँगा",
    "मुझे जीना नहीं है, जीने का दिल नहीं करता",
    "मेरे मरने के बाद किसी को फर्क नहीं पड़ेगा",
    "मैंने कलाई काट ली",
    "पुल से कूद जाऊँगा",
    "ख़ुदकुशी के बारे में सोचता हूँ",
    # mixed
    "मुझे लगता है suicide ही रास्ता है",
]

MEDIUM = [
    "I feel so empty inside",
    "I'm just a burden to my family",
    "kuch bhi accha nahi lagta yaar",
    "main bas ek bojh hoon",
    "मैं किसी काम का नहीं",
    "बहुत निराशा है",
]

SAFE = [
    "exam ne meri jaan le li",
    "mar gaya yaar itna kaam karke",
    "agar fail hua toh papa jaan se maar denge",
    "this assignment is killing me",
    "I'm dying to go home for Diwali",
    "I could die of embarrassment lol",
    "इस परीक्षा ने तो मेरी जान ले ली",
    "मर गया यार आज तो",
    "I can't sleep at night",
    "boss ne sabke saamne daanta",
    "kya mujhe depression hai?",
    "What is Tele-MANAS?",
    "my phone battery is dead",
    "दिल बहुत तेज़ धड़क रहा है",
]


@pytest.mark.parametrize("text", HIGH)
def test_lexicon_high(text: str) -> None:
    assert lexicon_check([text]).risk == "high", text


@pytest.mark.parametrize("text", MEDIUM)
def test_lexicon_medium(text: str) -> None:
    assert lexicon_check([text]).risk == "medium", text


@pytest.mark.parametrize("text", SAFE)
def test_lexicon_does_not_fire_on_idioms_or_everyday_text(text: str) -> None:
    assert lexicon_check([text]).risk == "none", (text, lexicon_check([text]).matches)


def test_transliterated_copy_is_also_checked() -> None:
    # A Hinglish phrasing the Roman patterns miss, but whose Devanagari copy matches.
    assert lexicon_check(["zyada sochti hoon", "मेरे जाने के बाद सब खुश"]).risk == "high"


def test_fold_handles_nukta_and_chandrabindu() -> None:
    assert fold("ज़िंदगी") == fold("जिंदगी")
    assert fold("हाँ") == fold("हां")


def test_max_risk() -> None:
    assert max_risk("none", "medium", "low") == "medium"
    assert max_risk("high", "none") == "high"


# --- LLM layer (mocked) ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"risk": "high"}', "high"),
        ('```json\n{"risk": "medium"}\n```', "medium"),
        ("Risk: LOW", "low"),
        ('{"risk": "extreme"}', None),
        ("I cannot help with that", None),
    ],
)
def test_parse_risk(raw: str, expected: str | None) -> None:
    assert parse_risk(raw) == expected


def test_or_combination_llm_catches_what_lexicon_misses() -> None:
    res = check_crisis(["shayad mere bina sab theek rahenge"], llm=lambda _p: '{"risk":"high"}')
    assert res.lexicon_risk in ("none", "high")
    assert res.risk == "high"
    assert res.flagged_for_review


def test_or_combination_lexicon_wins_when_llm_says_none() -> None:
    res = check_crisis(["main suicide karna chahti hoon"], llm=lambda _p: '{"risk":"none"}')
    assert res.lexicon_risk == "high" and res.llm_risk == "none"
    assert res.risk == "high"


def test_llm_failure_keeps_lexicon_result_and_flags_session() -> None:
    def broken(_p: str) -> str:
        raise TimeoutError("ollama down")

    res = check_crisis(["I feel okay today"], llm=broken)
    assert res.risk == "none"
    assert res.llm_error and "TimeoutError" in res.llm_error
    assert res.flagged_for_review


def test_classifier_sees_only_recent_history() -> None:
    seen: list[str] = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return '{"risk":"low"}'

    history = [f"message {i}" for i in range(10)]
    check_crisis(["latest"], history=history, llm=spy, history_turns=3)
    assert "message 9" in seen[0] and "message 7" in seen[0]
    assert "message 6" not in seen[0]
    assert "Latest message: latest" in seen[0]


def test_prompt_without_history() -> None:
    assert "Earlier messages" not in build_classifier_prompt("hi", [])


# --- Fixed protocol --------------------------------------------------------------------------

SCRIPTS = ["english", "roman_hindi", "devanagari", "mixed"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_protocol_has_helplines_and_consent_question(script: str) -> None:
    text = PROTOCOL[script]  # type: ignore[index]
    assert "14416" in text and "112" in text
    assert "?" in text  # asks whether they want help connecting to a counsellor


@pytest.mark.parametrize("script", SCRIPTS)
def test_followup_templates_mention_helpline(script: str) -> None:
    for table in (CONSENT_YES, CONSENT_NO, HELPLINE_REMINDER):
        assert "14416" in table[script]  # type: ignore[index]


@pytest.mark.parametrize(
    ("script", "expected"),
    [("english", "english"), ("roman_hindi", "roman_hindi"), ("devanagari", "devanagari")],
)
def test_protocol_is_written_in_the_users_script(script: str, expected: str) -> None:
    assert detect_script(PROTOCOL[script]) == expected  # type: ignore[index]


def test_mixed_protocol_keeps_english_names() -> None:
    assert "Tele-MANAS" in PROTOCOL["mixed"] and "counsellor" in PROTOCOL["mixed"]


def test_templates_are_gender_neutral_for_the_bot() -> None:
    for t in (PROTOCOL["roman_hindi"], PROTOCOL["devanagari"]):
        for gendered in ("karunga", "karungi", "sakta hoon", "sakti hoon", "करूंगा", "करूँगी"):
            assert gendered not in t


# --- Consent ---------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["haan", "Haan ji", "yes", "ok", "theek hai", "हाँ", "जी", "ठीक है"])
def test_consent_yes(text: str) -> None:
    assert classify_consent(text) == "yes"


@pytest.mark.parametrize("text", ["nahi", "no", "abhi nahi", "नहीं", "rehne do"])
def test_consent_no(text: str) -> None:
    assert classify_consent(text) == "no"


@pytest.mark.parametrize(
    "text", ["haan lekin mujhe darr lagta hai ki koi meri baat nahi samjhega", "hmm", "kyun?"]
)
def test_consent_unclear_goes_back_through_crisis_check(text: str) -> None:
    assert classify_consent(text) == "unclear"
