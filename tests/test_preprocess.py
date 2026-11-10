import pytest

from tai_bot import preprocess as pp
from tai_bot.preprocess import detect_script, make_preprocess_node, normalize


@pytest.fixture(autouse=True)
def no_xlit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never load IndicXlit; transliteration is exercised via the mock LLM."""
    monkeypatch.setattr(pp, "_get_xlit", lambda: None)


# --- normalize -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yaar mujhe bht anxiety ho rahi h", "yaar mujhe bahut anxiety ho rahi hai"),
        ("neend nhi aati", "neend nahi aati"),
        ("nhi.", "nahi."),  # punctuation preserved
        ("nahiiiii", "nahi"),  # stretched word recognised
        ("bahuuuut thak gaya", "bahut thak gaya"),
        ("sooooo tired", "soo tired"),  # unknown stretched word only collapsed to 2
        ("  extra    spaces  ", "extra spaces"),
        ("मुझे   नींद नहीं आती", "मुझे नींद नहीं आती"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_ambiguous_variant_only_in_aggressive_mode() -> None:
    assert normalize("He never listens") == "He never listens"
    assert normalize("kal se neend nahi aa rahi he", aggressive=True).endswith("rahi hai")


def test_english_see_not_turned_into_se() -> None:
    assert normalize("see you") == "see you"


# --- detect_script ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("मुझे रात को नींद नहीं आती", "devanagari"),
        ("yaar mujhe bahut anxiety ho rahi hai", "roman_hindi"),
        ("I can't stop overthinking at night", "english"),
        ("मुझे exam का बहुत stress है", "mixed"),
        ("kaise ho", "roman_hindi"),
        ("neend nahi", "roman_hindi"),
        ("haan", "roman_hindi"),
        ("hi", "english"),
        # regressions: English words that are also Hinglish markers
        ("The main problem is my exams", "english"),
        ("He doesn't talk to me anymore", "english"),
        ("I feel like I'm on par with nobody", "english"),
    ],
)
def test_detect_script(text: str, expected: str) -> None:
    assert detect_script(text) == expected


@pytest.mark.parametrize("prev", ["devanagari", "roman_hindi", "english", "mixed"])
@pytest.mark.parametrize("msg", ["hmm", "ok", "Okay.", "haan", "thanks"])
def test_short_message_inherits_previous_script(prev: str, msg: str) -> None:
    assert detect_script(msg, prev) == prev


def test_emoji_only_inherits_previous_script() -> None:
    assert detect_script("🙂🙏", "devanagari") == "devanagari"
    assert detect_script("🙂") == "english"


def test_short_message_without_history() -> None:
    assert detect_script("hmm") == "english"
    assert detect_script("ok") == "english"


# --- preprocess_node -------------------------------------------------------------------------


def test_node_hmm_after_mixed_gets_sensible_instruction() -> None:
    """Known issue from the prototype: 'hmm' after a mixed message."""
    out = make_preprocess_node()({"raw_input": "hmm", "script": "mixed"})
    assert out["script"] == "mixed"
    assert out["script_inherited"] is True
    instr = out["reply_instruction"]
    assert "Devanagari" in instr
    assert "short acknowledgement" in instr
    assert "keep the reply short" in instr.lower()


def test_node_fresh_message_not_inherited() -> None:
    out = make_preprocess_node()({"raw_input": "मुझे exam का बहुत stress है", "script": "english"})
    assert out["script"] == "mixed"
    assert out["script_inherited"] is False
    assert "short acknowledgement" not in out["reply_instruction"]


def test_node_english() -> None:
    out = make_preprocess_node()({"raw_input": "I can't sleep"})
    assert out["script"] == "english"
    assert out["retrieval_queries"] == ["I can't sleep"]
    assert out["crisis_check_texts"] == ["I can't sleep"]


def test_node_hinglish_uses_llm_fallback_for_devanagari_copy() -> None:
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "यार मुझे बहुत anxiety हो रही है"

    out = make_preprocess_node(llm=fake_llm)({"raw_input": "yaar mujhe bht anxiety ho rahi h"})
    assert out["script"] == "roman_hindi"
    assert out["retrieval_queries"] == [
        "yaar mujhe bahut anxiety ho rahi hai",
        "यार मुझे बहुत anxiety हो रही है",
    ]
    assert out["crisis_check_texts"] == out["retrieval_queries"]
    assert len(calls) == 1


def test_node_hinglish_without_any_transliterator_still_works() -> None:
    out = make_preprocess_node()({"raw_input": "mujhe neend nahi aati"})
    assert out["retrieval_queries"] == ["mujhe neend nahi aati"]


def test_llm_fallback_rejects_non_devanagari_output() -> None:
    out = make_preprocess_node(llm=lambda _p: "sorry, I can't")(
        {"raw_input": "mujhe neend nahi aati"}
    )
    assert out["retrieval_queries"] == ["mujhe neend nahi aati"]


def test_llm_fallback_survives_llm_error() -> None:
    def broken(_p: str) -> str:
        raise TimeoutError

    out = make_preprocess_node(llm=broken)({"raw_input": "mujhe neend nahi aati"})
    assert out["retrieval_queries"] == ["mujhe neend nahi aati"]


def test_aggressive_normalization_applied_for_hinglish() -> None:
    out = make_preprocess_node()({"raw_input": "kal se neend nahi aa rahi he"})
    assert out["clean_input"] == "kal se neend nahi aa rahi hai"
