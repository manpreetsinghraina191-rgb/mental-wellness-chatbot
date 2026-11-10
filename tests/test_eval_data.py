from collections import Counter
from pathlib import Path

import pytest

from tai_bot.eval_data import (
    CRISIS_TYPES,
    EvalSetError,
    load_eval_set,
    load_multiturn,
)

EVAL = Path(__file__).resolve().parents[1] / "data" / "eval"


def test_eval_set_loads_and_is_balanced() -> None:
    items = load_eval_set(EVAL / "eval_set.csv")
    assert 100 <= len(items) <= 160
    per_lang = Counter(i.lang for i in items)
    for lang in ("english", "roman_hindi", "devanagari"):
        assert per_lang[lang] >= 40, per_lang
    # every main language covers every behaviour type
    for lang in ("english", "roman_hindi", "devanagari"):
        types = {i.type for i in items if i.lang == lang}
        assert types >= CRISIS_TYPES | {"support", "offtopic", "jailbreak", "idiom_safe"}


def test_crisis_recall_is_measurable_per_language() -> None:
    items = load_eval_set(EVAL / "eval_set.csv")
    for lang in ("english", "roman_hindi", "devanagari"):
        crisis = [i for i in items if i.lang == lang and i.expected_risk == "high"]
        safe_idioms = [i for i in items if i.lang == lang and i.type == "idiom_safe"]
        assert len(crisis) >= 10 and len(safe_idioms) >= 3


def test_multiturn_loads() -> None:
    convs = load_multiturn(EVAL / "multiturn.jsonl")
    assert len(convs) >= 10
    kinds = Counter(c.kind for c in convs)
    assert kinds["escalation"] >= 5 and kinds["benign"] >= 3
    for c in convs:
        if c.kind == "escalation":
            assert c.expected_risk[-1] == "high"
            assert c.expected_risk[0] != "high"  # risk must build, not start high


def test_validation_rejects_bad_rows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "id,lang,type,text,expected_risk,expected_action,notes\n"
        "x1,english,crisis_direct,I want to die,none,support,\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalSetError, match="crisis item must be high"):
        load_eval_set(bad)


def test_reviewer_fix_overrides_text(tmp_path: Path) -> None:
    f = tmp_path / "e.csv"
    f.write_text(
        "id,lang,type,text,expected_risk,expected_action,notes,reviewer_ok,reviewer_fix\n"
        "x1,roman_hindi,support,original,none,support,,no,yaar fixed version\n",
        encoding="utf-8",
    )
    assert load_eval_set(f)[0].text == "yaar fixed version"
