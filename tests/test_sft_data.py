import itertools
import json
from pathlib import Path

from tai_bot.safety_terms import find_unsafe, has_crisis_hint
from tai_bot.sft_data import (
    build_sft,
    build_translate_prompt,
    extract_esconv,
    filter_records,
    merge_turns,
    parse_translation,
    read_jsonl,
    translate_samples,
)


def turn(speaker: str, text: str, strategy: str | None = None) -> dict[str, object]:
    t: dict[str, object] = {"speaker": speaker, "content": text, "annotation": {}}
    if strategy:
        t["annotation"] = {"strategy": strategy}
    return t


def dialog(turns: list[dict[str, object]], empathy: str | None = "5") -> dict[str, object]:
    seeker = {"empathy": empathy} if empathy else {}
    return {
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "survey_score": {"seeker": seeker},
        "dialog": turns,
    }


GOOD = dialog(
    [
        turn("seeker", "Hello"),
        turn("supporter", "Hi, what would you like to talk about?", "Question"),
        turn("seeker", "I am scared I will lose my job."),
        turn("seeker", "My boss keeps criticising me."),
        turn("supporter", "That sounds really stressful, being criticised every day.", "Reflection of feelings"),
        turn("seeker", "Yes it is."),
        turn("supporter", "I also lost my job once and it was so hard for me.", "Self-disclosure"),
        turn("seeker", "Really?"),
        turn("supporter", "It makes sense that you feel anxious when your work feels so uncertain.", "Affirmation and Reassurance"),
    ]
)  # fmt: skip


# --- safety terms ----------------------------------------------------------------------------


def test_find_unsafe() -> None:
    assert find_unsafe("Maybe ask your doctor about medication") == "medication"
    assert find_unsafe("It sounds like you have depression") == "sounds like you have"
    assert find_unsafe("Just calm down") == "calm down"
    assert find_unsafe("That sounds really hard. What helps you relax?") is None


def test_crisis_hint() -> None:
    assert has_crisis_hint("sometimes I want to die")
    assert not has_crisis_hint("this exam is killing me")  # idiom, not a crisis hint


# --- extract ---------------------------------------------------------------------------------


def test_merge_turns_merges_consecutive_speakers() -> None:
    merged = merge_turns(GOOD["dialog"])  # type: ignore[arg-type]
    roles = [t["role"] for t in merged]
    assert roles == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert merged[2]["text"] == "I am scared I will lose my job. My boss keeps criticising me."


def test_extract_skips_self_disclosure_and_keeps_good_turns() -> None:
    samples, stats = extract_esconv([GOOD], max_per_dialog=10)  # type: ignore[list-item]
    responses = [s["en"][-1]["text"] for s in samples]
    assert not any("I also lost my job" in r for r in responses)
    assert any("makes sense that you feel anxious" in r for r in responses)
    for s in samples:
        roles = [t["role"] for t in s["en"]]
        assert roles[0] == "user" and roles[-1] == "assistant"
        assert all(a != b for a, b in itertools.pairwise(roles))  # alternating
    assert stats["drop_turn:strategy"] >= 1


def test_extract_drops_low_empathy_and_crisis_dialogs() -> None:
    low = dialog(GOOD["dialog"], empathy="2")  # type: ignore[arg-type]
    missing = dialog(GOOD["dialog"], empathy=None)  # type: ignore[arg-type]
    crisis = dialog([*GOOD["dialog"], turn("seeker", "honestly I want to die")])  # type: ignore[list-item]
    samples, stats = extract_esconv([low, missing, crisis])
    assert samples == []
    assert stats["drop_dialog:low_or_missing_empathy"] == 2
    assert stats["drop_dialog:crisis_language"] == 1


# --- translate -------------------------------------------------------------------------------


def _sample(id_: str = "s1") -> dict[str, object]:
    return {
        "id": id_,
        "source": "esconv",
        "meta": {"problem": "job crisis", "emotion": "anxiety"},
        "en": [
            {"role": "user", "text": "I am scared I will lose my job."},
            {
                "role": "assistant",
                "text": "That sounds really stressful. What worries you the most?",
            },
        ],
    }


GOOD_TRANSLATION = {
    "hindi": [
        "मुझे डर है कि मेरी नौकरी चली जाएगी।",
        "यह सच में बहुत स्ट्रेस वाली बात है। आपको सबसे ज़्यादा किस बात की चिंता है?",
    ],
    "hinglish": [
        "Mujhe darr hai ki meri job chali jayegi.",
        "Yeh sach mein bahut stressful hai. Aapko sabse zyada kis baat ki chinta hai?",
    ],
}


def test_translate_prompt_contains_items() -> None:
    prompt = build_translate_prompt([_sample()])  # type: ignore[list-item]
    assert '"id": "s1"' in prompt and "lose my job" in prompt


def test_parse_translation_handles_fences_and_bad_items() -> None:
    raw = "```json\n" + json.dumps(
        {"items": [{"id": "s1", **GOOD_TRANSLATION}, {"id": "s2", "hindi": ["only one"], "hinglish": []}]}
    ) + "\n```"  # fmt: skip
    out, errors = parse_translation(raw, [_sample("s1"), _sample("s2")])  # type: ignore[list-item]
    assert [r["id"] for r in out] == ["s1"]
    assert errors == ["s2"]
    assert parse_translation("not json", [_sample()])[1] == ["s1"]  # type: ignore[list-item]


def test_translate_samples_is_resumable(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        ids = [i for i in ("s1", "s2", "s3") if f'"id": "{i}"' in prompt]
        return json.dumps({"items": [{"id": i, **GOOD_TRANSLATION} for i in ids]})

    out = tmp_path / "t.jsonl"
    samples = [_sample("s1"), _sample("s2")]
    translate_samples(samples, fake_llm, out, batch_size=5, log=lambda _: None)  # type: ignore[arg-type]
    assert len(calls) == 1
    stats = translate_samples([*samples, _sample("s3")], fake_llm, out, log=lambda _: None)  # type: ignore[list-item]
    assert len(calls) == 2 and '"id": "s1"' not in calls[1]
    assert stats["already_done"] == 2
    assert [r["id"] for r in read_jsonl(out)] == ["s1", "s2", "s3"]


# --- filter + format -------------------------------------------------------------------------


def test_filter_catches_wrong_script_and_unsafe_terms() -> None:
    good = {**_sample("good"), **GOOD_TRANSLATION}
    bad = {
        **_sample("bad"),
        "hindi": ["I am scared.", "That sounds stressful, what worries you?"],  # not Devanagari
        "hinglish": ["मुझे डर है", "Aapko medication lena chahiye, kya lagta hai aapko?"],
    }
    records, _stats = filter_records([good, bad])
    assert records[0]["keep"] == {"en": True, "hindi": True, "hinglish": True}
    assert records[1]["keep"]["hindi"] is False
    assert records[1]["keep"]["hinglish"] is False
    assert "script" in records[1]["problems"]["hindi"]
    assert "devanagari_in_hinglish" in records[1]["problems"]["hinglish"]
    assert "unsafe_term" in records[1]["problems"]["hinglish"]


def test_build_sft_splits_by_source_and_uses_language_system_prompt() -> None:
    recs = [{**_sample(f"s{i}"), **GOOD_TRANSLATION} for i in range(20)]
    filtered, _ = filter_records(recs)
    train, val, _stats = build_sft(filtered, val_frac=0.1)
    assert len(train) + len(val) == 60
    train_src = {e["source_id"] for e in train}
    val_src = {e["source_id"] for e in val}
    assert train_src.isdisjoint(val_src)
    hg = next(e for e in train if e["lang"] == "hinglish")
    assert hg["messages"][0]["role"] == "system"
    assert "Hinglish" in hg["messages"][0]["content"]
    assert "Tele-MANAS 14416" in hg["messages"][0]["content"]
    assert [m["role"] for m in hg["messages"][1:]] == ["user", "assistant"]


# --- regressions from the first pilot (2026-11-10) -------------------------------------------


def test_indirect_crisis_phrasing_is_caught() -> None:
    assert has_crisis_hint("sometimes I feel like it would be so much easier to just be dead")
    assert has_crisis_hint("I just want to end it all")


def test_rule_filters_for_supporter_turns() -> None:
    from tai_bot.sft_data import response_problem

    assert (
        response_problem("My school was in a small city and the professors were never busy.")
        == "self_disclosure"
    )
    assert (
        response_problem("Definitely. I am assuring you will be fine very soon.") == "false_promise"
    )
    assert (
        response_problem("Why not try to work for Uber or Lyft to supplement your income?")
        == "out_of_context"
    )
    assert (
        response_problem("That would be an absolutely terrible thing to do, honestly.")
        == "judgmental"
    )
    assert response_problem("It makes sense that you feel tired after such a long week.") is None


def test_window_starts_after_a_bad_earlier_supporter_turn() -> None:
    d = dialog(
        [
            turn("seeker", "I feel so alone these days."),
            turn("supporter", "I lost my job once too and it felt the same for me.", "Question"),
            turn("seeker", "Really? I just miss my friends."),
            turn(
                "supporter",
                "It sounds like you really miss the people who matter to you.",
                "Reflection of feelings",
            ),
        ]
    )
    samples, _ = extract_esconv([d], max_per_dialog=5)  # type: ignore[list-item]
    assert len(samples) == 1
    texts = [t["text"] for t in samples[0]["en"]]
    assert texts == [
        "Really? I just miss my friends.",
        "It sounds like you really miss the people who matter to you.",
    ]


def test_low_quality_score_drops_all_languages() -> None:
    rec = {**_sample("q"), **GOOD_TRANSLATION, "scores": [2], "issues": "false promise"}
    records, stats = filter_records([rec])
    assert records[0]["keep"] == {"en": False, "hindi": False, "hinglish": False}
    assert stats["dropped_low_quality_source"] == 1
    ok = {**_sample("ok"), **GOOD_TRANSLATION, "scores": [5]}
    assert filter_records([ok])[0][0]["keep"]["en"] is True


def test_parse_translation_reads_scores() -> None:
    raw = json.dumps({"items": [{"id": "s1", **GOOD_TRANSLATION, "scores": [4], "issues": ""}]})
    out, _ = parse_translation(raw, [_sample("s1")])  # type: ignore[list-item]
    assert out[0]["scores"] == [4]
    raw_bad = json.dumps({"items": [{"id": "s1", **GOOD_TRANSLATION, "scores": [4, 5]}]})
    assert parse_translation(raw_bad, [_sample("s1")])[0][0]["scores"] is None  # type: ignore[list-item]
