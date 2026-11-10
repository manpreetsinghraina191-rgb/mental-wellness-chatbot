"""Load and validate the evaluation set.

`data/eval/eval_set.csv` is the source of truth for single-turn prompts (edit it in Excel /
Google Sheets; keep UTF-8). `data/eval/multiturn.jsonl` holds conversations where risk builds
up over several turns, plus benign controls.

The eval set is hand-written and kept separate from all training data.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

LANGS = {"english", "roman_hindi", "devanagari", "mixed"}
RISKS = {"none", "low", "medium", "high"}
TYPES = {
    "support", "technique", "info", "greeting", "offtopic", "jailbreak", "med_request",
    "diagnosis_request", "crisis_direct", "crisis_indirect", "crisis_medium", "idiom_safe",
    "crisis_disguised",
}  # fmt: skip
ACTIONS = {
    "support", "technique", "inform", "greet", "redirect", "refuse", "decline_meds",
    "no_diagnosis", "crisis_protocol", "support_checkin", "escalation_consent",
    "crisis_followup",
}  # fmt: skip
CRISIS_TYPES = {"crisis_direct", "crisis_indirect", "crisis_disguised"}


@dataclass(frozen=True)
class EvalItem:
    id: str
    lang: str
    type: str
    text: str
    expected_risk: str
    expected_action: str
    notes: str = ""


@dataclass(frozen=True)
class Conversation:
    id: str
    lang: str
    kind: str
    turns: list[str]
    expected_risk: list[str]
    expected_action: list[str]
    notes: str = ""


class EvalSetError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise EvalSetError(msg)


def load_eval_set(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            item = EvalItem(
                id=row["id"].strip(),
                lang=row["lang"].strip(),
                type=row["type"].strip(),
                text=(row.get("reviewer_fix") or row["text"]).strip(),
                expected_risk=row["expected_risk"].strip(),
                expected_action=row["expected_action"].strip(),
                notes=row.get("notes", "").strip(),
            )
            _check(bool(item.text), f"{item.id}: empty text")
            _check(item.lang in LANGS, f"{item.id}: bad lang {item.lang!r}")
            _check(item.type in TYPES, f"{item.id}: bad type {item.type!r}")
            _check(item.expected_risk in RISKS, f"{item.id}: bad risk {item.expected_risk!r}")
            _check(item.expected_action in ACTIONS, f"{item.id}: bad action")
            if item.type in CRISIS_TYPES:
                _check(item.expected_risk == "high", f"{item.id}: crisis item must be high")
                _check(item.expected_action == "crisis_protocol", f"{item.id}: crisis action")
            items.append(item)
    ids = [i.id for i in items]
    _check(len(ids) == len(set(ids)), "duplicate ids in eval set")
    return items


def load_multiturn(path: Path) -> list[Conversation]:
    convs: list[Conversation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        c = Conversation(
            id=d["id"],
            lang=d["lang"],
            kind=d["kind"],
            turns=d["turns"],
            expected_risk=d["expected_risk"],
            expected_action=d["expected_action"],
            notes=d.get("notes", ""),
        )
        n = len(c.turns)
        _check(n > 0, f"{c.id}: no turns")
        _check(len(c.expected_risk) == n and len(c.expected_action) == n, f"{c.id}: lengths")
        _check(c.lang in LANGS, f"{c.id}: bad lang")
        _check(all(r in RISKS for r in c.expected_risk), f"{c.id}: bad risk")
        _check(all(a in ACTIONS for a in c.expected_action), f"{c.id}: bad action")
        convs.append(c)
    return convs
