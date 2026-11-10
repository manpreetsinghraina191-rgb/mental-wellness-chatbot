"""LangGraph wiring for Saathi.

  START -> preprocess --(waiting for consent?)--> consent_yes / consent_no -> END
                     \\-> crisis --high, protocol not shown recently--> protocol -> END
                                --high again (follow-up)-----------------> generate
                                --otherwise--> router --off-topic--> redirect -> END
                                                      --jailbreak--> refuse   -> END
                                                      --else-------> generate -> END

  generate = system prompt + recent history (+ retrieved context, step 5) -> local LLM
             -> output rails -> one regeneration -> fixed safe fallback

Every terminal node appends the user message and the reply to `messages`. State lives only in
the in-memory checkpointer for the session; nothing is written to disk.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from tai_bot.crisis import (
    CONSENT_NO,
    CONSENT_YES,
    HELPLINE_REMINDER,
    PROTOCOL,
    check_crisis,
    classify_consent,
)
from tai_bot.llm_clients import ChatLLM, ChatMessage
from tai_bot.output_check import SAFE_FALLBACK, check_reply, correction_note
from tai_bot.preprocess import Script, make_preprocess_node
from tai_bot.prompts import PERSONA
from tai_bot.router import REDIRECT, REFUSE, route_message

TextLLM = Callable[[str], str]
HISTORY_MESSAGES = 8  # last 4 exchanges go to the generator
PROTOCOL_REPEAT_AFTER = 12  # messages; show the full protocol again after this many


class ChatState(TypedDict, total=False):
    messages: Annotated[list[ChatMessage], operator.add]
    raw_input: str
    clean_input: str
    script: Script
    script_inherited: bool
    retrieval_queries: list[str]
    crisis_check_texts: list[str]
    reply_instruction: str
    risk_level: str
    crisis_matches: list[str]
    crisis_llm_risk: str | None
    intent: str
    route: str
    route_source: str
    context: list[str]
    response: str
    output_problems: list[str]
    used_fallback: bool
    flagged_for_review: bool
    awaiting_consent: bool
    protocol_shown_at: int
    escalations: list[dict[str, Any]]


INTENT_HINTS = {
    "venting": "Focus on listening and validating. Offer one small idea only if it clearly helps.",
    "technique": (
        "The user wants a practical coping technique. Briefly validate, then give ONE simple "
        "technique with clear steps (a short numbered list is fine)."
    ),
    "info": "Answer briefly and factually. If you are not sure, say so. Never diagnose.",
    "greeting": "Reply warmly and briefly, and gently invite them to share how they are feeling.",
    "med_request": (
        "The user asks about medicines. Do NOT name any medicine or pill. Kindly explain that a "
        "doctor or psychiatrist is the right person for medicines, then offer one non-medical "
        "idea that may help."
    ),
    "diagnosis_request": (
        "The user asks whether they have a condition. Do NOT diagnose. Validate their feelings, "
        "explain that only a professional can assess this, and mention that Tele-MANAS 14416 "
        "offers free, confidential counselling."
    ),
}
MEDIUM_RISK_HINT = (
    "The user may be struggling a lot. After validating, gently check in: ask softly how they "
    "are coping and whether they have someone they can talk to."
)
CRISIS_FOLLOWUP_HINT = (
    "The user may be in crisis and has already been given helpline numbers. Be calm, warm and "
    "brief. Do not lecture or give advice beyond staying safe and reaching out to someone."
)


def _finish(state: ChatState, response: str, **updates: Any) -> dict[str, Any]:
    turn = [
        {"role": "user", "content": state["raw_input"]},
        {"role": "assistant", "content": response},
    ]
    return {"response": response, "messages": turn, **updates}


def build_graph(
    chat: ChatLLM,
    classifier: TextLLM | None = None,
    router: TextLLM | None = None,
    transliterator: TextLLM | None = None,
    checkpointer: Any = None,
) -> Any:
    """`chat`: generator (local). `classifier`: crisis LLM. `router`: safety-router LLM.
    Any of the optional LLMs may be None (then rules / lexicon only)."""
    base_preprocess = make_preprocess_node(transliterator)

    def preprocess(state: ChatState) -> dict[str, Any]:
        out = base_preprocess(dict(state))
        # reset per-turn fields (the checkpointer keeps the previous turn's values)
        out.update(
            crisis_matches=[], crisis_llm_risk=None, intent="", route="", route_source="",
            output_problems=[], used_fallback=False, context=[],
        )  # fmt: skip
        return out

    def after_preprocess(state: ChatState) -> str:
        if state.get("awaiting_consent"):
            answer = classify_consent(state["clean_input"])
            if answer in ("yes", "no"):
                return f"consent_{answer}"
        return "crisis"

    def consent_yes(state: ChatState) -> dict[str, Any]:
        record = {"time": time.strftime("%Y-%m-%d %H:%M"), "consent": True, "risk": "high"}
        return _finish(
            state,
            CONSENT_YES[state["script"]],
            awaiting_consent=False,
            flagged_for_review=True,
            risk_level="high",
            route="consent_yes",
            escalations=[*state.get("escalations", []), record],
        )

    def consent_no(state: ChatState) -> dict[str, Any]:
        return _finish(
            state,
            CONSENT_NO[state["script"]],
            awaiting_consent=False,
            flagged_for_review=True,
            risk_level="high",
            route="consent_no",
        )

    def crisis(state: ChatState) -> dict[str, Any]:
        history = [m["content"] for m in state.get("messages", []) if m["role"] == "user"]
        res = check_crisis(state["crisis_check_texts"], history, classifier)
        return {
            "risk_level": res.risk,
            "crisis_matches": res.matches,
            "crisis_llm_risk": res.llm_risk,
            "flagged_for_review": state.get("flagged_for_review", False) or res.flagged_for_review,
            "awaiting_consent": False,
        }

    def after_crisis(state: ChatState) -> str:
        if state["risk_level"] != "high":
            return "router"
        shown_at = state.get("protocol_shown_at")
        n = len(state.get("messages", []))
        if shown_at is None or n - shown_at > PROTOCOL_REPEAT_AFTER:
            return "protocol"
        return "generate"

    def protocol(state: ChatState) -> dict[str, Any]:
        return _finish(
            state,
            PROTOCOL[state["script"]],
            awaiting_consent=True,
            protocol_shown_at=len(state.get("messages", [])),
            flagged_for_review=True,
            route="crisis_protocol",
        )

    def route(state: ChatState) -> dict[str, Any]:
        decision = route_message(state["clean_input"], router)
        return {"route": decision.route, "intent": decision.intent, "route_source": decision.source}

    def after_route(state: ChatState) -> str:
        return state["route"]

    def redirect(state: ChatState) -> dict[str, Any]:
        return _finish(state, REDIRECT[state["script"]])

    def refuse(state: ChatState) -> dict[str, Any]:
        return _finish(state, REFUSE[state["script"]])

    def generate(state: ChatState) -> dict[str, Any]:
        script = state["script"]
        followup = state.get("risk_level") == "high"
        hints = [
            CRISIS_FOLLOWUP_HINT if followup else INTENT_HINTS.get(state.get("intent", ""), "")
        ]
        if state.get("risk_level") == "medium":
            hints.append(MEDIUM_RISK_HINT)
        system = f"{PERSONA}\n\nLanguage: {state['reply_instruction']}\n\n" + " ".join(hints)
        if state.get("context"):
            system += "\n\nReference information (use only if relevant; do not invent facts):\n"
            system += "\n".join(f"- {c}" for c in state["context"])
        history = state.get("messages", [])[-HISTORY_MESSAGES:]
        msgs: list[ChatMessage] = [{"role": "system", "content": system}, *history]
        msgs.append({"role": "user", "content": state["clean_input"]})

        problems: list[str] = []
        reply = ""
        for attempt in range(2):
            try:
                reply = chat(msgs)
            except Exception as e:
                problems = [f"llm_error:{type(e).__name__}"]
                break
            result = check_reply(reply, script)
            if result.ok:
                problems = []
                break
            problems = result.problems
            if attempt == 0:
                note = correction_note(problems, state["reply_instruction"])
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": note},
                ]
        used_fallback = bool(problems)
        if used_fallback:
            reply = SAFE_FALLBACK[script]
        if followup:
            reply = f"{reply}\n\n{HELPLINE_REMINDER[script]}"
        return _finish(
            state,
            reply,
            output_problems=problems,
            used_fallback=used_fallback,
            route=state.get("route") or "crisis_followup",
        )

    g = StateGraph(ChatState)
    for name, fn in [
        ("preprocess", preprocess),
        ("consent_yes", consent_yes),
        ("consent_no", consent_no),
        ("crisis", crisis),
        ("protocol", protocol),
        ("router", route),
        ("redirect", redirect),
        ("refuse", refuse),
        ("generate", generate),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "preprocess")
    g.add_conditional_edges("preprocess", after_preprocess, ["consent_yes", "consent_no", "crisis"])
    g.add_conditional_edges("crisis", after_crisis, ["protocol", "generate", "router"])
    g.add_conditional_edges(
        "router", after_route, {"generate": "generate", "redirect": "redirect", "refuse": "refuse"}
    )
    for terminal in ("consent_yes", "consent_no", "protocol", "redirect", "refuse", "generate"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def respond(graph: Any, thread_id: str, text: str) -> dict[str, Any]:
    """Run one user turn. Returns the final state for this thread."""
    config = {"configurable": {"thread_id": thread_id}}
    result: dict[str, Any] = graph.invoke({"raw_input": text}, config)
    return result
