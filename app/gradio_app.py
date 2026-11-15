"""Saathi demo UI (Gradio).

  uv run python app/gradio_app.py      -> http://127.0.0.1:7860

Models come from .env (all local Ollama):
  CHAT_MODEL        generator                     (default qwen3.5:4b)
  CLASSIFIER_MODEL  crisis classifier + router    (default qwen3.5:4b)

Conversations live only in memory for the browser session; nothing is written to disk.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from tai_bot.graph import build_graph, respond
from tai_bot.llm_clients import make_chat_llm, make_llm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "qwen3.5:4b")
USE_RAG = os.getenv("USE_RAG", "1") == "1"
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"  # no gain at 14 cards (eval)
USE_TRANSLIT = os.getenv("USE_TRANSLIT", "1") == "1"  # Hinglish Recall@3 50% -> 100%


def load_retriever() -> Any:
    """Retrieval is optional: needs `uv sync --extra rag` and a built index."""
    if not USE_RAG:
        return None
    try:
        from tai_bot.retrieval import Retriever, load_bge_m3, load_reranker

        return Retriever.load(load_bge_m3(), load_reranker() if USE_RERANKER else None)
    except Exception as e:  # missing extra or index: run without retrieval
        print(f"Retrieval disabled ({type(e).__name__}: {e})")
        return None


RETRIEVER = load_retriever()
GRAPH = build_graph(
    chat=make_chat_llm(CHAT_MODEL),
    classifier=make_llm("ollama", CLASSIFIER_MODEL, json_mode=True, temperature=0.0),
    router=make_llm("ollama", CLASSIFIER_MODEL, json_mode=True, temperature=0.0),
    transliterator=make_llm("ollama", CLASSIFIER_MODEL, temperature=0.0) if USE_TRANSLIT else None,
    retriever=RETRIEVER,
)

TEXT = {
    "English": {
        "disclaimer": (
            "### Before we start\n"
            "Saathi is a supportive AI companion for stress, sleep, loneliness and everyday "
            "worries, in **English, Hindi or Hinglish**. It is **not a doctor or therapist**, "
            "cannot diagnose, and does not replace professional care. Your messages stay on this "
            "computer and are not saved after you close the page.\n\n"
            "**If you are in danger right now, call 112.**"
        ),
        "ok": "I understand, start chatting",
        "help_btn": "Need help now?",
        "help": (
            "### You are not alone\n"
            "- **Tele-MANAS: 14416** or **1-800-891-4416**: free, 24x7 mental-health support "
            "from the Government of India, in many Indian languages\n"
            "- **Emergency: 112**\n\n"
            "If you can, reach out to someone you trust right now."
        ),
        "placeholder": "Type in English, हिंदी or Hinglish…",
        "why": "Why this reply? (system view for the demo)",
        "reset": "New conversation",
    },
    "हिंदी": {
        "disclaimer": (
            "### शुरू करने से पहले\n"
            "साथी तनाव, नींद, अकेलेपन और रोज़मर्रा की चिंताओं के लिए एक सहायक AI साथी है, जो "
            "**हिंदी, अंग्रेज़ी या Hinglish** में बात करता है। यह **डॉक्टर या थेरेपिस्ट नहीं है**, "
            "कोई निदान नहीं करता और पेशेवर इलाज की जगह नहीं लेता। आपके संदेश इसी कंप्यूटर पर रहते "
            "हैं और पेज बंद करने के बाद सेव नहीं होते।\n\n"
            "**अगर आप अभी ख़तरे में हैं, तो 112 पर कॉल करें।**"
        ),
        "ok": "मैं समझ गया/गई, बात शुरू करें",
        "help_btn": "अभी मदद चाहिए?",
        "help": (
            "### आप अकेले नहीं हैं\n"
            "- **टेली-मानस: 14416** या **1-800-891-4416**: भारत सरकार की मुफ़्त, 24 घंटे "
            "मानसिक स्वास्थ्य सहायता, कई भारतीय भाषाओं में\n"
            "- **आपातकाल: 112**\n\n"
            "हो सके तो अभी किसी भरोसेमंद इंसान से बात करें।"
        ),
        "placeholder": "हिंदी, English या Hinglish में लिखें…",
        "why": "यह जवाब क्यों? (डेमो के लिए सिस्टम व्यू)",
        "reset": "नई बातचीत",
    },
}


def explain(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "detected_script": state.get("script"),
        "script_inherited_from_previous": state.get("script_inherited"),
        "risk_level": state.get("risk_level"),
        "crisis_llm_said": state.get("crisis_llm_risk"),
        "crisis_keyword_matches": state.get("crisis_matches"),
        "route": state.get("route"),
        "route_decided_by": state.get("route_source"),
        "intent": state.get("intent"),
        "retrieved_sources": state.get("sources", []),
        "output_check_problems": state.get("output_problems"),
        "used_safe_fallback": state.get("used_fallback"),
        "flagged_for_human_review": state.get("flagged_for_review"),
        "escalation_records": state.get("escalations", []),
        "models": {
            "generator": CHAT_MODEL,
            "classifier_router": CLASSIFIER_MODEL,
            "retrieval": ("BGE-M3" + (" + reranker" if USE_RERANKER else ""))
            if RETRIEVER
            else "off",
        },
    }


def send(
    text: str, history: list[dict[str, Any]], thread_id: str
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not text.strip():
        return "", history, {}
    state = respond(GRAPH, thread_id, text)
    history = [
        *history,
        {"role": "user", "content": text},
        {"role": "assistant", "content": state["response"]},
    ]
    return "", history, explain(state)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Saathi: mental-health support") as demo:
        thread = gr.State(lambda: str(uuid.uuid4()))
        with gr.Row():
            gr.Markdown("# Saathi 🌱\nA supportive listener in English, हिंदी and Hinglish")
            lang = gr.Radio(["English", "हिंदी"], value="English", label="Interface", scale=0)
            help_btn = gr.Button(TEXT["English"]["help_btn"], variant="stop", scale=0)
        help_box = gr.Markdown(TEXT["English"]["help"], visible=False)

        with gr.Group(visible=True) as disclaimer_box:
            disclaimer = gr.Markdown(TEXT["English"]["disclaimer"])
            ok_btn = gr.Button(TEXT["English"]["ok"], variant="primary")

        with gr.Group(visible=False) as chat_box:
            chatbot = gr.Chatbot(height=460, label="Saathi")
            msg = gr.Textbox(placeholder=TEXT["English"]["placeholder"], show_label=False)
            with gr.Row():
                reset = gr.Button(TEXT["English"]["reset"])
            with gr.Accordion(TEXT["English"]["why"], open=False) as why_box:
                why = gr.JSON()

        ok_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None,
                     [disclaimer_box, chat_box])  # fmt: skip
        help_btn.click(lambda: gr.update(visible=True), None, help_box)
        msg.submit(send, [msg, chatbot, thread], [msg, chatbot, why])
        reset.click(lambda: ([], {}, str(uuid.uuid4())), None, [chatbot, why, thread])

        def switch(language: str) -> tuple[Any, ...]:
            t = TEXT[language]
            return (
                gr.update(value=t["help_btn"]),
                gr.update(value=t["help"]),
                gr.update(value=t["disclaimer"]),
                gr.update(value=t["ok"]),
                gr.update(placeholder=t["placeholder"]),
                gr.update(value=t["reset"]),
                gr.update(label=t["why"]),
            )

        lang.change(switch, lang, [help_btn, help_box, disclaimer, ok_btn, msg, reset, why_box])
    return demo


if __name__ == "__main__":
    build_ui().launch()
