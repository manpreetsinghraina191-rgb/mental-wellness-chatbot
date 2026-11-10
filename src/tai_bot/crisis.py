"""Crisis detection and the fixed crisis protocol.

Two layers, combined with OR (the higher risk wins), because a missed crisis is the costly
error:

  1. Lexicon: regex patterns in English, Romanized Hindi and Devanagari (fast, explainable,
     works without any model). Both verb genders are covered (chahta/chahti, rahunga/rahungi).
     Common idioms ("exam ne meri jaan le li", "mar gaya yaar", "this is killing me") are
     deliberately NOT matched.
  2. LLM classifier over the last few USER turns (risk often builds up gradually). It returns
     only a label, so it stays fast on a small local model.

Crisis replies are fixed, reviewed templates, never generated text.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from tai_bot.preprocess import Script

Risk = Literal["none", "low", "medium", "high"]
RISK_ORDER: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}


def max_risk(*risks: str) -> Risk:
    best = max(risks, key=lambda r: RISK_ORDER.get(r, 0))
    return best if best in RISK_ORDER else "none"  # type: ignore[return-value]


# --- Layer 1: lexicon ------------------------------------------------------------------------


def fold(text: str) -> str:
    """Lowercase, NFC, and fold Devanagari spelling variation: drop nukta (ज़ -> ज),
    chandrabindu -> anusvara (हाँ -> हां)."""
    text = unicodedata.normalize("NFC", text.lower())
    text = unicodedata.normalize("NFD", text).replace("़", "")
    return unicodedata.normalize("NFC", text).replace("ँ", "ं")


# Hinglish building blocks
_WANT = r"(chahta|chahti|chahte|chahata|chahati)"
_AM = r"(hoon|hun|hu|hoo|h|hai)?"
_KHATAM = r"(khatam|khatm|khtm|khattam)"
_FUT_FIRST = r"(dunga|dungi|doonga|doongi|du|dun|doon)"

HIGH_PATTERNS: dict[str, list[str]] = {
    "english": [
        r"\bsuicid(e|al)\b",
        r"\bkill(ing)? my ?self\b",
        r"\bend(ing)? (my|it) (life|all)\b",
        r"\bend it all\b",
        r"\btake my (own )?life\b",
        r"\b(want|wanted|wanna|going) (to )?die\b",
        r"\bwish i (was|were) dead\b",
        r"\b(better off|be|being) dead\b",
        r"\b(don'?t|do not) want to (live|be alive|be here|wake up|exist)\b",
        r"\b(never|not) wake up\b",
        r"\bno (reason|point) (to|in) (live|living|go on)\b",
        r"\bnot worth living\b",
        r"\b(hurt|hurting|harm|harming|cut|cutting|burn|burned|burnt|burning|starve|starving) (my ?self)\b",
        r"\bself[- ]?harm",
        r"\bcut (myself|my wrists?)\b",
        r"\boverdos",
        r"\b(take|swallow|took) (them )?all\b.*\b(pills|tablets)\b",
        r"\b(pills|tablets)\b.*\b(take|swallow) (them )?all\b",
        r"\b(how many|lethal|fatal)\b.*\b(pills|tablets|paracetamol|dose)\b",
        r"\b(pills|tablets|paracetamol|dose)\b.*\b(fatal|lethal|kill|die)\b",
        r"\bbetter off without me\b",
        r"\b(won'?t|will not) be (a burden|around|here)\b",
        r"\bnot be (around|here) (much longer|after|anymore|tomorrow)\b",
        r"\b(goodbye|suicide) (note|letter)\b",
        r"\bletters? to my family\b.*\bjust in case\b",
        r"\bgiving away my (things|stuff|belongings)\b",
        r"\bjump(ing)? (off|from) (a|the) (building|bridge|roof)\b",
        r"\bhang my ?self\b",
        r"\bmake it (all )?stop\b",
        r"\bcan'?t go on\b|\bcannot go on\b",
        r"\bdisappear (forever|for good)\b",
        r"\b(if|when) i (wasn'?t|weren'?t|am not) (here|around|alive)\b",
    ],
    "roman_hindi": [
        r"\b(khudkhushi|khudkushi|aatmhatya|atmahatya|atmhatya)\b",
        rf"\bkhud ko (\w+ )?({_KHATAM}|maar|mar|hurt|nuksan|nuksaan|chot|kaat)",
        rf"\b(sab|sabkuch|sab kuch|zindagi|jindagi|life) {_KHATAM} (kar|karna|karne|{_FUT_FIRST}|karu|karun)",
        rf"\b{_KHATAM} kar {_FUT_FIRST}\b",
        r"\bjeene ka (mann|man|dil) nahi",
        rf"\b(jeena|jina|zinda) nahi (rehna|{_WANT})",
        rf"\b(marna|mar jana|mar jaana|mar jaun|mar jau|mar jaunga|mar jaungi) {_WANT}?",
        r"\bmar (jana|jaana|jaun|jau) (hi )?(behtar|better|accha)",
        r"\bkabhi (na|nahi) (uthu|uthun|uthoon|uthna)\b",
        r"\bhamesha ke liye (so|chala|chali|gayab|disappear)",
        r"\bmere (jaane|marne) ke baad\b",
        r"\b(mere|mujhe) bina (sab|sabhi|log)",
        r"\b(haath|hath|nas|nasein|kalai) (\w+ )?(kaat|kaata|kaati|kaatna|kaatne)",
        r"\b(saari|sari|bahut saari|sab) (goliyan|goliya|goli|tablets|pills)\b.*\b(kha|khaa|le)",
        r"\b(goliyan|goliya|tablets|pills)\b.*\b(kha lunga|kha lungi|kha lu|kha lun)\b",
        r"\bnahi (rahunga|rahungi|rahoonga|rahoongi|rahoon|rahun)\b",
        r"\b(kitni|kitne)\b.*\b(goli|goliyan|paracetamol|tablet|tablets|pills)\b.*\b(mar|maut|marta|mar jata)",
        r"\b(goodbye|alvida|suicide) (letter|note|chitthi|khat)\b",
        r"\b(aur|ab) nahi (jhel|seh|sah) (sakta|sakti|sakte|paunga|paungi)",
        r"\b(chhat|chat|pul|building) se (kood|koodna|kud|chhalang)",
        r"\btrain ke (aage|saamne)",
        r"\b(saari|sari|apni) (cheezein|cheezen|cheeze|chizein|samaan|saman)\b.*\b(de di|de dee|"
        r"de diya|de dun|de du|dedi|baant)",
        r"\b(zarurat|zaroorat|jarurat) nahi (padegi|rahegi)\b",
        r"\bdisappear ho (jaun|jau|jaana|jana) hamesha",
    ],
    "devanagari": [
        r"(आत्महत्या|खुदकुशी|सुसाइड)",
        r"खुद को (\S+ )?(खत्म|मार|नुकसान|चोट|काट)",
        r"(सब|सबकुछ|सब कुछ|जिंदगी|ज़िंदगी|जीवन) खत्म (कर|करना|करने|कर दूं|कर दूंगा|कर दूंगी)",
        r"जीने का (मन|दिल) नहीं",
        r"(जीना|जिंदा) नहीं (रहना|चाहता|चाहती)",
        r"(मरना|मर जाना|मर जाऊं|मर जाउं) (चाहता|चाहती|बेहतर|अच्छा)?",
        r"कभी (न|ना|नहीं) (उठूं|उठूँ|उठना)",
        r"हमेशा के लिए (सो|चला|चली|गायब)",
        r"मेरे (जाने|मरने) के बाद",
        r"(मेरे|मुझ) (बिना|बगैर) (सब|सभी|लोग)",
        r"(हाथ|नस|नसें|कलाई) (\S+ )?(काट|काटा|काटी|काटना)",
        r"(सारी|सब|बहुत सारी) (गोलियां|गोलियाँ|गोली).*(खा|ले)",
        r"नहीं (रहूंगा|रहूंगी|रहूँगा|रहूँगी)",
        r"(कितनी|कितने).*(गोली|गोलियां|पैरासिटामोल|टैबलेट).*(मर|मौत)",
        r"(अलविदा|सुसाइड) (चिट्ठी|पत्र|खत|नोट)",
        r"और नहीं (होता|सह|झेल)",
        r"(छत|पुल) से (कूद|छलांग)",
        r"(सारी|अपनी) (चीजें|चीजे|सामान).*(दे दी|दे दिया|दे दूं|बांट)",
        r"जरूरत नहीं (पड़ेगी|पडेगी|रहेगी)",
    ],
}

MEDIUM_PATTERNS: dict[str, list[str]] = {
    "english": [
        r"\bhopeless",
        r"\b(what'?s|what is) (even )?the point\b",
        r"\bno point (in )?(trying|anything)\b",
        r"\bfeel(ing)? (so |really |completely )?(empty|numb|worthless)\b",
        r"\bnothing (feels good|matters)\b",
        r"\b(never|not) going to get better\b",
        r"\b(i'?m|i am) (just )?a burden\b",
        r"\bcan'?t take (it|this) anymore\b",
        r"\bgive up on (everything|life)\b",
        r"\bruined everything\b",
        r"\b(hit|hits|beat|beats) me\b",
    ],
    "roman_hindi": [
        r"\b(koi|kya) (fayda|faayda|fayeda) (hi )?(nahi)?",
        r"\bhopeless\b",
        r"\b(khali|khaali) (khali|khaali|sa|pan)\b|\bkhalipan\b|\bempty\b",
        r"\bkuch (bhi )?(accha|acha) nahi (lagta|lag raha)",
        r"\bbojh (hoon|hun|hu|hai|ban)",
        r"\bkisi kaam ka nahi\b",
        r"\b(kuch|kabhi) theek nahi (hoga|hota)",
        r"\b(maara|mara|maarte|peeta|peet)\b",
    ],
    "devanagari": [
        r"निराश",
        r"(कोई|क्या) फायदा (ही )?नहीं",
        r"खालीपन|खाली खाली",
        r"कुछ (भी )?अच्छा नहीं लगता",
        r"बोझ (हूं|हूँ|हैं|बन)",
        r"किसी काम (का|की) नहीं",
        r"कुछ ठीक नहीं होगा",
        r"(मारा|मारते|पीटा|पीटते)",
    ],
}

_COMPILED = {
    level: [
        (lang, p, re.compile(fold(p) if lang == "devanagari" else p, re.IGNORECASE))
        for lang, pats in table.items()
        for p in pats
    ]
    for level, table in (("high", HIGH_PATTERNS), ("medium", MEDIUM_PATTERNS))
}


@dataclass
class LexiconHit:
    risk: Risk
    matches: list[str] = field(default_factory=list)


def lexicon_check(texts: list[str]) -> LexiconHit:
    """Check every text (original + transliterated copy). All languages' patterns are tried on
    every text, because users mix scripts."""
    folded = [fold(t) for t in texts if t]
    for level in ("high", "medium"):
        hits = [f"{lang}:{p}" for lang, p, rx in _COMPILED[level] for t in folded if rx.search(t)]
        if hits:
            return LexiconHit(risk=level, matches=sorted(set(hits)))
    return LexiconHit(risk="none")


# --- Layer 2: LLM classifier -----------------------------------------------------------------

CLASSIFIER_PROMPT = """You are a safety classifier for an Indian mental-health support chat. \
Messages may be in English, Hindi (Devanagari) or Hinglish (Hindi in Roman letters).
Rate the risk that the user may harm themselves or is in danger, for the LATEST message, using \
the earlier messages as context (risk often builds up slowly).

high = any thought of suicide, wanting to die, not waking up, self-harm, a plan or means \
(pills, jumping), saying goodbye, giving things away, or feeling others are better off without \
them. Direct OR indirect.
medium = hopelessness, emptiness, feeling worthless or a burden, without mention of death; or \
being hurt or unsafe at home.
low = sadness or stress with some negative words.
none = everyday stress, venting, questions, greetings, off-topic requests, and idioms such as \
"exam ne meri jaan le li", "mar gaya yaar itna kaam karke", "this assignment is killing me", \
"I'm dying to go home".
If unsure between two levels, choose the higher one.

{context}Latest message: {message}

Return only JSON: {{"risk": "none" | "low" | "medium" | "high"}}"""


def build_classifier_prompt(message: str, history: list[str]) -> str:
    context = ""
    if history:
        context = "Earlier messages from the user:\n" + "\n".join(f"- {h}" for h in history) + "\n"
    return CLASSIFIER_PROMPT.format(context=context, message=message)


def parse_risk(raw: str) -> Risk | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        value = json.loads(text).get("risk", "")
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r"\b(none|low|medium|high)\b", text.lower())
        value = m.group(1) if m else ""
    value = str(value).strip().lower()
    return value if value in RISK_ORDER else None  # type: ignore[return-value]


# --- Combined check --------------------------------------------------------------------------


@dataclass
class CrisisResult:
    risk: Risk
    lexicon_risk: Risk
    llm_risk: Risk | None
    matches: list[str]
    llm_error: str | None = None

    @property
    def flagged_for_review(self) -> bool:
        return RISK_ORDER[self.risk] >= RISK_ORDER["medium"] or self.llm_error is not None


def check_crisis(
    texts: list[str],
    history: list[str] | None = None,
    llm: Callable[[str], str] | None = None,
    history_turns: int = 4,
) -> CrisisResult:
    """`texts`: the current message (and its transliterated copy).
    `history`: earlier USER messages, oldest first. `llm`: prompt -> text, or None."""
    lex = lexicon_check(texts)
    llm_risk: Risk | None = None
    error: str | None = None
    if llm is not None and texts:
        prompt = build_classifier_prompt(texts[0], (history or [])[-history_turns:])
        try:
            llm_risk = parse_risk(llm(prompt))
            if llm_risk is None:
                error = "unparseable classifier output"
        except Exception as e:  # the lexicon result still stands; the session gets flagged
            error = f"{type(e).__name__}: {e}"
    risk = max_risk(lex.risk, llm_risk or "none")
    return CrisisResult(risk, lex.risk, llm_risk, lex.matches, error)


# --- Fixed protocol templates ----------------------------------------------------------------
# Reviewed, pre-written text. Never generated. Mixed-script users get the Devanagari version
# with the English names kept, matching how they write.

PROTOCOL: dict[Script, str] = {
    "english": (
        "I'm really sorry you're going through this much pain, and I'm glad you told me. "
        "You don't have to face this alone.\n\n"
        "Please reach out for help right now: call Tele-MANAS at 14416 (free, 24x7, in many "
        "Indian languages). If you are in immediate danger, call 112.\n\n"
        "If you can, please contact someone you trust right now, like a friend, a family member "
        "or anyone nearby, and let them know how you're feeling.\n\n"
        "Would you like help connecting with a counsellor? I'm here, and you can keep talking "
        "to me."
    ),
    "roman_hindi": (
        "Mujhe sach mein bahut dukh hai ki aap itna dard mehsoos kar rahe hain, aur accha kiya "
        "ki aapne mujhe bataya. Aapko yeh akele nahi jhelna hai.\n\n"
        "Abhi madad ke liye Tele-MANAS ko 14416 par call kijiye (free hai, 24x7, Hindi aur "
        "kai bhashaon mein). Agar aap abhi khatre mein hain, toh 112 par call kijiye.\n\n"
        "Ho sake toh abhi kisi bharosemand insaan ko, jaise dost, ghar wale ya koi paas mein, "
        "bataiye ki aap kaisa feel kar rahe hain.\n\n"
        "Kya aap chahenge ki main kisi counsellor se baat karne mein aapki madad karun? Main "
        "yahin hoon, aap mujhse baat karte rahiye."
    ),
    "devanagari": (
        "मुझे सच में बहुत दुख है कि आप इतना दर्द महसूस कर रहे हैं, और अच्छा किया कि आपने मुझे "
        "बताया। आपको यह अकेले नहीं झेलना है।\n\n"
        "अभी मदद के लिए टेली-मानस को 14416 पर कॉल कीजिए (यह मुफ़्त है, 24 घंटे, हिंदी और कई "
        "भाषाओं में)। अगर आप अभी ख़तरे में हैं, तो 112 पर कॉल कीजिए।\n\n"
        "हो सके तो अभी किसी भरोसेमंद इंसान को, जैसे दोस्त, घरवाले या कोई पास में, बताइए कि आप "
        "कैसा महसूस कर रहे हैं।\n\n"
        "क्या आप चाहेंगे कि मैं किसी काउंसलर से बात करने में आपकी मदद करूँ? मैं यहीं हूँ, आप "
        "मुझसे बात करते रहिए।"
    ),
    "mixed": "",  # filled below from the Devanagari text
}
PROTOCOL["mixed"] = (
    PROTOCOL["devanagari"].replace("टेली-मानस", "Tele-MANAS").replace("काउंसलर", "counsellor")
)

# After the user says yes: no false promise of a call-back; the fastest real route is the
# helpline. The session is flagged and an escalation record is created by the graph.
CONSENT_YES: dict[Script, str] = {
    "english": (
        "Thank you for trusting me with this. The quickest way to talk to a trained counsellor "
        "right now is Tele-MANAS: call 14416. It's free and open 24x7. If you feel you might act "
        "on these thoughts, please call 112 or go to the nearest hospital.\n\n"
        "I'm still here with you. Would you like to tell me what has been happening?"
    ),
    "roman_hindi": (
        "Mujh par bharosa karne ke liye shukriya. Abhi kisi trained counsellor se baat karne ka "
        "sabse jaldi tareeka hai Tele-MANAS: 14416 par call kijiye. Yeh free hai aur 24x7 khula "
        "hai. Agar aapko lag raha hai ki aap khud ko nuksan pahuncha sakte hain, toh 112 par "
        "call kijiye ya sabse paas ke hospital jaiye.\n\n"
        "Main abhi bhi yahin hoon. Kya aap mujhe bataana chahenge ki kya ho raha hai?"
    ),
    "devanagari": (
        "मुझ पर भरोसा करने के लिए शुक्रिया। अभी किसी प्रशिक्षित काउंसलर से बात करने का सबसे जल्दी "
        "तरीका है टेली-मानस: 14416 पर कॉल कीजिए। यह मुफ़्त है और 24 घंटे खुला है। अगर आपको लग "
        "रहा है कि आप खुद को नुकसान पहुँचा सकते हैं, तो 112 पर कॉल कीजिए या सबसे पास के अस्पताल "
        "जाइए।\n\nमैं अभी भी यहीं हूँ। क्या आप मुझे बताना चाहेंगे कि क्या हो रहा है?"
    ),
    "mixed": "",
}
CONSENT_YES["mixed"] = (
    CONSENT_YES["devanagari"].replace("टेली-मानस", "Tele-MANAS").replace("काउंसलर", "counsellor")
)

CONSENT_NO: dict[Script, str] = {
    "english": (
        "That's completely okay. You don't have to do anything you're not ready for. I'm here "
        "to listen, without judgement. If things ever feel like too much, Tele-MANAS (14416) "
        "and 112 are always there.\n\nWhat is weighing on you the most right now?"
    ),
    "roman_hindi": (
        "Bilkul theek hai. Aapko kuch bhi aisa karne ki zaroorat nahi jiske liye aap abhi taiyaar "
        "nahi hain. Main bina judge kiye aapki baat sunne ke liye yahin hoon. Agar kabhi sab bahut "
        "zyada lage, toh Tele-MANAS (14416) aur 112 hamesha hain.\n\n"
        "Abhi aapko sabse zyada kya pareshan kar raha hai?"
    ),
    "devanagari": (
        "बिल्कुल ठीक है। आपको ऐसा कुछ भी करने की ज़रूरत नहीं जिसके लिए आप अभी तैयार नहीं हैं। मैं "
        "बिना जज किए आपकी बात सुनने के लिए यहीं हूँ। अगर कभी सब बहुत ज़्यादा लगे, तो टेली-मानस "
        "(14416) और 112 हमेशा हैं।\n\nअभी आपको सबसे ज़्यादा क्या परेशान कर रहा है?"
    ),
    "mixed": "",
}
CONSENT_NO["mixed"] = CONSENT_NO["devanagari"].replace("टेली-मानस", "Tele-MANAS")

# Appended (fixed text) to supportive replies while a session stays high-risk after the full
# protocol has been shown once, so it is not repeated word for word every turn.
HELPLINE_REMINDER: dict[Script, str] = {
    "english": "If you feel unsafe, please call Tele-MANAS at 14416 or 112. I'm here with you.",
    "roman_hindi": (
        "Agar aap safe feel nahi kar rahe, toh Tele-MANAS (14416) ya 112 par call kijiye. "
        "Main aapke saath hoon."
    ),
    "devanagari": (
        "अगर आप सुरक्षित महसूस नहीं कर रहे, तो टेली-मानस (14416) या 112 पर कॉल कीजिए। मैं आपके साथ हूँ।"
    ),
    "mixed": "",
}
HELPLINE_REMINDER["mixed"] = HELPLINE_REMINDER["devanagari"].replace("टेली-मानस", "Tele-MANAS")


# --- Consent -------------------------------------------------------------------------------

Consent = Literal["yes", "no", "unclear"]
_YES = re.compile(
    r"^(haan|haa|ha|han|hanji|haan ji|ji|ji haan|yes|yeah|yep|yup|ok|okay|sure|please|"
    r"theek hai|thik hai|thik h|theek h|chalo|zaroor|jaroor|bilkul|"
    r"हां|हा|जी|जी हां|ठीक है|जरूर|बिल्कुल|हां जी)$"
)
_NO = re.compile(
    r"^(nahi|nahin|nhi|na|no|nope|not now|mat|rehne do|abhi nahi|"
    r"नहीं|ना|मत|रहने दो|अभी नहीं)$"
)


def classify_consent(text: str) -> Consent:
    """Only short replies count as a consent answer; anything longer goes back through the
    normal crisis check."""
    t = fold(text).strip(" .!?।")
    t = re.sub(r"\s+", " ", t)
    if len(t.split()) > 4:
        return "unclear"
    if _NO.match(t):
        return "no"
    if _YES.match(t):
        return "yes"
    return "unclear"


def protocol_text(script: Script) -> str:
    return PROTOCOL[script]
