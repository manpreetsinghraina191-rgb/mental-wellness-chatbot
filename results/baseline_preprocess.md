# Baselines (no LLM)

Eval items: 149

## Script detection accuracy

| Language | Accuracy |
|---|---|
| english | 100% (47/47) |
| roman_hindi | 100% (47/47) |
| devanagari | 100% (47/47) |
| mixed | 100% (8/8) |
| **all** | **100% (149/149)** |

Misses:

## Crisis detection: keyword list only (current `CRISIS_HINTS`)

| Language | Recall on high-risk | False alarms on safe idioms |
|---|---|---|
| english | 27% (3/11) | 0% (0/3) |
| roman_hindi | 9% (1/11) | 0% (0/3) |
| devanagari | 0% (0/11) | 0% (0/3) |
| mixed | 50% (1/2) | 0% (0/1) |

Missed crisis messages are the costly error. This baseline motivates the multilingual lexicon + LLM classifier in step 3.

## Script detection on independent real data

| Data | Expected | Accuracy |
|---|---|---|
| ESConv seeker messages (real English) | english | 100% (1999/2000) |
| erc-hinglish sitcom lines (real Hinglish) | roman_hindi | 80% (306/383) |

The eval-set accuracy above is optimistic (word list was extended after a first run on it); these independent numbers are the ones to report. Some erc-hinglish lines are actually English, so the Hinglish figure is a lower bound.
