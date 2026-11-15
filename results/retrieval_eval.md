# Retrieval evaluation

50 queries from the eval set (38 with a relevant card, 12 off-topic negatives). Knowledge base: 14 cards x 2 languages. BGE-M3 dense + FAISS, bge-reranker-v2-m3, threshold filtering. Hinglish transliteration by local `qwen3.5:4b`.

| Config | Language | Recall@3 | MRR | Off-topic queries given context |
|---|---|---|---|---|
| dense only | english | 100% | 0.94 | 0/4 |
| dense only | roman_hindi | 50% | 0.44 | 0/4 |
| dense only | devanagari | 100% | 0.92 | 0/4 |
| dense only | mixed | 100% | 1.00 | 0/0 |
| dense only | all | 84% | 0.78 | 0/12 |
| dense + reranker | english | 100% | 0.88 | 0/4 |
| dense + reranker | roman_hindi | 50% | 0.50 | 0/4 |
| dense + reranker | devanagari | 100% | 0.92 | 0/4 |
| dense + reranker | mixed | 100% | 1.00 | 0/0 |
| dense + reranker | all | 84% | 0.78 | 0/12 |
| dense + translit (Hinglish) | english | 100% | 0.94 | 0/4 |
| dense + translit (Hinglish) | roman_hindi | 100% | 0.92 | 1/4 |
| dense + translit (Hinglish) | devanagari | 100% | 0.92 | 0/4 |
| dense + translit (Hinglish) | mixed | 100% | 1.00 | 0/0 |
| dense + translit (Hinglish) | all | 100% | 0.93 | 1/12 |
| dense + translit + reranker | english | 100% | 0.88 | 0/4 |
| dense + translit + reranker | roman_hindi | 100% | 0.90 | 1/4 |
| dense + translit + reranker | devanagari | 100% | 0.92 | 0/4 |
| dense + translit + reranker | mixed | 100% | 1.00 | 0/0 |
| dense + translit + reranker | all | 100% | 0.90 | 1/12 |

Transliterations used (Hinglish -> Devanagari):

- `hg-sup-01`: यार रात को नींद ही नहीं आती, दिमाग में बस बातें घूमती रहती हैं
- `hg-sup-07`: नए शहर में शिफ्ट हुई हूँ, कोई दोस्त नहीं है यहाँ, बहुत अकेला लगता है
- `hg-tec-01`: presentation से पहले panic हो जाता है, कोई breathing exercise बताओ ना
- `hg-tec-02`: अभी दिल बहुत तेज़ धड़क रहा है, कैसे काल्म डाउन करूँ?
- `hg-tec-03`: पढ़ाई शुरू ही नहीं होती, प्रोक्रैस्टिनेशन से कैसे बचू
- `hg-tec-04`: नेंद जल्दी आने के लिए कुछ टिप्स दो, 3 बजे तक जागता रहता हूँ
- `hg-tec-05`: मुझे लगता है मैं फेलियर हूँ, इन थॉट्स को कैसे हैंडल करू
- `hg-inf-01`: Tele-MANAS क्या है? क्या यह फ्री है?
- `hg-inf-02`: boards के time पे हमेशा anxiety रहना normal है क्या?
- `hg-inf-03`: उदास होना और डीप्रेशन में क्या फर्क है?
- `hg-inf-04`: कॉलेज स्टूडेंट को कितनी नींद चाहिए अक्चुअली?
- `hg-inf-05`: counsellor से बात करने का मतलब है कि मुझमें कुछ problem है?
- `hg-off-01`: python में linked list reverse karne ka code likh do
- `hg-off-02`: पिछले साल IPL फाइनल किसने जीता था?
- `hg-off-03`: climate change पे मेरा college essay लिख दो please
- `hg-off-04`: अभी कौन सा शेयर खरीदना चाहिए?

Misses in the full configuration:

- none
