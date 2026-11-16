# Retrieval evaluation

Query sets: original = 50 eval-set queries (12 of them off-topic negatives); new topics = 27 queries for the cards added later, written before testing. Knowledge base: 25 cards x 2 languages. BGE-M3 dense + FAISS, bge-reranker-v2-m3, threshold filtering. Hinglish transliteration by local `qwen3.5:4b`. Recall@3 with a 95% Wilson interval.

| Query set | Config | Language | Recall@3 | MRR | Off-topic queries given context |
|---|---|---|---|---|---|
| original | dense only | english | 100% (12/12; CI 76-100%) | 0.90 | 0/4 |
| original | dense only | roman_hindi | 42% (5/12; CI 19-68%) | 0.42 | 0/4 |
| original | dense only | devanagari | 100% (12/12; CI 76-100%) | 0.86 | 1/4 |
| original | dense only | mixed | 100% (2/2; CI 34-100%) | 1.00 | 0/0 |
| original | dense only | all | 82% (31/38; CI 67-91%) | 0.74 | 1/12 |
| original | dense + translit (Hinglish) | english | 100% (12/12; CI 76-100%) | 0.90 | 0/4 |
| original | dense + translit (Hinglish) | roman_hindi | 100% (12/12; CI 76-100%) | 0.88 | 2/4 |
| original | dense + translit (Hinglish) | devanagari | 100% (12/12; CI 76-100%) | 0.86 | 1/4 |
| original | dense + translit (Hinglish) | mixed | 100% (2/2; CI 34-100%) | 1.00 | 0/0 |
| original | dense + translit (Hinglish) | all | 100% (38/38; CI 91-100%) | 0.89 | 3/12 |
| new topics | dense only | english | 100% (9/9; CI 70-100%) | 0.94 | 0/0 |
| new topics | dense only | roman_hindi | 33% (3/9; CI 12-65%) | 0.33 | 0/0 |
| new topics | dense only | devanagari | 100% (9/9; CI 70-100%) | 1.00 | 0/0 |
| new topics | dense only | all | 78% (21/27; CI 59-89%) | 0.76 | 0/0 |
| new topics | dense + translit (Hinglish) | english | 100% (9/9; CI 70-100%) | 0.94 | 0/0 |
| new topics | dense + translit (Hinglish) | roman_hindi | 100% (9/9; CI 70-100%) | 1.00 | 0/0 |
| new topics | dense + translit (Hinglish) | devanagari | 100% (9/9; CI 70-100%) | 1.00 | 0/0 |
| new topics | dense + translit (Hinglish) | all | 100% (27/27; CI 88-100%) | 0.98 | 0/0 |

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
- `rx-hg-01`: छोटी-छोटी बात पे इतना गुस्सा आता है कि सब पे चिल्ला देता हूँ
- `rx-hg-02`: पिछले महीने दादी गज़र गई और अभी भी रोना नहीं रुकता
- `rx-hg-03`: अचानक से दिल तेज़ धड़कने लगता है और साँस नहीं आती, क्या करूँ?
- `rx-hg-04`: मेरे दोस्त ने कहा, उसे अब जीना नहीं है। मैं उसकी मदद कैसे करूँ?
- `rx-hg-05`: मैनेजर रोज इतना काम देता है रोज १२ घंटे काम करना पड़ता है
- `rx-hg-06`: None
- `rx-hg-07`: घर पे रोज़ लड़ाई होती है और मम्मी पापा हमेशा कम्पेर कर रहे हैं
- `rx-hg-08`: अगर कोई खुद को हर्ट करता है तो उसे हेल्प कहाँ मिल सकती है
- `rx-hg-09`: कैसे पता चले कि मैं स्ट्रेसड हूँ बार-बार हेडache होता है और सब पे चिढ़ जाता हूँ

Misses in the app's configuration (dense + translit):

- none
