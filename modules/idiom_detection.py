import os
import json
import math
import requests
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import time
import csv




GROQ_API_KEY = "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"

def get_dynamic_threshold(lang: str) -> float:
    lang = lang.upper()
    if lang in ["ES-EC", "ZH"]:
        return 0.55  
    elif lang in ["KA", "KK", "NO", "SK", "TR", "IG", "UZ"]:
        return 0.40  
    return 0.45  

def _groq_call(system: str, user: str, max_tokens: int = 50) -> str:
    for attempt in range(10): 
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            time.sleep(2)  
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e):
                wait = 2 * (attempt + 1)  
                print(f"[Groq] Rate limit, {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                raise
    return ""


def llama_physically_possible(sentence: str, phrase: str) -> bool:
    """
    Returns True if the phrase is physically possible in the sentence context.
    'raining cats and dogs' → False → fast-path idiom label.
    """
    system = (
    "You are a linguistics expert. Answer ONLY with 'yes' or 'no'."
)

    user = (
    f'Sentence: "{sentence}"\n'
    f'Phrase: "{phrase}"\n\n'
    f'Step 1: Identify the subject and object in the sentence.\n'
    f'Step 2: Ask — do the subject and object refer to real, physical entities '
    f'that could literally perform or experience this phrase?\n'
    f'Step 3: If YES to both — the phrase is literal. If the subject/object '
    f'are abstract or metaphorical — it is figurative.\n\n'
    f'Is "{phrase}" physically and literally occurring in this sentence '
    f'with real entities? Answer only yes or no.'
    )
    answer = _groq_call(system, user, max_tokens=5).lower()
    return answer.startswith("y")


def llama_literal_or_idiomatic_score(sentence: str, phrase: str) -> float:
    """
    Asks LLaMA if phrase is used literally or idiomatically.
    Returns a score: 1.0 = definitely idiomatic, 0.0 = definitely literal.
    Soft scoring based on response wording.
    """
    system = (
    "You are a linguistics expert. "
    "Respond with ONLY one of: "
    "'definitely literal', 'probably literal', 'uncertain', "
    "'probably idiomatic', 'definitely idiomatic'. "
    "No other words."
)
    user = (
    f'Sentence: "{sentence}"\n'
    f'Phrase: "{phrase}"\n\n'
    f'In this sentence, does "{phrase}" refer to something '
    f'real and physical that is actually happening? '
    f'Or is it a figure of speech?\n\n'
    f'If a real physical event/object is described → lean toward literal.\n'
    f'If the meaning is abstract or metaphorical → lean toward idiomatic.'
)
    answer = _groq_call(system, user, max_tokens=10).lower()

    score_map = {
        "definitely idiomatic": 1.0,
        "probably idiomatic":   0.75,
        "uncertain":            0.5,
        "probably literal":     0.25,
        "definitely literal":   0.0,
    }
    for key, val in score_map.items():
        if key in answer:
            return val
    if "idiomatic" in answer:
        return 0.8
    if "literal" in answer:
        return 0.2
    return 0.5

bge_model = SentenceTransformer('BAAI/bge-m3')

def bge_cosine_signal(sentence: str, phrase: str) -> float:
    """
    Computes cosine similarity locally using sentence-transformers.
    """
    try:
        vec_phrase = bge_model.encode(phrase)
        vec_context = bge_model.encode(sentence)

        dot = np.dot(vec_phrase, vec_context)
        norm_p = np.linalg.norm(vec_phrase)
        norm_c = np.linalg.norm(vec_context)
        cosine_sim = dot / (norm_p * norm_c + 1e-9)

        return float(1.0 - cosine_sim)
    except Exception as e:
        print(f"[BGE-M3 Local] Error: {e}")
        return 0.5


def detect_idiom(sentence: str, phrase: str, lang: str) -> dict:
    """
    Full idiom detection pipeline.

    Returns:
        {
          "is_idiomatic": bool,
          "confidence": float,        # 0..1 (1 = certain idiom)
          "method": str,              # which signal decided
        }
    """
    result = {
        "is_idiomatic": False,
        "confidence": 0.0,
        "method": None,
        "wiktionary_def": None,
    }

    possible = llama_physically_possible(sentence, phrase)
    possible_score = 0
    bge_score = 0
    if not possible:
        possible_score = 0.70
    else:
        bge_score = bge_cosine_signal(sentence, phrase)

    llm_score = llama_literal_or_idiomatic_score(sentence, phrase)
    blended   = (bge_score * 0.4) + (llm_score * 0.6) + (possible_score * 0.4)

    threshold = get_dynamic_threshold(lang)
    is_idiomatic = blended >= threshold

    result.update({
        "is_idiomatic": is_idiomatic,
        "confidence": blended,
        "method": "blended_bge_llm",
    })
    print(
        f"[IdiomDetect] BGE={bge_score:.2f} LLM={llm_score:.2f} "
        f"Blended={blended:.2f} → {'IDIOMATIC' if is_idiomatic else 'LITERAL'}"
    )
    return result


if __name__ == "__main__":
    test_cases = csv.reader(open("output.csv", newline="", encoding="utf-8"))
    op = {}
    for sent, phrase in test_cases:
        print(f"\n{'='*60}")
        print(f"Sentence : {sent}")
        print(f"Phrase   : {phrase}")
        r = detect_idiom(sent, phrase)
        print(f"Result   : {json.dumps(r, indent=2)}")
        op[(sent, phrase)] = r
    csv.writer(open("idiom_detection_results.csv", "w", newline="", encoding="utf-8")).writerows(
        [("sentence", "phrase", "is_idiomatic", "confidence", "method", "wiktionary_def")] +
        [(s, p, r["is_idiomatic"], r["confidence"], r["method"], r["wiktionary_def"]) for (s, p), r in op.items()])
