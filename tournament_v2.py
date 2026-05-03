import os
import base64
import itertools
import requests
from pathlib import Path

LAMBDA_PENALTY   = 1.5
OPENROUTER_API_KEY = ""
MODEL            = "qwen/qwen3-vl-8b-instruct"
API_URL          = "https://openrouter.ai/api/v1/chat/completions"


def _encode_image(image_path: str) -> str | None:
    p = Path(image_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_user_content(
    path_a: str | None,
    path_b: str | None,
    cand_a: dict,
    cand_b: dict,
    query_prefix: str,
    question: str,
) -> list:
    content = [{"type": "text", "text": query_prefix}]

    b64_a = _encode_image(path_a) if path_a else None
    if b64_a:
        content.append({"type": "text", "text": "Image A:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_a}"}
        })
        content.append({"type": "text", "text": f"Caption A: {cand_a.get('caption', '')}\n"})
    else:
        content.append({"type": "text", "text": f"Image A (caption only): {cand_a.get('caption', '')}\n"})

    b64_b = _encode_image(path_b) if path_b else None
    if b64_b:
        content.append({"type": "text", "text": "Image B:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_b}"}
        })
        content.append({"type": "text", "text": f"Caption B: {cand_b.get('caption', '')}\n"})
    else:
        content.append({"type": "text", "text": f"Image B (caption only): {cand_b.get('caption', '')}\n"})

    content.append({"type": "text", "text": question})
    return content


def qwen_pairwise_compare(
    cand_a: dict,
    cand_b: dict,
    match_query: str,
    is_idiomatic: bool,
    image_dir: str | None = None,
) -> str:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    def resolve_path(cand):

        p = cand.get("image_path", "")
        if p and p != "NOT_FOUND" and Path(p).exists():
            return p

        if not image_dir:
            return None
        name = cand.get("image_name", "")
        if not name:
            return None
        flat = Path(image_dir) / name
        if flat.exists():
            return str(flat)
        for sub in Path(image_dir).iterdir():
            if sub.is_dir():
                candidate = sub / name
                if candidate.exists():
                    return str(candidate)
        return None

    path_a = resolve_path(cand_a)
    path_b = resolve_path(cand_b)
    has_images = bool(path_a or path_b)


    if is_idiomatic:
        system = (
            "You are a multimodal image ranking assistant. "
            "You will be shown two images (or descriptions) and the actual, real-world meaning of an idiom. "
            "Decide which image best depicts this real-world human situation, context, and its underlying atmosphere. "
            "CRITICAL INSTRUCTION: Reject images that show literal animal metaphors, surreal objects, or cartoons. "
            "Focus strictly on the human and environmental reality described in the text. "
            "Reply ONLY with a single letter: A or B."
        )
        query_prefix = f"Real-world situation and atmosphere to match: {match_query}\n\n"
        question     = "Which image (A or B) better depicts this real-world situation? Answer only A or B."
    else:
        system = (
            "You are a multimodal image ranking assistant. "
            "You will be shown two images (or descriptions) and a phrase. "
            "Decide which image more directly and literally depicts the phrase. "
            "Reply ONLY with a single letter: A or B."
        )
        query_prefix = f"Phrase to match literally: {match_query}\n\n"
        question     = "Which image (A or B) more literally depicts this phrase? Answer only A or B."

    user_content = _build_user_content(path_a, path_b, cand_a, cand_b, query_prefix, question)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 5,
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/admire-pipeline",  
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        winner = "A" if "A" in answer else "B"
        mode   = "vision" if has_images else "caption"
        print(f"    [{mode}] {cand_a.get('image_name','A')} vs {cand_b.get('image_name','B')} → {winner}")
        return winner
    except Exception as e:
        print(f"    [OpenRouter ERROR] {e}")
        try:
            print(f"    [Detail]: {r.text}")
        except:
            pass
        print("    → defaulting A")
        return "A"


def run_tournament_v2(
    candidates:   list[dict],
    match_query:  str,
    penalties:    dict[str, float],
    is_idiomatic: bool,
    image_dir:    str | None = None,
) -> tuple[list[dict], list[dict]]:
    if not OPENROUTER_API_KEY:
        print("API key not found. Running in MOCK mode with keyword overlap heuristic.")

    n    = len(candidates)
    wins = {c["image_id"]: 0 for c in candidates}
    match_log = []

    mode_str = f"OpenRouter/{MODEL}" if OPENROUTER_API_KEY else "MOCK(keyword)"
    print(f"\n  [Tournament-v2] {mode_str} | idiomatic={is_idiomatic} | {n*(n-1)//2} matches")

    for idx_a, idx_b in itertools.combinations(range(n), 2):
        ca, cb = candidates[idx_a], candidates[idx_b]

        if OPENROUTER_API_KEY:
            w = qwen_pairwise_compare(ca, cb, match_query, is_idiomatic, image_dir)

        winner_id = ca["image_id"] if w == "A" else cb["image_id"]
        wins[winner_id] += 1
        match_log.append({"a": ca["image_id"], "b": cb["image_id"], "winner": winner_id})

    results = []
    for c in candidates:
        iid     = c["image_id"]
        w       = wins[iid]
        penalty = penalties.get(iid, 0.0) if is_idiomatic else 0.0
        score   = w - LAMBDA_PENALTY * penalty
        results.append({
            "image_id":   iid,
            "image_name": c.get("image_name", ""),
            "caption":    c.get("caption", ""),
            "wins":       w,
            "penalty":    round(penalty, 3),
            "score":      round(score, 3),
        })

    results.sort(key=lambda x: (x["score"], x["wins"]), reverse=True)
    return results, match_log



if __name__ == "__main__":
    import json

    test_query = "Ablam evin en önemli gelir kaynağı oldu. Atmosphere: Wealth, relief, and pride."
    test_dir   = r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk"

    test_candidates = [
        {"image_id": "img1", "image_name": "60724405930.png", "caption": "An older man lying on a couch covered with a large amount of money."},
        {"image_id": "img2", "image_name": "47911094135.png", "caption": "A cartoon man's face with dollar signs in his eyes and gold coins surrounding him."},
        {"image_id": "img3", "image_name": "16634799208.png", "caption": "A close-up of a woman's hand with an orange gemstone ring."},
        {"image_id": "img4", "image_name": "80371640998.png", "caption": "A cartoon chicken with orange and white feathers sitting on golden eggs."},
        {"image_id": "img5", "image_name": "99148916362.png", "caption": "A stylized cartoon chicken with a ruffled white and orange body."},
    ]

    penalties = {"img1": 0.0, "img2": 0.0, "img3": 0.0, "img4": 0.0, "img5": 0.0}

    print("\n" + "="*60)
    print("TURNUVA BAŞLIYOR (Qwen2.5-VL-3B via OpenRouter)")
    print("="*60)

    results, match_log = run_tournament_v2(
        candidates   = test_candidates,
        match_query  = test_query,
        penalties    = penalties,
        is_idiomatic = True,
        image_dir    = test_dir,
    )

    print(f"\n{'='*60}")
    print("MAÇ LOGLARI:")
    for log in match_log:
        print(f"  {log['a']} vs {log['b']} → {log['winner']}")

    print(f"\n{'='*60}")
    print("FİNAL SIRALAMASI:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
