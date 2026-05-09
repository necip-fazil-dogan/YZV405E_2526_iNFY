import os
import json
import torch
from PIL import Image
from groq import Groq
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import time

# ── Config & Model Initialization ────────────────────────────────────────────

GROQ_API_KEY = "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"

# GroundingDINO'yu global olarak başlatıyoruz ki her görselde baştan yüklenmesin
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[GroundingDINO] Model {device.upper()} üzerinde başlatılıyor...")

try:
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
except Exception as e:
    print(f"[GroundingDINO] Yükleme hatası: {e}\nLütfen internet bağlantınızı kontrol edin veya kütüphaneleri güncelleyin.")



def _groq_call(system: str, user: str, max_tokens: int = 50) -> str:
    for attempt in range(10): 
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
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
                print(f"[Groq] Rate limit, {wait}s wait...")
                time.sleep(wait)
            else:
                raise
    return ""


# ── A. Paraphrase ────────────────────────────────────────────────────────────

def paraphrase_sentence(sentence: str, phrase: str, lang: str) -> str:
    """
    Restate the sentence using different words but same meaning.
    Replaces the idiomatic phrase with its literal equivalent in meaning.
    """
    system = (
        "You are a paraphrasing assistant. "
        "Restate the sentence using plain language. "
        "Replace any idiomatic expressions with their actual meaning. "
        "Keep it short — just a few words or one sentence."
        f"CRITICAL: ALWAYS REPLY IN THE SOURCE LANGUAGE ({lang.upper()}). DO NOT USE ENGLISH UNLESS THE SOURCE IS ENGLISH."
    )
    few_shot_examples = ""
    lang_upper = lang.upper()
    
    if lang_upper == "TR":
        few_shot_examples = (
            "\nExample Guideline:\n"
            "Sentence: 'O kadar yorgunum ki ayaklarıma kara sular indi.'\n"
            "Phrase: 'ayaklarına kara sular inmek'\n"
            "Answer: 'Çok yoruldum.'\n\n"
            "Now do the following:\n"
        )
    elif lang_upper == "UZ":
        few_shot_examples = (
            "\nExample Guideline:\n"
            "Sentence: 'Imtihondan yiqilganini eshitib, tarvuzi qo'ltig'idan tushdi.'\n"
            "Phrase: 'tarvuzi qo'ltig'idan tushdi'\n"
            "Answer: 'Juda xafa bo'ldi.'\n\n"
            "Now do the following:\n"
        )
    elif lang_upper == "IG":
        few_shot_examples = (
            "\nExample Guideline:\n"
            "Sentence: 'Nwa a nwere ntị ike, anaghị anụ ihe.'\n"
            "Phrase: 'ntị ike'\n"
            "Answer: 'Isi ike.'\n\n"
            "Now do the following:\n"
        )
    elif lang_upper == "KA":
        few_shot_examples = (
            "\nExample Guideline:\n"
            "Sentence: 'ნუ ცდილობ ჩემთვის თვალებში ნაცრის შეყრას, სიმართლე ვიცი.'\n"
            "Phrase: 'თვალებში ნაცრის შეყრა'\n"
            "Answer: 'მოტყუება.'\n\n"
            "Now do the following:\n"
        )

    user = (
        f'{few_shot_examples}'
        f'Sentence: "{sentence}"\n'
        f'The phrase "{phrase}" is used idiomatically. '
        "Paraphrase this sentence with the same meaning but different words. "
        "Short answer only."
    )
    result = _groq_call(system, user, max_tokens=80)
    print(f"[Paraphrase] {result}")
    return result


def extract_atmosphere(sentence: str, phrase: str) -> str:
    """
    What emotional or atmospheric situation does the idiom evoke?
    Used as the figurative matching signal in the tournament.
    """
    system = (
        "You are a literary analyst. "
        "Describe the emotional or atmospheric situation evoked by a phrase in a sentence. "
        "Just a few words — no full sentences."
    )
    user = (
        f'Sentence: "{sentence}"\n'
        f'Phrase: "{phrase}"\n'
        "What atmospheric or emotional situation does this phrase evoke? "
        "Answer in just a few words."
    )
    result = _groq_call(system, user, max_tokens=30)
    print(f"[Atmosphere] {result}")
    return result


def extract_physical_objects(phrase: str) -> str:
    system = (
        "You are an object extraction assistant for computer vision. "
        "Given a phrase in any language, extract ONLY the tangible, physical nouns/objects. "
        "CRITICAL: YOU MUST TRANSLATE AND OUTPUT THE WORDS STRICTLY IN ENGLISH. "
        "Return them as a comma-separated list. No explanations, no extra words.\n"
    )
    user = f'Phrase: "{phrase}"'
    result = _groq_call(system, user, max_tokens=20)
    print(f"[PhysicalObjects] '{phrase}' -> {result}")
    return result

def grounding_dino_penalty(image_path: str, text_query: str) -> float:
    try:
        image = Image.open(image_path).convert("RGB")
        
        query = text_query.lower().strip()
        if not query.endswith("."):
            query += "."
            
        inputs = dino_processor(images=image, text=query, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        
        results = dino_processor.image_processor.post_process_object_detection(
            outputs, threshold=0.25, target_sizes=target_sizes
        )[0]
        
        scores = results["scores"]
        
        if len(scores) == 0:
            print(f"[GroundingDINO] '{text_query}' -> {os.path.basename(image_path)}: No matches (0.0)")
            return 0.0
            
        max_score = float(scores.max().item())
        
        if max_score < 0.50:
            print(f"[GroundingDINO] '{text_query}' -> {os.path.basename(image_path)}: {max_score:.3f} (Weak match, likely not literal)")
            return 0.0
            
        print(f"[GroundingDINO] '{text_query}' -> {os.path.basename(image_path)}: {max_score:.3f} (Strong literal match!)")
        return max_score

    except Exception as e:
        print(f"[GroundingDINO] Error - {image_path}: {e}")
        return 0.0


def grounding_dino_penalty_batch(
    candidates: list[dict],
    physical_query: str,
) -> dict[str, float]:
    penalties = {}
    for cand in candidates:
        img_id = cand["image_id"]
        img_path = cand["image_path"] 
        score = grounding_dino_penalty(img_path, physical_query)
        penalties[img_id] = round(score, 3)
    return penalties


def extract_context(
    sentence: str,
    phrase: str,
    candidates: list[dict],
    is_idiomatic: bool,
    lang: str,
    wiktionary_def: str | None = None,
) -> dict:
    """
    Full context extraction step.

    Returns:
        {
          "paraphrase":   str,    # plain-language restatement
          "atmosphere":   str,    # emotional/atmospheric keywords
          "penalties":    dict,   # {image_id: float} literal grounding penalty
          "match_query":  str,    # query to use in tournament
        }
    """
    print(f"\n[ContextExtract] sentence='{sentence}' | idiomatic={is_idiomatic}")
    physical_query = extract_physical_objects(phrase)
    penalties = grounding_dino_penalty_batch(candidates, physical_query)

    if not is_idiomatic:
        return {
            "paraphrase":  phrase,
            "atmosphere":  "",
            "penalties":   penalties,
            "match_query": phrase,
        }

    if wiktionary_def:
        paraphrase = wiktionary_def
        print(f"[ContextExtract] Using Wiktionary def as paraphrase: {paraphrase[:60]}")
    else:
        paraphrase = paraphrase_sentence(sentence, phrase,lang)

    atmosphere = extract_atmosphere(sentence, phrase)

    match_query = f"{paraphrase}. Atmosphere: {atmosphere}"

    return {
        "paraphrase":  paraphrase,
        "atmosphere":  atmosphere,
        "penalties":   penalties,
        "match_query": match_query,
    }



if __name__ == "__main__":
    sample = {
        "sentence":  "Yüksek maaşlı bir işe başlayan ablam, bir anda evin altın yumurtlayan tavuğu oldu.",
        "phrase":    "altın yumurtlayan tavuk",
        "is_idiomatic": True,
        "wiktionary_def": None,
        "candidates": [
            {"image_id": "img1", "image_path": r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk\16634799208.png"}, 
            {"image_id": "img2", "image_path": r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk\47911094135.png"},
            {"image_id": "img1", "image_path": r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk\60724405930.png"}, 
            {"image_id": "img2", "image_path": r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk\80371640998.png"},
            {"image_id": "img1", "image_path": r"C:\Users\necip\OneDrive\Desktop\NLP Codes\images\Turkish\altın yumurtlayan tavuk\99148916362.png"}, 
        ],
    }

    ctx = extract_context(
        sentence      = sample["sentence"],
        phrase        = sample["phrase"],
        candidates    = sample["candidates"],
        is_idiomatic  = sample["is_idiomatic"],
        wiktionary_def= sample["wiktionary_def"],
    )

    print(f"\n{'='*60}")
    print(json.dumps(ctx, indent=2))