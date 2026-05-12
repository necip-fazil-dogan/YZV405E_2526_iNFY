import os
import sys
import csv
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from modules.idiom_detection    import detect_idiom
from modules.context_extraction import extract_context
from modules.tournament_v2      import run_tournament_v2

DEFAULT_TSV       = "submission_Turkish.tsv"
DEFAULT_IMAGE_DIR = "images/"
DEFAULT_OUTPUT    = "outputs/submission_Turkish_filled.tsv"


def parse_tsv(tsv_path: str) -> list[dict]:
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            candidates = []
            for j in range(1, 6):
                name_key    = f"image{j}_name"
                caption_key = f"image{j}_caption"
                img_name    = row.get(name_key, "").strip()
                caption     = row.get(caption_key, "").strip()
                if img_name:
                    candidates.append({
                        "image_id":   f"img{j}",
                        "image_name": img_name,
                        "caption":    caption,
                    })
            rows.append({
                "row_index":      i,
                "compound":       row.get("compound", "").strip(),
                "sentence":       row.get("sentence", "").strip(),
                "expected_order": row.get("expected_order", "").strip(),
                "candidates":     candidates,
                "_raw":           row,
            })
    return rows

def resolve_image_paths(candidates: list[dict], image_dir: str):
    for cand in candidates:
        name = cand["image_name"]
        p = Path(image_dir) / name
        if p.exists():
            cand["image_path"] = str(p)
            continue
            
        found = False
        if Path(image_dir).exists():
            for sub in Path(image_dir).iterdir():
                if sub.is_dir():
                    candidate_path = sub / name
                    if candidate_path.exists():
                        cand["image_path"] = str(candidate_path)
                        found = True
                        break
        if not found:
            cand["image_path"] = "NOT_FOUND"


def process_row(row: dict, image_dir: str, lang: str) -> tuple[str, dict]:
    sentence   = row["sentence"]
    compound   = row["compound"]
    candidates = row["candidates"]

    print(f"\n{'─'*60}")
    print(f"Compound : {compound}")
    print(f"Sentence : {sentence[:80]}{'...' if len(sentence)>80 else ''}")

    resolve_image_paths(candidates, image_dir)

    # 1. Idiom Detection
    idiom = detect_idiom(sentence, compound,lang)
    is_idiomatic = idiom["is_idiomatic"]

    # 2. Context Extraction
    ctx = extract_context(
        sentence=sentence,
        phrase=compound,
        candidates=candidates,
        is_idiomatic=is_idiomatic,
        lang=lang,
        wiktionary_def=idiom.get("wiktionary_def")
    )

    # 3. Tournament
    ranked, match_log = run_tournament_v2(
        candidates=candidates,
        match_query=ctx["match_query"],
        penalties=ctx["penalties"],
        caption_sims=ctx["caption_sims"],
        is_idiomatic=is_idiomatic,
        image_dir=image_dir,
    )

    id_to_name = {c["image_id"]: c["image_name"] for c in candidates}
    ordered_names = [id_to_name[r["image_id"]] for r in ranked]
    result = str(ordered_names)
    print(f"[Result] {result}")

    debug_info = {
        "row_index": row["row_index"],
        "idiom": compound,
        "sentence": sentence,
        "is_idiomatic": is_idiomatic,
        "confidence": idiom["confidence"],
        "paraphrase": ctx["paraphrase"],
        "atmosphere": ctx["atmosphere"],
        "match_query": ctx["match_query"],
        "penalties": ctx["penalties"],
        "caption_sims": ctx["caption_sims"],
        "final_ranking": [
            {
                "image_name": r["image_name"],
                "wins":       r["wins"],
                "dino_score": r["dino_score"],
                "cap_sim":    r["cap_sim"],
                "penalty":    r["penalty"],
                "score":      r["score"],
            }
            for r in ranked
        ]
    }
    
    return result, debug_info



def main():
    parser = argparse.ArgumentParser(description="AdMIRe 2 Local Submission Pipeline")
    parser.add_argument("--tsv",       default=DEFAULT_TSV,       help="Input TSV path")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Folder with image files")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,    help="Output TSV path")
    parser.add_argument("--limit",     type=int, default=None,    help="Process only first N rows")
    parser.add_argument("--threads",   type=int, default=10,      help="Parallel thread number (default: 10)")
    parser.add_argument("--lang", type=str, default="TR", help="Language short code (e.g. TR, UZ, KA)")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    rows = parse_tsv(args.tsv)
    if args.limit:
        rows = rows[:args.limit]

    t_start = time.time()
    results    = [None] * len(rows)   
    debug_logs = [None] * len(rows)

    def worker(i, row):
        try:
            order, debug_info = process_row(row, args.image_dir,args.lang)
            return i, order, debug_info
        except Exception as e:
            print(f"\n  [Error] Row {row['row_index']}: {e}")
            order = str([c["image_name"] for c in row["candidates"]])
            return i, order, {"row_index": row["row_index"], "error": str(e)}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]["_raw"].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t").writeheader()

    import threading
    write_lock = threading.Lock()

    def save_row(row, order):
        with write_lock:
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                out_row = dict(row["_raw"])
                out_row["expected_order"] = order
                writer.writerow(out_row)

    print(f"🚀 {len(rows)} row, {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(worker, i, row): i for i, row in enumerate(rows)}
        done = 0
        for future in as_completed(futures):
            i, order, debug_info = future.result()
            results[i]    = (rows[i], order)
            debug_logs[i] = debug_info
            save_row(rows[i], order)
            done += 1
            print(f"\r  ✓ {done}/{len(rows)} completed", end="", flush=True)

    debug_file_path = out_path.with_name(out_path.stem + "_rapor.json")
    with open(debug_file_path, "w", encoding="utf-8") as f:
        json.dump(debug_logs, f, indent=2, ensure_ascii=False)

    elapsed = round(time.time() - t_start, 1)
    print(f"\n\n✅ Process Completed! {len(results)} rows processed in {elapsed} seconds.")
    print(f"📁 Output TSV: {out_path}")
    print(f"🔎 Detailed Analysis Report: {debug_file_path}")


if __name__ == "__main__":
    main()