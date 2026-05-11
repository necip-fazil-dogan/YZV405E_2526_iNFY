"""
Birleştirme: tam TSV (base) içinde compound + sentence eşleşen satırlarda
yalnızca expected_order alanını yeni pipeline çıktısından günceller.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def row_key(row: dict) -> tuple[str, str]:
    return (row.get("compound", "").strip(), row.get("sentence", "").strip())


def format_expected_order_like_legacy(names: list[str]) -> str:
    """submission_old ile aynı: ['a.png', 'b.png'] (tek tırnak, csv'' kaçışı yok)."""
    return str(names)


def normalize_expected_order(cell: str) -> str:
    """JSON / literal liste / boşlukla ayrılmış dosya adlarını legacy str(list)'e çevir."""
    s = (cell or "").strip()
    if not s:
        return s
    parsed = None
    try:
        parsed = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            pass
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return format_expected_order_like_legacy(parsed)
    # Pipeline hata yedeği veya eski çıktı: "a.png b.png c.png"
    parts = s.split()
    if parts and all(_is_image_filename_token(p) for p in parts):
        return format_expected_order_like_legacy(parts)
    return s


def _is_image_filename_token(p: str) -> bool:
    if not p or any(ch in p for ch in "[]'\""):
        return False
    if "." not in p:
        return False
    ext = p.rsplit(".", 1)[-1].lower()
    return ext in ("png", "jpg", "jpeg", "webp", "gif")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge expected_order by (compound, sentence).")
    p.add_argument("--base", type=Path, default=Path("submission_old.tsv"))
    p.add_argument("--new", type=Path, default=Path("output/submission_TR.tsv"))
    p.add_argument("--out", type=Path, default=Path("submission_merged.tsv"))
    args = p.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Base dosya yok: {args.base}")
    if not args.new.exists():
        raise SystemExit(f"Yeni çıktı yok: {args.new}")

    with args.new.open(newline="", encoding="utf-8") as f:
        new_rows = list(csv.DictReader(f, delimiter="\t"))
    # Aynı (compound, sentence) birden fazlaysa sondaki geçerli
    new_by_key: dict[tuple[str, str], dict] = {}
    for r in new_rows:
        new_by_key[row_key(r)] = r

    with args.base.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames or "expected_order" not in fieldnames:
            raise SystemExit("Base TSV beklenen sütunları içermiyor.")
        base_rows = list(reader)

    updated = 0
    for r in base_rows:
        k = row_key(r)
        if k in new_by_key and "expected_order" in new_by_key[k]:
            r["expected_order"] = new_by_key[k]["expected_order"]
            updated += 1

    for r in base_rows:
        r["expected_order"] = normalize_expected_order(r.get("expected_order", ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(base_rows)

    print(f"Base satır: {len(base_rows)} | Yeni satır: {len(new_rows)} | Yeni benzersiz anahtar: {len(new_by_key)}")
    print(f"Güncellenen expected_order: {updated}")
    print(f"Yazıldı: {args.out.resolve()}")


if __name__ == "__main__":
    main()
