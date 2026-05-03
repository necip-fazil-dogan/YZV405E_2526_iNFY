import os
import subprocess
from pathlib import Path


os.environ["GROQ_API_KEY"]       = ""   
os.environ["OPENROUTER_API_KEY"] = ""   


THREADS = 3   


LANGUAGES = [
    ["Turkish","TR"],
    #["Chinese","ZH"],
    #["Georgian","KA"],
    #["Greek","EL"],
    #["Igbo","IG"],
    #["Kazakh","KK"],
    #["Norwegian","NO"],
    #["Portuguese-Brazil","PT-BR"],
    #["Portuguese-Portugal","PT-PT"],
    #["Russian","RU"],
    #["Serbian","SR"],
    #["Slovak","SK"],
    #["Slovenian","SL"],
    #["Spanish-Ecuador","ES-EC"],
    #["Turkish","TR"],
    #["Uzbek","UZ"]
]


TSV_DIR    = "."        
IMAGE_DIR  = "images"  
OUTPUT_DIR = "output"  


Path(OUTPUT_DIR).mkdir(exist_ok=True)

for lang in LANGUAGES:
    tsv = f"{TSV_DIR}/submission_{lang[0]}.tsv"
    imgs = f"{IMAGE_DIR}/{lang[0]}"
    out  = f"{OUTPUT_DIR}/submission_{lang[1]}.tsv"

    if not Path(tsv).exists():
        print(f"⚠️  {tsv} bulunamadı, atlanıyor...")
        continue

    print(f"\n{'='*50}")
    print(f"    Language: {lang[0]} ({lang[1]})")
    print(f"   TSV    : {tsv}")
    print(f"   Imgs   : {imgs}")
    print(f"   Out    : {out}")
    print(f"   Threads: {THREADS}")
    print(f"{'='*50}")

    result = subprocess.run(
        ["python", "admire_pipeline.py",
         "--tsv",       tsv,
         "--image_dir", imgs,
         "--output",    out,
         "--threads",   str(THREADS)],env=os.environ
    )

    if result.returncode == 0:
        print(f"✅ {lang[0]} ({lang[1]}) completed → {out}")
    else:
        print(f"❌ {lang[0]} ({lang[1]}) failed, continuing...")

print(f"\n🎉 Done! Output files: {OUTPUT_DIR}/")

