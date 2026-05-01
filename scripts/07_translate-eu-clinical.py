"""
07_translate-eu-clinical.py

Builds a parallel EU-CA corpus from pre-sampled Basque paragraphs.

INPUT
-----
File   : parallel_corpus_eu_ca_sampled.json
Format : list of records with fields:
         doc_id, language, publication_date, source, source_url, doc_type,
         licence, authors, url, para_id, eu, ca (ca="" untranslated)
         (produced by sample_eu_clinical_pretranslation.py)

PIPELINE
--------
1. Load pre-sampled paragraphs from parallel_corpus_eu_ca_sampled.json.
2. Translate each Basque paragraph into Catalan using NLLB-200 3.3B
   (EU -> CA, eus_Latn -> cat_Latn) with GPU batching.
3. Align translated paragraphs back to their source by document and
   paragraph index.

OUTPUT
------
parallel_corpus_eu_ca.json  — full aligned corpus with all metadata fields
                              + eu + ca, one record per paragraph.

REQUIREMENTS
------------
    pip install transformers torch tqdm sentencepiece

USAGE
-----
    python 03_translate-eu-clinical.py
    python 03_translate-eu-clinical.py --resume
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

INPUT_JSON  = Path("sampled-data/eu-clinical_sampled100k.json")
OUTPUT_JSON = Path("backtranslated-corpus/eu-clinical_backtranslated.json")

NLLB_MODEL      = "facebook/nllb-200-3.3B"
NLLB_SRC        = "eus_Latn"
NLLB_TGT        = "cat_Latn"
NLLB_BATCH_SIZE = 8
MAX_NEW_TOKENS  = 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def load_sampled(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}\nRun sample_eu_clinical_pretranslation.py first.")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Loaded {len(rows):,} paragraphs from {path}")
    return rows


def load_done_doc_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        existing = json.load(f)
    done = {r["doc_id"] for r in existing if r.get("ca", "").strip()}
    print(f"  Resume: {len(done)} already-translated doc_ids found in {path.name}")
    return done


def load_nllb() -> tuple:
    print(f"Loading NLLB model: {NLLB_MODEL} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model) -> list[str]:
    tokenizer.src_lang = NLLB_SRC
    forced_bos         = tokenizer.convert_tokens_to_ids(NLLB_TGT)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_NEW_TOKENS,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=MAX_NEW_TOKENS,
            num_beams=4,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def run_translation(rows: list[dict]) -> list[dict]:
    print(f"\nTranslating EU -> CA with NLLB-200 3.3B ({len(rows):,} paragraphs)")
    tokenizer, model = load_nllb()
    texts = [r["eu"] for r in rows]

    for i in tqdm(range(0, len(texts), NLLB_BATCH_SIZE), desc="EU->CA"):
        batch = texts[i : i + NLLB_BATCH_SIZE]
        try:
            translated = translate_batch(batch, tokenizer, model)
        except Exception as e:
            print(f"  [WARN] Batch {i} failed: {e}")
            translated = [""] * len(batch)
        for j, ca in enumerate(translated):
            rows[i + j]["ca"] = ca

    del model
    torch.cuda.empty_cache()
    return rows


def align_by_doc(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)

    aligned = []
    for doc_id, doc_rows in tqdm(groups.items(), desc="Aligning"):
        doc_rows.sort(key=lambda r: int(r["para_id"]))
        for i, row in enumerate(doc_rows):
            row["para_id"] = i
            aligned.append(row)

    print(f"  Aligned rows: {len(aligned):,}")
    return aligned


def save_json(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"JSON -> {path} ({len(rows):,} records)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate pre-sampled Basque clinical paragraphs to Catalan with NLLB-200.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip doc_ids already translated in the output JSON.",
    )
    args = parser.parse_args()

    print("Step 1: Loading pre-sampled paragraphs")
    rows     = load_sampled(INPUT_JSON)
    skip_ids = load_done_doc_ids(OUTPUT_JSON) if args.resume else set()

    existing_rows = []
    if args.resume and OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing_rows = json.load(f)

    rows = [r for r in rows if r["doc_id"] not in skip_ids]
    if not rows:
        print("Nothing new to translate.")
        return

    print(f"  Paragraphs to translate: {len(rows):,}")

    print("\nStep 2: Translating EU -> CA")
    rows = run_translation(rows)

    print("\nStep 3: Aligning by document")
    rows = align_by_doc(rows)

    if args.resume and existing_rows:
        print(f"  Merging {len(existing_rows):,} existing rows with {len(rows):,} new rows")
        rows = existing_rows + rows
        rows.sort(key=lambda r: (str(r.get("doc_id", "")), int(r.get("para_id") or 0)))

    print("\nStep 4: Saving output")
    save_json(rows, OUTPUT_JSON)

    print("\nDone.")


if __name__ == "__main__":
    main()