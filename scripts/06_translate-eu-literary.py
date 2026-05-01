"""
02_translate-eu-literary.py

Translates Basque literary paragraphs into Catalan using
facebook/nllb-200-3.3B (eus_Latn -> cat_Latn).

INPUT
-----
File   : literary/eu-literary/output/sampled_pretranslation.jsonl
Format : one JSON object per line with fields:
         doc_id, para_id, offset_eu, source_eu
         (produced by sample_eu_literary_pretranslation.py)

PIPELINE
--------
1. Load pre-sampled paragraphs from sampled_pretranslation.jsonl.
2. Translate each paragraph EU -> CA in batches using NLLB-200 3.3B.
3. Align translated paragraphs back to source by index, compute CA offsets.
4. Write one JSONL file per document + a merged all.jsonl.

OUTPUT
------
literary/eu-literary/output/<doc_id>.jsonl  — per-document aligned JSONL
literary/eu-literary/output/all.jsonl       — merged JSONL, all documents

Each line:
    {
      "doc_id":         "AzkarateGaltzaundi",
      "para_id":        0,
      "offset_eu":      0,
      "offset_ca":      0,
      "source_eu":      "...",
      "ca_translation": "..."
    }

REQUIREMENTS
------------
    pip install transformers torch tqdm sentencepiece

USAGE
-----
    python 02_translate-eu-literary.py
    python 02_translate-eu-literary.py --resume
    python 02_translate-eu-literary.py --batch-size 16 --max-tokens 256
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

INPUT_DIR   = Path("literary/eu-literary")
OUTPUT_DIR  = INPUT_DIR / "output"
INPUT_JSONL = OUTPUT_DIR / "sampled_pretranslation.jsonl"

NLLB_MODEL      = "facebook/nllb-200-3.3B"
NLLB_SRC        = "eus_Latn"
NLLB_TGT        = "cat_Latn"
NLLB_BATCH_SIZE = 8
MAX_NEW_TOKENS  = 512
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def load_sampled(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}\nRun sample_eu_literary_pretranslation.py first.")
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                groups[rec["doc_id"]].append(rec)
    for doc_id in groups:
        groups[doc_id].sort(key=lambda r: int(r["para_id"]))
    total = sum(len(v) for v in groups.values())
    print(f"Loaded {total:,} paragraphs across {len(groups)} documents from {path}")
    return groups


def compute_offsets(paragraphs: list[str]) -> list[int]:
    offsets, pos = [], 0
    for p in paragraphs:
        offsets.append(pos)
        pos += len(p) + 2
    return offsets


def load_nllb() -> tuple:
    print(f"Loading {NLLB_MODEL} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    print("  Model loaded.")
    return tokenizer, model


def translate_batch(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_tokens: int,
) -> list[str]:
    tokenizer.src_lang = NLLB_SRC
    forced_bos         = tokenizer.convert_tokens_to_ids(NLLB_TGT)
    results            = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Translating", leave=False):
        batch = texts[i : i + batch_size]
        try:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_tokens,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=max_tokens,
                    num_beams=4,
                    do_sample=False,
                )
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        except Exception as e:
            print(f"  [WARN] Batch {i} failed: {e}")
            results.extend([""] * len(batch))

    return results


def load_done_stems(output_dir: Path) -> set[str]:
    done = set()
    for f in output_dir.glob("*.jsonl"):
        if f.stem not in ("all", "sampled_pretranslation"):
            done.add(f.stem)
    if done:
        print(f"  Resume: {len(done)} document(s) already translated.")
    return done


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate pre-sampled Basque literary paragraphs to Catalan with NLLB-200.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip documents already present in the output directory.",
    )
    parser.add_argument("--batch-size", type=int, default=NLLB_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_output = OUTPUT_DIR / "all.jsonl"

    groups     = load_sampled(INPUT_JSONL)
    done_stems = load_done_stems(OUTPUT_DIR) if args.resume else set()
    pending    = {doc_id: rows for doc_id, rows in groups.items() if doc_id not in done_stems}

    if not pending:
        print("Nothing new to translate.")
        return

    print(f"Processing {len(pending)} document(s)...")
    tokenizer, model = load_nllb()

    for doc_id, doc_rows in tqdm(pending.items(), desc="Documents"):
        print(f"\n{doc_id}")
        eu_paras = [r["source_eu"] for r in doc_rows]

        print(f"  Paragraphs: {len(eu_paras)}")
        ca_paras = translate_batch(eu_paras, tokenizer, model, args.batch_size, args.max_tokens)

        offsets_ca = compute_offsets(ca_paras)
        records = [
            {
                "doc_id":         doc_id,
                "para_id":        doc_rows[i]["para_id"],
                "offset_eu":      doc_rows[i]["offset_eu"],
                "offset_ca":      offsets_ca[i],
                "source_eu":      eu_paras[i],
                "ca_translation": ca_paras[i],
            }
            for i in range(min(len(eu_paras), len(ca_paras)))
            if eu_paras[i] and ca_paras[i]
        ]

        out_file = OUTPUT_DIR / f"{doc_id}.jsonl"
        save_jsonl(records, out_file)
        append_jsonl(records, all_output)
        print(f"  Saved {len(records)} aligned pairs -> {out_file.name}")

    del model
    torch.cuda.empty_cache()
    print(f"\nDone. Merged output -> {all_output}")


if __name__ == "__main__":
    main()