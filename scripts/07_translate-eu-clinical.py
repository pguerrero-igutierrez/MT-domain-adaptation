"""
07_translate-eu-clinical.py

Builds a parallel EU-CA corpus from pre-sampled Basque clinical paragraphs
using HiTZ/Latxa-Llama-3.1-8B-Instruct via vLLM offline batching:
  Pass 1: source_eu -> Catalan      (ca)

INPUT
-----
File   : sampled-data/eu-clinical_sampled100k.json
Format : JSON list of records with fields:
         doc_id, language, publication_date, source, source_url, doc_type,
         licence, authors, url, para_id, eu, ca (ca="" untranslated)

PIPELINE
--------
1. Load pre-sampled paragraphs from eu-clinical_sampled100k.json.
2. Translate eu -> Catalan using Latxa (vLLM offline batch).
3. Align by document and paragraph index.
4. Save merged output JSON preserving all metadata fields.

OUTPUT
------
backtranslated-corpus/eu-clinical_backtranslated.json

Each output record preserves all original metadata fields, plus:
    {
      ...,
      "eu":             str,   (original Basque)
      "ca":             str    (translated Catalan)
    }

REQUIREMENTS
------------
    pip install vllm tqdm

USAGE
-----
    python 07_translate-eu-clinical.py
    python 07_translate-eu-clinical.py --resume
    python 07_translate-eu-clinical.py --batch-size 64 --max-tokens 512
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

INPUT_JSON   = Path("sampled-data/eu-clinical_sampled100k.json")
OUTPUT_JSON  = Path("backtranslated-corpus/eu-clinical_backtranslated.json")

MODEL_ID        = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
BATCH_SIZE      = 64
MAX_TOKENS      = 512
TEMPERATURE     = 0.0
TENSOR_PARALLEL = 1

EU_TO_CA_SYSTEM = (
    "Ets un traductor professional. "
    "Tradueix el text en basc següent al català. "
    "Proporciona només la traducció, sense explicacions."
)

def load_sampled(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
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


def build_prompts(texts: list[str], system: str) -> list[str]:
    return [
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        for text in texts
    ]


def run_inference(
    llm: LLM,
    texts: list[str],
    system: str,
    batch_size: int,
    max_tokens: int,
    desc: str,
) -> list[str]:
    sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens)
    results: list[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        prompts = build_prompts(batch_texts, system)
        try:
            outputs = llm.generate(prompts, sampling)
            for out in outputs:
                results.append(out.outputs[0].text.strip())
        except Exception as e:
            print(f"  [WARN] Batch {i} failed: {e}")
            results.extend([""] * len(batch_texts))
    return results


def align_by_doc(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)
    aligned = []
    for doc_rows in groups.values():
        doc_rows.sort(key=lambda r: int(r["para_id"]))
        aligned.extend(doc_rows)
    return aligned


def save_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows):,} records -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--tensor-parallel", type=int, default=TENSOR_PARALLEL)
    args = parser.parse_args()

    rows     = load_sampled(INPUT_JSON)
    skip_ids = load_done_doc_ids(OUTPUT_JSON) if args.resume else set()

    existing_rows: list[dict] = []
    if args.resume and OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, encoding="utf-8") as f:
            existing_rows = json.load(f)

    rows = [r for r in rows if r["doc_id"] not in skip_ids]
    if not rows:
        print("Nothing new to translate.")
        return

    print(f"Paragraphs to process: {len(rows):,}")
    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, tensor_parallel_size=args.tensor_parallel)
    print("  Model loaded.")

    eu_texts = [r["eu"] for r in rows]

    print("\nStep 1: EU -> CA")
    ca_translations = run_inference(
        llm, eu_texts, EU_TO_CA_SYSTEM,
        args.batch_size, args.max_tokens, "EU->CA",
    )

    for i, row in enumerate(rows):
        row["ca"] = ca_translations[i] if i < len(ca_translations) else ""

    rows = [r for r in rows if r.get("ca")]
    print(f"\nValid records after filtering empties: {len(rows):,}")

    rows = align_by_doc(rows)

    if args.resume and existing_rows:
        print(f"Merging {len(existing_rows):,} existing + {len(rows):,} new rows")
        rows = existing_rows + rows
        rows.sort(key=lambda r: (str(r.get("doc_id", "")), int(r.get("para_id") or 0)))

    save_json(rows, OUTPUT_JSON)
    print("\nDone.")


if __name__ == "__main__":
    main()
