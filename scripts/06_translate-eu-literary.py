"""
06_translate-eu-literary.py

Translates Spanish literary sentences in two independent passes using
HiTZ/Latxa-Llama-3.1-8B-Instruct via vLLM offline batching:
  Pass 1: source_es -> Basque   (eu_backtrans)
  Pass 2: source_es -> Catalan  (ca_translation)

Both passes use the same Spanish source sentence as input.

INPUT
-----
File   : sampled-data/ehuhac_parallel.jsonl
Format : one JSON object per line with fields:
         doc_id, para_id, offset_es, offset_eu, source_es, source_eu

PIPELINE
--------
1. Load parallel pairs from ehuhac_parallel.jsonl.
2. Translate source_es -> Basque using Latxa (vLLM offline batch).
3. Translate source_es -> Catalan using Latxa (vLLM offline batch).
4. Write per-document JSONL files + merged all.jsonl.

OUTPUT
------
backtranslated-corpus/<doc_id>.jsonl
backtranslated-corpus/eu-literary-trilingual.jsonl

Each output line:
    {
      "doc_id":         str,
      "para_id":        int,
      "offset_es":      int,
      "offset_eu":      int,
      "offset_ca":      int,
      "source_es":      str,
      "source_eu":      str,
      "eu_backtrans":   str,
      "ca_translation": str
    }

REQUIREMENTS
------------
    pip install vllm tqdm

USAGE
-----
    python 06_translate-eu-literary.py
    python 06_translate-eu-literary.py --resume
    python 06_translate-eu-literary.py --batch-size 64 --max-tokens 512
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

INPUT_JSONL  = Path("sampled-data/ehuhac_sampled_parallel.jsonl")
OUTPUT_DIR   = Path("backtranslated-corpus")

MODEL_ID     = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
BATCH_SIZE   = 64
MAX_TOKENS   = 512
TEMPERATURE  = 0.0
TENSOR_PARALLEL = 1


ES_TO_EU_SYSTEM = (
    "Zara itzultzaile profesional bat. "
    "Itzuli ondorengo gaztelaniazko testua euskarara. "
    "Eman itzulpena soilik, azalpenik gabe."
)

ES_TO_CA_SYSTEM = (
    "Ets un traductor professional. "
    "Tradueix el text en castellà següent al català. "
    "Proporciona només la traducció, sense explicacions."
)


def load_sampled(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
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
    print(f"Loaded {total:,} pairs across {len(groups)} document(s) from {path}")
    return groups


def compute_offsets(texts: list[str]) -> list[int]:
    offsets, pos = [], 0
    for t in texts:
        offsets.append(pos)
        pos += len(t.encode("utf-8")) + 1
    return offsets


def build_prompts(texts: list[str], system: str) -> list[str]:
    prompts = []
    for text in texts:
        prompts.append(
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )
    return prompts


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


def load_done_stems(output_dir: Path) -> set[str]:
    done = {f.stem for f in output_dir.glob("*.jsonl") if f.stem != "all"}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--tensor-parallel", type=int, default=TENSOR_PARALLEL)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_output = OUTPUT_DIR / "eu-literary-all.jsonl"

    groups = load_sampled(INPUT_JSONL)
    done_stems = load_done_stems(OUTPUT_DIR) if args.resume else set()
    pending = {doc_id: rows for doc_id, rows in groups.items() if doc_id not in done_stems}

    if not pending:
        print("Nothing new to translate.")
        return

    print(f"Loading {MODEL_ID}...")
    llm = LLM(model=MODEL_ID, tensor_parallel_size=args.tensor_parallel)
    print("  Model loaded.")

    for doc_id, doc_rows in tqdm(pending.items(), desc="Documents"):
        print(f"\n{doc_id} ({len(doc_rows):,} pairs)")

        es_texts = [r["source_es"] for r in doc_rows]

        print("  Step 1: ES -> EU backtranslation")
        eu_backtrans = run_inference(
            llm, es_texts, ES_TO_EU_SYSTEM,
            args.batch_size, args.max_tokens, "  ES->EU",
        )

        print("  Step 2: ES -> CA translation")
        ca_translations = run_inference(
            llm, es_texts, ES_TO_CA_SYSTEM,
            args.batch_size, args.max_tokens, "  ES->CA",
        )

        offsets_es = compute_offsets(es_texts)
        offsets_eu = compute_offsets([r["source_eu"] for r in doc_rows])
        offsets_ca = compute_offsets(ca_translations)

        records = [
            {
                "doc_id":         doc_id,
                "para_id":        doc_rows[i]["para_id"],
                "offset_es":      offsets_es[i],
                "offset_eu":      offsets_eu[i],
                "offset_ca":      offsets_ca[i],
                "source_es":      es_texts[i],
                "source_eu":      doc_rows[i]["source_eu"],
                "eu_backtrans":   eu_backtrans[i],
                "ca_translation": ca_translations[i],
            }
            for i in range(len(doc_rows))
            if eu_backtrans[i] and ca_translations[i]
        ]

        out_file = OUTPUT_DIR / f"eu-literary-{doc_id}.jsonl"
        save_jsonl(records, out_file)
        append_jsonl(records, all_output)
        print(f"  Saved {len(records):,} records -> {out_file.name}")

    print(f"\nDone. Merged output -> {all_output}")


if __name__ == "__main__":
    main()