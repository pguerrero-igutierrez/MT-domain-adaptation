"""
01_sample_general.py

Randomly samples 50,000 sentence pairs from the
`projecte-aina/CA-EU_Parallel_Corpus` training split and writes them as a
JSON list for the general-domain fine-tuning pipeline.

Input
-----
Hugging Face dataset: `projecte-aina/CA-EU_Parallel_Corpus`

Output
------
sampled-data/ca_eu_50k.json
    List of records with `ca` and `eu` fields.

Requirements
------------
    pip install datasets
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "projecte-aina/CA-EU_Parallel_Corpus"
SAMPLE_SIZE = 50_000
OUTPUT_FILE = Path("sampled-data/ca_eu_50k.json")
SEED = 42

def main():
    print(f"Loading dataset '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME, split="train")
    total = len(ds)
    print(f"Total rows available: {total:,}")

    n = min(SAMPLE_SIZE, total)
    random.seed(SEED)
    indices = random.sample(range(total), n)

    print(f"Sampling {n:,} rows...")
    sample = ds.select(indices)

    records = [{"ca": row["ca"], "eu": row["eu"]} for row in sample]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. {n:,} records saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
