"""
Randomly samples 50,000 rows from the projecte-aina/CA-EU_Parallel_Corpus dataset
and saves them to a JSON file. The dataset contains two columns: 'ca' (Catalan)
and 'eu' (Basque). Requires: pip install datasets
"""

import json
import random
from datasets import load_dataset

DATASET_NAME = "projecte-aina/CA-EU_Parallel_Corpus"
SAMPLE_SIZE = 100_000
OUTPUT_FILE = "ca_eu_100k.json"
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

    print(f"Saving to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. {n:,} records saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()