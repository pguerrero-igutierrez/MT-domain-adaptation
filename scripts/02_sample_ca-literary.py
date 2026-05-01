"""
Pre-translation subsampling for the CA-ES corpus pipeline.
Reads corpus/corpus_ca_es.csv, filters to the most recent years,
samples 100k rows, and writes a JSON file ready to be passed directly
to 05_translate.py instead of the full corpus.

Output: corpus/corpus_ca_es_100k_recent.json
        List of records with all original columns preserved.
Requires: no extra dependencies (stdlib only)
"""

import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INPUT_CSV   = Path("corpus/corpus_ca_es.csv")
OUTPUT_JSON = Path("corpus/corpus_ca_es_100k_recent.json")

SAMPLE_SIZE = 100_000
YEAR_COLUMN = "year"
SEED        = 42

csv.field_size_limit(sys.maxsize)


def parse_year(value) -> int | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(s, fmt).year
        except ValueError:
            continue
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 4:
        candidate = int(digits[:4])
        if 1000 <= candidate <= 2100:
            return candidate
    return None


def main():
    print(f"Loading {INPUT_CSV}...")
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows       = list(reader)
    print(f"Total rows: {len(rows):,}")

    if YEAR_COLUMN not in fieldnames:
        raise ValueError(f"Column '{YEAR_COLUMN}' not found. Available: {fieldnames}")

    year_to_rows = defaultdict(list)
    skipped = 0
    for row in rows:
        year = parse_year(row.get(YEAR_COLUMN))
        if year is None:
            skipped += 1
        else:
            year_to_rows[year].append(row)

    if skipped:
        print(f"Skipped {skipped:,} rows with unparseable year.")

    sorted_years = sorted(year_to_rows.keys(), reverse=True)
    print(f"Years found (most recent first): {sorted_years[:10]}{'...' if len(sorted_years) > 10 else ''}")

    accumulated = []
    used_years  = []
    for year in sorted_years:
        year_rows = year_to_rows[year]
        accumulated.extend(year_rows)
        used_years.append(year)
        print(f"  Year {year}: {len(year_rows):,} rows  |  cumulative: {len(accumulated):,}")
        if len(accumulated) >= SAMPLE_SIZE:
            break

    print(f"\nYears used: {used_years}")
    print(f"Total rows in selected years: {len(accumulated):,}")

    random.seed(SEED)
    if len(accumulated) > SAMPLE_SIZE:
        accumulated = random.sample(accumulated, SAMPLE_SIZE)
        print(f"Downsampled to {SAMPLE_SIZE:,} rows.")

    accumulated.sort(key=lambda r: (r.get("doc_id", ""), int(r.get("para_id") or 0)))

    print(f"Saving to {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(accumulated, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(accumulated):,} rows saved to {OUTPUT_JSON}")
    print(f"Years covered: {min(used_years)} – {max(used_years)}")
    print(f"\nNow run: python scripts/05_translate.py --langs eu ca")
    print(f"         (update INPUT_CSV in that script to point to {OUTPUT_JSON})")


if __name__ == "__main__":
    main()