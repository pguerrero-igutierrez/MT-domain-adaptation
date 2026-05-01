"""
Pre-translation subsampling for the EU clinical JSON pipeline.
Reads JSON files from clinical/eu-clinical/, extracts paragraphs
(same logic as 03_translate-eu-clinical.py), sorts documents by
publication_date (most recent first), accumulates until 100k paragraphs
are reached, then writes a JSON file ready to be fed to the translation step.

Output: parallel_corpus_eu_ca_sampled.json
        List of records with all metadata + eu + ca="" fields.
        Drop-in replacement: load and pass directly to run_translation().
Requires: no extra dependencies (stdlib only)
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path

EU_DIR      = Path("clinical/eu-clinical")
OUTPUT_JSON = Path("parallel_corpus_eu_ca_sampled.json")

SAMPLE_SIZE  = 100_000
MIN_PARA_LEN = 20
SEED         = 42


def load_json_file(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path.name}: {e}")
        return {}


def extract_meta(data: dict) -> dict:
    authors_raw = data.get("authors", [])
    authors = ", ".join(
        a.get("author", "") for a in authors_raw
        if isinstance(a, dict) and a.get("author", "") not in ("", "NO", "AUTHOR")
    ) if isinstance(authors_raw, list) else ""
    return {
        "doc_id":           data.get("id", ""),
        "language":         data.get("language", ""),
        "publication_date": data.get("publication_date", ""),
        "source":           data.get("source", ""),
        "source_url":       data.get("source_url", ""),
        "doc_type":         data.get("type", ""),
        "licence":          data.get("licence", ""),
        "authors":          authors,
        "url":              data.get("url", ""),
    }


def parse_date(value) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 4:
        try:
            return datetime(int(digits[:4]), 1, 1)
        except ValueError:
            pass
    return None


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r'\n{2,}', text)
    paragraphs = []
    for block in blocks:
        block = re.sub(r'\s+', ' ', block.replace('\n', ' ')).strip()
        if len(block) > MIN_PARA_LEN:
            paragraphs.append(block)
    return paragraphs


def main():
    json_files = sorted(EU_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {EU_DIR}")
    print(f"Found {len(json_files):,} JSON files in {EU_DIR}")

    print("Parsing publication dates...")
    file_meta = []
    skipped = 0
    for path in json_files:
        data = load_json_file(path)
        if not data:
            continue
        meta = extract_meta(data)
        text = data.get("text", "").strip()
        if not text:
            continue
        date = parse_date(meta["publication_date"])
        if date is None:
            skipped += 1
        file_meta.append((date, path, meta, text))

    if skipped:
        print(f"  {skipped:,} files with unparseable publication_date (kept, sorted last).")

    file_meta.sort(key=lambda x: (x[0] is None, x[0] if x[0] else datetime.min), reverse=True)

    print("Accumulating paragraphs from most recent documents...")
    accumulated = []
    used_docs = []
    for date, path, meta, text in file_meta:
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            continue
        rows = [
            {**meta, "para_id": i, "eu": para, "ca": ""}
            for i, para in enumerate(paragraphs)
        ]
        accumulated.extend(rows)
        used_docs.append((meta["doc_id"], str(date)[:10] if date else "unknown", len(paragraphs)))
        print(f"  {path.name}: {len(paragraphs):,} paragraphs  |  cumulative: {len(accumulated):,}")
        if len(accumulated) >= SAMPLE_SIZE:
            break

    print(f"\nDocuments used: {len(used_docs)}")
    print(f"Total paragraphs accumulated: {len(accumulated):,}")

    random.seed(SEED)
    if len(accumulated) > SAMPLE_SIZE:
        accumulated = random.sample(accumulated, SAMPLE_SIZE)
        accumulated.sort(key=lambda r: (str(r.get("doc_id", "")), int(r.get("para_id") or 0)))
        print(f"Downsampled to {SAMPLE_SIZE:,} paragraphs.")

    print(f"Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(accumulated, f, ensure_ascii=False, indent=2)

    dates_used = [d for _, d, _ in used_docs if d != "unknown"]
    print(f"Done. {len(accumulated):,} paragraphs saved to {OUTPUT_JSON}")
    if dates_used:
        print(f"Date range: {min(dates_used)} – {max(dates_used)}")
    print(f"\nNext step: load {OUTPUT_JSON} and pass to run_translation() in 03_translate-eu-clinical.py")


if __name__ == "__main__":
    main()