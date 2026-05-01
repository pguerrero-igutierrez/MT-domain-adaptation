"""
Pre-translation subsampling for the EU literary SGML pipeline.
Reads .sgml files from literary/eu-literary/, extracts and parses paragraphs
(same logic as 02_translate-eu-literary.py), then selects 100k paragraphs
from the most recently modified source files and writes a single all.jsonl
with only source_eu filled in — ready to be fed to the translation step.

Output: sampled-data/eu-literary_sampled100k.jsonl
        Each line: { "doc_id", "para_id", "offset_eu", "source_eu" }
Requires: no extra dependencies (stdlib only)
"""

import html
import json
import re
from pathlib import Path





INPUT_DIR   = Path("data/literary/eu-literary")
OUTPUT_DIR = Path("sampled-data")
OUTPUT_FILE = OUTPUT_DIR / "eu-literary_sampled100k.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_SIZE = 100_000
MIN_PARA_LEN = 20

TEXT_TAGS = re.compile(
    r'<(?:p|head|item|l|lg|docTitle|title|note|bibl|catDesc|label)(?:\s[^>]*)?>',
    re.IGNORECASE,
)
TAG_STRIP  = re.compile(r'<[^>]+>')
ENTITY_MAP = {
    "&ntilde;": "ñ", "&laquo;": "«", "&raquo;": "»",
    "&aacute;": "á", "&eacute;": "é", "&iacute;": "í",
    "&oacute;": "ó", "&uacute;": "ú", "&amp;": "&",
    "&lt;": "<", "&gt;": ">", "&quot;": '"',
}


def resolve_entities(text: str) -> str:
    for entity, char in ENTITY_MAP.items():
        text = text.replace(entity, char)
    try:
        text = html.unescape(text)
    except Exception:
        pass
    return text


def extract_paragraphs(path: Path) -> list[str]:
    try:
        raw = path.read_text(encoding="latin-1")
    except Exception as e:
        print(f"  [WARN] Could not read {path.name}: {e}")
        return []
    raw = resolve_entities(raw)
    paragraphs = []
    for match in TEXT_TAGS.finditer(raw):
        start = match.end()
        close = re.search(r'</\w+>', raw[start:])
        if not close:
            continue
        inner   = raw[start : start + close.start()]
        cleaned = TAG_STRIP.sub(" ", inner)
        cleaned = re.sub(r'\s+', " ", cleaned).strip()
        if len(cleaned) >= MIN_PARA_LEN:
            paragraphs.append(cleaned)
    return paragraphs


def compute_offsets(paragraphs: list[str]) -> list[int]:
    offsets, pos = [], 0
    for p in paragraphs:
        offsets.append(pos)
        pos += len(p) + 2
    return offsets


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sgml_files = sorted(INPUT_DIR.glob("*.sgml"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not sgml_files:
        raise FileNotFoundError(f"No .sgml files found in {INPUT_DIR}")
    print(f"Found {len(sgml_files)} .sgml files, sorted by most recently modified.")

    accumulated = []
    used_files  = []
    for sgml_file in sgml_files:
        paragraphs = extract_paragraphs(sgml_file)
        if not paragraphs:
            print(f"  [SKIP] {sgml_file.name} — no usable paragraphs")
            continue
        offsets = compute_offsets(paragraphs)
        records = [
            {
                "doc_id":    sgml_file.stem,
                "para_id":   i,
                "offset_eu": offsets[i],
                "source_eu": paragraphs[i],
            }
            for i in range(len(paragraphs))
        ]
        accumulated.extend(records)
        used_files.append(sgml_file.name)
        print(f"  {sgml_file.name}: {len(paragraphs):,} paragraphs  |  cumulative: {len(accumulated):,}")
        if len(accumulated) >= SAMPLE_SIZE:
            break

    print(f"\nFiles used: {used_files}")
    print(f"Total paragraphs accumulated: {len(accumulated):,}")

    if len(accumulated) > SAMPLE_SIZE:
        import random
        random.seed(42)
        accumulated = random.sample(accumulated, SAMPLE_SIZE)
        accumulated.sort(key=lambda r: (r["doc_id"], r["para_id"]))
        print(f"Downsampled to {SAMPLE_SIZE:,} paragraphs.")

    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in accumulated:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. {len(accumulated):,} paragraphs ready for translation -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()