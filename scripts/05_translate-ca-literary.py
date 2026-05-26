"""
05_translate-ca-literary.py

Translates literary Spanish paragraphs from CTILC to Basque (ES->EU), then
realigns the translated text to the original Catalan paragraph structure with
an in-script DP alignment procedure based on `multilingual-e5-large`
embeddings.

Model:
    es->eu : HiTZ/Latxa-Llama-3.1-8B-Instruct  (vLLM offline batching)

Reads:  sampled-data/corpus_ca_es_100k_lit.json
Output: backtranslated-corpus/ca-literary_trilingual.json
        Fields: all original fields + text_eu
        para_id, offset_eu and aligned_paragraphs are recomputed after alignment.

Usage:
    python scripts/05_translate-ca-literary.py
    python scripts/05_translate-ca-literary.py --resume
    python scripts/05_translate-ca-literary.py --batch-size 256 --max-tokens 512
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(sys.maxsize)

CORPUS_DIR   = Path(__file__).parent.parent / "sampled-data"
INPUT_JSON   = CORPUS_DIR / "corpus_ca_es_100k_lit.json"
OUTPUT_JSON  = Path("backtranslated-corpus/ca-literary_trilingual.json")
CACHE_DIR    = CORPUS_DIR / ".translation_cache"

LATXA_MODEL   = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
EMBED_MODEL   = "intfloat/multilingual-e5-large"

LATXA_SYSTEM    = "Itzultzaile profesional bat zara. Emandako testua euskarara itzuli behar duzu, jatorrizko esanahia eta tonua mantenduz."
LATXA_USER_TMPL = "Itzuli testu hau euskarara:\n\n{text}"

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading multilingual-e5-large...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def embed(texts: list[str], prefix: str = "passage") -> np.ndarray:
    model = get_embed_model()
    prefixed = [f"{prefix}: {t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def make_overlap_embeddings(vecs: np.ndarray, max_merge: int = 2) -> dict[tuple, np.ndarray]:
    n = len(vecs)
    blocks = {}
    for i in range(n):
        for length in range(1, max_merge + 1):
            if i + length > n:
                break
            mean_vec = vecs[i : i + length].mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            blocks[(i, length)] = mean_vec
    return blocks


def vecalign_dp(
    paras_src: list[str],
    paras_tgt: list[str],
    max_merge: int = 2,
    gap_penalty: float = 0.2,
) -> list[tuple[str, str]]:
    if not paras_src or not paras_tgt:
        return []

    vecs_src   = embed(paras_src)
    vecs_tgt   = embed(paras_tgt)
    blocks_src = make_overlap_embeddings(vecs_src, max_merge)
    blocks_tgt = make_overlap_embeddings(vecs_tgt, max_merge)

    n, m  = len(paras_src), len(paras_tgt)
    NEG_INF = -1e9
    dp    = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    back  = {}
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == NEG_INF:
                continue
            for lsrc in range(1, max_merge + 1):
                for ltgt in range(1, max_merge + 1):
                    ni, nj = i + lsrc, j + ltgt
                    if ni > n or nj > m:
                        continue
                    sim     = cosine_sim(blocks_src[(i, lsrc)], blocks_tgt[(j, ltgt)])
                    penalty = gap_penalty * (lsrc + ltgt - 2)
                    score   = dp[i][j] + sim - penalty
                    if score > dp[ni][nj]:
                        dp[ni][nj] = score
                        back[(ni, nj)] = (i, j, lsrc, ltgt)

    path = []
    i, j = n, m
    while (i, j) != (0, 0):
        if (i, j) not in back:
            break
        pi, pj, lsrc, ltgt = back[(i, j)]
        path.append((pi, lsrc, pj, ltgt))
        i, j = pi, pj
    path.reverse()

    pairs = []
    for si, lsrc, sj, ltgt in path:
        src_text = " ".join(paras_src[si : si + lsrc]).strip()
        tgt_text = " ".join(paras_tgt[sj : sj + ltgt]).strip()
        if src_text and tgt_text:
            pairs.append((src_text, tgt_text))
    return pairs


def align_to_source(
    source_paras: list[str],
    translated_paras: list[str],
) -> list[str]:
    if len(source_paras) == len(translated_paras):
        return translated_paras

    print(f"    Realigning: src={len(source_paras)} tgt={len(translated_paras)}")
    pairs = vecalign_dp(source_paras, translated_paras)

    aligned = {p[0]: p[1] for p in pairs}
    result  = []
    for src in source_paras:
        result.append(aligned.get(src, ""))
    return result


def compute_offsets(paragraphs: list[str]) -> list[int]:
    offsets, pos = [], 0
    for p in paragraphs:
        offsets.append(pos)
        pos += len(p) + 2
    return offsets


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


def group_by_doc(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["doc_id"]].append(row)
    for doc_id in groups:
        groups[doc_id].sort(key=lambda r: int(r.get("para_id", 0)))
    return groups


def load_cache(lang: str) -> dict[str, str]:
    path = CACHE_DIR / f"{lang}.csv"
    if not path.exists():
        return {}
    cache = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache[row["text_es"]] = row["translation"]
    print(f"  Cache loaded: {len(cache)} entries for {lang}")
    return cache


def save_cache(lang: str, cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{lang}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_es", "translation"])
        writer.writeheader()
        for text_es, translation in cache.items():
            writer.writerow({"text_es": text_es, "translation": translation})


def translate_eu_batch(
    texts: list[str],
    batch_size: int,
    max_tokens: int,
) -> list[str]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LATXA_MODEL)

    def make_prompt(text: str) -> str:
        messages = [
            {"role": "system", "content": LATXA_SYSTEM},
            {"role": "user",   "content": LATXA_USER_TMPL.format(text=text)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    llm = LLM(
        model=LATXA_MODEL,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    prompts = [make_prompt(t) for t in texts]
    print(f"  EU: running vLLM on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


def run_translation(
    unique_texts: list[str],
    cache: dict[str, str],
    batch_size: int,
    max_tokens: int,
) -> dict[str, str]:
    pending = [t for t in unique_texts if t not in cache]
    print(f"EU: {len(unique_texts)} unique paragraphs, {len(pending)} not cached")

    if pending:
        translated = translate_eu_batch(pending, batch_size, max_tokens)
        for text, translation in zip(pending, translated):
            cache[text] = translation
        save_cache("eu", cache)

    return cache


def build_output(
    rows: list[dict],
    groups: dict[str, list[dict]],
    eu_cache: dict[str, str],
) -> list[dict]:
    out_rows = []

    for doc_id, doc_rows in groups.items():
        source_paras = [r["text_es"] for r in doc_rows]

        eu_raw = [eu_cache.get(t, "") for t in source_paras]
        eu_aligned = align_to_source(source_paras, [t for t in eu_raw if t])

        while len(eu_aligned) < len(source_paras):
            eu_aligned.append("")

        aligned_n  = len(source_paras)
        offsets_eu = compute_offsets(eu_aligned)

        for i, row in enumerate(doc_rows):
            out_row = dict(row)
            out_row["para_id"]            = i
            out_row["aligned_paragraphs"] = aligned_n
            out_row["text_eu"]            = eu_aligned[i]
            out_row["offset_eu"]          = offsets_eu[i]
            out_rows.append(out_row)

        print(f"  [ALIGNED] {doc_id}: {aligned_n} paragraphs")

    return out_rows


def write_output(out_rows: list[dict]) -> None:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"\nOutput -> {OUTPUT_JSON}")
    print(f"  Rows: {len(out_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate ES corpus to EU, realigned to original paragraph structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Use cached translations from previous runs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    rows = load_rows(INPUT_JSON)
    groups = group_by_doc(rows)
    unique_es = list({r["text_es"] for r in rows if r.get("text_es")})

    eu_cache = load_cache("eu") if args.resume else {}
    eu_cache = run_translation(unique_es, eu_cache, args.batch_size, args.max_tokens)

    print("\nRealigning translations to original paragraph structure...")
    out_rows = build_output(rows, groups, eu_cache)

    write_output(out_rows)
    print("Done.")


if __name__ == "__main__":
    main()
