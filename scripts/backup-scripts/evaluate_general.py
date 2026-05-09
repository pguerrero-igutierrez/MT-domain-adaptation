"""
evaluate-general.py

Evaluates fine-tuned general translation models and baseline models
on the held-out 5 % test split used during training.

Pass --test-file to load a pre-saved JSON test set.

Usage:
    python evaluate-general.py --models HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_JSON   = Path("sampled-data/ca_eu_50k.json")
OUTPUT_DIR   = Path("outputs/eval")

SEED          = 42
TRAIN_SPLIT   = 0.90
VALID_SPLIT   = 0.05
MIN_SRC_CHARS = 20
MIN_TGT_CHARS = 20

INSTRUCTION = {
    "eu2ca": "Itzuli testu hau euskaratik katalanera:\n\n{source}",
    "ca2eu": "Tradueix aquest text del català al basc:\n\n{source}",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    samples = []
    for r in rows:
        ca = (r.get("ca") or "").strip()
        eu = (r.get("eu") or "").strip()
        if len(ca) >= MIN_SRC_CHARS and len(eu) >= MIN_TGT_CHARS:
            samples.append({"source": eu, "target": ca, "direction": "eu2ca"})
            samples.append({"source": ca, "target": eu, "direction": "ca2eu"})
    return samples


def reconstruct_test_set() -> list[dict]:
    set_seed(SEED)
    samples = load_data(INPUT_JSON)

    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * TRAIN_SPLIT)
    valid_end = train_end + int(n * VALID_SPLIT)
    test_samples = samples[valid_end:]
    
    return test_samples


def build_prompt(sample: dict) -> str:
    return INSTRUCTION[sample["direction"]].format(source=sample["source"])


def load_model_and_tokenizer(model_path: str, use_4bit: bool):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    peft_config_path = Path(model_path) / "adapter_config.json"
    is_peft = peft_config_path.exists()

    if is_peft:
        with open(peft_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", model_path)
        print(f"  Detected PEFT adapter. Base model: {base_model_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print("  Loading as full model checkpoint.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
    )
    return [d.strip() for d in decoded]


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    chrf  = CHRF(word_order=2)
    bleu  = BLEU(effective_order=True)
    ter   = TER()

    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    ter_score  = ter.corpus_score(hypotheses, [references]).score

    hyp_lens = [len(h.split()) for h in hypotheses]
    ref_lens = [len(r.split()) for r in references]
    length_ratio = np.mean(hyp_lens) / max(np.mean(ref_lens), 1)

    return {
        "chrF++":       round(chrf_score, 2),
        "BLEU":         round(bleu_score, 2),
        "TER":          round(ter_score, 2),
        "length_ratio": round(length_ratio, 3),
        "n_samples":    len(hypotheses),
    }


def print_table(results: dict) -> None:
    directions = [k for k in results if k != "overall"]
    all_dirs = directions + ["overall"]

    col_w = 12
    header = f"{'Direction':<12}" + "".join(f"{'chrF++':>{col_w}}{'BLEU':>{col_w}}{'TER':>{col_w}}{'LenRatio':>{col_w}}{'N':>{col_w}}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for d in all_dirs:
        if d not in results:
            continue
        m = results[d]
        row = (
            f"{d:<12}"
            f"{m['chrF++']:>{col_w}.2f}"
            f"{m['BLEU']:>{col_w}.2f}"
            f"{m['TER']:>{col_w}.2f}"
            f"{m['length_ratio']:>{col_w}.3f}"
            f"{m['n_samples']:>{col_w}}"
        )
        if d == "overall":
            print("-" * len(header))
        print(row)
    print("=" * len(header))


def print_samples(per_sample: list[dict], n: int = 5) -> None:
    print(f"\n--- {n} random predictions ---")
    for s in random.sample(per_sample, min(n, len(per_sample))):
        print(f"  [{s['direction']}]")
        print(f"  SRC : {s['source'][:100]!r}")
        print(f"  REF : {s['reference'][:100]!r}")
        print(f"  HYP : {s['hypothesis'][:100]!r}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",          nargs="+", required=True,
                        help="List of model paths or HF base models (e.g., HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1)")
    parser.add_argument("--test-file",       default=None,
                        help="Pre-saved JSON test set. If not given, reconstructs from data files.")
    parser.add_argument("--directions",      nargs="+", choices=["eu2ca", "ca2eu"],
                        default=["eu2ca", "ca2eu"],
                        help="Which directions to evaluate")
    parser.add_argument("--max-samples",     type=int, default=None,
                        help="Cap samples per direction (useful for quick smoke-test)")
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--max-new-tokens",  type=int, default=256)
    parser.add_argument("--no-4bit",         action="store_true")
    parser.add_argument("--seed",            type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test_file:
        print(f"Loading test set from {args.test_file}...")
        with open(args.test_file, encoding="utf-8") as f:
            test_samples = json.load(f)
        print(f"  {len(test_samples):,} samples loaded.")
    else:
        print("Reconstructing test set (seed=42, same split as training)...")
        test_samples = reconstruct_test_set()
        print(f"  {len(test_samples):,} test samples reconstructed.")

    test_samples = [s for s in test_samples if s["direction"] in args.directions]

    by_dir: dict[str, list[dict]] = {}
    for s in test_samples:
        by_dir.setdefault(s["direction"], []).append(s)

    if args.max_samples:
        for d in by_dir:
            by_dir[d] = random.sample(by_dir[d], min(args.max_samples, len(by_dir[d])))
        test_samples = [s for slist in by_dir.values() for s in slist]

    print(f"\nTest split summary:")
    for d, items in sorted(by_dir.items()):
        print(f"  {d}: {len(items):,} samples")

    use_4bit = not args.no_4bit and torch.cuda.is_available()

    # LOOP OVER MULTIPLE MODELS
    for model_path in args.models:
        run_name = Path(model_path).name.replace("/", "_")
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_path, use_4bit)

        all_results: dict[str, dict] = {}
        all_hypotheses: list[str] = []
        all_references: list[str] = []
        per_sample_out: list[dict] = []

        for direction in args.directions:
            samples = by_dir.get(direction, [])
            if not samples:
                print(f"\n  No samples for {direction}, skipping.")
                continue

            print(f"\nEvaluating {direction} ({len(samples):,} samples)...")
            prompts    = [build_prompt(s) for s in samples]
            references = [s["target"] for s in samples]
            hypotheses = []

            t0 = time.time()
            for i in tqdm(range(0, len(prompts), args.batch_size), desc=direction):
                batch_prompts = prompts[i : i + args.batch_size]
                batch_hyps    = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)
                hypotheses.extend(batch_hyps)
            elapsed = time.time() - t0

            metrics = compute_metrics(hypotheses, references)
            metrics["seconds"] = round(elapsed, 1)
            all_results[direction] = metrics

            for s, hyp, ref in zip(samples, hypotheses, references):
                per_sample_out.append({
                    "direction":  direction,
                    "source":     s["source"],
                    "reference":  ref,
                    "hypothesis": hyp,
                })

            all_hypotheses.extend(hypotheses)
            all_references.extend(references)

            print(f"  chrF++={metrics['chrF++']:.2f}  BLEU={metrics['BLEU']:.2f}  TER={metrics['TER']:.2f}  ({elapsed:.0f}s)")

        if len(args.directions) > 1 and all_hypotheses:
            all_results["overall"] = compute_metrics(all_hypotheses, all_references)

        print_table(all_results)
        print_samples(per_sample_out)

        out_path = OUTPUT_DIR / f"{run_name}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model":      model_path,
                    "run_name":   run_name,
                    "metrics":    all_results,
                    "per_sample": per_sample_out,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nResults saved -> {out_path}")

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()