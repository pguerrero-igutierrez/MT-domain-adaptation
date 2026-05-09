"""
evaluate_all.py

Unified evaluation script for General, Literary, and Clinical translation models.
Evaluates fine-tuned LoRA models and baselines on their respective test sets.

Usage:
    python evaluate_all.py --task general --models HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1 --test-file outputs/test_set_general.json
    python evaluate_all.py --task literary --models outputs/literaryv2 --test-file outputs/test_set_literary.json
    python evaluate_all.py --task clinical --models outputs/clinicalv1 --test-file outputs/test_set_clinical.json
"""

import argparse
import gc
import json
import random
import time
import logging
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF, TER
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
from comet import download_model as download_comet
from comet import load_from_checkpoint as load_comet

DATA_PATHS = {
    "general": Path("sampled-data/ca_eu_50k.json"),
    "literary_ca": Path("backtranslated-corpus/ca-literary_trilingual.json"),
    "literary_eu": Path("backtranslated-corpus/eu-literary-EhuHac.jsonl"),
    "clinical": Path("backtranslated-corpus/eu-clinical_backtranslated.json"),
}
OUTPUT_DIR = Path("outputs/eval")

SEED = 42
TRAIN_SPLIT = 0.90
VALID_SPLIT = 0.05
MIN_SRC_CHARS = 20
MIN_TGT_CHARS = 20
MAX_LEN_RATIO = 3.0

INSTRUCTION = {
    "eu2ca": "Itzuli testu hau euskaratik katalanera:\n\n{source}",
    "ca2eu": "Tradueix aquest text del català al basc:\n\n{source}",
}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_json_data(path: Path, direction: str, src_key: str, tgt_key: str, apply_ratio_filter: bool) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    samples = []
    for r in rows:
        src = (r.get(src_key) or "").strip()
        tgt = (r.get(tgt_key) or "").strip()
        if len(src) < MIN_SRC_CHARS or len(tgt) < MIN_TGT_CHARS:
            continue
        if apply_ratio_filter:
            ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
            if ratio > MAX_LEN_RATIO:
                continue
        samples.append({"source": src, "target": tgt, "direction": direction})
    return samples

def load_jsonl_data(path: Path, direction: str, src_key: str, tgt_key: str, apply_ratio_filter: bool) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = (r.get(src_key) or "").strip()
            tgt = (r.get(tgt_key) or "").strip()
            if len(src) < MIN_SRC_CHARS or len(tgt) < MIN_TGT_CHARS:
                continue
            if apply_ratio_filter:
                ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
                if ratio > MAX_LEN_RATIO:
                    continue
            samples.append({"source": src, "target": tgt, "direction": direction})
    return samples

def reconstruct_test_set(task: str, domain_weight: int) -> list[dict]:
    set_seed(SEED)
    
    if task == "general":
        with open(DATA_PATHS["general"], encoding="utf-8") as f:
            rows = json.load(f)
        samples = []
        for r in rows:
            ca = (r.get("ca") or "").strip()
            eu = (r.get("eu") or "").strip()
            if len(ca) >= MIN_SRC_CHARS and len(eu) >= MIN_TGT_CHARS:
                samples.append({"source": eu, "target": ca, "direction": "eu2ca"})
                samples.append({"source": ca, "target": eu, "direction": "ca2eu"})
    elif task == "literary":
        ca_samples = load_json_data(DATA_PATHS["literary_ca"], "eu2ca", "text_eu", "text_ca", True)
        eu_samples = load_jsonl_data(DATA_PATHS["literary_eu"], "ca2eu", "ca_translation", "source_eu", True)
        samples = (ca_samples + eu_samples) * domain_weight
    elif task == "clinical":
        samples = load_json_data(DATA_PATHS["clinical"], "ca2eu", "ca", "eu", True) * domain_weight

    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * TRAIN_SPLIT)
    valid_end = train_end + int(n * VALID_SPLIT)
    
    return samples[valid_end:]

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
    if peft_config_path.exists():
        with open(peft_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", model_path)
        print(f"  Detected PEFT adapter. Base model: {base_model_name}")
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print("  Loading as full model checkpoint.")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer

def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=768
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
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [d.strip() for d in decoded]

def compute_metrics(sources: list[str], hypotheses: list[str], references: list[str], comet_evaluator) -> dict:
    chrf = CHRF(word_order=2)
    bleu = BLEU(effective_order=True)
    ter = TER()
    
    comet_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    comet_score = comet_evaluator.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0).system_score

    return {
        "chrF++": round(chrf.corpus_score(hypotheses, [references]).score, 2),
        "BLEU": round(bleu.corpus_score(hypotheses, [references]).score, 2),
        "TER": round(ter.corpus_score(hypotheses, [references]).score, 2),
        "COMET": round(comet_score * 100, 2),
        "length_ratio": round(np.mean([len(h.split()) for h in hypotheses]) / max(np.mean([len(r.split()) for r in references]), 1), 3),
        "n_samples": len(hypotheses),
    }

def print_table(results: dict) -> None:
    if not results:
        return
    all_dirs = [k for k in results if k != "overall"] + (["overall"] if "overall" in results else [])
    col_w = 10
    header = f"{'Direction':<12}" + "".join(f"{m:>{col_w}}" for m in ["chrF++", "BLEU", "TER", "COMET", "LenRatio", "N"])
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for d in all_dirs:
        if d not in results: continue
        m = results[d]
        if d == "overall": print("-" * len(header))
        print(f"{d:<12}{m.get('chrF++', 0.0):>{col_w}.2f}{m.get('BLEU', 0.0):>{col_w}.2f}{m.get('TER', 0.0):>{col_w}.2f}{m.get('COMET', 0.0):>{col_w}.2f}{m.get('length_ratio', 0.0):>{col_w}.3f}{m.get('n_samples', 0):>{col_w}}")
    print("=" * len(header))

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Evaluator for MT Models")
    parser.add_argument("--task",            required=True, choices=["general", "literary", "clinical"])
    parser.add_argument("--models",          nargs="+", required=True, help="List of model paths or HF base models")
    parser.add_argument("--test-file",       default=None, help="Pre-saved JSON test set.")
    parser.add_argument("--domain-weight",   type=int, default=1, help="Must match domain weight used in training")
    parser.add_argument("--directions",      nargs="+", default=None, help="Which directions to evaluate (overrides defaults)")
    parser.add_argument("--max-samples",     type=int, default=None, help="Cap samples per direction")
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--max-new-tokens",  type=int, default=1024)
    parser.add_argument("--no-4bit",         action="store_true")
    parser.add_argument("--seed",            type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading COMET evaluation model (wmt22-comet-da)...")
    comet_path = download_comet("Unbabel/wmt22-comet-da")
    comet_evaluator = load_comet(comet_path)
    comet_evaluator.eval()

    if args.directions is None:
        if args.task == "clinical":
            args.directions = ["ca2eu"]
        else:
            args.directions = ["eu2ca", "ca2eu"]

    if args.test_file:
        print(f"\nLoading test set from {args.test_file}...")
        with open(args.test_file, encoding="utf-8") as f:
            test_samples = json.load(f)
    else:
        print(f"\nReconstructing {args.task} test set (seed={args.seed})...")
        test_samples = reconstruct_test_set(args.task, args.domain_weight)

    test_samples = [s for s in test_samples if s["direction"] in args.directions]
    
    by_dir = {}
    for s in test_samples:
        by_dir.setdefault(s["direction"], []).append(s)

    if args.max_samples:
        for d in by_dir:
            by_dir[d] = random.sample(by_dir[d], min(args.max_samples, len(by_dir[d])))
        test_samples = [s for slist in by_dir.values() for s in slist]

    print("\nTest split summary:")
    for d, items in sorted(by_dir.items()):
        print(f"  {d}: {len(items):,} samples")

    use_4bit = not args.no_4bit and torch.cuda.is_available()

    for model_path in args.models:
        run_name = Path(model_path).name.replace("/", "_")
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_path}")
        print(f"{'='*70}")

        model, tokenizer = load_model_and_tokenizer(model_path, use_4bit)

        all_results, all_hypotheses, all_references, all_sources, per_sample_out = {}, [], [], [], []

        for direction in args.directions:
            samples = by_dir.get(direction, [])
            if not samples: continue

            print(f"\nEvaluating {direction} ({len(samples):,} samples)...")
            sources    = [s["source"] for s in samples]
            prompts    = [build_prompt(s) for s in samples]
            references = [s["target"] for s in samples]
            hypotheses = []

            backup_path = OUTPUT_DIR / f"{run_name}_{args.task}_{direction}_backup.json"
            t0 = time.time()
            elapsed = 0.0

            if backup_path.exists():
                print(f"  [INFO] Backup found at {backup_path}. Skipping inference...")
                with open(backup_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                
                for item in saved_data:
                    hypotheses.append(item["hypothesis"])
                    per_sample_out.append(item)
            else:
                print("  Generating translations...")
                for i in tqdm(range(0, len(prompts), args.batch_size), desc=direction):
                    batch_prompts = prompts[i : i + args.batch_size]
                    hypotheses.extend(generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens))
                elapsed = time.time() - t0

                direction_out = []
                for s, hyp, ref in zip(samples, hypotheses, references):
                    item = {"direction": direction, "source": s["source"], "reference": ref, "hypothesis": hyp}
                    direction_out.append(item)
                    per_sample_out.append(item)
                
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump(direction_out, f, ensure_ascii=False, indent=2)
                print(f"  [Backup] Inference completed and saved to {backup_path}")

            print("  Calculating metrics (COMET, BLEU, etc.)...")
            try:
                metrics = compute_metrics(sources, hypotheses, references, comet_evaluator)
                if elapsed > 0:
                    metrics["seconds"] = round(elapsed, 1)
                all_results[direction] = metrics
                print(f"  chrF++={metrics['chrF++']:.2f}  BLEU={metrics['BLEU']:.2f}  TER={metrics['TER']:.2f}  COMET={metrics['COMET']:.2f}")
            except Exception as e:
                print(f"  [ERROR] Failed to calculate metrics for {direction}: {e}")
                print("  Translations are safely stored in the backup file.")
                continue

            all_sources.extend(sources)
            all_hypotheses.extend(hypotheses)
            all_references.extend(references)

        if len(args.directions) > 1 and all_hypotheses:
            try:
                print("\nCalculating overall metrics...")
                all_results["overall"] = compute_metrics(all_sources, all_hypotheses, all_references, comet_evaluator)
            except Exception as e:
                print(f"  [ERROR] Failed to calculate overall metrics: {e}")

        print_table(all_results)

        out_path = OUTPUT_DIR / f"{run_name}_{args.task}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_path, "run_name": run_name, "task": args.task, "metrics": all_results, "per_sample": per_sample_out}, f, ensure_ascii=False, indent=2)
        print(f"\nFinal results saved -> {out_path}")
        
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()