"""
09.2_finetuning_literaryv1_tokenmatched.py

Fine-tunes `HiTZ/Latxa-Qwen3-VL-8B-Instruct` with LoRA for Catalan->Basque
literary translation (`ca2eu` only) using a token-budget-matched subset of the
literary corpus. The train and validation splits are matched to the clinical
domain's source-token budgets to support a fair literary-vs-clinical
comparison.

Data sources
------------
backtranslated-corpus/eu-literary-EhuHac.jsonl
    source = `ca_translation`  (Catalan)
    target = `source_eu`       (Basque)

backtranslated-corpus/eu-clinical_backtranslated.json
    used only to derive the clinical reference token budgets

Split
-----
1. Shuffle the clinical corpus and take fixed-size reference splits
   (`1174` train, `65` valid, remainder test).
2. Compute source-token budgets for the clinical train and valid splits.
3. Consume shuffled literary examples until those budgets are matched.
4. Use the remaining literary examples as test data.

Output
------
outputs/literaryv1_tokenmatched/   – LoRA adapters + tokenizer
outputs/test_set_literary.json     – held-out literary test set, saved only if absent

Usage
-----
    python scripts/09.2_finetuning_literaryv1_tokenmatched.py
    python scripts/09.2_finetuning_literaryv1_tokenmatched.py --no-4bit
    python scripts/09.2_finetuning_literaryv1_tokenmatched.py --output-dir outputs/my_tokenmatched_run
"""

import argparse
import json
import random
from pathlib import Path

from sacrebleu.metrics import BLEU
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    #EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL    = "HiTZ/Latxa-Qwen3-VL-8B-Instruct"
LITERARY_JSONL= Path("backtranslated-corpus/eu-literary-EhuHac.jsonl")
CLINICAL_JSON = Path("backtranslated-corpus/eu-clinical_backtranslated.json")
OUTPUT_DIR    = Path("outputs/literaryv1_tokenmatched")

SEED          = 42
MAX_LENGTH    = 768 
INSTRUCTION   = {"ca2eu": "Tradueix aquest text del català al basc:\n\n{source}"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_clinical_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for r in data:
        src = r.get("ca", "").strip()
        tgt = r.get("eu", "").strip()
        if src and tgt:
            samples.append({"source": src, "target": tgt})
    return samples

def load_literary_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            src = r.get("ca_translation", "").strip()
            tgt = r.get("source_eu", "").strip()
            if len(src) > 20 and len(tgt) > 20:
                samples.append({"source": src, "target": tgt, "direction": "ca2eu"})
    return samples

def get_token_stats(samples, tokenizer):
    if not samples: return 0, 0.0, 0.0
    counts = [len(tokenizer.encode(s["source"], add_special_tokens=False)) for s in samples]
    return sum(counts), np.mean(counts), np.std(counts)

def print_comparison(lit_splits, clin_splits, tokenizer):
    print(f"\n{'DOMAIN':<20} | {'SPLIT':<10} | {'SAMPLES':<10} | {'TOTAL TOKENS':<15} | {'AVG TOK':<8} | {'STD TOK':<8}")
    print("-" * 90)
    for base_name, suffix, d_data in [("Clinical", "(Ref)", clin_splits), ("Literary", "(Match)", lit_splits)]:
        for s_name in ["train", "valid", "test"]:
            samples = d_data.get(s_name, [])
            if not samples: continue
            display_name = f"{base_name} {suffix}" if s_name in ["train", "valid"] else base_name
            total, avg, std = get_token_stats(samples, tokenizer)
            print(f"{display_name:<20} | {s_name:<10} | {len(samples):<10,} | {total:<15,} | {avg:<8.1f} | {std:<8.1f}")
        print("-" * 90)

def tokenize_fn(examples, tokenizer, max_length):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    
    for prompt, target in zip(examples["prompt"], examples["target"]):
        # 1. We keep a space to be safe, but we will find the exact index
        full_text = prompt + " " + target + tokenizer.eos_token
        
        enc_full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        enc_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
        
        full_ids = enc_full["input_ids"]
        prompt_ids = enc_prompt["input_ids"]
        
        # 2. Find the junction
        # We look for the first index where the full_ids differ from prompt_ids
        # or where the prompt ends.
        junction_idx = 0
        min_len = min(len(full_ids), len(prompt_ids))
        while junction_idx < min_len and full_ids[junction_idx] == prompt_ids[junction_idx]:
            junction_idx += 1
            
        # 3. Create labels: Mask everything up to the junction
        # This ensures the model learns every token that isn't part of the pure prompt
        labels = [-100] * junction_idx + full_ids[junction_idx:]
        
        input_ids_list.append(full_ids)
        attention_mask_list.append(enc_full["attention_mask"])
        labels_list.append(labels)
        
    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def make_compute_metrics(tokenizer):
    bleu_metric = BLEU()
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        result = bleu_metric.corpus_score(decoded_preds, decoded_labels)
        return {"bleu": result.score}
    return compute_metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=BASE_MODEL)
    parser.add_argument("--output-dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--max-length",  type=int,   default=MAX_LENGTH)
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--lora-drop",   type=float, default=0.05)
    parser.add_argument("--no-4bit",     action="store_true")
    parser.add_argument("--seed",        type=int,   default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and matching budgets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. Process Clinical Reference to get budget
    clinical_all = load_clinical_json(CLINICAL_JSON)
    random.shuffle(clinical_all)
    clin_train = clinical_all[:1174]
    clin_valid = clinical_all[1174:1174+65]
    clin_test  = clinical_all[1174+65:]
    clinical_splits = {"train": clin_train, "valid": clin_valid, "test": clin_test}
    
    train_budget, _, _ = get_token_stats(clin_train, tokenizer)
    valid_budget, _, _ = get_token_stats(clin_valid, tokenizer)

    # 2. Match Literary samples to that budget
    all_literary = load_literary_jsonl(LITERARY_JSONL)
    random.shuffle(all_literary)

    def take_by_budget(samples, budget):
        taken, current = [], 0
        for s in samples:
            toks = len(tokenizer.encode(s["source"], add_special_tokens=False))
            current += toks
            taken.append(s)
            if current >= budget: break
        return taken

    lit_train = take_by_budget(all_literary, train_budget)
    remainder = all_literary[len(lit_train):]
    lit_valid = take_by_budget(remainder, valid_budget)
    lit_test  = remainder[len(lit_valid):]

    literary_splits = {"train": lit_train, "valid": lit_valid, "test": lit_test}
    print_comparison(literary_splits, clinical_splits, tokenizer)

    for s in lit_train + lit_valid + lit_test:
        s["prompt"] = INSTRUCTION["ca2eu"].format(source=s["source"])

    train_ds = Dataset.from_list(lit_train)
    eval_ds  = Dataset.from_list(lit_valid)

    # 3. Model Prep
    use_4bit = not args.no_4bit and torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # 4. Tokenize Datasets
    tok_kwargs = dict(tokenizer=tokenizer, max_length=args.max_length)
    train_ds = train_ds.map(lambda ex: tokenize_fn(ex, **tok_kwargs), batched=True, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(lambda ex: tokenize_fn(ex, **tok_kwargs), batched=True, remove_columns=eval_ds.column_names)

    # --- NEW SANITY CHECK ---
    print("\n" + "="*50)
    print("LOGGING TOKENIZED SAMPLES FOR VERIFICATION")
    print("="*50)
    for i in range(min(2, len(train_ds))):
        sample = train_ds[i]
        print(f"\n[Sample {i}] FULL INPUT DECODED:")
        print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        
        # Decode only the tokens where label is not -100
        label_ids = [l for l in sample["labels"] if l != -100]
        print(f"\n[Sample {i}] TARGET TOKENS (LEARNED PART):")
        print(tokenizer.decode(label_ids, skip_special_tokens=False))
    print("="*50 + "\n")

    # 5. Training Arguments (Synchronized with clinical run)
    bf16_avail = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=bf16_avail,
        fp16=not bf16_avail and torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100),
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving adapters to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_set_path = Path("outputs/test_set_literary.json")
    if not test_set_path.exists():
        with open(test_set_path, "w", encoding="utf-8") as f:
            json.dump(lit_test, f, ensure_ascii=False, indent=2)
        print(f"Test set saved -> {test_set_path}")

    print("Done.")

if __name__ == "__main__":
    main()
