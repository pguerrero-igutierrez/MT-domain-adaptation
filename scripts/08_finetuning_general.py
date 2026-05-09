"""
08_finetuning_general.py

Fine-tunes HiTZ/Latxa-Qwen3-VL-8B-Instruct with LoRA on general-domain
Catalan–Basque parallel data from AINA for bidirectional translation in a
single model using explicit direction instructions per training example.

Data source
-----------
sampled-data/ca_eu_50k.json
    Fields expected: "ca" (Catalan), "eu" (Basque)
    direction eu2ca: source = eu  →  target = ca
    direction ca2eu: source = ca  →  target = eu
    Both directions are generated from every pair to maximise data usage.

Instruction templates
---------------------
eu2ca: "Itzuli testu hau euskaratik katalanera:\n\n{source}"
ca2eu: "Tradueix aquest text del català al basc:\n\n{source}"

Split
-----
90 % train / 10 % eval+test.
The eval set is saved to outputs/test_set_general.json after training so
downstream scripts (clinicalv2, literaryv2) can use it as a shared test set.

Output
------
outputs/generalv1/              – LoRA adapters + tokenizer
outputs/test_set_general.json   – held-out 10 % test set

Usage
-----
    python 08_finetuning_general.py
    python 08_finetuning_general.py --no-4bit
    python 08_finetuning_general.py --epochs 5 --lr 2e-4
    python 08_finetuning_general.py --max-train-samples 10000
    python 08_finetuning_general.py --output-dir outputs/my_general_run
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_MODEL    = "HiTZ/Latxa-Qwen3-VL-8B-Instruct"
INPUT_JSON    = Path("sampled-data/ca_eu_50k.json")
OUTPUT_DIR    = Path("outputs/generalv1")

SEED              = 42
MAX_LENGTH        = 768
TRAIN_SPLIT       = 0.90
VALID_SPLIT       = 0.05
MAX_TRAIN_SAMPLES = None

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_prompt(sample: dict) -> str:
    return INSTRUCTION[sample["direction"]].format(source=sample["source"])


def print_stats(samples: list[dict], label: str) -> None:
    by_dir = {}
    for s in samples:
        by_dir.setdefault(s["direction"], []).append(s)
    print(f"\n{label}: {len(samples):,} total")
    for d, items in sorted(by_dir.items()):
        src_lens = [len(i["source"]) for i in items]
        tgt_lens = [len(i["target"]) for i in items]
        print(
            f"  {d}: {len(items):,} samples | "
            f"src avg={int(np.mean(src_lens))} | "
            f"tgt avg={int(np.mean(tgt_lens))}"
        )


def print_examples(samples: list[dict], n: int = 3) -> None:
    print(f"\n--- {n} random examples ---")
    for s in random.sample(samples, min(n, len(samples))):
        print(f"  direction : {s['direction']}")
        print(f"  prompt    : {build_prompt(s)[:120]!r}")
        print(f"  target    : {s['target'][:80]!r}")
        print()


def tokenize_fn(examples, tokenizer, max_length):
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for prompt, target in zip(examples["prompt"], examples["target"]):
        full_text  = prompt + target + tokenizer.eos_token
        enc_full   = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        enc_prompt = tokenizer(prompt,    truncation=True, max_length=max_length, padding=False)
        prompt_len = len(enc_prompt["input_ids"])
        labels     = [-100] * prompt_len + enc_full["input_ids"][prompt_len:]
        input_ids_list.append(enc_full["input_ids"])
        attention_mask_list.append(enc_full["attention_mask"])
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}






def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tokenizer):
    chrf_metric = CHRF(word_order=2)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]
        result = chrf_metric.corpus_score(decoded_preds, decoded_labels)
        return {"chrf": result.score}

    return compute_metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",             default=BASE_MODEL)
    parser.add_argument("--output-dir",        default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=8)
    parser.add_argument("--lr",                type=float, default=5e-5)
    parser.add_argument("--max-length",        type=int,   default=MAX_LENGTH)
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--lora-drop",         type=float, default=0.05)
    parser.add_argument("--no-4bit",           action="store_true")
    parser.add_argument("--seed",              type=int,   default=SEED)
    parser.add_argument("--max-train-samples", type=int,   default=MAX_TRAIN_SAMPLES,
                        help="Cap training samples after shuffle (None = use all)")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    all_samples = load_data(INPUT_JSON)
    print(f"  Raw pairs loaded: {len(all_samples) // 2:,}  →  {len(all_samples):,} samples (both directions)")

    random.shuffle(all_samples)

    n = len(all_samples)
    train_end = int(n * TRAIN_SPLIT)
    valid_end = train_end + int(n * VALID_SPLIT)

    train_samples = all_samples[:train_end]
    valid_samples = all_samples[train_end:valid_end]
    test_samples  = all_samples[valid_end:]

    if args.max_train_samples is not None:
        train_samples = train_samples[:args.max_train_samples]
        print(f"  Training capped at {len(train_samples):,} samples (--max-train-samples)")

    print_stats(train_samples, "TRAIN")
    print_stats(valid_samples, "VALID")
    print_stats(test_samples,  "TEST")
    print_examples(train_samples)

    for s in train_samples:
        s["prompt"] = build_prompt(s)
    for s in valid_samples:
        s["prompt"] = build_prompt(s)
    for s in test_samples:
        s["prompt"] = build_prompt(s)

    train_ds = Dataset.from_list(train_samples)
    eval_ds  = Dataset.from_list(valid_samples)

    use_4bit = not args.no_4bit and torch.cuda.is_available()
    print(f"\nLoading model: {args.model}  (4-bit={use_4bit})")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()

    tok_kwargs = dict(tokenizer=tokenizer, max_length=args.max_length)
    train_ds = train_ds.map(
        lambda ex: tokenize_fn(ex, **tok_kwargs),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: tokenize_fn(ex, **tok_kwargs),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    bf16_avail = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=bf16_avail,
        fp16=not bf16_avail and torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=16,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving adapters to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_set_path = Path("outputs/test_set_general.json")
    test_set_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_set_path.exists():
        with open(test_set_path, "w", encoding="utf-8") as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        print(f"Test set saved -> {test_set_path}")
    else:
        print(f"Test set already exists, not overwriting -> {test_set_path}")

    print("Done.")


if __name__ == "__main__":
    main()