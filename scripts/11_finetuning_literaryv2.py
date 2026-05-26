"""
11_finetuning_literaryv2.py

Continues fine-tuning from a general-purpose Basque-Catalan translation
checkpoint using the same literary back-translation corpora as
`09_finetuning_literaryv1.py`.
Evaluates using BLEU score during training.

Base model
----------
By default the script expects `--model` to point to a merged/full general
checkpoint. If `--is-peft` is used, `--model` must point to a local PEFT
adapter directory so `adapter_config.json` can be read before merging.

Data sources
------------
Literary eu2ca : backtranslated-corpus/ca-literary_trilingual.json
    source = text_eu  →  target = text_ca

Literary ca2eu : backtranslated-corpus/eu-literary-EhuHac.jsonl
    source = ca_translation  →  target = source_eu

Direction instructions (same as v1)
------------------------------------
eu2ca: "Itzuli testu literario hau euskaratik katalanera:\n\n{source}"
ca2eu: "Tradueix aquest text literari del català al basc:\n\n{source}"

Strategy
--------
- Load the HF baseline with its existing LoRA adapters (PEFT) or as a
  plain causal-LM checkpoint and apply fresh LoRA on top.
- Reuse the same literary training data as literaryv1 so that the main
  experimental difference is the starting checkpoint rather than the
  corpus composition.
- Cosine LR schedule with a short warmup continuing from a low LR so we
  do not overwrite the general weights.

Split
-----
90 % train / 5 % valid / 5 % test

Output
------
outputs/literaryv2/              – LoRA adapters + tokenizer
outputs/test_set_literary.json   – held-out 5 % test set

Usage
-----
    python scripts/11_finetuning_literaryv2.py --model outputs/generalv1 --is-peft
    python scripts/11_finetuning_literaryv2.py --model your-org/merged-general-checkpoint
    python scripts/11_finetuning_literaryv2.py --model your-org/merged-general-checkpoint --no-4bit --epochs 2
"""

import argparse
import json
import random
from pathlib import Path

from sacrebleu.metrics import BLEU
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = "your-org/latxa-qwen3-8b-general-eucat"

CA_JSON       = Path("backtranslated-corpus/ca-literary_trilingual.json")
EU_JSONL      = Path("backtranslated-corpus/eu-literary-EhuHac.jsonl")
OUTPUT_DIR    = Path("outputs/literaryv2")

SEED              = 42
MAX_LENGTH        = 768
TRAIN_SPLIT       = 0.90
VALID_SPLIT       = 0.05
MAX_TRAIN_SAMPLES = None

MIN_SRC_CHARS = 20
MIN_TGT_CHARS = 20
MAX_LEN_RATIO = 3.0

INSTRUCTION = {
    "eu2ca": "Itzuli testu literario hau euskaratik katalanera:\n\n{source}",
    "ca2eu": "Tradueix aquest text literari del català al basc:\n\n{source}",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ca_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    samples = []
    for r in rows:
        src = (r.get("text_eu") or "").strip()
        tgt = (r.get("text_ca") or "").strip()
        if len(src) < MIN_SRC_CHARS or len(tgt) < MIN_TGT_CHARS:
            continue
        ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
        if ratio > MAX_LEN_RATIO:
            continue
        samples.append({"source": src, "target": tgt, "direction": "eu2ca"})
    return samples


def load_eu_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = (r.get("ca_translation") or "").strip()
            tgt = (r.get("source_eu") or "").strip()
            if len(src) < MIN_SRC_CHARS or len(tgt) < MIN_TGT_CHARS:
                continue
            ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
            if ratio > MAX_LEN_RATIO:
                continue
            samples.append({"source": src, "target": tgt, "direction": "ca2eu"})
    return samples


def build_prompt(sample: dict) -> str:
    return INSTRUCTION[sample["direction"]].format(source=sample["source"])


def print_stats(samples: list[dict], label: str) -> None:
    by_dir = {}
    for s in samples:
        by_dir.setdefault(s["direction"], []).append(s)
    print(f"\n{label}: {len(samples):,} total")
    for d, items in sorted(by_dir.items()):
        print(
            f"  {d}: {len(items):,} | "
            f"src_avg={int(np.mean([len(i['source']) for i in items]))} | "
            f"tgt_avg={int(np.mean([len(i['target']) for i in items]))}"
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
        enc_full   = tokenizer(full_text,  truncation=True, max_length=max_length, padding=False)
        enc_prompt = tokenizer(prompt,     truncation=True, max_length=max_length, padding=False)
        prompt_len = len(enc_prompt["input_ids"])
        labels     = [-100] * prompt_len + enc_full["input_ids"][prompt_len:]
        input_ids_list.append(enc_full["input_ids"])
        attention_mask_list.append(enc_full["attention_mask"])
        labels_list.append(labels)
    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}


def oversample(samples: list[dict], factor: int) -> list[dict]:
    return samples * factor


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tokenizer):
    bleu_metric = BLEU()

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True
        )

        decoded_labels = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = bleu_metric.corpus_score(
            decoded_preds,
            decoded_labels
        )

        return {"bleu": result.score}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default=DEFAULT_BASE_MODEL,
                        help="HF repo of the general fine-tuned checkpoint")
    parser.add_argument("--is-peft",          action="store_true",
                        help="Set if the HF repo is a PEFT/LoRA checkpoint (not merged)")
    parser.add_argument("--output-dir",       default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=4)
    parser.add_argument("--grad-accum",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=5e-5)
    parser.add_argument("--max-length",       type=int,   default=MAX_LENGTH)
    parser.add_argument("--lora-r",           type=int,   default=16)
    parser.add_argument("--lora-alpha",       type=int,   default=32)
    parser.add_argument("--lora-drop",        type=float, default=0.05)
    parser.add_argument("--no-4bit",          action="store_true")
    parser.add_argument("--seed",             type=int,   default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=MAX_TRAIN_SAMPLES,
                        help="Cap training samples after shuffle (None = use all)")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ca_samples      = load_ca_json(CA_JSON)
    eu_samples      = load_eu_jsonl(EU_JSONL)
    all_samples = ca_samples + eu_samples

    print(f"  ca-literary (eu2ca)  : {len(ca_samples):,}")
    print(f"  eu-literary (ca2eu)  : {len(eu_samples):,}")

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
    print(f"\nLoading base model from HF: {args.model}  (4-bit={use_4bit})")

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

    if args.is_peft:
        with open(Path(args.model) / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path")
        print(f"Loading base model: {base_model_name}")
        
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base, args.model)
        model = model.merge_and_unload()
    else:
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)],
    )

    print("\nStarting continued literary fine-tuning...")
    trainer.train()

    print(f"\nSaving adapters to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_set_path = Path("outputs/test_set_literary.json")
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
