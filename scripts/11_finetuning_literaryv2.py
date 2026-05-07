"""
11_finetuning_literaryv2.py

Continue fine-tuning from a general-purpose Basque–Catalan translation
baseline (08_finetuning_general.py output, hosted on HuggingFace Hub) by
mixing general parallel data with both literary backtranslation corpora.

Base model
----------
A LoRA-merged (or full) checkpoint uploaded to HF after the general
fine-tuning stage, e.g. "your-org/latxa-qwen3-8b-general-eucat".
Override with --model.

Data sources  (all three used simultaneously)
---------------------------------------------
General data: loaded via --general-jsonl  (same format as ehuhac_parallel)
    source = source_es / source_eu fields; treated as supplementary signal

Literary eu2ca : backtranslated-corpus/ca-literary_trilingual.json
    source = text_eu  →  target = text_ca

Literary ca2eu : backtranslated-corpus/eu-literary-trilingual.jsonl
    source = ca_translation  →  target = source_eu

Direction instructions (same as v1)
------------------------------------
eu2ca: "Itzuli testu hau euskaratik katalanera:\n\n{source}"
ca2eu: "Tradueix aquest text del català al basc:\n\n{source}"

The general data (eu↔es sentences) is included purely as regularisation;
it is labelled with its own direction tags so the model does not confuse
language pairs.

Strategy
--------
- Load the HF baseline with its existing LoRA adapters (PEFT) or as a
  plain causal-LM checkpoint and apply fresh LoRA on top.
- Use a higher literary sampling weight to bias the mix toward the target
  domain without forgetting general translation ability.
- Cosine LR schedule with a short warmup continuing from a low LR so we
  do not overwrite the general weights.

Output
------
outputs/literaryv2/   – LoRA adapters + tokenizer

Usage
-----
    python 11_finetuning_literaryv2.py --model your-org/latxa-qwen3-8b-general-eucat
    python 11_finetuning_literaryv2.py --model your-org/... --no-4bit --epochs 2
    python 11_finetuning_literaryv2.py --model your-org/... --literary-weight 3
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASE_MODEL = "your-org/latxa-qwen3-8b-general-eucat"

CA_JSON       = Path("backtranslated-corpus/ca-literary_trilingual.json")
EU_JSONL      = Path("backtranslated-corpus/eu-literary-trilingual.jsonl")
OUTPUT_DIR    = Path("outputs/literaryv2")

SEED              = 42
MAX_LENGTH        = 512
TRAIN_SPLIT       = 0.90
MAX_TRAIN_SAMPLES = None

MIN_SRC_CHARS = 20
MIN_TGT_CHARS = 20

INSTRUCTION = {
    "eu2ca": "Itzuli testu hau euskaratik katalanera:\n\n{source}",
    "ca2eu": "Tradueix aquest text del català al basc:\n\n{source}",
    "eu2es": "Itzuli testu hau euskaratik gaztelaniara:\n\n{source}",
    "es2eu": "Itzuli testu hau gaztelaniatik euskarara:\n\n{source}",
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
        if len(src) >= MIN_SRC_CHARS and len(tgt) >= MIN_TGT_CHARS:
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
            if len(src) >= MIN_SRC_CHARS and len(tgt) >= MIN_TGT_CHARS:
                samples.append({"source": src, "target": tgt, "direction": "ca2eu"})
    return samples


def load_general_jsonl(path: Path) -> list[dict]:
    if not path or not path.exists():
        return []
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            es = (r.get("source_es") or "").strip()
            eu = (r.get("source_eu") or "").strip()
            if len(es) >= MIN_SRC_CHARS and len(eu) >= MIN_TGT_CHARS:
                samples.append({"source": eu, "target": es, "direction": "eu2es"})
                samples.append({"source": es, "target": eu, "direction": "es2eu"})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            default=DEFAULT_BASE_MODEL,
                        help="HF repo of the general fine-tuned checkpoint")
    parser.add_argument("--is-peft",          action="store_true",
                        help="Set if the HF repo is a PEFT/LoRA checkpoint (not merged)")
    parser.add_argument("--general-jsonl",    default=None,
                        help="Path to general parallel data JSONL (ehuhac_parallel format)")
    parser.add_argument("--literary-weight",  type=int, default=2,
                        help="Repeat literary samples N times to up-weight domain")
    parser.add_argument("--output-dir",       default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch-size",       type=int,   default=4)
    parser.add_argument("--grad-accum",       type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=5e-5)
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
    general_path    = Path(args.general_jsonl) if args.general_jsonl else None
    general_samples = load_general_jsonl(general_path)

    literary = oversample(ca_samples + eu_samples, args.literary_weight)
    all_samples = literary + general_samples

    print(f"  ca-literary (eu2ca)  : {len(ca_samples):,}  × {args.literary_weight} = {len(ca_samples)*args.literary_weight:,}")
    print(f"  eu-literary (ca2eu)  : {len(eu_samples):,}  × {args.literary_weight} = {len(eu_samples)*args.literary_weight:,}")
    print(f"  general (eu↔es)      : {len(general_samples):,}")

    random.shuffle(all_samples)

    split = int(len(all_samples) * TRAIN_SPLIT)
    train_samples = all_samples[:split]
    eval_samples  = all_samples[split:]

    if args.max_train_samples is not None:
        train_samples = train_samples[:args.max_train_samples]
        print(f"  Training capped at {len(train_samples):,} samples (--max-train-samples)")

    print_stats(train_samples, "TRAIN")
    print_stats(eval_samples,  "EVAL")
    print_examples(train_samples)

    for s in train_samples:
        s["prompt"] = build_prompt(s)
    for s in eval_samples:
        s["prompt"] = build_prompt(s)

    train_ds = Dataset.from_list(train_samples)
    eval_ds  = Dataset.from_list(eval_samples)

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
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
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
            json.dump(eval_samples, f, ensure_ascii=False, indent=2)
        print(f"Test set saved -> {test_set_path}")
    else:
        print(f"Test set already exists, not overwriting -> {test_set_path}")

    print("Done.")


if __name__ == "__main__":
    main()