""" 12_finetuning_clinicalv2.py

Continue fine-tuning from 08_finetuning_general.py 
General data + clinical data (backtranslated-corpus/eu-clinical_backtranslated.json)

"""

"""
12_finetuning_clinicalv2.py

Continue fine-tuning from 08_finetuning_general.py 
General data + clinical data (backtranslated-corpus/eu-clinical_backtranslated.json)
"""

import json
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "HiTZ/Latxa-Qwen3-8B-Instruct""
CHECKPOINT_PATH = "outputs/latxa-ca-eu-bidirectional"  # from script 08

GENERAL_DATA_PATH = "sampled-data/ca_eu_50k.json"
CLINICAL_DATA_PATH = "backtranslated-corpus/eu-clinical_backtranslated.json"

OUTPUT_DIR = "outputs/latxa-ca-eu-clinical-v2"

MAX_LENGTH = 768

# =========================
# LOAD GENERAL DATA
# =========================
with open(GENERAL_DATA_PATH, "r", encoding="utf-8") as f:
    general_raw = json.load(f)

def format_bidirectional(example):
    ca = example["ca"]
    eu = example["eu"]

    return [
        {
            "text": f"<|user|>\ntranslate from catalan to basque:\n{ca}\n<|assistant|>\n{eu}"
        },
        {
            "text": f"<|user|>\ntranslate from basque to catalan:\n{eu}\n<|assistant|>\n{ca}"
        }
    ]

general_expanded = []
for ex in general_raw:
    general_expanded.extend(format_bidirectional(ex))

general_dataset = Dataset.from_list(general_expanded)

# =========================
# LOAD CLINICAL DATA
# =========================
with open(CLINICAL_DATA_PATH, "r", encoding="utf-8") as f:
    clinical_raw = json.load(f)

clinical_expanded = []
for ex in clinical_raw:
    clinical_expanded.extend(format_bidirectional(ex))  # assumes same keys (ca, eu)

clinical_dataset = Dataset.from_list(clinical_expanded)

# =========================
# MERGE DATASETS
# =========================
dataset = concatenate_datasets([general_dataset, clinical_dataset])
dataset = dataset.train_test_split(test_size=0.05)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZATION (same as script 08)
# =========================
def tokenize(example):
    full_text = example["text"]

    if "<|assistant|>" not in full_text:
        raise ValueError("Missing assistant tag")

    prompt, target = full_text.split("<|assistant|>")
    prompt += "<|assistant|>"

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# =========================
# LOAD MODEL + LORA
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False
)

# Load LoRA weights from previous training
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

model.print_trainable_parameters()

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,  # slightly fewer epochs for continued tuning
    learning_rate=1e-4,  # lower LR for stability
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    report_to="none",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=data_collator
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE
# =========================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
