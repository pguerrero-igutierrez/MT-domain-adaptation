"""
08_finetuning_general.py

finetune latxa HiTZ/Latxa-Llama-3.1-8B-Instruct on general-domain parallel data from AINA
located at sampled-data/ca_eu_50k.json (catalan-euskera parallel data)
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG
# =========================
MODEL_NAME = "HiTZ/Latxa-Qwen3-8B-Instruct""
DATA_PATH = "sampled-data/ca_eu_50k.json"
OUTPUT_DIR = "outputs/latxa-ca-eu-bidirectional"

MAX_LENGTH = 768

# =========================
# LOAD DATA
# =========================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

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

expanded_data = []
for ex in raw_data:
    expanded_data.extend(format_bidirectional(ex))

dataset = Dataset.from_list(expanded_data)

# Optional split
dataset = dataset.train_test_split(test_size=0.05)

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# TOKENIZATION (completion-only loss)
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
    labels[:prompt_len] = [-100] * prompt_len  # mask prompt

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# =========================
# MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False
)

# =========================
# LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # switch to bf16=True if supported
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
