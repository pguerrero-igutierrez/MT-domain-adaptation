# Scripts

This directory contains all pipeline scripts for data sampling, synthetic data generation, model fine-tuning, and evaluation. Scripts are numbered to reflect execution order.

---

## Overview

| Stage | Script(s) | Description |
|---|---|---|
| Data sampling | `01–04` | Extract and subsample domain corpora |
| Synthetic data generation | `05–07` | Translate or back-translate to build CA–EU parallel data |
| Fine-tuning | `08–12`, `09.2` | Train LoRA adapters on sampled, synthetic, and token-matched data |
| Evaluation | `13` | Score all models with BLEU, chrF++, TER, COMET |

---

## Data Sampling (`01–04`)

### `01_sample_general.py`
Loads the `projecte-aina/CA-EU_Parallel_Corpus` dataset from HuggingFace and randomly samples 50,000 CA–EU sentence pairs.

---

### `02_sample_ca-literary.py`
Reads `data/literary/CTILC-paula-ca-es-literary/corpus_ca_es.csv`, filters to the most recent document years, and samples up to 100,000 CA–ES paragraph pairs for subsequent pivot translation.

---

### `03_sample_eu-literary.py`
Reads the aligned EhuHac ES–EU plain-text files (`EhuHac.es-eu.es`, `EhuHac.es-eu.eu`), filters short lines, and down-samples to 100,000 aligned sentence pairs.

---

### `04_sample_eu-clinical.py`
Reads JSON documents from `data/eu-clinical/`, extracts paragraphs, sorts documents by publication date (most recent first), and accumulates up to 100,000 paragraphs. Prepares records with an empty `ca` field ready for translation.

---

## Synthetic Data Generation (`05–07`)

All translation uses `HiTZ/Latxa-Llama-3.1-8B-Instruct` via vLLM offline batching with greedy decoding.

### `05_translate-ca-literary.py`
Translates Spanish paragraphs from the CTILC corpus to Basque (ES→EU) using Latxa. After translation, applies a DP-based paragraph alignment (vecalign-style, using `multilingual-e5-large` embeddings) to re-align the translated EU text to the original CA paragraph structure, producing trilingual CA–ES–EU records.

---

### `06_translate-eu-literary.py`
Translates Spanish sentences from EhuHac (ES→CA) using Latxa, producing synthetic Catalan translations aligned to existing Basque originals.

---

### `07_translate-eu-clinical.py`
Translates Basque clinical paragraphs to Catalan (EU→CA) using Latxa. Long documents are split into character-level chunks (default 2,000 chars) before translation and rejoined afterwards to avoid truncation by the model's context limit.

---

## Fine-tuning (`08–12`)

All fine-tuning scripts share the same core architecture:

- **Base model**: `HiTZ/Latxa-Qwen3-VL-8B-Instruct`
- **Adaptation**: LoRA (r=16, α=32, dropout=0.05) targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Quantization**: 4-bit NF4 (bitsandbytes), with bfloat16 compute
- **Sequence length**: 768 tokens
- **Training**: 3 epochs, batch size 4, gradient accumulation 8, lr=5e-5 cosine
- **Checkpoint selection**: best BLEU on validation set (early stopping patience=3)
- **Instruction format**: explicit direction prefix injected per sample

### Instruction Templates

| Setting | Prompt |
|---|---|
| EU→CA general | `Itzuli testu hau euskaratik katalanera:\n\n{source}` |
| CA→EU general | `Tradueix aquest text del català al basc:\n\n{source}` |
| EU→CA literary | `Itzuli testu literario hau euskaratik katalanera:\n\n{source}` |
| CA→EU literary | `Tradueix aquest text literari del català al basc:\n\n{source}` |
| CA→EU clinical | `Tradueix aquest text clínic del català al basc:\n\n{source}` |

---

### `08_finetuning_general.py`
Fine-tunes on the AINA general parallel corpus. Each CA–EU pair generates two instruction samples (EU→CA and CA→EU). 

---

### `09_finetuning_literaryv1.py`
Direct literary SFT from the base model. Combines CA-literary (EU→CA direction from `ca-literary_trilingual.json`) and EU-literary (CA→EU direction from `eu-literary-EhuHac.jsonl`) into a single bidirectional training set. 

---

### `09.2_finetuning_literaryv1_tokenmatched.py`
Variant of `09` that trains only on CA→EU literary data matched to the same token budget as the clinical training set (~360k source tokens). Enables a fair data-controlled comparison between literary and clinical domain learnability.

---

### `10_finetuning_clinicalv1.py`
Direct clinical SFT from the base model. Single direction (CA→EU) only, using the back-translated clinical corpus. The model is prompted with a domain-specific clinical instruction.

---

### `11_finetuning_literaryv2.py`
Continued fine-tuning of the `generalv1` checkpoint on literary data. Loads the PEFT checkpoint, merges adapters (`merge_and_unload`), then applies fresh LoRA for continued domain adaptation.

**Usage**: `--model outputs/generalv1 --is-peft`

---

### `12_finetuning_clinicalv2.py`
Continued fine-tuning of the `generalv1` checkpoint on clinical data, following the same merge-and-retrain strategy as `11`.

**Usage**: `--model outputs/generalv1 --is-peft`

---

## Evaluation (`13`)

### `13_evaluate_all.py`
Unified evaluation script that loads one or more model checkpoints (full or PEFT), generates translations on a held-out test set, and computes four automatic metrics.

**Metrics**: chrF++ (word order=2), BLEU (effective order), TER, COMET (`Unbabel/wmt22-comet-da`)

**Key flags**:

| Flag | Description |
|---|---|
| `--task` | `general`, `literary`, or `clinical` |
| `--models` | One or more model paths or HF repo IDs |
| `--test-file` | Pre-saved JSON test set (recommended) |
| `--directions` | Override default directions for the task |
| `--max-samples` | Cap samples per direction for quick runs |
| `--batch-size` | Generation batch size (default 4) |
| `--max-new-tokens` | Max output tokens per sample (default 1024) |

---

## Common Arguments (fine-tuning scripts)

| Argument | Default | Description |
|---|---|---|
| `--model` | `HiTZ/Latxa-Qwen3-VL-8B-Instruct` | Base model path or HF repo |
| `--output-dir` | `outputs/{model_name}/` | Where to save adapters and tokenizer |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 4 | Per-device train batch size |
| `--grad-accum` | 8 | Gradient accumulation steps |
| `--lr` | 5e-5 | Peak learning rate |
| `--max-length` | 768 | Maximum tokenized sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA scaling factor |
| `--no-4bit` | False | Disable 4-bit quantization (requires more VRAM) |
| `--seed` | 42 | Global random seed |
| `--max-train-samples` | None | Cap on training samples (for ablations) |


---
**Paula Guerrero & Iker Gutierrez** · University of the Basque Country (EHU) · Machine Translation and Multilingualism 2025–2026
