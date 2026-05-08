# MT Domain Adaptation: Catalan-Basque Translation

Fine-tuning LLM machine translation models for domain-specific translation tasks. This project adapts the Latxa/Qwen base model to three domains: general, literary, and clinical for bidirectional Catalan-Basque translation.

## Overview

This project compares two training strategies for domain-specific neural machine translation:

- **v1 Models (scripts 09, 10)**: Fine-tune from scratch on domain-specific data alone (literary domain, clinical domain)
- **v2 Models (scripts 11, 12)**: Fine-tune from the general model, mixing general + domain-specific data (transfer learning approach)

The hypothesis is that the v2 approach (starting from a general baseline) will achieve better results than training from scratch (v1), especially for low-resource scenarios. Both literary and clinical models support bidirectional or unidirectional translation depending on data availability.

The pipeline flow:
1. Sample and prepare domain-specific corpora (scripts 01-04)
2. Generate synthetic back-translations using Latxa-Llama-3.1-8B (scripts 05-07)
3. Train models: v1 from base, v2 from general baseline (scripts 08-12)
4. Evaluate with chrF++, BLEU, and TER metrics (evaluate_*.py)

## Data Sources & Preparation

The project uses three domains with different data preparation strategies:

| Domain | Source Corpus | Preparation | Pairs | Directions |
|--------|---------------|-----------|-------|-----------|
| **General** | AINA CA-EU Parallel Corpus (HuggingFace) | Direct 50K random sample | 50K | CA ↔ EU (both) |
| **Literary CA** | CTILC Spanish-Catalan literary corpus | Filter recent years (100K), translate ES→EU/CA using Latxa, realign with vecalign | ~100K | EU → CA (Spanish-pivot) |
| **Literary EU** | EhuHac Spanish-Basque aligned sentences | Load aligned ES-EU pairs (100K), translate ES→CA/EU using Latxa | ~100K | EU ↔ CA (both) |
| **Clinical EU** | E3C Basque clinical documents | Extract paragraphs (100K), sort by publication date, translate EU→CA→EU backtranslation | ~100K | CA → EU (only) |

## Project Structure

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `scripts/` | 12 numbered Python scripts | Data sampling, translation, fine-tuning, evaluation |
| `backtranslated-corpus/` | JSON/JSONL files | Back-translated domain-specific data ready for fine-tuning |
| `sampled-data/` | `ca_eu_50k.json` | General domain sample (50K parallel pairs) |
| `outputs/` | Model directories | Fine-tuned LoRA adapters and test sets |

## Scripts Workflow

### Data Preparation Phase

| Script | What it does | Input | Output |
|--------|------|-------|--------|
| `01_sample_general.py` | Randomly samples 50K CA-EU parallel pairs from AINA HF dataset | HF: projecte-aina/CA-EU_Parallel_Corpus | `sampled-data/ca_eu_50k.json` |
| `02_sample_ca-literary.py` | Filters CTILC Spanish-Catalan corpus to recent years, subsamples 100K rows with all metadata preserved | `data/literary/CTILC-paula-ca-es-literary/corpus_ca_es.csv` | `sampled-data/corpus_ca_es_100k_lit.json` |
| `03_sample_eu-literary.py` | Aligns Spanish-Basque EhuHac literary text files, creates parallel JSONL with aligned sentence pairs, subsamples 100K | `data/literary/opus-EhuHac-literary/es-eu.txt/` (ES & EU files) | `sampled-data/ehuhac_sampled_parallel.jsonl` |
| `04_sample_eu-clinical.py` | Loads Basque clinical JSON files, extracts paragraphs, sorts by most recent publication, subsamples 100K paragraphs | `data/eu-clinical/` (JSON files) | `sampled-data/eu-clinical_sampled100k.json` |

### Translation Phase (Back-translation & Synthetic Data)

| Script | What it does | Input | Output |
|--------|------|-------|--------|
| `05_translate-ca-literary.py` | Translates CTILC Spanish literary paragraphs ES→EU and ES→CA using Latxa-Llama-3.1-8B-Instruct (vLLM). Re-aligns output paragraphs back to original structure using vecalign alignment. Preserves all original metadata. | `sampled-data/corpus_ca_es_100k_lit.json` | `backtranslated-corpus/ca-literary_trilingual.json` (fields: text_eu, text_ca_back) |
| `06_translate-eu-literary.py` | Translates EhuHac Spanish literary sentences in two sequential passes: Pass 1: ES→EU (backtranslation), Pass 2: ES→CA (translation). Outputs per-document JSONL + merged file. | `sampled-data/ehuhac_sampled_parallel.jsonl` | `backtranslated-corpus/eu-literary-EhuHac.jsonl` + `eu-literary-trilingual.jsonl` |
| `07_translate-eu-clinical.py` | Translates Basque clinical paragraphs via Catalan bridge. Pass 1: EU→CA (translation), Pass 2: CA→EU (backtranslation to validate quality). Preserves all clinical metadata (doc_id, publication_date, authors, source, etc). | `sampled-data/eu-clinical_sampled100k.json` | `backtranslated-corpus/eu-clinical_backtranslated.json` |

### Fine-tuning Phase (Single Domain)

| Script | What it does | Architecture | Languages | Input | Output |
|--------|------|-----------|-----------|--------|--------|
| `08_finetuning_general.py` | Fine-tunes Latxa-Qwen3-VL-8B-Instruct with LoRA for bidirectional translation. Both CA→EU and EU→CA in single model with explicit direction instructions. 90/10 train/test split. | LoRA on Latxa-Qwen3-VL-8B-Instruct | CA ↔ EU (bidirectional) | `sampled-data/ca_eu_50k.json` | `outputs/generalv1/` (LoRA adapters) |
| `09_finetuning_literaryv1.py` | Fine-tunes Latxa-Qwen3-VL-8B-Instruct with LoRA from scratch on literary data. Combines two literary sources (CA-ES trilingual + EhuHac EU-ES aligned). Bidirectional EU↔CA. 90/5/5 train/valid/test split. | LoRA on Latxa-Qwen3-VL-8B-Instruct | EU ↔ CA (bidirectional) | `backtranslated-corpus/ca-literary_trilingual.json` + `eu-literary-EhuHac.jsonl` | `outputs/literaryv1/` + test set |
| `10_finetuning_clinicalv1.py` | Fine-tunes Latxa-Qwen3-VL-8B-Instruct with LoRA from scratch on clinical data. Only one direction available (CA→EU) since corpus is Catalan→Basque backtranslated. 90/5/5 train/valid/test split. | LoRA on Latxa-Qwen3-VL-8B-Instruct | CA → EU (unidirectional) | `backtranslated-corpus/eu-clinical_backtranslated.json` | `outputs/clinicalv1/` + test set |

### Fine-tuning Phase (Transfer Learning)

| Script | What it does | Strategy | Input | Output |
|--------|------|-----------|--------|--------|
| `11_finetuning_literaryv2.py` | Continues training from the general model (v1) by mixing general data + both literary corpora together. Uses literary sampling weight to bias toward literary domain. Cosine LR schedule with short warmup from low LR. Bidirectional. | Load general-domain model, stack fresh LoRA on top, mix datasets | General model (HF Hub) + `backtranslated-corpus/ca-literary_trilingual.json` + `eu-literary-EhuHac.jsonl` | `outputs/literaryv2/` + test set |
| `12_finetuning_clinicalv2.py` | Continues training from the general model (v1) by mixing general data + clinical corpus. Uses cosine LR schedule. Unidirectional (CA→EU only). | Load general-domain model, stack fresh LoRA, regularize with general data | General model (HF Hub) + `backtranslated-corpus/eu-clinical_backtranslated.json` | `outputs/clinicalv2/` + test set |

### Evaluation Phase

| Script | Models Evaluated | Metrics | Input | Output |
|--------|--------|---------|-------|--------|
| `evaluate_general.py` | generalv1 (any LoRA checkpoint from 08_finetuning_general.py) | chrF++ (primary), BLEU, TER, length ratio | `sampled-data/ca_eu_50k.json` or test set JSON | stdout table + JSON results in `outputs/eval/` |
| `evaluate_literary.py` | literaryv1 & literaryv2 (LoRA from 09 or 11) | chrF++ (primary), BLEU, TER, length ratio | `backtranslated-corpus/ca-literary_trilingual.json` + `eu-literary-EhuHac.jsonl` or test set JSON | stdout table + JSON results in `outputs/eval/` |
| `evaluate_clinical.py` | clinicalv1 & clinicalv2 (LoRA from 10 or 12) | chrF++ (primary), BLEU, TER, length ratio | `backtranslated-corpus/eu-clinical_backtranslated.json` or test set JSON | stdout table + JSON results in `outputs/eval/` |

## Setup

Requirements: Python 3.11+

```bash
pip install -r requirements.txt
```

## Usage

### Typical Pipeline

**1. Data Preparation** (one-time):
```bash
python scripts/01_sample_general.py
python scripts/02_sample_ca-literary.py
python scripts/03_sample_eu-literary.py
python scripts/04_sample_eu-clinical.py
```

**2. Translation / Back-translation** (one-time):
```bash
python scripts/05_translate-ca-literary.py --langs eu ca
python scripts/06_translate-eu-literary.py
python scripts/07_translate-eu-clinical.py
```

**3a. Train v1 Models** (from scratch):
```bash
python scripts/08_finetuning_general.py
python scripts/09_finetuning_literaryv1.py
python scripts/10_finetuning_clinicalv1.py
```

**3b. Train v2 Models** (transfer learning from general):
```bash
python scripts/11_finetuning_literaryv2.py --model <hf-repo-or-path>
python scripts/12_finetuning_clinicalv2.py --model <hf-repo-or-path>
```

**4. Evaluate all models**:
```bash
python scripts/evaluate_general.py --model outputs/generalv1
python scripts/evaluate_literary.py --model outputs/literaryv1 --test-file outputs/test_set_literary.json
python scripts/evaluate_literary.py --model outputs/literaryv2 --test-file outputs/test_set_literary.json
python scripts/evaluate_clinical.py --model outputs/clinicalv1 --test-file outputs/test_set_clinical.json
python scripts/evaluate_clinical.py --model outputs/clinicalv2 --test-file outputs/test_set_clinical.json
```

### Common Options

Most scripts support CLI flags:

- `--epochs N`: Number of training epochs
- `--lr FLOAT`: Learning rate (default 2e-4)
- `--batch-size N`: Batch size for inference/training
- `--max-train-samples N`: Limit training samples (useful for testing)
- `--output-dir PATH`: Custom output directory
- `--no-4bit`: Disable 4-bit quantization (use full precision)
- `--resume`: Resume from checkpoint (for translation scripts)
- `--run-name NAME`: Custom name for evaluation results

See each script's docstring for full options: `head -n 50 scripts/XX_name.py`




