# Domain adaptation for Catalan–Basque machine translation via synthetic data and continued fine-tuning

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/paper.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/pguerrero-igutierrez/mt-domain-adaptation-ca-eu)

**Paula Guerrero & Iker Gutierrez** · University of the Basque Country (EHU) · Machine Translation and Multilingualism 2025–2026

---

## Overview

This repository contains the code, data pipelines, and evaluation framework for domain-adapted Catalan↔Basque (CA–EU) neural machine translation. The project addresses the scarcity of domain-specific parallel corpora for this under-resourced language pair by constructing synthetic bilingual data from monolingual and pivot-language resources, then fine-tuning a Basque-centric instruction-tuned LLM with LoRA adapters.

Two specialized domains are targeted:

| Domain | Translation Direction | 
|---|---|
| Literary | CA↔EU (bidirectional) | 
| Clinical | CA→EU only | 

The general-domain model serves as both a standalone baseline and a warm-start checkpoint for continued domain-specific fine-tuning.


---

## Models

| Model ID | Script | Description |
|---|---|---|
| `generalv1` | `08_finetuning_general.py` | Bidirectional CA↔EU model fine-tuned on AINA general corpus |
| `literaryv1` | `09_finetuning_literaryv1.py` | Direct literary SFT from base model (bidirectional) |
| `literaryv1_tokenmatched` | `09.2_finetuning_literaryv1_tokenmatched.py` | CA→EU literary token-matched variant for fair comparison with the clinical setup |
| `literaryv2` | `11_finetuning_literaryv2.py` | Literary SFT continued from `generalv1` checkpoint |
| `clinicalv1` | `10_finetuning_clinicalv1.py` | Direct clinical SFT from base model (CA→EU) |
| `clinicalv2` | `12_finetuning_clinicalv2.py` | Clinical SFT continued from `generalv1` checkpoint |

All models use `HiTZ/Latxa-Qwen3-VL-8B-Instruct` as base, with LoRA adapters (r=16, α=32) and 4-bit NF4 quantization (bitsandbytes).

---

## Pipeline

| Stage | Scripts | Description |
|---|---|---|
| Data sampling | `01–04` | Prepare general, literary, and clinical datasets |
| Synthetic data generation | `05–07` | Translate or back-translate domain corpora |
| Fine-tuning | `08–12` | Train general, domain-specific, and token-matched LoRA adapters |
| Evaluation | `13` | Evaluate models with BLEU, chrF++, TER, and COMET |

---

## Datasets

All datasets use a split of 90/5/5% for train/validation/test.

### General
Sampled from `projecte-aina/CA-EU_Parallel_Corpus` (50k pairs). 

| Split | CA→EU | EU→CA |
|---|---|---|
| Train | 43,661 | 43,568 |
| Validation | 2,388 | 2,458 |
| Test | 2,412 | 2,435 |

### Literary
Built from two sources via pivot translation (ES as pivot language):

- **CTILC CA–ES** ([`guerreropaula/synthetic-corpus-ca-es`](https://github.com/guerreropaula/synthetic-corpus-ca-es)): ~30k pairs; ES pivot → EU synthetic (back-translation to produce CA–EU pairs)
- **EhuHac ES–EU**: 100k pairs; ES pivot → CA synthetic

| Split | CA→EU | EU→CA |
|---|---|---|
| Train | 75,444 | 26,900 |
| Validation | 4,217 | 1,468 |
| Test | 4,261 | 1,426 |

Average sentence length: ~110 tokens.

### Clinical
Built from Basque clinical documents (E3C corpus) translated to Catalan. Single direction only (CA→EU), with Catalan as synthetic source and Basque as the high-quality reference target.

| Split | CA→EU |
|---|---|
| Train | 1,174 |
| Validation | 65 |
| Test | 66 |

Average document length: ~1,078 tokens.


**Back-translation setup**: machine-translated texts serve as the source language during fine-tuning, while original high-quality texts are the target. This ensures the model learns from clean references.

All translation for data construction uses `HiTZ/Latxa-Llama-3.1-8B-Instruct` via vLLM offline batching.

---

## Key Findings

1. **Domain-specific fine-tuning is the primary driver of improvement**, particularly for the clinical domain where gains over zero-shot are largest.
2. **Cross-domain transfer helps but is insufficient**: the general model improves over zero-shot in-domain but falls well short of specialized models.
3. **Direct SFT outperforms continued adaptation** for the literary domain (literaryv1 > literaryv2 across all metrics).
4. **Synthetic data via pivot translation and back-translation successfully compensates** for the complete absence of in-domain CA–EU parallel corpora.
5. **Clinical domains are more "learnable"**: under token-matched conditions, clinical SFT yields substantially higher scores than literary SFT, likely due to terminological regularity.

---

## Data Sources

Raw corpora are not included in this repository due to size. Download them from the following links and place them under `data/`.

| Corpus | Domain | Link |
|---|---|---|
| projecte-aina/CA-EU_Parallel_Corpus | General | https://huggingface.co/datasets/projecte-aina/CA-EU_Parallel_Corpus |
| CTILC CA–ES literary corpus | Literary | https://github.com/guerreropaula/synthetic-corpus-ca-es |
| EhuHac ES–EU literary corpus | Literary | https://opus.nlpl.eu/datasets/EhuHac?pair=eu&es |
| E3C Basque clinical corpus | Clinical | https://github.com/hltfbk/E3C-Corpus |

---

## Quick Start


```bash
pip install -r requirements.txt
```

### General model

```bash
python scripts/01_sample_general.py
python scripts/08_finetuning_general.py
```

### Literary models

```bash
python scripts/02_sample_ca-literary.py
python scripts/03_sample_eu-literary.py
python scripts/05_translate-ca-literary.py --resume
python scripts/06_translate-eu-literary.py --resume
python scripts/09_finetuning_literaryv1.py
python scripts/09.2_finetuning_literaryv1_tokenmatched.py
python scripts/11_finetuning_literaryv2.py --is-peft --model outputs/generalv1
```

### Clinical models

```bash
python scripts/04_sample_eu-clinical.py
python scripts/07_translate-eu-clinical.py --resume
python scripts/10_finetuning_clinicalv1.py
python scripts/12_finetuning_clinicalv2.py --is-peft --model outputs/generalv1
```

### Evaluation

```bash
# General
python scripts/13_evaluate_all.py \
  --task general \
  --models HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1 \
  --test-file outputs/test_set_general.json

# Literary
python scripts/13_evaluate_all.py \
  --task literary \
  --models outputs/literaryv1 outputs/literaryv1_tokenmatched outputs/literaryv2 \
  --test-file outputs/test_set_literary.json

# Clinical
python scripts/13_evaluate_all.py \
  --task clinical \
  --models outputs/clinicalv1 outputs/clinicalv2 \
  --test-file outputs/test_set_clinical.json
```

---

## Repository Structure

```
.
├── data/                          # Raw source corpora (not included for size constraints)
├── sampled-data/                  # Sampled/preprocessed inputs 
├── backtranslated-corpus/         # Synthetic parallel corpora
├── outputs/
│   ├── generalv1/                 
│   ├── literaryv1/
│   ├── literaryv2/
│   ├── clinicalv1/
│   ├── clinicalv2/
│   ├── literaryv1_tokenmatched/
│   ├── test_set_general.json
│   ├── test_set_literary.json
│   ├── test_set_clinical.json
│   └── eval/                      # Per-model evaluation JSON results
├── paper/
│   ├── README.md
│   └── paper.pdf                  
├── poster/                        
└── scripts/
    ├── 01_sample_general.py
    ├── 02_sample_ca-literary.py
    ├── 03_sample_eu-literary.py
    ├── 04_sample_eu-clinical.py
    ├── 05_translate-ca-literary.py
    ├── 06_translate-eu-literary.py
    ├── 07_translate-eu-clinical.py
    ├── 08_finetuning_general.py
    ├── 09_finetuning_literaryv1.py
    ├── 09.2_finetuning_literaryv1_tokenmatched.py
    ├── 10_finetuning_clinicalv1.py
    ├── 11_finetuning_literaryv2.py
    ├── 12_finetuning_clinicalv2.py
    └── 13_evaluate_all.py
```


---

## Citation

If you use this work, please cite:

```bibtex
@misc{guerrero-gutierrez-2026-caeu-mt,
  title        = {Domain Adaptation for Catalan-Basque Machine Translation via Synthetic Data and Continued Fine-Tuning},
  author       = {Guerrero, Paula and Gutierrez, Iker},
  year         = {2026},
  note         = {Unpublished manuscript}
}
```

---

## Contact

- [pguerrero005@ikasle.ehu.eus](mailto:pguerrero005@ikasle.ehu.eus)
- [igutierrez134@ikasle.ehu.eus](mailto:igutierrez134@ikasle.ehu.eus)

*MSc in Language Analysis and Processing (EHU) · Machine Translation 2025–26*
