# Domain adaptation for Catalan–Basque machine translation via synthetic data and continued fine-tuning

**Preprint:** The arXiv link will be added here once the paper is announced. The manuscript PDF is not distributed in this repository.
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/pguerrero-igutierrez/mt-domain-adaptation-ca-eu)

**Paula Guerrero-Castelló & Iker Gutiérrez-Fandiño**<br>

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

| Model ID | Script | Description | Released adapter |
|---|---|---|---|
| `generalv1` | `08_finetuning_general.py` | Bidirectional CA↔EU model fine-tuned on AINA general corpus | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-General-eu-ca) |
| `literaryv1` | `09_finetuning_literaryv1.py` | Direct literary SFT from base model (bidirectional) | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v1-ca-eu) |
| `literaryv1_tokenmatched` | `09b_finetuning_literaryv1_tokenmatched.py` | CA→EU literary token-matched variant | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v1-tokenmatched-ca-eu) |
| `literaryv2` | `11_finetuning_literaryv2.py` | Literary SFT continued from `generalv1` checkpoint | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v2-ca-eu) |
| `clinicalv1` | `10_finetuning_clinicalv1.py` | Direct clinical SFT from base model (CA→EU) | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Clinical-v1-ca-eu) |
| `clinicalv2` | `12_finetuning_clinicalv2.py` | Clinical SFT continued from `generalv1` checkpoint | [Hugging Face](https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Clinical-v2-ca-eu) |

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

The reported experiments use a split of 90/5/5% for train/validation/test. Counts below describe those experiments; the currently released clinical corpus is a different version, documented in [PROVENANCE.md](PROVENANCE.md).

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

## Released data and generated artifacts

The synthetic corpora in `backtranslated-corpus/` can be used directly for training. The clinical file used by all scripts is `backtranslated-corpus/eu-clinical_eu2ca.json`.

- `sampled-data/ca_eu_50k.json` is generated by step 01 and is not tracked. Its random seed is 42; the historical upstream dataset revision is not recorded, so regeneration does not guarantee the exact historical sample.
- The aligned ES–EU intermediate `sampled-data/ehuhac_sampled_parallel.jsonl` is generated by step 03. Download the sentence-aligned EhuHac text files `EhuHac.es-eu.es` and `EhuHac.es-eu.eu` and place them in `data/literary/opus-EhuHac-literary/es-eu.txt/` before running it.
- The tracked `sampled-data/eu-literary_sampled100k.jsonl` is a legacy monolingual intermediate. It lacks the `source_es` field required by step 06 and cannot substitute for the aligned intermediate, even if renamed. Reconstructing step 06 requires step 03; training can instead use the released translated corpus.
- Checkpoint directories under `outputs/` are generated locally and ignored by Git. For evaluation without retraining, download an adapter from the model table into the corresponding local directory; it must include its adapter configuration, weights and tokenizer. Step 11/12 can then consume a downloaded `generalv1` checkpoint using `--is-peft`.

**Clinical evaluation provenance:** `outputs/test_set_clinical_reported.json` contains the 66 ordered inputs and references shared by all four published clinical prediction caches, matching the reported evaluation size. It is reconstructed by `python scripts/prepare_reported_clinical_test.py` and is used in the evaluation example below. The existing `outputs/test_set_clinical.json` contains 79 different examples and is retained separately. A local archived corpus was identified that reproduces the reported split and all 66 evaluation inputs exactly; its checksum and filtering procedure are recorded in [PROVENANCE.md](PROVENANCE.md). The archive's distribution and exact historical run provenance still require author confirmation. Use `--clinical-data /path/to/the/archive.json` in clinical training, token matching or test-set reconstruction to select it explicitly. The default released corpus produces a different split; do not interpret a rerun on that version as exact reproduction of the reported training.

## Quick Start

Run commands from the repository root with **Python 3.11**. Training and corpus generation require appropriate NVIDIA GPU/CUDA support. Use separate environments: COMET requires Transformers 4, whereas the released MT setup pins Transformers 5. Corpus generation uses vLLM's compatible Torch/Transformers stack.

```bash
# Fine-tuning and translation evaluation
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# COMET scoring, called by the evaluator in a separate process
python3.11 -m venv .venv-comet
.venv-comet/bin/python -m pip install --upgrade pip
.venv-comet/bin/python -m pip install -r requirements-comet.txt

# Optional: needed only to reconstruct synthetic corpora with steps 05–07
python3.11 -m venv .venv-generation
.venv-generation/bin/python -m pip install --upgrade pip
.venv-generation/bin/python -m pip install -r requirements-generation.txt
```

These dependency files describe the release setup; they do not establish the exact historical environments used for every experiment.

### General model

```bash
.venv/bin/python scripts/01_sample_general.py
.venv/bin/python scripts/08_finetuning_general.py
```

### Literary models

```bash
# Optional corpus reconstruction; requires raw source corpora.
.venv/bin/python scripts/02_sample_ca_literary.py
.venv/bin/python scripts/03_sample_eu_literary.py
.venv-generation/bin/python scripts/05_translate_ca_literary.py --resume
.venv-generation/bin/python scripts/06_translate_eu_literary.py --resume

# Training can use the synthetic corpora already present in the repository.
.venv/bin/python scripts/09_finetuning_literaryv1.py
.venv/bin/python scripts/09b_finetuning_literaryv1_tokenmatched.py
.venv/bin/python scripts/11_finetuning_literaryv2.py --is-peft --model outputs/generalv1
```

### Clinical models

```bash
# Optional corpus reconstruction; the translated clinical corpus is already released.
.venv/bin/python scripts/04_sample_eu_clinical.py
.venv-generation/bin/python scripts/07_translate_eu_clinical.py --resume

.venv/bin/python scripts/10_finetuning_clinicalv1.py
.venv/bin/python scripts/12_finetuning_clinicalv2.py --is-peft --model outputs/generalv1
```

### Evaluation

```bash
# General
.venv/bin/python scripts/13_evaluate_all.py \
  --task general \
  --models HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1 \
  --test-file outputs/test_set_general.json

# Literary
.venv/bin/python scripts/13_evaluate_all.py \
  --task literary \
  --models outputs/literaryv1 outputs/literaryv1_tokenmatched outputs/literaryv2 \
  --test-file outputs/test_set_literary.json

# Clinical
.venv/bin/python scripts/13_evaluate_all.py \
  --task clinical \
  --models outputs/clinicalv1 outputs/clinicalv2 \
  --test-file outputs/test_set_clinical_reported.json
```

---

The evaluator defaults to `.venv-comet/bin/python`; use `--comet-python /other/environment/bin/python` to select a different COMET environment. Translation caches are reused only when their ordered sources and references match the selected test set. Use a fresh `--output-dir outputs/eval_recheck` when changing the sample subset or settings. Existing matching caches are scored without loading a translation model.

To score an existing prediction file directly, without importing the MT stack:

```bash
.venv-comet/bin/python scripts/score_comet.py \
  --input outputs/eval/generalv1_clinical_ca2eu_backup.json \
  --output outputs/comet_recheck/generalv1_clinical_scores.json
```

The scorer retains the original `Unbabel/wmt22-comet-da` model, batch size 8 and ×100 reporting scale. Pass `--gpus 1` for GPU scoring, or use the CPU default. Training settings and released numerical results are unchanged.

Run the release workflow regression tests with `.venv/bin/python -m unittest discover -s tests -v`. These exercise data handoffs, cache checks and the scoring subprocess with a deterministic test double; they do not train or evaluate the translation models.

## Repository Structure

The model directories shown below are generated artifacts, not tracked checkpoints; their released adapters are linked above.

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
│   ├── test_set_clinical_reported.json
│   └── eval/                      # Per-model evaluation JSON results
├── paper/
│   └── README.md                  # Preprint link to be added after announcement
├── poster/                        
└── scripts/
    ├── 01_sample_general.py
    ├── 02_sample_ca_literary.py
    ├── 03_sample_eu_literary.py
    ├── 04_sample_eu_clinical.py
    ├── 05_translate_ca_literary.py
    ├── 06_translate_eu_literary.py
    ├── 07_translate_eu_clinical.py
    ├── 08_finetuning_general.py
    ├── 09_finetuning_literaryv1.py
    ├── 09b_finetuning_literaryv1_tokenmatched.py
    ├── 10_finetuning_clinicalv1.py
    ├── 11_finetuning_literaryv2.py
    ├── 12_finetuning_clinicalv2.py
    ├── 13_evaluate_all.py
    ├── comet_bridge.py
    └── score_comet.py
```


---

## Citation

The arXiv identifier and link will be added once announced. Article, code, dataset and base-model licensing are separate decisions; a repository code licence is pending author agreement.

If you use this work, please cite:

```bibtex
@misc{guerrero-gutierrez-2026-caeu-mt,
  title        = {Domain Adaptation for Catalan-Basque machine translation via synthetic data and continued fine-tuning},
  author       = {Guerrero-Castelló, Paula and Gutiérrez-Fandiño, Iker},
  year         = {2026},
  note         = {Unpublished manuscript}
}
```

---

## Contact

**Iker Gutiérrez-Fandiño**<br>
*Computational linguist*<br>
[GitHub Profile](https://github.com/iker-gutierrez) | [LinkedIn](https://www.linkedin.com/in/iker-gutierrez-fandino)

**Paula Guerrero-Castelló**<br>
*Computational linguist*<br>
[GitHub Profile](https://github.com/guerreropaula) | [LinkedIn](https://www.linkedin.com/in/paula-guerrero-castelló)

