# Machine Translation Domain Adaptation: Catalan–Basque

Fine-tuning HiTZ/Latxa-Qwen3-VL-8B-Instruct for domain-specific Catalan↔Basque translation across general, literary, and clinical domains using LoRA and synthetic data augmentation.

The project fine-tunes `HiTZ/Latxa-Qwen3-VL-8B-Instruct` with LoRA adapters. Literary and clinical datasets are expanded with synthetic data generated using `HiTZ/Latxa-Llama-3.1-8B-Instruct`.

## Pipeline

| Stage | Scripts | Description |
|---|---|---|
| Data sampling | `01–04` | Prepare general, literary, and clinical datasets |
| Synthetic data generation | `05–07` | Translate or backtranslate domain data |
| Fine-tuning | `08–12` | Train general and domain-specific LoRA adapters |
| Evaluation | `13` | Evaluate models with BLEU, chrF++, TER, and COMET |

## Datasets

### General

`01_sample_general.py` samples 50k Catalan–Basque pairs from:

```text
projecte-aina/CA-EU_Parallel_Corpus
```

### Literary

The literary dataset is built from:

- CTILC CA–ES literary corpus
- EhuHac ES–EU literary corpus


The literary models are trained bidirectionally:

```text
Basque → Catalan
Catalan → Basque
```

### Clinical

The clinical dataset is built from Basque clinical documents and translated into Catalan.

```text
Catalan → Basque
```

Catalan is the synthetic source text; Basque is the original reference text.

## Models

| Model | Script | Description |
|---|---|---|
| `generalv1` | `08_finetuning_general.py` | General CA↔EU model trained on AINA |
| `literaryv1` | `09_finetuning_literaryv1.py` | Direct literary fine-tuning |
| `clinicalv1` | `10_finetuning_clinicalv1.py` | Direct clinical fine-tuning |
| `literaryv2` | `11_finetuning_literaryv2.py` | Literary fine-tuning continued from `generalv1` |
| `clinicalv2` | `12_finetuning_clinicalv2.py` | Clinical fine-tuning continued from `generalv1` |


## Quick Start

Install dependencies:

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
python scripts/11_finetuning_literaryv2.py --is-peft --model outputs/literaryv1
```

### Clinical models

```bash
python scripts/04_sample_eu-clinical.py
python scripts/07_translate-eu-clinical.py --resume

python scripts/10_finetuning_clinicalv1.py
python scripts/12_finetuning_clinicalv2.py --is-peft --model outputs/clinicalv1
```


## Evaluation

General evaluation:

```bash
python scripts/13_evaluate_all.py \
  --task general \
  --models HiTZ/Latxa-Qwen3-VL-8B-Instruct outputs/generalv1 \
  --test-file outputs/test_set_general.json
```

Literary evaluation:

```bash
python scripts/13_evaluate_all.py \
  --task literary \
  --models outputs/literaryv1 outputs/literaryv2 \
  --test-file outputs/test_set_literary.json
```

Clinical evaluation:

```bash
python scripts/13_evaluate_all.py \
  --task clinical \
  --models outputs/clinicalv1 outputs/clinicalv2 \
  --test-file outputs/test_set_clinical.json
```

Evaluation metrics:

- BLEU
- chrF++
- TER
- COMET
- length ratio



## Repository Structure

```text
.
├── data/
├── sampled-data/
├── backtranslated-corpus/
├── outputs/
│   └── eval/
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
    ├── 10_finetuning_clinicalv1.py
    ├── 11_finetuning_literaryv2.py
    ├── 12_finetuning_clinicalv2.py
    └── 13_evaluate_all.py
```

## Notes

- Clinical translation is only trained and evaluated in the `ca2eu` direction.
- `literaryv2` and `clinicalv2` require a previously trained general checkpoint.


---

## Citation

If you use this work, please cite:

```bibtex
@misc{x,
  title        = {x},
  author       = {x},
  year         = {2026},
  note         = {Unpublished manuscript}
}
```

Also cite the CLARA-MeD dataset:

```bibtex

```
---


### Contact

- [pguerrero005@ikasle.ehu.eus](mailto:pguerrero005@ikasle.ehu.eus)  
- [igutierrez134@ikasle.ehu.eus](mailto:igutierrez134@ikasle.ehu.eus)  
---
*Project for Machine Translation 2025-26*   
*MSc in Language Analysis and Processing (EHU)*  
