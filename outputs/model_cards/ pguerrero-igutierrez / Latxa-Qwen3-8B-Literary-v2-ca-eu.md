---
base_model: HiTZ/Latxa-Qwen3-VL-8B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- transformers
- machine-translation
- domain-adaptation
- literary
- catalan
- basque
language:
- ca
- eu
---

# Latxa-Qwen3-8B-Literary-v2-ca-eu

`Latxa-Qwen3-8B-Literary-v2-ca-eu` is a LoRA adapter for Catalan-Basque literary machine translation. It continues fine-tuning from the project general-domain checkpoint and targets bidirectional literary translation for the `ca->eu` and `eu->ca` directions.

## Model details

- Developed by: Paula Guerrero and Iker Gutierrez
- Affiliation: University of the Basque Country (EHU)
- Model type: LoRA adapter for `HiTZ/Latxa-Qwen3-VL-8B-Instruct`
- Languages: Catalan (`ca`), Basque (`eu`)
- Domain: Literary translation
- Base model: `HiTZ/Latxa-Qwen3-VL-8B-Instruct`
- Continued from: `pguerrero-igutierrez/Latxa-Qwen3-8B-General-eu-ca`
- Repository: `pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v2-ca-eu`
- Collection: `pguerrero-igutierrez/mt-domain-adaptation-ca-eu`

## Sources

- Hugging Face repository: https://huggingface.co/pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v2-ca-eu
- Hugging Face collection: https://huggingface.co/collections/pguerrero-igutierrez/mt-domain-adaptation-ca-eu
- Project repository: https://github.com/pguerrero-igutierrez/MT-domain-adaptation
- Paper source: https://github.com/pguerrero-igutierrez/MT-domain-adaptation/tree/main/paper

## Intended use

This model is intended for research on low-resource Catalan-Basque literary translation, especially in settings where in-domain parallel data is scarce and synthetic back-translation data is used for adaptation.

Supported prompting directions:

- `eu->ca`: `Itzuli testu hau euskaratik katalanera:\n\n{source}`
- `ca->eu`: `Tradueix aquest text del català al basc:\n\n{source}`

## Out-of-scope use

- High-stakes use without human review
- Professional literary publishing without post-editing
- Medical, legal, or safety-critical translation workflows
- General multilingual tasks outside Catalan-Basque translation

## Training data

This adapter was trained on the same literary corpora as `literaryv1`, built through Spanish-pivot synthetic data generation and back-translation:

- `backtranslated-corpus/ca-literary_trilingual.json`
- `backtranslated-corpus/eu-literary-EhuHac.jsonl`

The model was then continued from the project general checkpoint rather than trained directly from the base model.

## Training procedure

- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Quantization: 4-bit NF4
- Max sequence length: 768
- Epochs: 3
- Batch size: 4
- Gradient accumulation: 8
- Learning rate: `5e-5`
- Scheduler: cosine
- Warmup ratio: 0.05
- Seed: 42
- Checkpoint selection: best validation BLEU

## Evaluation

Results on the literary held-out test set:

| Direction | chrF++ | BLEU | TER | COMET |
|---|---:|---:|---:|---:|
| `eu->ca` | 34.51 | 7.44 | 87.66 | 68.72 |
| `ca->eu` | 25.87 | 2.31 | 100.74 | 64.34 |
| Overall | 30.02 | 5.17 | 93.81 | 65.44 |

In the project experiments, this continued-adaptation literary model performed slightly below the direct literary SFT model (`literaryv1`) across the reported literary metrics.

## Limitations

- Trained on synthetic literary supervision rather than human-translated in-domain CA-EU parallel data
- Literary quality aspects such as style, voice, and fluency are only partially captured by automatic metrics
- CA->EU literary performance remains challenging, especially under word-level metrics such as BLEU

## Usage

This repository contains adapter weights, so it must be loaded on top of the base model.

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

base_id = "HiTZ/Latxa-Qwen3-VL-8B-Instruct"
adapter_id = "pguerrero-igutierrez/Latxa-Qwen3-8B-Literary-v2-ca-eu"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_id)

prompt = "Tradueix aquest text del català al basc:\n\nBon vespre."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

If you use this model, please cite the project repository:

```bibtex
@misc{guerrero-gutierrez-2026-caeu-mt,
  title        = {Domain Adaptation for Catalan-Basque Machine Translation via Synthetic Data and Continued Fine-Tuning},
  author       = {Guerrero, Paula and Gutierrez, Iker},
  year         = {2026},
  note         = {Unpublished manuscript}
}
```

## Contact

- Paula Guerrero: `pguerrero005@ikasle.ehu.eus`
- Iker Gutierrez: `igutierrez134@ikasle.ehu.eus`

