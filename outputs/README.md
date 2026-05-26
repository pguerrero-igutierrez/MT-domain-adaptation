# Outputs

This directory contains all artifacts produced by the fine-tuning and evaluation pipeline: trained LoRA adapters, held-out test sets, token-matched comparison checkpoints, and per-model evaluation results.

---

## Fine-tuned Models

Each subdirectory stores a LoRA adapter checkpoint and the corresponding tokenizer, produced by the respective fine-tuning script. Adapters are compatible with the `peft` library and require the base model `HiTZ/Latxa-Qwen3-VL-8B-Instruct` to be loaded separately.

| Directory | Script | Description |
|---|---|---|
| `generalv1/` | `08_finetuning_general.py` | Bidirectional CA↔EU general-domain model |
| `literaryv1/` | `09_finetuning_literaryv1.py` | Bidirectional CA↔EU literary model (direct SFT) |
| `literaryv2/` | `11_finetuning_literaryv2.py` | Bidirectional CA↔EU literary model (continued from `generalv1`) |
| `clinicalv1/` | `10_finetuning_clinicalv1.py` | CA→EU clinical model (direct SFT) |
| `clinicalv2/` | `12_finetuning_clinicalv2.py` | CA→EU clinical model (continued from `generalv1`) |
| `literaryv1_tokenmatched/` | `09.2_finetuning_literaryv1_tokenmatched.py` | CA→EU literary model trained on a token-matched subset |


---

## Test Sets

Test sets are written once by the first fine-tuning script to run for each domain and are not overwritten by subsequent scripts. They are used as fixed evaluation benchmarks across all models within a domain.

| File | Domain | Pairs | Direction(s) |
|---|---|---|---|
| `test_set_general.json` | General | ~4,847 | EU→CA and CA→EU |
| `test_set_literary.json` | Literary | ~9,687 | EU→CA and CA→EU |
| `test_set_clinical.json` | Clinical | 66 | CA→EU only |


---

## Evaluation Results (`eval/`)

The `eval/` subdirectory is populated by `13_evaluate_all.py`. For each model and task, two files are written:

### Per-direction translation backup

```
eval/{model_name}_{task}_{direction}_backup.json
```

Contains the raw translation output before metric computation. Used for fault tolerance: if the evaluation script is interrupted after inference but before metric scoring, translations are not regenerated on re-run.


### Full results file

```
eval/{model_name}_{task}_results.json
```

Contains aggregated metrics per direction (and overall) alongside all per-sample predictions.


### Evaluation Metrics

| Metric | Tool | Notes |
|---|---|---|
| chrF++ | `sacrebleu` (word order=2) | Character n-gram F-score with word-level n-grams |
| BLEU | `sacrebleu` (effective order) | Corpus-level BLEU |
| TER | `sacrebleu` | Translation Edit Rate (lower is better) |
| COMET | `Unbabel/wmt22-comet-da` | Neural reference-based quality estimate |
| Length ratio | — | Mean hypothesis/reference token length ratio |


---

**Paula Guerrero & Iker Gutierrez** · University of the Basque Country (EHU) · Machine Translation and Multilingualism 2025–2026
