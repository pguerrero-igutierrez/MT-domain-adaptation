"""
04_build_finetuning_corpus.py

Builds fine-tuning datasets for CA<->EU translation using:
  - projecte-aina/CA-EU_Parallel_Corpus  (base general-domain pairs)
  - corpus/parallel_corpus_eu_ca.csv     (clinical domain, from build_parallel_corpus.py)
  - literary/eu-literary/output/all.jsonl (literary domain, from 02_translate-eu-literary.py)

Produces three fine-tuning splits:
  - finetuning/base/     : general domain only (CA-EU_Parallel_Corpus)
  - finetuning/clinical/ : general + clinical
  - finetuning/literary/ : general + literary

Each split contains train.jsonl, val.jsonl formatted for seq2seq fine-tuning:
    {"translation": {"ca": "...", "eu": "..."}}

Both directions (CA->EU and EU->CA) are included in every split.

Base model for fine-tuning: facebook/nllb-200-distilled-1.3B
Language codes: cat_Latn (CA), eus_Latn (EU)

REQUIREMENTS
------------
    pip install datasets transformers torch tqdm

USAGE
-----
    python 06_build_finetuning_corpus.py
    python 06_build_finetuning_corpus.py --val-ratio 0.02 --max-base 500000
    python 06_build_finetuning_corpus.py --domain
"""
