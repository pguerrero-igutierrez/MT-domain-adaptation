# Paper

## Title

*Domain adaptation for Catalan-Basque machine translation via synthetic data and continued fine-tuning*

## Summary

This paper studies domain adaptation for Catalan-Basque machine translation in low-resource settings. It investigates whether synthetic bilingual data built from monolingual and pivot-language resources can improve translation quality in specialized domains, and compares direct domain-specific fine-tuning against continued fine-tuning from a general-domain checkpoint.

The experiments use `HiTZ/Latxa-Qwen3-VL-8B-Instruct` with LoRA adapters across three domains: general, literary, and clinical. Literary and clinical corpora are constructed through pivot translation and back-translation with `HiTZ/Latxa-Llama-3.1-8B-Instruct`. The main findings are that domain-specific fine-tuning clearly improves over zero-shot and general-domain transfer, direct fine-tuning works better than continued adaptation in the specialized domains, and under token-matched conditions the clinical domain is easier to learn than the literary one.


**Paula Guerrero & Iker Gutierrez** · University of the Basque Country (EHU) · Machine Translation and Multilingualism 2025–2026