# Machine Translation Domain Adaptation: Catalan–Basque

Fine-tuning HiTZ/Latxa-Qwen3-VL-8B-Instruct for domain-specific Catalan↔Basque translation across general, literary, and clinical domains using LoRA and synthetic data augmentation.

## Overview

This project adapts a multilingual vision-language model for low-resource Catalan–Basque translation through parameter-efficient fine-tuning. The pipeline processes three distinct domains: general (AINA parallel corpus), literary (CTILC + EhuHac), and clinical (domain-specific EU texts). Backtranslation augments training data in literary and clinical domains where parallel data is scarce. Final models are evaluated with BLEU, CHRF, and COMET metrics.

