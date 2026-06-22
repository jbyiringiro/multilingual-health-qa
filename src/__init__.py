"""Multilingual Health QA - source package.

Modules
-------
config        : YAML-backed experiment configuration.
utils         : seeding, logging, experiment registry.
preprocessing : text cleaning + prompt construction (language-aware).
data          : load CSVs, build HF datasets, tokenize.
metrics       : multilingual-friendly ROUGE-1 / ROUGE-L F1.
train         : Seq2Seq fine-tuning entry point.
predict       : inference + Zindi submission generation.
"""
__all__ = ["config", "utils", "preprocessing", "data", "metrics", "train", "predict"]
