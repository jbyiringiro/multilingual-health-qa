"""Data loading, prompt building, and tokenization.

Keeps a clean separation:
  * load_frames()      -> raw pandas DataFrames (used by EDA too)
  * build_datasets()   -> HF DatasetDict with 'source'/'target' text columns
  * tokenize_datasets()-> adds input_ids / labels for the Trainer
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .preprocessing import build_source, build_target


def load_frames(cfg: Config) -> dict[str, pd.DataFrame]:
    d = Path(cfg.data_dir)
    frames = {
        "train": pd.read_csv(d / cfg.train_file),
        "val": pd.read_csv(d / cfg.val_file),
        "test": pd.read_csv(d / cfg.test_file),
    }
    # Guard against missing answers (none observed, but keep training robust).
    for split in ("train", "val"):
        frames[split] = frames[split].dropna(subset=["output"]).reset_index(drop=True)
    return frames


def _to_examples(df: pd.DataFrame, cfg: Config, with_target: bool) -> dict[str, list]:
    sources = [
        build_source(q, s, style=cfg.prompt_style,
                     use_language_tag=cfg.use_language_tag, clean=cfg.clean_text)
        for q, s in zip(df["input"], df["subset"])
    ]
    out = {"id": df["ID"].tolist(), "subset": df["subset"].tolist(), "source": sources}
    if with_target:
        out["target"] = [build_target(a, clean=cfg.clean_text) for a in df["output"]]
    return out


def build_datasets(cfg: Config):
    """Return a DatasetDict with text columns. Import datasets lazily."""
    from datasets import Dataset, DatasetDict

    frames = load_frames(cfg)
    if cfg.balance_subsets:
        # Oversample minority subsets (and cap majority) to balance_target each, so
        # low-resource languages (Amharic, Akan, ...) get equal training exposure.
        parts = []
        for subset, g in frames["train"].groupby("subset"):
            replace = len(g) < cfg.balance_target
            parts.append(g.sample(n=cfg.balance_target, replace=replace, random_state=cfg.seed))
        frames["train"] = pd.concat(parts).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_train_samples:
        frames["train"] = frames["train"].sample(
            n=min(cfg.max_train_samples, len(frames["train"])),
            random_state=cfg.seed).reset_index(drop=True)

    ds = DatasetDict({
        "train": Dataset.from_dict(_to_examples(frames["train"], cfg, True)),
        "val": Dataset.from_dict(_to_examples(frames["val"], cfg, True)),
        "test": Dataset.from_dict(_to_examples(frames["test"], cfg, False)),
    })
    return ds


def tokenize_datasets(ds, tokenizer, cfg: Config):
    def _tok(batch):
        model_inputs = tokenizer(batch["source"], max_length=cfg.max_source_len,
                                 truncation=True)
        if "target" in batch:
            labels = tokenizer(text_target=batch["target"],
                               max_length=cfg.max_target_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keep_cols = ["id", "subset"]
    tokenized = {}
    for split in ds:
        remove = [c for c in ds[split].column_names if c not in keep_cols]
        tokenized[split] = ds[split].map(_tok, batched=True, remove_columns=remove,
                                         desc=f"tokenizing {split}")
    from datasets import DatasetDict
    return DatasetDict(tokenized)
