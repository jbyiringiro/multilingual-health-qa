"""Experiment configuration.

A single dataclass holds every knob an experiment can vary. Experiments are
defined as YAML files in ``configs/`` and loaded here, so each leaderboard
submission is reproducible from exactly one config + one seed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # ---- identity ----
    name: str = "baseline_mt5_small"          # experiment id (used for output dirs)
    notes: str = ""                           # short human description for the log

    # ---- data ----
    data_dir: str = "data"
    train_file: str = "Train.csv"
    val_file: str = "Val.csv"
    test_file: str = "Test.csv"
    sample_submission: str = "SampleSubmission.csv"
    max_train_samples: Optional[int] = None   # subsample for fast iteration
    max_eval_samples: Optional[int] = 800     # cap val for speed during training
    balance_subsets: bool = False             # oversample minority subsets to balance_target
    balance_target: int = 4000                # per-subset target size when balancing

    # ---- prompt / preprocessing ----
    prompt_style: str = "tagged"              # 'plain' | 'tagged' | 'instruct'
    use_language_tag: bool = True
    clean_text: bool = True

    # ---- model ----
    model_name: str = "google/mt5-small"
    max_source_len: int = 320
    max_target_len: int = 320

    # ---- training ----
    seed: int = 42
    epochs: float = 3.0
    lr: float = 3e-4
    train_batch_size: int = 4
    eval_batch_size: int = 4
    grad_accum: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "adafactor"                  # mT5 trains best with Adafactor
    early_stopping_patience: int = 2

    # ---- generation / inference ----
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    gen_max_new_tokens: int = 320
    do_sample: bool = False

    # ---- bookkeeping ----
    output_root: str = "outputs"

    # ---- PEFT / QLoRA (only used by the decoder-LLM ceiling experiment) ----
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @property
    def model_dir(self) -> str:
        return str(Path(self.output_root) / "models" / self.name)

    @property
    def submission_path(self) -> str:
        return str(Path(self.output_root) / "submissions" / f"{self.name}.csv")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
