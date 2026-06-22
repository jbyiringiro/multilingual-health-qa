"""Reproducibility helpers and a lightweight experiment registry.

The registry appends one row per training/inference run to
``experiments/results.csv`` so the report's experiment table and the
leaderboard-progression plot can be regenerated from a single source of truth.
"""
from __future__ import annotations

import csv
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_CSV = Path("experiments/results.csv")
RESULTS_FIELDS = [
    "timestamp", "experiment", "model_name", "prompt_style", "seed",
    "epochs", "lr", "max_source_len", "max_target_len", "num_beams",
    "val_rouge1_f1", "val_rougeL_f1", "val_mean_f1",
    "public_lb", "private_lb", "notes",
]


def set_seed(seed: int = 42) -> None:
    """Seed every RNG we touch. The competition requires deterministic reruns."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "healthqa") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def log_experiment(row: dict[str, Any]) -> None:
    """Append a result row, creating the CSV with a header if needed."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {**{k: "" for k in RESULTS_FIELDS}, **row}
    row["timestamp"] = row.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
