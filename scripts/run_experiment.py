"""Run a training experiment from a config, with optional overrides.

Thin CLI around src.train so we can cap data / epochs / lengths for fast local
iteration without editing the canonical config files. Every override is printed
and saved into the model's config.yaml, so the run stays reproducible.

Examples:
    python scripts/run_experiment.py --config configs/mt5_small_tagged.yaml
    python scripts/run_experiment.py --config configs/mt5_small_tagged.yaml \
        --max-train 10000 --epochs 3 --src-len 256 --tgt-len 256 --name mt5_small_tagged_fast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import Config          # noqa: E402
from src.train import train            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="override experiment name / output dir")
    ap.add_argument("--model-name", default=None, help="override base model (e.g. warm-start from a local dir)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--src-len", type=int, default=None)
    ap.add_argument("--tgt-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--beams", type=int, default=None)
    ap.add_argument("--no-submission", action="store_true")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip per-epoch generation eval (robust+fast on Windows/8GB)")
    ap.add_argument("--balance", action="store_true",
                    help="oversample minority subsets to balance_target each (fairness experiment)")
    a = ap.parse_args()

    cfg = Config.from_yaml(a.config)
    over = {}
    if a.name: cfg.name = a.name; over["name"] = a.name
    if a.model_name: cfg.model_name = a.model_name; over["model_name"] = a.model_name
    if a.balance: cfg.balance_subsets = True; over["balance_subsets"] = True
    if a.max_train is not None: cfg.max_train_samples = a.max_train; over["max_train_samples"] = a.max_train
    if a.max_eval is not None: cfg.max_eval_samples = a.max_eval; over["max_eval_samples"] = a.max_eval
    if a.epochs is not None: cfg.epochs = a.epochs; over["epochs"] = a.epochs
    if a.src_len is not None: cfg.max_source_len = a.src_len; over["max_source_len"] = a.src_len
    if a.tgt_len is not None:
        cfg.max_target_len = a.tgt_len; cfg.gen_max_new_tokens = a.tgt_len
        over["max_target_len"] = a.tgt_len
    if a.batch is not None: cfg.train_batch_size = a.batch; cfg.eval_batch_size = a.batch; over["batch"] = a.batch
    if a.grad_accum is not None: cfg.grad_accum = a.grad_accum; over["grad_accum"] = a.grad_accum
    if a.beams is not None: cfg.num_beams = a.beams; over["num_beams"] = a.beams

    print(f"[run] config={a.config} overrides={over} no_eval={a.no_eval}")
    train(cfg, write_submission=not a.no_submission, eval_during_training=not a.no_eval)


if __name__ == "__main__":
    main()
