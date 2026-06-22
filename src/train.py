"""Seq2Seq fine-tuning entry point.

Usage:
    python -m src.train --config configs/mt5_small_baseline.yaml

Trains an encoder-decoder model on the health-QA data, selects the best
checkpoint by mean ROUGE F1 on the validation set, logs the result to
experiments/results.csv, and (optionally) writes a test submission.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .utils import set_seed, get_logger, log_experiment
from .data import build_datasets, tokenize_datasets
from .metrics import compute_rouge

log = get_logger()


def _strip_special(text: str) -> str:
    # mT5 sometimes emits a leading <extra_id_*> or stray pad markers; drop them.
    return text.replace("<extra_id_0>", "").strip()


def train(cfg: Config, write_submission: bool = True, eval_during_training: bool = True):
    """Fine-tune a seq2seq model.

    eval_during_training=False skips the per-epoch generation evaluation. On
    Windows/8GB this generation eval can deadlock and is slow; disabling it makes
    training robust and faster. Validation ROUGE is then computed separately
    (e.g. scripts/eval_submission.py) instead of inside the loop.
    """
    import numpy as np
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback,
    )

    set_seed(cfg.seed)
    # Optional hard cap on the share of GPU memory this process may use, so the
    # rest stays free for other apps on a shared GPU. Set e.g. TRAIN_GPU_MEM_FRACTION=0.62.
    import os
    _frac = os.environ.get("TRAIN_GPU_MEM_FRACTION")
    if _frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
        log.info(f"[{cfg.name}] capped GPU memory to {float(_frac):.0%} of device")

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    cfg.save(Path(cfg.model_dir) / "config.yaml")

    log.info(f"[{cfg.name}] loading tokenizer + model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = build_datasets(cfg)
    tok = tokenize_datasets(ds, tokenizer, cfg)
    eval_ds = tok["val"]
    if cfg.max_eval_samples and len(eval_ds) > cfg.max_eval_samples:
        eval_ds = eval_ds.select(range(cfg.max_eval_samples))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest",
                                      label_pad_token_id=-100)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dpred = [_strip_special(t) for t in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        dref = tokenizer.batch_decode(labels, skip_special_tokens=True)
        m = compute_rouge(dpred, dref)
        return {"rouge1_f1": m["rouge1_f1"], "rougeL_f1": m["rougeL_f1"], "mean_f1": m["mean_f1"]}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.model_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing,
        bf16=cfg.bf16, fp16=cfg.fp16,
        optim=cfg.optim,  # "adafactor" is a valid optim value; preferred for mT5 stability
        predict_with_generate=eval_during_training,
        generation_max_length=cfg.gen_max_new_tokens,
        generation_num_beams=cfg.num_beams,
        eval_strategy="epoch" if eval_during_training else "no",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=eval_during_training,
        metric_for_best_model="mean_f1",
        greater_is_better=True,
        dataloader_num_workers=0,   # avoid Windows dataloader-worker deadlocks
        report_to="none",
        seed=cfg.seed,
    )

    # transformers renamed `tokenizer` -> `processing_class` in v5. Stay compatible
    # with both (Colab may ship 4.x, local may ship 5.x).
    import inspect
    tok_kw = ("processing_class" if "processing_class"
              in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer")
    callbacks = ([EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
                 if eval_during_training else [])
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tok["train"],
        eval_dataset=eval_ds if eval_during_training else None,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_during_training else None,
        callbacks=callbacks,
        **{tok_kw: tokenizer},
    )

    log.info(f"[{cfg.name}] training on {len(tok['train'])} examples "
             f"(eval_during_training={eval_during_training})")
    trainer.train()
    metrics = trainer.evaluate() if eval_during_training else {}
    if metrics:
        log.info(f"[{cfg.name}] best val metrics: {metrics}")

    trainer.save_model(cfg.model_dir)
    tokenizer.save_pretrained(cfg.model_dir)

    log_experiment({
        "experiment": cfg.name, "model_name": cfg.model_name,
        "prompt_style": cfg.prompt_style, "seed": cfg.seed,
        "epochs": cfg.epochs, "lr": cfg.lr,
        "max_source_len": cfg.max_source_len, "max_target_len": cfg.max_target_len,
        "num_beams": cfg.num_beams,
        "val_rouge1_f1": round(metrics.get("eval_rouge1_f1", 0), 4),
        "val_rougeL_f1": round(metrics.get("eval_rougeL_f1", 0), 4),
        "val_mean_f1": round(metrics.get("eval_mean_f1", 0), 4),
        "notes": cfg.notes,
    })

    if write_submission:
        from .predict import generate_submission
        generate_submission(cfg)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-submission", action="store_true")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    train(cfg, write_submission=not args.no_submission)


if __name__ == "__main__":
    main()
