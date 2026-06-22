"""Evaluate a trained seq2seq model on a stratified Val sample.

Because training runs with eval_during_training=False (robust on Windows/8GB),
we score the saved model here instead. Generates answers for a per-subset
stratified Val sample and reports overall + macro ROUGE-1/L F1, and per-subset
breakdown - so you know the local score before spending a Zindi submission.

Usage:
    python scripts/eval_model_val.py --config configs/mt5_small_tagged.yaml --n 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import Config              # noqa: E402
from src.metrics import compute_rouge      # noqa: E402
from src.preprocessing import build_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--n", type=int, default=400, help="stratified val sample size")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    cfg = Config.from_yaml(a.config)
    src_dir = a.model_dir or cfg.model_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(src_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(src_dir).to(device).eval()
    if cfg.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)

    val = pd.read_csv(Path(cfg.data_dir) / cfg.val_file).dropna(subset=["output"])
    # stratified sample: proportional per subset, min 5 each
    frac = min(1.0, a.n / len(val))
    sample = (val.groupby("subset", group_keys=False)
              .apply(lambda g: g.sample(max(5, int(len(g) * frac)), random_state=cfg.seed)))
    print(f"evaluating on {len(sample)} val rows ({sample['subset'].nunique()} subsets)")

    preds = []
    bs = cfg.eval_batch_size
    rows = sample.reset_index(drop=True)
    for i in tqdm(range(0, len(rows), bs), desc="generating"):
        chunk = rows.iloc[i:i + bs]
        srcs = [build_source(q, s, style=cfg.prompt_style,
                             use_language_tag=cfg.use_language_tag, clean=cfg.clean_text)
                for q, s in zip(chunk["input"], chunk["subset"])]
        enc = tok(srcs, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_source_len).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=cfg.gen_max_new_tokens,
                                 num_beams=cfg.num_beams,
                                 no_repeat_ngram_size=cfg.no_repeat_ngram_size)
        preds.extend(t.replace("<extra_id_0>", "").strip()
                     for t in tok.batch_decode(out, skip_special_tokens=True))

    rows = rows.assign(pred=preds)
    overall = compute_rouge(rows["pred"].tolist(), rows["output"].tolist())
    print(f"\nOVERALL  R1={overall['rouge1_f1']:.4f}  RL={overall['rougeL_f1']:.4f}  "
          f"mean={overall['mean_f1']:.4f}")
    macro = []
    for s, g in rows.groupby("subset"):
        m = compute_rouge(g["pred"].tolist(), g["output"].tolist())
        macro.append(m["mean_f1"])
        print(f"  {s:9} mean_f1={m['mean_f1']:.4f}  (n={len(g)})")
    print(f"MACRO mean_f1={np.mean(macro):.4f}")
    print("\nSample generation:\n", rows.iloc[0]["pred"][:200])


if __name__ == "__main__":
    main()
