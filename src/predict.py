"""Inference + Zindi submission generation.

The submission has 4 columns: ID, TargetRLF1, TargetR1F1, TargetLLM. We submit
the *same* generated answer into all three target columns - the leaderboard
scores that one text against ROUGE-L, ROUGE-1, and an LLM judge respectively.

Usage:
    python -m src.predict --config configs/mt5_small_baseline.yaml
    python -m src.predict --model-dir outputs/models/foo --name foo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .config import Config
from .utils import get_logger, set_seed
from .preprocessing import build_source

log = get_logger()
TARGET_COLS = ["TargetRLF1", "TargetR1F1", "TargetLLM"]


def _batch(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def generate_predictions(cfg: Config, model_dir: str | None = None) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    set_seed(cfg.seed)
    src_dir = model_dir or cfg.model_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[{cfg.name}] loading model from {src_dir} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(src_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(src_dir).to(device).eval()
    if cfg.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)

    test = pd.read_csv(Path(cfg.data_dir) / cfg.test_file)
    sources = [build_source(q, s, style=cfg.prompt_style,
                            use_language_tag=cfg.use_language_tag, clean=cfg.clean_text)
               for q, s in zip(test["input"], test["subset"])]

    preds: list[str] = []
    for chunk in tqdm(list(_batch(sources, cfg.eval_batch_size)), desc="generating"):
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                        max_length=cfg.max_source_len).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=cfg.gen_max_new_tokens,
                num_beams=cfg.num_beams, no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                length_penalty=cfg.length_penalty, do_sample=cfg.do_sample,
            )
        preds.extend(t.replace("<extra_id_0>", "").strip()
                     for t in tokenizer.batch_decode(out, skip_special_tokens=True))

    return pd.DataFrame({"ID": test["ID"], "pred": preds})


def generate_submission(cfg: Config, model_dir: str | None = None,
                        out_path: str | None = None) -> str:
    df = generate_predictions(cfg, model_dir=model_dir)
    # Empty generations would score 0 and can crash some graders - backstop them.
    df["pred"] = df["pred"].fillna("").map(lambda t: t if t.strip() else "No information available.")
    sub = pd.DataFrame({"ID": df["ID"]})
    for c in TARGET_COLS:
        sub[c] = df["pred"]

    out_path = out_path or cfg.submission_path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"[{cfg.name}] wrote submission -> {out_path}  ({len(sub)} rows)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=None, help="generation batch size (bigger=faster)")
    ap.add_argument("--beams", type=int, default=None)
    ap.add_argument("--name", default=None, help="override experiment name (output paths)")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.name:
        cfg.name = args.name
    if args.batch:
        cfg.eval_batch_size = args.batch
    if args.beams:
        cfg.num_beams = args.beams
    generate_submission(cfg, model_dir=args.model_dir, out_path=args.out)


if __name__ == "__main__":
    main()
