"""Exp 0 - zero-cost retrieval baseline (no GPU, no training).

For each test question, find the most similar *training* question within the
same subset (character n-gram TF-IDF cosine) and copy its answer. This:
  * gives an honest leaderboard floor before any fine-tuning,
  * validates the submission format end-to-end,
  * is multilingual-robust (char n-grams need no language-specific tokenizer).

Usage:  python scripts/baseline_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import clean_text  # noqa: E402

DATA = Path("data")
OUT = Path("outputs/submissions/baseline_retrieval.csv")
TARGET_COLS = ["TargetRLF1", "TargetR1F1", "TargetLLM"]


def main() -> None:
    train = pd.read_csv(DATA / "Train.csv")
    test = pd.read_csv(DATA / "Test.csv")
    train["q"] = train["input"].map(clean_text)
    test["q"] = test["input"].map(clean_text)

    preds = pd.Series(index=test.index, dtype="object")
    for subset, te in test.groupby("subset"):
        tr = train[train.subset == subset]
        if tr.empty:  # fallback to global pool
            tr = train
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        tr_mat = vec.fit_transform(tr["q"])
        te_mat = vec.transform(te["q"])
        sims = linear_kernel(te_mat, tr_mat)         # cosine (tfidf is L2-normed)
        best = sims.argmax(axis=1)
        preds.loc[te.index] = tr["output"].to_numpy()[best]
        print(f"{subset:9} test={len(te):4d}  matched from train={len(tr)}")

    sub = pd.DataFrame({"ID": test["ID"]})
    for c in TARGET_COLS:
        sub[c] = preds.values
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(sub)} rows)")


if __name__ == "__main__":
    main()
