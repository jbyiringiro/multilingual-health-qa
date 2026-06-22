"""Per-column ('mixture-of-experts') submission.

The Zindi submission has three independently-scored columns:
    TargetR1F1  -> scored by ROUGE-1 F1
    TargetRLF1  -> scored by ROUGE-L F1
    TargetLLM   -> scored by an LLM judge

Different systems win different metrics: TF-IDF retrieval copies real answers
(high lexical overlap -> high ROUGE) but does not address the specific question
(LLM-judge ~0); the fine-tuned AfriTeVa model writes a genuine answer (high
LLM-judge) but lower ROUGE. Since the columns are scored separately, we put the
best system in each column:

    ROUGE columns  <- retrieval baseline
    LLM column     <- generative model

Derived public-score weighting (fit exactly on 3 submissions):
    Public = 0.37*R1 + 0.37*RL + 0.26*LLM

Usage:
    python scripts/make_combined_submission.py \
        --rouge outputs/submissions/baseline_retrieval.csv \
        --llm   outputs/submissions/afriteva_v2_base_full.csv \
        --out   outputs/submissions/combined_retrieval_afriteva.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rouge", required=True, help="submission to use for the two ROUGE columns")
    ap.add_argument("--llm", required=True, help="submission to use for the LLM column")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rouge = pd.read_csv(a.rouge)
    llm = pd.read_csv(a.llm)
    # Align on ID so row order can differ between the two source files.
    m = rouge[["ID", "TargetR1F1", "TargetRLF1"]].merge(
        llm[["ID", "TargetLLM"]], on="ID", how="left", validate="one_to_one")

    assert m["TargetLLM"].notna().all(), "LLM source is missing some test IDs"
    assert len(m) == len(rouge) == len(llm), "row count mismatch between sources"

    out = m[["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8")
    print(f"wrote {a.out}  ({len(out)} rows)")
    print("  ROUGE cols (R1F1, RLF1) <-", Path(a.rouge).name)
    print("  LLM col   (TargetLLM)   <-", Path(a.llm).name)


if __name__ == "__main__":
    main()
