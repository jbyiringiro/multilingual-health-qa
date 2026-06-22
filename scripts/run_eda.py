"""Exploratory Data Analysis: generates the EDA figures used in the report.

Produces report/figures/eda_*.png (subset distribution, answer lengths, question vs
answer length) and prints the key numbers behind the preprocessing decisions. No GPU needed.
Run:  python scripts/run_eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("report/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})

SUBSET_INFO = {"Eng_Uga": ("English", "Uganda"), "Aka_Gha": ("Akan", "Ghana"),
               "Eng_Gha": ("English", "Ghana"), "Eng_Eth": ("English", "Ethiopia"),
               "Lug_Uga": ("Luganda", "Uganda"), "Eng_Ken": ("English", "Kenya"),
               "Swa_Ken": ("Swahili", "Kenya"), "Amh_Eth": ("Amharic", "Ethiopia")}


def main():
    tr = pd.read_csv("data/Train.csv"); va = pd.read_csv("data/Val.csv"); te = pd.read_csv("data/Test.csv")
    tr["in_chars"] = tr["input"].str.len(); tr["out_chars"] = tr["output"].astype(str).str.len()
    tr["out_words"] = tr["output"].astype(str).str.split().str.len()

    # 1) subset distribution
    dist = pd.concat([tr.subset.value_counts().rename("train"),
                      va.subset.value_counts().rename("val"),
                      te.subset.value_counts().rename("test")], axis=1).fillna(0).astype(int)
    dist = dist.loc[dist.train.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 4))
    dist.plot(kind="bar", ax=ax)
    ax.set_title("Examples per language/region subset"); ax.set_ylabel("count")
    ax.set_xticklabels([f"{s}\n({SUBSET_INFO[s][0]})" for s in dist.index], rotation=0, fontsize=8)
    fig.savefig(FIG / "eda_subset_distribution.png", bbox_inches="tight"); plt.close(fig)

    # 2) answer length distribution + by subset
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].hist(tr.out_chars.clip(upper=2000), bins=50, color="#4c72b0")
    ax[0].axvline(tr.out_chars.median(), color="red", ls="--", label=f"median {tr.out_chars.median():.0f}")
    ax[0].set_title("Answer length (characters)"); ax[0].set_xlabel("chars"); ax[0].legend()
    order = tr.groupby("subset").out_chars.median().sort_values().index
    ax[1].boxplot([tr[tr.subset == s].out_chars.clip(upper=2000) for s in order], labels=list(order),
                  showfliers=False)
    ax[1].set_title("Answer length by subset"); ax[1].tick_params(axis="x", rotation=45, labelsize=8)
    fig.savefig(FIG / "eda_answer_lengths.png", bbox_inches="tight"); plt.close(fig)

    # 3) input vs output length
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(tr.in_chars, tr.out_chars.clip(upper=2500), s=4, alpha=.15, color="#55a868")
    ax.set_xlabel("question length (chars)"); ax.set_ylabel("answer length (chars)")
    ax.set_title("Short questions -> long answers\n(answers ~5x longer on average)")
    fig.savefig(FIG / "eda_input_output_lengths.png", bbox_inches="tight"); plt.close(fig)

    # numbers for the findings doc
    dup_q = int(tr.input.duplicated().sum())
    dup_pair = int(tr.duplicated(["input", "output"]).sum())
    overlap = len(set(tr.input) & set(te.input))
    pcts = tr.out_chars.quantile([.5, .9, .95, .99]).round(0).astype(int).to_dict()

    print("wrote EDA figures to report/figures/")
    print(f"  imbalance: Eng_Uga={dist.loc['Eng_Uga','train']}, Amh_Eth={dist.loc['Amh_Eth','train']}")
    print(f"  answer median/p99 chars: {pcts[0.5]}/{pcts[0.99]} | dup_q={dup_q} dup_pair={dup_pair} overlap={overlap}")


if __name__ == "__main__":
    main()
