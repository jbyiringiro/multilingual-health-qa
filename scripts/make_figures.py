"""Generate the report's result figures from real submission data.

All leaderboard numbers below are the verified public-score breakdowns recorded
from the Zindi leaderboard (see experiments/results.csv and the report). The
public score is exactly  0.37*ROUGE1 + 0.37*ROUGE-L + 0.26*LLM-judge.

Outputs PNGs to report/figures/ and a tidy experiment table CSV.
Run:  python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("report/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 11})

W_R1, W_RL, W_LLM = 0.37, 0.37, 0.26  # verified public-score weights

# --- Submitted entries, in chronological order (from leaderboard screenshots) ---
SUBS = pd.DataFrame([
    # label,                    R1,     RL,     LLM,    public
    ("Retrieval (word)",       0.4378, 0.3545, 0.0000, 0.293151),
    ("Retrieval (char)",       0.4838, 0.4045, 0.0000, 0.328671),
    ("AfriTeVa (3ep)",         0.3828, 0.2787, 0.4812, 0.369867),
    ("Combined\n(retr+AfriTeVa)", 0.4838, 0.4045, 0.5035, 0.459581),
    ("Combined\n(retr+warm5_long)", 0.4838, 0.4045, 0.5335, 0.467381),
    ("Combined\n(retr+warm9)", 0.4838, 0.4045, 0.5457, 0.470553),
    ("Combined\n(retr+balanced)", 0.4838, 0.4045, 0.5597, 0.474193),
], columns=["label", "R1", "RL", "LLM", "public"])

LEADER = 0.728509  # current #1


def fig_progression():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(SUBS))
    ax.plot(x, SUBS["public"], "o-", lw=2.2, ms=8, color="#1f77b4", label="Our public score")
    for i, (s, p) in enumerate(zip(SUBS["label"], SUBS["public"])):
        ax.annotate(f"{p:.3f}", (i, p), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(LEADER, ls="--", color="#d62728", alpha=.7, label=f"Leaderboard #1 ({LEADER:.3f})")
    ax.axhline(0.589, ls=":", color="grey", alpha=.8, label="Retrieval-ROUGE ceiling (0.589)")
    ax.set_xticks(list(x)); ax.set_xticklabels(SUBS["label"], fontsize=8.5)
    ax.set_ylabel("Public leaderboard score"); ax.set_ylim(0.25, 0.78)
    ax.set_title("Leaderboard progression across submissions")
    ax.legend(loc="center right", fontsize=9); ax.grid(alpha=.3)
    fig.savefig(FIG / "leaderboard_progression.png"); plt.close(fig)


def fig_decomposition():
    """Stacked bar: weighted contribution of each metric to the public score."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    c_r1 = W_R1 * SUBS["R1"]; c_rl = W_RL * SUBS["RL"]; c_llm = W_LLM * SUBS["LLM"]
    x = range(len(SUBS))
    ax.bar(x, c_r1, label=f"0.37 x ROUGE-1", color="#4c72b0")
    ax.bar(x, c_rl, bottom=c_r1, label="0.37 x ROUGE-L", color="#55a868")
    ax.bar(x, c_llm, bottom=c_r1 + c_rl, label="0.26 x LLM-judge", color="#c44e52")
    for i, p in enumerate(SUBS["public"]):
        ax.annotate(f"{p:.3f}", (i, p), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(SUBS["label"], fontsize=8.5)
    ax.set_ylabel("Contribution to public score")
    ax.set_title("Score decomposition: how each metric contributes\n"
                 "(the 'combined' jump = retrieval ROUGE + model LLM in separate columns)")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.savefig(FIG / "score_decomposition.png"); plt.close(fig)


def fig_proxy_validation():
    """Local Val ROUGE vs leaderboard ROUGE (same-text submissions) -> proxy is trustworthy."""
    pts = pd.DataFrame([
        ("char R1", 0.4622, 0.4838), ("char RL", 0.3934, 0.4045),
        ("AfriTeVa R1", 0.3637, 0.3828), ("AfriTeVa RL", 0.2847, 0.2787),
    ], columns=["metric", "local", "lb"])
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0.2, 0.55], [0.2, 0.55], "--", color="grey", alpha=.7, label="y = x")
    ax.scatter(pts["local"], pts["lb"], s=90, color="#1f77b4", zorder=3)
    for _, r in pts.iterrows():
        ax.annotate(r["metric"], (r["local"], r["lb"]), textcoords="offset points",
                    xytext=(6, -2), fontsize=9)
    ax.set_xlabel("Local Val ROUGE (our proxy)"); ax.set_ylabel("Leaderboard ROUGE")
    ax.set_title("Local proxy tracks the leaderboard\n(points hug the diagonal)")
    ax.legend(); ax.grid(alpha=.3); ax.set_xlim(0.2, 0.55); ax.set_ylim(0.2, 0.55)
    fig.savefig(FIG / "proxy_validation.png"); plt.close(fig)


def fig_per_subset():
    """Per-subset Val mean-F1 for the retrieval baseline (the weak-language story)."""
    data = pd.DataFrame([
        ("Eng_Ken", 0.608), ("Swa_Ken", 0.603), ("Eng_Uga", 0.543), ("Eng_Eth", 0.538),
        ("Lug_Uga", 0.530), ("Aka_Gha", 0.258), ("Eng_Gha", 0.252), ("Amh_Eth", 0.164),
    ], columns=["subset", "mean_f1"]).sort_values("mean_f1")
    colors = ["#c44e52" if v < 0.30 else "#4c72b0" for v in data["mean_f1"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(data["subset"], data["mean_f1"], color=colors)
    for i, v in enumerate(data["mean_f1"]):
        ax.text(v + .008, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Val mean ROUGE-F1"); ax.set_xlim(0, 0.7)
    ax.set_title("Per-subset performance (retrieval baseline)\n"
                 "Amharic, Akan, Eng-Ghana lag - a bias/fairness concern")
    ax.grid(alpha=.3, axis="x")
    fig.savefig(FIG / "per_subset_rouge.png"); plt.close(fig)


def fig_learning_curve():
    """Val mean-F1 vs cumulative training epochs (AfriTeVa-base, warm-started)."""
    pts = [(3, 0.324), (5, 0.348), (7, 0.356), (9, 0.361)]
    ep = [p[0] for p in pts]; val = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, val, "o-", lw=2.3, ms=9, color="#4c72b0")
    for e, v in pts:
        ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, fontweight="bold")
    # label the diminishing deltas between points
    for (e0, v0), (e1, v1) in zip(pts, pts[1:]):
        ax.annotate(f"+{v1 - v0:.3f}", ((e0 + e1) / 2, (v0 + v1) / 2 - 0.006),
                    ha="center", fontsize=8, color="#c44e52")
    ax.set_xlabel("Cumulative training epochs (AfriTeVa-base, warm-started)")
    ax.set_ylabel("Val mean ROUGE-F1")
    ax.set_title("Learning curve: Val ROUGE vs training epochs\n"
                 "Gains diminish (+0.024, +0.008, +0.005) - approaching the model's ceiling")
    ax.set_xticks(ep); ax.grid(alpha=.3); ax.set_ylim(0.31, 0.375)
    fig.savefig(FIG / "learning_curve.png"); plt.close(fig)


def experiment_table():
    df = pd.read_csv("experiments/results.csv")
    cols = ["experiment", "model_name", "epochs", "val_mean_f1", "public_lb", "notes"]
    df[cols].to_csv(FIG / "experiment_table.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    fig_progression(); fig_decomposition(); fig_proxy_validation(); fig_per_subset()
    fig_learning_curve(); experiment_table()
    print("wrote figures to", FIG, ":")
    for p in sorted(FIG.glob("*")):
        print("  ", p.name)
