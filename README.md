# Multilingual Health QA in Low-Resource African Languages

Fine-tuning encoder-decoder models to answer health questions across 8 language/region
subsets - English, Akan, Luganda, Swahili and Amharic across Ghana, Uganda, Kenya and
Ethiopia - for the
[Zindi challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge).

Final course project (ML Techniques I). Final public score **0.474**, private **0.4626
(rank 399)**. The full write-up is in [`report/`](report/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jbyiringiro/multilingual-health-qa/blob/main/notebooks/02_Train.ipynb)
&larr; opens `notebooks/02_Train.ipynb` and runs end-to-end on Colab (see [Run on Colab](#run-on-colab)).

## Task

Given a health **question** (`input`) and its **subset** (language + country), generate an
**answer** (`output`). Each submission has three independently-scored columns - `TargetR1F1`,
`TargetRLF1`, `TargetLLM` - scored by ROUGE-1 F1, ROUGE-L F1, and an LLM-as-judge respectively.

| Split | Rows | Columns |
|------|-----:|---------|
| Train | 29,815 | `ID, input, output, subset` |
| Val   | 6,686  | `ID, input, output, subset` |
| Test  | 2,618  | `ID, input, subset` |

## Approach

Encoder-decoder (seq2seq) fine-tuning, the architecture covered in class and the natural fit
for conditional text generation:

1. **Retrieval baselines** (TF-IDF nearest neighbour) - a strong ROUGE floor.
2. **AfriTeVa-V2-base** - a T5 pretrained on African languages (my primary model), fine-tuned
   with a `<subset>` language-tag prompt, warm-start training, longer targets, and balanced
   sampling.
3. **Per-column submission** - because the three columns are scored independently, I place the
   retrieval answer in the two ROUGE columns and the model's answer in the LLM column.

The public score is exactly `0.37 x ROUGE-1 + 0.37 x ROUGE-L + 0.26 x LLM-judge`. The twelve
experiments, with analysis, are in [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md).

## Results

Public score improved **0.293 -> 0.474** across submissions; final private score **0.4626
(rank 399)** - only 0.012 below public, showing the solution generalized rather than
overfitting the public split.

| Submission | Public | What changed |
|-----------|:------:|--------------|
| Retrieval (word) | 0.293 | TF-IDF nearest answer |
| Retrieval (char) | 0.329 | better lexical match |
| AfriTeVa-base | 0.370 | first generative model (earns LLM-judge) |
| Combined | 0.460 | per-column: retrieval ROUGE + model LLM |
| Combined (warm9) | 0.471 | more training |
| **Combined (balanced)** | **0.474** | balanced subset sampling |

Figures (progression, score decomposition, learning curve, per-subset, leaderboard
screenshots) are in [`report/figures/`](report/figures/).

## Repository layout

```
.
├── src/             # pipeline: config, preprocessing, data, metrics, train, predict, utils
├── configs/         # one YAML per experiment (fully reproducible)
├── scripts/         # retrieval baseline, training runner, submission + figure generation
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Train.ipynb
│   └── 03_FutureWork_AfriTeVa_large_Colab.ipynb
├── experiments/     # EXPERIMENTS.md (12 experiments) + results.csv
├── report/          # REPORT.md, the PDF, and figures/
├── data/            # competition CSVs (download from Zindi; not committed)
└── requirements.txt
```

## Run on Colab

The repository runs end-to-end on Google Colab with minimal setup:

1. Click the **Open in Colab** badge above (opens `notebooks/02_Train.ipynb`).
2. `Runtime -> Change runtime type -> GPU` (T4 or L4).
3. `Runtime -> Run all`.

The first cell clones the repo and installs dependencies automatically (it detects Colab).
The only manual step is the **data cell**: when prompted, upload `Train.csv`, `Val.csv`,
`Test.csv` and `SampleSubmission.csv` (downloaded from the Zindi page; CC-BY-SA, not
committed). The notebook then fine-tunes the model and writes a submission file. For a fast
check, uncomment the smoke-test line to run the full loop on a tiny subsample in ~2 minutes.

## Setup & run (local)

```bash
pip install -r requirements.txt          # on Colab, the notebook does this for you
# (local Windows / RTX 50-series: install torch first from the cu128 index)

# Train an experiment end-to-end (trains, evaluates, logs, writes a submission)
python -m src.train --config configs/afriteva_v2_base.yaml

# Re-generate a submission from a trained model
python -m src.predict --config configs/afriteva_v2_base.yaml
```

Download `Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv` from the Zindi page into
`data/`. They are CC-BY-SA and intentionally not committed.

## Reproducibility

Every experiment is one YAML config + a fixed seed (42), set for Python, NumPy and PyTorch in
`src/utils.set_seed`. The exact config is saved next to each trained model, and
`experiments/results.csv` is the single source of truth for all reported numbers.

## Ethics

A fluent but wrong health answer can cause harm, and the data imbalance means weaker support
for Amharic and Akan. I treat the system as a research prototype, not medical advice; the
report's Ethics section covers misinformation, bias, and deployment considerations.
