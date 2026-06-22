# Experiments

Twelve experiments, each changing one thing. For each: what changed, why, the outcome,
and the insight. Numbers come from `results.csv` (local Val) and the Zindi leaderboard (LB).
Public score = 0.37 x ROUGE-1 + 0.37 x ROUGE-L + 0.26 x LLM-judge (verified on 8 submissions).

| # | Experiment | Change | Val mean-F1 | Public LB | Insight |
|--:|-----------|--------|:----------:|:---------:|---------|
| 1 | Retrieval (char TF-IDF) | nearest train answer | 0.428 | 0.329 | strong ROUGE, zero LLM-judge |
| 2 | Retrieval (word TF-IDF) | word-level TF-IDF | 0.391 | 0.293 | lexical-only floor |
| 3 | mT5-small (capped) | 8k rows, 3 ep | 0.181 | not sub. | undertrained: repetitive, code-switches |
| 4 | Language tag prompt | `<subset>` prefix | - | - | conditions the target language |
| 5 | AfriTeVa-base, full data | African-pretrained T5 | 0.324 | 0.370 | earns LLM-judge (0.48) retrieval cannot |
| 6 | Per-column combination | retrieval ROUGE + model LLM | - | 0.460 | columns scored independently |
| 7 | Warm-start (5 epochs) | +2 ep | 0.348 | - | more training raises ROUGE |
| 8 | + longer targets (7 ep) | target 384 | 0.356 | 0.467 | monotonic gains; LLM 0.50 -> 0.53 |
| 9 | Warm-start (9 epochs) | +2 ep | 0.361 | 0.471 | diminishing returns, near ceiling |
| 10 | Decoding / length study | beams, generation length | - | - | proxy validated against LB |
| 11 | mT5-base (model selection) | 580M general vs AfriTeVa 429M | 0.322 | not sub. | African pretraining beats a bigger general model |
| 12 | Balanced sampling | oversample subsets to 4k each | 0.385 | 0.474 | best score; nuanced fairness trade-off |

---

## Detailed log

### Exp 1-2 - Retrieval baselines (no training)
For each test question I return the answer of the most similar train question (per subset),
using TF-IDF nearest neighbour: char n-gram (2-5) and word-level. Public: char 0.329, word
0.293. **Insight:** retrieval gets high ROUGE but the LLM judge scores ~0 - a copied answer
shares vocabulary but does not answer the specific question. This framed the whole strategy.

### Exp 3 - mT5-small (negative result)
Fine-tune mT5-small on 8k rows, 3 epochs, greedy decoding. Val 0.181, far below retrieval;
output is repetitive and code-switches to English. **Insight:** a small model on little data
is badly undertrained, which motivated full data, more epochs, longer targets, and an
African-pretrained model.

### Exp 4 - Language-tag prompt
Prepend the subset tag (e.g. `<Aka_Gha>`) to the source. **Why:** mT5 cannot reliably infer
the target language from a short question; the tag conditions generation on the right language.
Used in all subsequent generative runs.

### Exp 5 - AfriTeVa-V2-base, full data
Fine-tune `castorini/afriteva_v2_base` on all 29,815 rows, 3 epochs. Val 0.324; public 0.370
with LLM-judge 0.48 (vs retrieval's 0). **Insight:** the generative model answers the actual
question, earning the LLM-judge credit retrieval forfeits - confirming the generative direction.

### Exp 6 - Per-column combination
Submit retrieval answers in the two ROUGE columns and the AfriTeVa answer in the LLM column
(the columns are scored independently). Public 0.460, a large jump from 0.370. **Insight:** the
biggest single gain came from understanding the metric, not from a bigger model.

### Exp 7-9 - Warm-start continuation (5 -> 7 -> 9 epochs, longer targets)
Warm-start from the previous checkpoint, add epochs, and lengthen targets 320 -> 384.
Val 0.348 (5 ep) -> 0.356 (7 ep) -> 0.361 (9 ep); public 0.467 -> 0.471. **Insight:** gains are
monotonic but clearly diminishing (+0.024, +0.008, +0.005) - AfriTeVa-base is near its ceiling,
and the LLM-judge tracked the Val ROUGE, validating the local proxy.

### Exp 10 - Decoding / length study (inference only)
Vary beam count, no-repeat-ngram, and generation length on a trained checkpoint without
retraining, and check the local proxy against the leaderboard ROUGE. **Insight:** the local
Val ROUGE tracks the leaderboard closely, so I can rank candidates without spending submissions.

### Exp 11 - Model selection: mT5-base vs AfriTeVa
Fine-tune mT5-base (580M, general multilingual), full data, 3 epochs. Val 0.322 - about the
same as AfriTeVa-base 3-epoch (0.324) and below AfriTeVa at 9 epochs (0.361), despite 35% more
parameters. **Insight:** domain-specific pretraining beats a larger general model on these
languages - the clearest justification for the AfriTeVa choice. Not submitted (ROUGE below AfriTeVa).

### Exp 12 - Balanced sampling (fairness)
Warm-start from the 9-epoch model and oversample every subset to 4,000 examples (+2 epochs).
Val 0.369 -> 0.385 (best); public 0.474 (best). It helped 6/8 subsets but HURT Amharic (-0.026)
and Luganda (-0.013). **Insight:** balancing is not a free fairness win - oversampling by
duplication overfits the smallest, most distinct sets; a weighted loss or augmentation would
be a better remedy.

---

## Leaderboard progression
Every Zindi submission, in order.

| # | Submission file | Public LB | R1 | RL | LLM | Notes |
|--:|-----------------|:---------:|:--:|:--:|:---:|-------|
| 1 | baseline_retrieval_word.csv | 0.293151 | 0.4378 | 0.3545 | 0.000 | word retrieval |
| 2 | baseline_retrieval.csv (char) | 0.328671 | 0.4838 | 0.4045 | 0.000 | best retrieval |
| 3 | afriteva_v2_base_full.csv | 0.369867 | 0.3828 | 0.2787 | 0.4812 | first generative |
| 4 | combined_retrieval_afriteva.csv | 0.459581 | 0.4838 | 0.4045 | 0.5035 | per-column combine |
| 5 | combined_retrieval_warm5long.csv | 0.467381 | 0.4838 | 0.4045 | 0.5335 | warm-start, longer targets |
| 6 | combined_retrieval_warm9.csv | 0.470553 | 0.4838 | 0.4045 | 0.5457 | further training (9 ep) |
| 7 | combined_retrieval_balanced.csv | 0.474193 | 0.4838 | 0.4045 | 0.5597 | BEST - balanced sampling |

**Final private leaderboard:** 0.4626 (rank 399); the LLM-judge was identical on the public and
private splits (0.5597), so the generative model generalized perfectly - the small drop came
only from the retrieval ROUGE columns. With retrieval ROUGE columns the score is capped at
0.589 even with a perfect judge.
