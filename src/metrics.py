"""Multilingual-friendly ROUGE-1 / ROUGE-L F1.

The Zindi leaderboard scores ROUGE-1 F1, ROUGE-L F1 and an LLM judge. We
reproduce the two ROUGE metrics locally so we can iterate without spending a
submission. Two deliberate choices make ROUGE behave sensibly across Akan,
Amharic, Luganda, Swahili and English:

  * use_stemmer=False  - the Porter stemmer is English-only and would mangle
    non-English tokens.
  * a plain unicode-aware word tokenizer (split on non-word chars, keep the
    Amharic/Latin letters) instead of rouge_score's default English tokenizer.

These approximate the official metric closely enough to rank experiments; the
leaderboard remains the final arbiter.
"""
from __future__ import annotations

import re
from typing import Sequence

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

_TOKEN = re.compile(r"\w+", re.UNICODE)


class _UnicodeWordTokenizer(Tokenizer):
    """Lowercase + split on unicode word boundaries (script-agnostic)."""

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN.findall(text.lower())


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rougeL"], use_stemmer=False, tokenizer=_UnicodeWordTokenizer()
)


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    s = _SCORER.score(reference or "", prediction or "")
    return {"rouge1_f1": s["rouge1"].fmeasure, "rougeL_f1": s["rougeL"].fmeasure}


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Corpus-level mean ROUGE F1s, plus the mean of the two (our model-selection metric)."""
    assert len(predictions) == len(references), "preds/refs length mismatch"
    r1 = r_l = 0.0
    n = max(len(predictions), 1)
    for p, r in zip(predictions, references):
        s = score_pair(p, r)
        r1 += s["rouge1_f1"]
        r_l += s["rougeL_f1"]
    r1, r_l = r1 / n, r_l / n
    return {"rouge1_f1": r1, "rougeL_f1": r_l, "mean_f1": (r1 + r_l) / 2}
