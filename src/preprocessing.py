"""Text cleaning and prompt construction.

The dataset mixes 8 language/region *subsets*. mT5 has no explicit language
tokens, so we inject a lightweight language tag into the source text - this is
one of the experiment levers (see EXPERIMENTS.md, Exp 3).
"""
from __future__ import annotations

import re
import unicodedata

# Subset code -> (language, country). Used to build readable language tags.
SUBSET_INFO: dict[str, tuple[str, str]] = {
    "Eng_Uga": ("English", "Uganda"),
    "Aka_Gha": ("Akan", "Ghana"),
    "Eng_Gha": ("English", "Ghana"),
    "Eng_Eth": ("English", "Ethiopia"),
    "Lug_Uga": ("Luganda", "Uganda"),
    "Eng_Ken": ("English", "Kenya"),
    "Swa_Ken": ("Swahili", "Kenya"),
    "Amh_Eth": ("Amharic", "Ethiopia"),
}

_WS = re.compile(r"[ \t ]+")
_MULTINL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Conservative cleaning: normalize unicode + whitespace only.

    We deliberately do NOT strip punctuation, lowercase, or touch diacritics -
    Akan/Amharic/Luganda meaning depends on them, and ROUGE is computed on the
    raw target. Over-cleaning here measurably hurt val ROUGE (Exp 2).
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def language_tag(subset: str) -> str:
    lang, country = SUBSET_INFO.get(subset, (subset, ""))
    return f"{lang} ({country})" if country else lang


def build_source(question: str, subset: str, *, style: str = "tagged",
                 use_language_tag: bool = True, clean: bool = True) -> str:
    """Construct the model input string from a question + subset.

    styles
    ------
    plain    : just the (cleaned) question.
    tagged   : compact tag prefix - '<Aka_Gha> question: ...'  (default).
    instruct : natural-language instruction, useful for decoder LLMs.
    """
    q = clean_text(question) if clean else str(question)
    if style == "plain":
        return q
    if style == "instruct":
        lang = language_tag(subset)
        return (f"You are a careful health information assistant. Answer the "
                f"following health question in {lang}. Be accurate, clear, and "
                f"concise.\nQuestion: {q}\nAnswer:")
    # 'tagged' (default)
    if use_language_tag:
        return f"<{subset}> answer health question: {q}"
    return f"answer health question: {q}"


def build_target(answer: str, *, clean: bool = True) -> str:
    return clean_text(answer) if clean else str(answer)
