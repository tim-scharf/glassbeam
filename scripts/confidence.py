"""
confidence.py
---------------
Confidence scoring for classify.py's predictions, built from two
independent, explainable signals (no embeddings, matching the rest of
this pipeline):

1. Rule tier -- every extractor's classify_*_detailed() function reports
   which tier of its rule cascade actually fired (e.g. an explicit
   keyword vs. a last-resort weak heuristic, or a unique category match
   vs. two categories' patterns colliding). Each tier maps to a base
   confidence score in TIER_SCORES below.
2. Novelty -- the fraction of a description's alphabetic tokens that
   never appeared anywhere in the training corpus (output/glassbeam_data.csv,
   6,139 unique descriptions) the rules were built/tuned against. High
   novelty means the rules are operating outside their validated
   vocabulary -- exactly the kind of gap the Tanner sample review had to
   surface by hand (MYOCARDIAL, TRANSTHORACIC ECHO, PSMA, ...).

confidence = tier_base_score - NOVELTY_PENALTY_WEIGHT * novelty, clamped to [0, 1].

Usage:
    from confidence import compute_novelty, tier_confidence
    novelty = compute_novelty(study_desc_raw)
    score = tier_confidence("laterality", "bare_letter", novelty)
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = ROOT_DIR / "output" / "glassbeam_data.csv"

_TOKEN_RE = re.compile(r"[A-Za-z]+")

# How much a fully-novel description (novelty=1.0) drags every confidence
# score down, regardless of which tier fired.
NOVELTY_PENALTY_WEIGHT = 0.4

TIER_SCORES = {
    "region_focus": {
        "explicit": 0.90,
        "combo": 0.85,
        "glued_fallback": 0.60,
        "last_resort": 0.35,
        "none": 0.70,
    },
    "laterality": {
        "explicit": 0.92,
        "bare_letter": 0.55,
        "none": 0.85,
    },
    "contrast": {
        "explicit": 0.92,
        "contrast_named_no_timing": 0.70,
        "tail_heuristic": 0.55,
        "none": 0.80,
    },
    "technique_study_type": {
        "unique_match": 0.90,
        "collision": 0.55,
        "none": 0.75,
    },
}


@lru_cache(maxsize=1)
def load_corpus_vocabulary(csv_path=str(DEFAULT_CSV)):
    """Every alphabetic token that appears anywhere in the training corpus
    (cached -- this is a one-time cost per process, not per row)."""
    df = pd.read_csv(csv_path)
    vocab = set()
    for raw in df["study_desc_raw"].dropna():
        vocab.update(tok.upper() for tok in _TOKEN_RE.findall(str(raw)))
    return vocab


def compute_novelty(raw_text, vocabulary=None):
    """Fraction of raw_text's alphabetic tokens never seen in the training corpus.

    0.0 = every token is familiar; 1.0 = none of them are. This is a
    distributional check ("have we ever seen this word at all"), not a
    check against any one model's keyword list -- a description can be
    low-novelty and still get "Unspecified" everywhere (it's just a kind
    of study none of the 4 attributes apply to), or high-novelty and
    still get correctly classified (a keyword rule happened to match
    anyway). It's an independent signal, meant to be combined with tier.
    """
    if vocabulary is None:
        vocabulary = load_corpus_vocabulary()
    if not isinstance(raw_text, str) or not raw_text.strip():
        return 0.0
    tokens = _TOKEN_RE.findall(raw_text.upper())
    if not tokens:
        return 0.0
    unseen = sum(1 for tok in tokens if tok not in vocabulary)
    return unseen / len(tokens)


def tier_confidence(attribute, tier, novelty):
    """Blend a rule-tier's base score with the novelty penalty.

    Returns None if tier is None or unrecognized -- callers should only
    invoke this once they know the attribute was actually applicable to
    this modality (a not-applicable attribute has no tier and no score).
    """
    if tier is None:
        return None
    base = TIER_SCORES.get(attribute, {}).get(tier)
    if base is None:
        return None
    score = base - NOVELTY_PENALTY_WEIGHT * novelty
    return round(max(0.0, min(1.0, score)), 3)
