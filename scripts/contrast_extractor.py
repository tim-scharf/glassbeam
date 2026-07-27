"""
contrast_extractor.py
----------------------
Classifies each raw study description into a contrast-timing bucket
(With / Without / With and without / Unspecified) using substitution
rules derived from examining every unique value of study_desc_raw in
output/glassbeam_data.csv.

Approach
========
1. Normalize the raw text: fix the OCR/typing typos and shorthand seen
   in the data (CONRAST, CONTAST, CONTR, ONTRAST, CNTRST, CONSTRAST,
   CONT, CM, DYE -> CONTRAST; WITHOU/WITHUOT/WITHOUG -> WITHOUT;
   "W/O" -> WO; glued "WWO" -> "WO W"; glued "WCONTRAST" -> "W CONTRAST").
2. Search the normalized text for independent WITH-family and
   WITHOUT-family tokens. If both families are present (whether written
   as an explicit combo like "WITH AND WITHOUT" or as two separate
   mentions), the study is "With and without".
3. If only WITHOUT-family tokens are present -> "Without".
   If only WITH-family tokens are present -> "With".
4. The bare tokens "W" / "WO" are only trustworthy as contrast markers
   when either (a) the word CONTRAST (post-normalization) is present
   somewhere in the string, or (b) the modality is MR/CT and the token
   sits in the last 3 words of the string -- this is the convention
   scanner protocol names use (e.g. "MRI BRAIN WITHOUT", "CT LEFT KNEE
   WITHOUT; MAKO", "ABDOMEN^1_ABDPEL_WO_W (ADULT)"). Outside of those
   two cases, bare "W"/"WO" are dropped since they are usually part of
   unrelated words/phrases ("WITH IMPLANTS", "WITHOUT DUPLEX DOPPLER",
   "WITH IMAGING GUIDANCE") rather than contrast timing.
5. Anything left with no contrast signal at all -> "Unspecified".

Usage:
    python3 scripts/contrast_extractor.py
    python3 scripts/contrast_extractor.py --csv output/glassbeam_data.csv --out data/contrast_timing.json
    python3 scripts/contrast_extractor.py --query "CT ABDOMEN PELVIS WITHOUT AND WITH CONTRAST" --modality CT
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = ROOT_DIR / "output" / "glassbeam_data.csv"
DEFAULT_OUT = ROOT_DIR / "data" / "contrast_timing.json"

CATEGORIES = ["With and without", "Without", "With", "Unspecified"]

# Modalities where "contrast" is a defined extractable attribute
# (see MULTI_LABEL_EXTRACTION['contrast'] in ontology_mapper.py).
TAIL_HEURISTIC_MODALITIES = {"MR", "CT"}
TAIL_WINDOW = 3

# ── Substitution rules ──────────────────────────────────────────────────────

# Typos/shorthand -> canonical "CONTRAST". Found by fuzzy-matching every
# alphabetic token in the corpus against "CONTRAST" (CM = contrast media,
# DYE = contrast dye, CONT = truncated CONTRAST).
CONTRAST_WORD_SUBS = {
    "CNTRST": "CONTRAST",
    "CONRAST": "CONTRAST",
    "CONSTRAST": "CONTRAST",
    "CONTAST": "CONTRAST",
    "CONTR": "CONTRAST",
    "CONTRAS": "CONTRAST",
    "ONTRAST": "CONTRAST",
    "CONT": "CONTRAST",
    "CM": "CONTRAST",
    "DYE": "CONTRAST",
}

# Typos of WITHOUT.
WITHOUT_WORD_SUBS = {
    "WITHOU": "WITHOUT",
    "WITHUOT": "WITHOUT",
    "WITHOUG": "WITHOUT",
}

_WORD_SUB_RE = re.compile(
    r"\b(" + "|".join(sorted({**CONTRAST_WORD_SUBS, **WITHOUT_WORD_SUBS}, key=len, reverse=True)) + r")\b"
)
_ALL_WORD_SUBS = {**CONTRAST_WORD_SUBS, **WITHOUT_WORD_SUBS}

_WCONTRAST_RE = re.compile(r"\bWCONTRAST\b")
_SLASH_WO_RE = re.compile(r"\bW\s*/\s*O\b")  # "W/O" -> "WO"
_GLUED_WWO_RE = re.compile(r"\bWWO\b")  # "WWO" -> "WO W" (exposes both tokens)

WITHOUT_FAMILY_RE = re.compile(r"\bWITHOUT\b|\bWO\b")
WITH_FAMILY_RE = re.compile(r"\bWITH\b|\bW\b")
CONTRAST_RE = re.compile(r"\bCONTRAST\b")

_TOKEN_RE = re.compile(r"[A-Za-z]+")


_SEPARATOR_RE = re.compile(r"[_^]")


def normalize(text):
    """Apply substitution rules to canonicalize contrast-related shorthand."""
    t = text.upper()
    # Scanner protocol codes use "_"/"^" as field separators (e.g.
    # "ABDOMEN^1_ABDPEL_WO_W (ADULT)") -- treat them as word boundaries
    # so \b-based rules below can see the glued tokens on either side.
    t = _SEPARATOR_RE.sub(" ", t)
    t = _WCONTRAST_RE.sub("W CONTRAST", t)
    t = _SLASH_WO_RE.sub("WO", t)
    t = _GLUED_WWO_RE.sub("WO W", t)
    t = _WORD_SUB_RE.sub(lambda m: _ALL_WORD_SUBS[m.group(1)], t)
    return t


def classify_contrast(raw_text, modality=None):
    """Return one of CATEGORIES for a single raw study description."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "Unspecified"

    norm = normalize(raw_text)
    has_contrast_word = bool(CONTRAST_RE.search(norm))

    if has_contrast_word:
        search_space = norm
    elif modality in TAIL_HEURISTIC_MODALITIES:
        tokens = _TOKEN_RE.findall(norm)
        search_space = " ".join(tokens[-TAIL_WINDOW:])
    else:
        return "Unspecified"

    has_without = bool(WITHOUT_FAMILY_RE.search(search_space))
    has_with = bool(WITH_FAMILY_RE.search(search_space))

    if has_without and has_with:
        return "With and without"
    if has_without:
        return "Without"
    if has_with:
        return "With"
    if has_contrast_word:
        # Contrast material is named (e.g. "BARIUM ENEMA SGL CONTRAST",
        # "CORONARY CTA WITH CONTRAST") but no explicit timing qualifier
        # -> contrast was administered.
        return "With"
    return "Unspecified"


def build_contrast_timing(df):
    """Classify every unique study_desc_raw and bucket them by category."""
    buckets = defaultdict(set)
    for raw, modality in df[["study_desc_raw", "modality"]].itertuples(index=False):
        if not isinstance(raw, str):
            continue
        category = classify_contrast(raw, modality)
        buckets[category].add(raw)

    return {category: sorted(buckets.get(category, [])) for category in CATEGORIES}


def main():
    parser = argparse.ArgumentParser(description="Classify raw study descriptions into contrast-timing buckets.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to glassbeam_data.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write contrast_timing.json")
    parser.add_argument("--query", default=None, help="Classify a single string instead of processing the CSV")
    parser.add_argument("--modality", default=None, help="Modality code to use with --query (e.g. MR, CT)")
    args = parser.parse_args()

    if args.query is not None:
        result = classify_contrast(args.query, args.modality)
        print(f"{result}  <-  {args.query!r} (modality={args.modality})")
        return

    df = pd.read_csv(args.csv)
    result = build_contrast_timing(df)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    total = sum(len(v) for v in result.values())
    print(f"Classified {total} unique study descriptions -> {out_path}")
    for category in CATEGORIES:
        print(f"  {category:<20} {len(result[category]):>5}")


if __name__ == "__main__":
    main()
