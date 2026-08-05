"""
laterality_extractor.py
------------------------
Classifies each raw study description into a laterality bucket
(Right / Left / Bilateral / Unspecified) using substitution rules derived
from examining every unique value of study_desc_raw in output/glassbeam_data.csv.

Approach
========
1. Normalize the raw text: fix typos (LFT/RIGH/RIGTH/BILATERA -> canonical),
   mark the "*L*"/"*R*" asterisk-wrapped laterality convention used by MR/CT
   extremity protocols (e.g. "MR KNEE*L* W/O CM") with unambiguous tokens,
   strip "R/O" ("rule out" shorthand -- not the letter R for Right), and
   strip "RIGHT TO LEFT" / "LEFT TO RIGHT" shunt-direction phrases (these
   describe blood-flow direction in NM shunt studies, not which side of the
   body was imaged).
2. If "BILAT"/"BILATERAL" is present anywhere -> "Bilateral" (this takes
   priority even when a single side is also mentioned, e.g. "BILAT AP
   STANDING AND LAT LEFT" -- the primary study is bilateral).
3. Otherwise look for independent Left-family and Right-family signals:
   LEFT, LT, the "*L*" marker, and the bare token "L"; RIGHT, RT, the "*R*"
   marker, and the bare token "R". Only text before the word "COMPARISON"
   is searched, since a handful of descriptions record the primary side
   first and a secondary comparison side after that keyword (e.g. "XRAY
   KNEE 4 VIEW RIGHT WITH COMPARISON LEFT" is a Right study).
4. The bare token "L" is dropped as a laterality signal when the string
   contains "SPINE" -- "L SPINE"/"L-SPINE" denotes the lumbar spine level,
   not a body side, and there is no comparable ambiguity for "R".
5. If both Left-family and Right-family signals survive -> "Bilateral".
   If only one family is present -> that side. Otherwise -> "Unspecified".

Usage:
    python3 scripts/laterality_extractor.py
    python3 scripts/laterality_extractor.py --csv output/glassbeam_data.csv --out data/laterality.json
    python3 scripts/laterality_extractor.py --query "XRAY KNEE 4 VIEW RIGHT WITH COMPARISON LEFT"
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = ROOT_DIR / "output" / "glassbeam_data.csv"
DEFAULT_OUT = ROOT_DIR / "data" / "laterality.json"

sys.path.insert(0, str(SCRIPT_DIR))
from modality_architecture import load_modality_architecture, attribute_applies

CATEGORIES = ["Unspecified", "Right", "Left", "Bilateral"]

# ── Substitution rules ──────────────────────────────────────────────────────

# Typos -> canonical form. Found by fuzzy-matching every alphabetic token in
# the corpus against LEFT/RIGHT/BILATERAL.
WORD_SUBS = {
    "LFT": "LEFT",
    "RIGH": "RIGHT",
    "RIGTH": "RIGHT",
    "BILATERA": "BILATERAL",
}

_WORD_SUB_RE = re.compile(r"\b(" + "|".join(sorted(WORD_SUBS, key=len, reverse=True)) + r")\b")

# "MR KNEE*L* W/O CM" / "CT FEMUR *R* W/O CM" -- the asterisk-wrapped form is
# always a genuine laterality marker (unlike the bare letter, it never
# collides with "L SPINE"/"L-SPINE"), so mark it with an unambiguous token
# before generic bare-letter detection runs.
_ASTERISK_L_RE = re.compile(r"\*\s*L\s*\*")
_ASTERISK_R_RE = re.compile(r"\*\s*R\s*\*")

# "R/O" = "rule out", not the letter R for Right (e.g. "...PRE TEVAR R/O
# DISSECTION"). Strip before bare-R detection runs.
_RULE_OUT_RE = re.compile(r"\bR\s*/\s*O\b")

# "NM LUNG RIGHT TO LEFT SHUNT" / "NM RIGHT TO LEFT CARDIAC SHUNT MAA" --
# blood-flow direction in a shunt study, not the side of the body imaged.
_SHUNT_DIRECTION_RE = re.compile(r"\b(?:RIGHT|LEFT)\s+TO\s+(?:RIGHT|LEFT)\b")

# Only text before "COMPARISON" reflects the side actually being studied;
# text after it names a secondary side kept only for reference.
_COMPARISON_RE = re.compile(r"\bCOMPARISON\b")

BILATERAL_RE = re.compile(r"\bBILAT\b|\bBILATERAL\b")
LEFT_FAMILY_RE = re.compile(r"\bLEFT\b|\bLT\b|\bASTERISKLEFT\b")
RIGHT_FAMILY_RE = re.compile(r"\bRIGHT\b|\bRT\b|\bASTERISKRIGHT\b")
BARE_L_RE = re.compile(r"\bL\b")
BARE_R_RE = re.compile(r"\bR\b")


def normalize(text):
    """Apply substitution rules to canonicalize laterality-related shorthand."""
    t = text.upper()
    t = _ASTERISK_L_RE.sub(" ASTERISKLEFT ", t)
    t = _ASTERISK_R_RE.sub(" ASTERISKRIGHT ", t)
    t = _RULE_OUT_RE.sub(" ", t)
    t = _SHUNT_DIRECTION_RE.sub(" ", t)
    t = _WORD_SUB_RE.sub(lambda m: WORD_SUBS[m.group(1)], t)
    return t


def classify_laterality_detailed(raw_text):
    """Return (category, tier) for a single raw study description.

    tier is "explicit" when the winning side(s) came from an unambiguous
    marker (LEFT/RIGHT/LT/RT/BILAT/the asterisk convention), "bare_letter"
    when the *weakest* contributing side came only from a bare L/R token,
    and "none" when nothing matched at all. For Bilateral, the weaker of
    the two contributing sides determines the tier (weakest link).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "Unspecified", "none"

    norm = normalize(raw_text)

    if BILATERAL_RE.search(norm):
        return "Bilateral", "explicit"

    search_space = _COMPARISON_RE.split(norm)[0]

    has_left_explicit = bool(LEFT_FAMILY_RE.search(search_space))
    has_right_explicit = bool(RIGHT_FAMILY_RE.search(search_space))

    # Bare "L" is untrustworthy near "SPINE" (lumbar spine level, not a
    # side); bare "R" has no such collision in this corpus.
    has_left_bare = not has_left_explicit and bool(BARE_L_RE.search(search_space)) and "SPINE" not in norm
    has_right_bare = not has_right_explicit and bool(BARE_R_RE.search(search_space))

    has_left = has_left_explicit or has_left_bare
    has_right = has_right_explicit or has_right_bare

    if has_left and has_right:
        tier = "explicit" if (has_left_explicit and has_right_explicit) else "bare_letter"
        return "Bilateral", tier
    if has_left:
        return "Left", "explicit" if has_left_explicit else "bare_letter"
    if has_right:
        return "Right", "explicit" if has_right_explicit else "bare_letter"
    return "Unspecified", "none"


def classify_laterality(raw_text):
    """Return one of CATEGORIES for a single raw study description."""
    return classify_laterality_detailed(raw_text)[0]


def build_laterality(df, architecture):
    """Classify every unique study_desc_raw and bucket them by category.

    Modalities where "laterality" isn't listed for them in
    modality_model_architecture.json are forced to "Unspecified" -- the
    routing table is authoritative, not just a side effect of the text
    happening to be silent for those modalities.
    """
    buckets = defaultdict(set)
    for raw, modality in df[["study_desc_raw", "modality"]].itertuples(index=False):
        if not isinstance(raw, str):
            continue
        if attribute_applies(modality, "laterality", architecture):
            category = classify_laterality(raw)
        else:
            category = "Unspecified"
        buckets[category].add(raw)

    return {category: sorted(buckets.get(category, [])) for category in CATEGORIES}


def main():
    parser = argparse.ArgumentParser(description="Classify raw study descriptions into laterality buckets.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to glassbeam_data.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write laterality.json")
    parser.add_argument("--query", default=None, help="Classify a single string instead of processing the CSV")
    parser.add_argument("--modality", default=None, help="Modality code to use with --query (e.g. MR, NM)")
    args = parser.parse_args()

    architecture = load_modality_architecture()

    if args.query is not None:
        if args.modality and not attribute_applies(args.modality, "laterality", architecture):
            result = "Unspecified"
            print(f"{result}  <-  {args.query!r} (modality={args.modality} excluded from laterality by modality_model_architecture.json)")
        else:
            result = classify_laterality(args.query)
            print(f"{result}  <-  {args.query!r}")
        return

    df = pd.read_csv(args.csv)
    result = build_laterality(df, architecture)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    total = sum(len(v) for v in result.values())
    print(f"Classified {total} unique study descriptions -> {out_path}")
    for category in CATEGORIES:
        print(f"  {category:<12} {len(result[category]):>5}")


if __name__ == "__main__":
    main()
