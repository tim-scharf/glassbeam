"""
technique_extractor.py
------------------------
Classifies each raw study description into a technique/study-type bucket
using substitution rules derived from examining every unique value of
study_desc_raw in output/glassbeam_data.csv.

Scope
=====
Unlike region/focus, laterality, and contrast, this attribute is NOT run
across all modalities. A keyword scan of every modality showed that only
NM and MG have a small, closed set of specialty techniques that partitions
nearly the whole modality:
  - MR/CT have real technique signal (arthrogram, angio, enterography, ...)
    but it's a long tail covering only ~11-17% of rows -- most of the
    modality would land in a meaningless "Standard" catch-all.
  - CR/DX show almost no technique signal (<1%).
  - RF/XA are already procedure-name-heavy (arthrogram, myelogram,
    injection, IR line placement, ...) and would need a much bigger,
    separate taxonomy -- out of scope here.
NM and MG each get their own fixed category list (see NM_CATEGORIES /
MG_CATEGORIES below), and data/modality_model_architecture.json lists
"technique_study_type" only for those two modalities.

Approach
========
Single-label per modality: each raw description is matched against an
ordered list of (category, keyword pattern) rules, evaluated top to
bottom; the first match wins. Order matters where categories could
otherwise collide (e.g. NM "THYROID CANCER THERAPY" should be Therapy,
not Thyroid/Parathyroid -- so Therapy is checked first; MG "STEREOTACTIC
... NEEDLE LOCALIZATION" should be Localization, not Biopsy -- so the
Biopsy rule requires an explicit BIOPSY/CORE keyword, not just
STEREOTACTIC alone). Anything matching nothing falls into "Other".

Usage:
    python3 scripts/technique_extractor.py
    python3 scripts/technique_extractor.py --csv output/glassbeam_data.csv --out data/technique_study_type.json
    python3 scripts/technique_extractor.py --query "NM THYROID UPTAKE AND SCAN" --modality NM
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
DEFAULT_OUT = ROOT_DIR / "data" / "technique_study_type.json"

sys.path.insert(0, str(SCRIPT_DIR))
from modality_architecture import load_modality_architecture, attribute_applies

# ── NM rules ─────────────────────────────────────────────────────────────────
# Therapy is checked first since therapy studies often also name the organ
# being treated (e.g. "THYROID CANCER THERAPY", "IV THERAPY LUTETIUM").
# Hepatobiliary is checked before Pulmonary so "LIVER TO LUNG SHUNT" lands
# on LIVER, not LUNG.
NM_CATEGORIES = [
    "Therapy/Theranostics",
    "Neuro/Brain",
    "Cardiac",
    "Hepatobiliary",
    "Pulmonary (V/Q)",
    "GI",
    "Lymphatic",
    "Infection/Inflammation (WBC/Gallium)",
    "Bone",
    "Thyroid/Parathyroid",
    "Renal/GU",
    "QC/Admin/Research",
    "Other",
]

NM_RULES = [
    ("Therapy/Theranostics", re.compile(
        r"THERAPY|ABLATION|\bI[- ]?131\b|\bI[- ]?123\b|LUTETIUM|RADIUM|THERASPHERE|DOTATATE|\bY90\b|TUMOR"
    )),
    ("Neuro/Brain", re.compile(r"\bBRAIN\b|DATSCAN|CSF SHUNT")),
    ("Cardiac", re.compile(r"CARDIAC|\bMUGA\b|MYOCARDIAL|STRESS TEST|\bHEART\b|\bCARD\b")),
    ("Hepatobiliary", re.compile(r"\bHIDA\b|HEPATOBILIARY|\bLIVER\b|\bSPLEEN\b|GALLBLADDER")),
    ("Pulmonary (V/Q)", re.compile(r"\bLUNG\b|VENTILATION|PERFUSION|V/Q|SHUNT")),
    ("GI", re.compile(r"GASTRIC|GASTROINTESTINAL|GI BLEED|GI BLOOD|MECKEL")),
    ("Lymphatic", re.compile(r"LYMPH|SENTINEL NODE")),
    ("Infection/Inflammation (WBC/Gallium)", re.compile(r"\bWBC\b|WHITE BLOOD CELL|INFLAMMAT|INDIUM|CERETEC|TAGGED|GALLIUM")),
    ("Bone", re.compile(r"\bBONE\b|MARROW")),
    ("Thyroid/Parathyroid", re.compile(r"THYROID|PARATHYROID")),
    ("Renal/GU", re.compile(r"KIDNEY|RENAL|\bGFR\b|LASIX|CAPTOPRIL")),
    ("QC/Admin/Research", re.compile(
        r"\bQC\b|OUTSIDE FILM|REFERENCE ONLY|^NUCLEAR-\d|^NRP\d|DXA COMPOSITION"
    )),
]

# ── MG rules ─────────────────────────────────────────────────────────────────
# Research/specimen/contrast/DEXA are checked first since they're the most
# specific signals. Post-procedure/call-back is checked before generic
# Screening/Diagnostic since "follow-up view after a prior procedure" is a
# more specific descriptor than the generic exam category. Biopsy requires
# an explicit BIOPSY/CORE keyword (not just STEREOTACTIC alone) so that
# "STEREOTACTIC ... NEEDLE LOCALIZATION" rows fall through to Localization.
MG_CATEGORIES = [
    "Research protocol",
    "Specimen imaging",
    "Contrast-enhanced mammography",
    "Bone densitometry (DEXA)",
    "Image-guided biopsy",
    "Needle/wire/seed localization",
    "Post-procedure/call-back",
    "Screening mammogram",
    "Diagnostic mammogram",
    "Consult/other",
    "Other",
]

MG_RULES = [
    ("Research protocol", re.compile(r"\bTMIST\b|IMGBI\d|IMGFL\d|IMGPT\d")),
    ("Specimen imaging", re.compile(r"SPECIMEN|SURG SPEC|\bTBBX\b")),
    ("Contrast-enhanced mammography", re.compile(r"W CONTRAST|WITH CONTRAST")),
    ("Bone densitometry (DEXA)", re.compile(r"\bDEXA\b")),
    ("Image-guided biopsy", re.compile(
        r"\bBIOPSY\b|VAC CORE|TOMO BIOPSY|\bBX\b|TOMO BX"
    )),
    ("Needle/wire/seed localization", re.compile(
        r"LOCALIZATION|WIRE PLACEMENT|MAG SEED|RADAR|WIRELESS LOCALIZ|GUIDED MARKER|MARKER PLACEMENT"
    )),
    ("Post-procedure/call-back", re.compile(
        r"CALL BACK|POST BIOPSY|POST MR\b|POST MRI\b|POST TOMO\b|POST SBB\b|POST US\b|ADD ?TIME"
    )),
    ("Screening mammogram", re.compile(r"SCREEN|\bSCR\b|^SC\b|\bTHD\b|\bTOMOHD\b")),
    ("Diagnostic mammogram", re.compile(r"DIAGNOSTIC|\bDIAG\b|^DX\b|2D DIAGNOSTIC")),
    ("Consult/other", re.compile(r"CONSULT|OUTSIDE FILM")),
]

RULES_BY_MODALITY = {"NM": NM_RULES, "MG": MG_RULES}
CATEGORIES_BY_MODALITY = {"NM": NM_CATEGORIES, "MG": MG_CATEGORIES}


def classify_technique(raw_text, modality):
    """Return a technique/study-type category for a single raw study description."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "Other"

    rules = RULES_BY_MODALITY.get(modality)
    if rules is None:
        return "Other"

    norm = raw_text.upper()
    for category, pattern in rules:
        if pattern.search(norm):
            return category
    return "Other"


def build_technique(df, architecture):
    """Classify every unique study_desc_raw (NM/MG only) and bucket by modality -> category.

    Modalities where "technique_study_type" isn't listed for them in
    modality_model_architecture.json are skipped entirely -- this attribute
    only exists for NM and MG.
    """
    result = {}
    for modality, categories in CATEGORIES_BY_MODALITY.items():
        if not attribute_applies(modality, "technique_study_type", architecture):
            continue
        buckets = defaultdict(set)
        subset = df[df["modality"] == modality]["study_desc_raw"].dropna()
        for raw in subset:
            buckets[classify_technique(raw, modality)].add(raw)
        result[modality] = {category: sorted(buckets.get(category, [])) for category in categories}

    return result


def main():
    parser = argparse.ArgumentParser(description="Classify raw study descriptions into technique/study-type buckets (NM/MG only).")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to glassbeam_data.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write technique_study_type.json")
    parser.add_argument("--query", default=None, help="Classify a single string instead of processing the CSV")
    parser.add_argument("--modality", default=None, help="Modality code to use with --query (e.g. NM, MG)")
    args = parser.parse_args()

    architecture = load_modality_architecture()

    if args.query is not None:
        if not args.modality or not attribute_applies(args.modality, "technique_study_type", architecture):
            print(f"Other  <-  {args.query!r} (modality={args.modality} excluded from technique_study_type by modality_model_architecture.json)")
        else:
            result = classify_technique(args.query, args.modality)
            print(f"{result}  <-  {args.query!r} (modality={args.modality})")
        return

    df = pd.read_csv(args.csv)
    result = build_technique(df, architecture)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    total = sum(len(v) for cats in result.values() for v in cats.values())
    print(f"Classified {total} unique study descriptions -> {out_path}")
    for modality, cats in result.items():
        print(f"  {modality}:")
        for category in CATEGORIES_BY_MODALITY[modality]:
            print(f"    {category:<40} {len(cats[category]):>5}")


if __name__ == "__main__":
    main()
