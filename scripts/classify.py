"""
classify.py
-------------
Single entry point for the whole pipeline: takes the two fields that
matter (modality, study_desc_raw), consults the routing table in
data/modality_model_architecture.json exactly once, and calls into
whichever of the 4 attribute models actually apply to that modality --
region/focus, laterality, contrast, technique_study_type. Each model's own
extractor script still owns its classification logic (classify_region_focus,
classify_laterality, classify_contrast, classify_technique); this file only
orchestrates the routing, it does not reimplement any rules.

An attribute that doesn't apply to a modality is reported as `None`
("not applicable"), distinct from a model running and finding no signal
(e.g. laterality's "Unspecified", technique's "Other") -- so a consumer can
tell "MG doesn't get a contrast model" apart from "MG got the contrast
model and it found nothing."

Usage:
    python3 scripts/classify.py --modality CT --text "CT CHEST ABDOMEN PELVIS W AND WO CONTRAST"
    python3 scripts/classify.py --csv output/glassbeam_data.csv --out data/classified_studies.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = ROOT_DIR / "output" / "glassbeam_data.csv"
DEFAULT_OUT = ROOT_DIR / "data" / "classified_studies.json"

sys.path.insert(0, str(SCRIPT_DIR))
from modality_architecture import load_modality_architecture, attribute_applies
from region_focus_extractor import classify_region_focus
from laterality_extractor import classify_laterality
from contrast_extractor import classify_contrast
from technique_extractor import classify_technique


def classify(modality, study_desc_raw, architecture):
    """Route (modality, study_desc_raw) through every applicable model.

    Returns a dict with one key per attribute. A value of None means the
    routing table excludes that attribute for this modality; any other
    value is whatever the underlying model returned.
    """
    region_focus = (
        classify_region_focus(study_desc_raw)
        if attribute_applies(modality, "region", architecture)
        else None
    )
    laterality = (
        classify_laterality(study_desc_raw)
        if attribute_applies(modality, "laterality", architecture)
        else None
    )
    contrast = (
        classify_contrast(study_desc_raw, modality)
        if attribute_applies(modality, "contrast", architecture)
        else None
    )
    technique_study_type = (
        classify_technique(study_desc_raw, modality)
        if attribute_applies(modality, "technique_study_type", architecture)
        else None
    )

    return {
        "modality": modality,
        "study_desc_raw": study_desc_raw,
        "region_focus": region_focus,
        "laterality": laterality,
        "contrast": contrast,
        "technique_study_type": technique_study_type,
    }


def build_classified(df, architecture):
    """Classify every unique (modality, study_desc_raw) pair in the CSV."""
    pairs = df[["modality", "study_desc_raw"]].dropna().drop_duplicates()
    return [
        classify(modality, raw, architecture)
        for modality, raw in pairs.itertuples(index=False)
    ]


def main():
    parser = argparse.ArgumentParser(description="Route (modality, study_desc_raw) through all applicable models.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to glassbeam_data.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write classified_studies.json")
    parser.add_argument("--modality", default=None, help="Modality code for a single --text query (e.g. CT, NM, MG)")
    parser.add_argument("--text", default=None, help="Classify a single string instead of processing the CSV")
    args = parser.parse_args()

    architecture = load_modality_architecture()

    if args.text is not None:
        if not args.modality:
            parser.error("--text requires --modality")
        result = classify(args.modality, args.text, architecture)
        print(json.dumps(result, indent=2, default=sorted))
        return

    df = pd.read_csv(args.csv)
    results = build_classified(df, architecture)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2, default=sorted))

    print(f"Classified {len(results)} unique (modality, study_desc_raw) pairs -> {out_path}")
    by_modality = df["modality"].value_counts()
    for modality in sorted(by_modality.index):
        applicable = [attr for attr in ("region", "focus", "laterality", "contrast", "technique_study_type")
                      if attribute_applies(modality, attr, architecture)]
        print(f"  {modality:<4} -> {applicable}")


if __name__ == "__main__":
    main()
