"""
classify.py
-------------
Single entry point for the whole pipeline: takes the two fields that
matter (modality, study_desc_raw), consults the routing table in
data/modality_model_architecture.json exactly once, and calls into
whichever of the 4 attribute models actually apply to that modality --
region/focus, laterality, contrast, technique_study_type. Each model's own
extractor script still owns its classification logic (classify_region_focus,
classify_laterality, classify_contrast, classify_technique, and their
*_detailed() siblings that additionally report which rule tier fired);
this file only orchestrates the routing, it does not reimplement any rules.

An attribute that doesn't apply to a modality is reported as `None`
("not applicable"), distinct from a model running and finding no signal
(e.g. laterality's "Unspecified", technique's "Other") -- so a consumer can
tell "MG doesn't get a contrast model" apart from "MG got the contrast
model and it found nothing."

Every prediction also gets a confidence score (see confidence.py): a
rule-tier base score, penalized by how much of the description's
vocabulary was never seen in the training corpus at all ("novelty").
`novelty_score` itself is reported once per record since it's a property
of the text, not any one attribute.

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
from region_focus_extractor import classify_region_focus_detailed
from laterality_extractor import classify_laterality_detailed
from contrast_extractor import classify_contrast_detailed
from technique_extractor import classify_technique_detailed
from confidence import compute_novelty, tier_confidence


def classify(modality, study_desc_raw, architecture):
    """Route (modality, study_desc_raw) through every applicable model.

    Returns a dict with one key per attribute plus a `<attribute>_confidence`
    sibling for each. A value of None (for both the prediction and its
    confidence) means the routing table excludes that attribute for this
    modality; any other value is whatever the underlying model returned.
    """
    novelty = compute_novelty(study_desc_raw)

    if attribute_applies(modality, "region", architecture):
        region_detail = classify_region_focus_detailed(study_desc_raw)
        region_focus = {region: sorted(info["foci"]) for region, info in region_detail.items()}
        if region_detail:
            region_focus_confidence = {
                region: tier_confidence("region_focus", info["tier"], novelty)
                for region, info in region_detail.items()
            }
        else:
            region_focus_confidence = tier_confidence("region_focus", "none", novelty)
    else:
        region_focus = None
        region_focus_confidence = None

    if attribute_applies(modality, "laterality", architecture):
        laterality, lat_tier = classify_laterality_detailed(study_desc_raw)
        laterality_confidence = tier_confidence("laterality", lat_tier, novelty)
    else:
        laterality = None
        laterality_confidence = None

    if attribute_applies(modality, "contrast", architecture):
        contrast, contrast_tier = classify_contrast_detailed(study_desc_raw, modality)
        contrast_confidence = tier_confidence("contrast", contrast_tier, novelty)
    else:
        contrast = None
        contrast_confidence = None

    if attribute_applies(modality, "technique_study_type", architecture):
        technique_study_type, tech_tier = classify_technique_detailed(study_desc_raw, modality)
        technique_confidence = tier_confidence("technique_study_type", tech_tier, novelty)
    else:
        technique_study_type = None
        technique_confidence = None

    return {
        "modality": modality,
        "study_desc_raw": study_desc_raw,
        "novelty_score": round(novelty, 3),
        "region_focus": region_focus,
        "region_focus_confidence": region_focus_confidence,
        "laterality": laterality,
        "laterality_confidence": laterality_confidence,
        "contrast": contrast,
        "contrast_confidence": contrast_confidence,
        "technique_study_type": technique_study_type,
        "technique_study_type_confidence": technique_confidence,
    }


def build_classified(df, architecture):
    """Classify every unique (modality, study_desc_raw) pair in the CSV."""
    pairs = df[["modality", "study_desc_raw"]].dropna().drop_duplicates()
    warned = set()
    results = []
    for modality, raw in pairs.itertuples(index=False):
        if modality not in architecture and modality not in warned:
            print(f"WARNING: unknown modality {modality!r} -- not in modality routing table, all attributes will be None", file=sys.stderr)
            warned.add(modality)
        results.append(classify(modality, raw, architecture))
    return results


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
        if args.modality not in architecture:
            print(f"WARNING: unknown modality {args.modality!r} -- not in modality routing table, all attributes will be None", file=sys.stderr)
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
