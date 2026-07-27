"""
region_focus_extractor.py
--------------------------
Classifies each raw study description into one or more global anatomical
regions and, within each detected region, into a specific imaging focus --
using keyword rules, no embeddings (same style as contrast_extractor.py /
laterality_extractor.py, applied to a multi-label attribute this time).

Unlike contrast/laterality, region is MULTI-LABEL: a single description can
span several regions (an explicit "CT CHEST ABDOMEN PELVIS" study, or a
combo shorthand token like "CAP"/"ABDPEL"), so the same raw description can
land in more than one region's bucket. Region-level counts therefore do NOT
sum to the number of unique descriptions the way contrast/laterality did.

Approach
========
1. Strip the "C-ARM"/"C ARM" fluoroscopy-equipment phrase before tokenizing
   -- it is not the upper-extremity "arm".
2. Tokenize into whole alphabetic words (splitting on digits, spaces, and
   scanner-protocol separators "^ _ - /" for free, since none of those are
   letters) and test membership against a curated region-keyword
   dictionary (e.g. LIVER/KIDNEY/RENAL -> Abdomen). A handful of tokens map
   to MULTIPLE regions at once: "CAP" (a scanner CT protocol abbreviation
   for Chest+Abdomen+Pelvis) and "ABDPEL"/"ABDPELV" (Abdomen+Pelvis).
3. "EXTREM"/"EXTREMITY"/"EXTREMITIES" combined with "UP"/"UPPER" or
   "LOW"/"LOWER" anywhere in the same description (verified against the
   corpus: no unrelated "FOLLOW UP"/"LOW DOSE" ever co-occurs with an
   extremity mention here) resolves to Upper/Lower extremity even when no
   specific joint is named.
4. "WHOLE" + "BODY" together -> Whole Body.
5. A few scanner protocol names glue words with NO separator at all (e.g.
   "HEAD^1_HELICALHEAD", "SPINE^1_MAZORLUMBARSPINE", "LSPINE^ROUTINE").
   When no region matched yet, a substring fallback checks whether HEAD/
   BRAIN/SPINE/CHEST/ABDOMEN/PELVIS/NECK appears inside any single token --
   verified safe: no other token in the corpus contains these stems as a
   false substring.
6. If still nothing matched, two weak last-resort signals apply: bare
   BONE/BONES -> Whole Body (bone survey/marrow studies), bare STROKE ->
   Head (code-stroke CT protocols default to head/neck vascular imaging).
7. Anything left unmatched goes into a top-level "Unspecified" bucket added
   alongside the 9 fixed regions -- same convention as the "Unspecified"
   category in contrast_timing.json and laterality.json.
8. Within each matched region, a second keyword pass assigns a specific
   focus (e.g. Abdomen -> liver/kidney/spleen/pancreas/adrenal/gallbladder).
   A region hit with no specific focus keyword goes into that region's own
   "unspecified" bucket. "Whole Body" has no sub-foci; everything there is
   bucketed as "unspecified".

The region set (9 GLOBAL_REGIONS, imported from ontology_mapper.py) is
fixed. The foci are not -- FOCUS_KEYWORDS is meant to keep growing as new
raw descriptions surface anatomy terms not yet covered.

Usage:
    python3 scripts/region_focus_extractor.py
    python3 scripts/region_focus_extractor.py --csv output/glassbeam_data.csv --out data/region_focus_ontology.json
    python3 scripts/region_focus_extractor.py --query "CT CHEST ABDOMEN PELVIS WITH CONTRAST"
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
DEFAULT_OUT = ROOT_DIR / "data" / "region_focus_ontology.json"

sys.path.insert(0, str(SCRIPT_DIR))
from ontology_mapper import GLOBAL_REGIONS
from modality_architecture import load_modality_architecture, attribute_applies

# ── Region keyword rules ────────────────────────────────────────────────────

REGION_KEYWORDS = {
    "Head": {
        "HEAD", "BRAIN", "SKULL", "ORBIT", "ORBITS", "SINUS", "SINUSES",
        "TEMPORAL", "MANDIBLE", "FACE", "FACIAL", "MAXILLOFACIAL",
        "PITUITARY", "MASTOID", "CAROTID", "CAROTIDS", "CEREBRAL",
    },
    "Neck": {"NECK", "THYROID", "LARYNX", "ESOPHAGUS"},
    "Chest": {
        "CHEST", "THORAX", "LUNG", "LUNGS", "HEART", "CARDIAC", "CORONARY",
        "AORTA", "AORTIC", "RIBS", "BREAST", "STERNUM", "CLAVICLE",
        "SCAPULA", "MEDIASTINUM", "PLEURA", "PULMONARY", "THORACENTESIS",
    },
    "Abdomen": {
        "ABDOMEN", "ABDOMINAL", "ABD", "LIVER", "KIDNEY", "KIDNEYS",
        "RENAL", "SPLEEN", "PANCREAS", "ADRENAL", "ADRENALS", "STOMACH",
        "COLON", "GALLBLADDER", "BLADDER", "PARACENTESIS",
        "RETROPERITONEAL", "RETROPERITONEUM",
    },
    "Pelvis": {
        "PELVIS", "PELVIC", "PELV", "PEL", "PROSTATE", "UTERUS", "CERVIX",
        "OVARY", "OVARIES", "TESTICLE", "TESTICLES", "TESTIS", "TESTICULAR",
        "SCROTUM", "ANUS",
    },
    "Spine": {
        "SPINE", "CERVICAL", "THORACIC", "LUMBAR", "SACRUM", "SACROILIAC",
        "COCCYX", "MYELOGRAM", "SCOLIOSIS",
    },
    "Upper extremity": {
        "SHOULDER", "ELBOW", "FOREARM", "WRIST", "HAND", "HANDS", "FINGER",
        "FINGERS", "HUMERUS", "ARM", "BRACHIAL", "SCAPHOID",
    },
    "Lower extremity": {
        "HIP", "HIPS", "KNEE", "KNEES", "ANKLE", "FOOT", "FEET", "FEMUR",
        "TIBIA", "FIBULA", "THIGH", "TOE", "TOES", "CALCANEUS", "PATELLA",
        "LEG",
    },
    # "Whole Body" has no standalone keyword set -- it's only reached via
    # the WHOLE+BODY phrase rule and the BONE/BONES last-resort signal.
}

# Tokens that name more than one region at once (scanner CT protocol
# abbreviations for combined-region studies).
COMBO_KEYWORDS = {
    "CAP": {"Chest", "Abdomen", "Pelvis"},
    "ABDPEL": {"Abdomen", "Pelvis"},
    "ABDPELV": {"Abdomen", "Pelvis"},
    # Axilla straddles the chest wall and the upper-extremity vascular/nodal
    # exam (LOINC lists it under both) -- e.g. "US AXILLA NON VASCULAR".
    "AXILLA": {"Chest", "Upper extremity"},
}

# Bare C/T/L level abbreviations ("C SPINE", "T-SPINE", "CT L SPINE W/O
# CM") -- only ever trusted once "SPINE" itself is already a token
# (see classify_region_focus), so they can't leak into region detection.
SPINE_LEVEL_LETTERS = {"C": "cervical", "T": "thoracic", "L": "lumbar"}

EXTREMITY_TOKENS = {"EXTREM", "EXTREMITY", "EXTREMITIES", "EXTRM"}
UPPER_TOKENS = {"UP", "UPPER"}
LOWER_TOKENS = {"LOW", "LOWER"}

# Only checked when no region matched via whole-word/combo rules -- covers
# scanner protocol names that glue words with no separator at all (e.g.
# "HELICALHEAD", "MAZORLUMBARSPINE").
GLUED_STEM_FALLBACK = {
    "HEAD": "Head",
    "BRAIN": "Head",
    "SPINE": "Spine",
    "CHEST": "Chest",
    "ABDOMEN": "Abdomen",
    "PELVIS": "Pelvis",
    "NECK": "Neck",
}

# Only checked when nothing above matched anything at all.
LAST_RESORT_KEYWORDS = {
    "BONE": "Whole Body",
    "BONES": "Whole Body",
    "STROKE": "Head",
}

# ── Focus keyword rules (per region) ────────────────────────────────────────

FOCUS_KEYWORDS = {
    "Head": {
        "brain": {"BRAIN"},
        "skull": {"SKULL"},
        "sinus": {"SINUS", "SINUSES"},
        "orbits": {"ORBIT", "ORBITS"},
        "temporal": {"TEMPORAL"},
        "mandible": {"MANDIBLE"},
        "face": {"FACE", "FACIAL", "MAXILLOFACIAL"},
        "pituitary": {"PITUITARY"},
        "mastoid": {"MASTOID"},
        "carotid": {"CAROTID", "CAROTIDS"},
        "brain": {"CEREBRAL"},
    },
    "Neck": {
        "thyroid": {"THYROID"},
        "larynx": {"LARYNX"},
        "esophagus": {"ESOPHAGUS"},
        "vascular": {"VASCULAR"},
    },
    "Chest": {
        "heart": {"HEART", "CARDIAC", "CORONARY"},
        "lungs": {"LUNG", "LUNGS", "PULMONARY"},
        "aorta": {"AORTA", "AORTIC"},
        "ribs": {"RIBS"},
        "breast": {"BREAST"},
        "sternum": {"STERNUM"},
        "clavicle": {"CLAVICLE"},
        "scapula": {"SCAPULA"},
        "mediastinum": {"MEDIASTINUM"},
        "pleura": {"PLEURA", "THORACENTESIS"},
        "axilla": {"AXILLA"},
    },
    "Abdomen": {
        "liver": {"LIVER"},
        "kidney": {"KIDNEY", "KIDNEYS", "RENAL"},
        "spleen": {"SPLEEN"},
        "pancreas": {"PANCREAS"},
        "adrenal": {"ADRENAL", "ADRENALS"},
        "gallbladder": {"GALLBLADDER"},
        "bladder": {"BLADDER"},
        "peritoneum": {"PARACENTESIS"},
        "retroperitoneum": {"RETROPERITONEAL", "RETROPERITONEUM"},
    },
    "Pelvis": {
        "prostate": {"PROSTATE"},
        "uterus": {"UTERUS", "CERVIX"},
        "ovary": {"OVARY", "OVARIES"},
        "testicle": {"TESTICLE", "TESTICLES", "TESTIS", "TESTICULAR"},
        "scrotum": {"SCROTUM"},
        "anus": {"ANUS"},
    },
    "Spine": {
        "cervical": {"CERVICAL", "CSPINE"},
        "thoracic": {"THORACIC"},
        "lumbar": {"LUMBAR", "LSPINE", "MAZORLUMBARSPINE"},
        "sacrum": {"SACRUM", "SACROILIAC", "COCCYX"},
        "myelogram": {"MYELOGRAM"},
        "scoliosis": {"SCOLIOSIS"},
    },
    "Upper extremity": {
        "shoulder": {"SHOULDER"},
        "arm": {"ARM", "HUMERUS", "BRACHIAL"},
        "elbow": {"ELBOW"},
        "forearm": {"FOREARM"},
        "wrist": {"WRIST", "SCAPHOID"},
        "hand": {"HAND", "HANDS"},
        "fingers": {"FINGER", "FINGERS"},
        "axilla": {"AXILLA"},
    },
    "Lower extremity": {
        "hip": {"HIP", "HIPS"},
        "thigh": {"THIGH", "FEMUR"},
        "knee": {"KNEE", "KNEES", "PATELLA"},
        "ankle": {"ANKLE"},
        "foot": {"FOOT", "FEET", "TOE", "TOES", "CALCANEUS"},
        "leg": {"LEG", "TIBIA", "FIBULA"},
    },
    "Whole Body": {},
}

_CARM_RE = re.compile(r"\bC[\s\-_/^]*ARM\b")
_TOKEN_RE = re.compile(r"[A-Za-z]+")


def classify_region_focus(raw_text):
    """Return {region: {focus, ...}} for a single raw study description."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {}

    text = _CARM_RE.sub(" ", raw_text.upper())
    tokens = set(_TOKEN_RE.findall(text))

    regions = set()
    for region, keywords in REGION_KEYWORDS.items():
        if tokens & keywords:
            regions.add(region)
    for combo_token, combo_regions in COMBO_KEYWORDS.items():
        if combo_token in tokens:
            regions.update(combo_regions)
    if tokens & EXTREMITY_TOKENS:
        if tokens & UPPER_TOKENS:
            regions.add("Upper extremity")
        if tokens & LOWER_TOKENS:
            regions.add("Lower extremity")
    if "WHOLE" in tokens and "BODY" in tokens:
        regions.add("Whole Body")

    if not regions:
        for tok in tokens:
            for stem, region in GLUED_STEM_FALLBACK.items():
                if stem in tok:
                    regions.add(region)

    if not regions:
        for kw, region in LAST_RESORT_KEYWORDS.items():
            if kw in tokens:
                regions.add(region)

    result = {}
    for region in regions:
        foci = {focus for focus, kws in FOCUS_KEYWORDS.get(region, {}).items() if tokens & kws}
        if region == "Spine" and "SPINE" in tokens:
            # Level abbreviations only ever appear glued to/beside "SPINE"
            # itself ("C SPINE", "T-SPINE", "CT L SPINE W/O CM"), so this
            # bare-letter check is scoped to when SPINE is already present
            # -- it never runs as a region signal on its own.
            foci |= {SPINE_LEVEL_LETTERS[letter] for letter in SPINE_LEVEL_LETTERS if letter in tokens}
        result[region] = foci if foci else {"unspecified"}
    return result


def build_region_focus(df, architecture):
    """Classify every unique study_desc_raw and bucket them by region -> focus.

    Modalities where "region" isn't listed for them in
    modality_model_architecture.json are forced to "Unspecified" -- the
    routing table is authoritative, not just a side effect of the text
    happening to be silent for those modalities (e.g. MG, where region is
    always breast and this model deliberately isn't run).
    """
    buckets = {region: defaultdict(set) for region in GLOBAL_REGIONS}
    unspecified = set()

    for raw, modality in df[["study_desc_raw", "modality"]].itertuples(index=False):
        if not isinstance(raw, str):
            continue
        region_foci = classify_region_focus(raw) if attribute_applies(modality, "region", architecture) else {}
        if not region_foci:
            unspecified.add(raw)
            continue
        for region, foci in region_foci.items():
            for focus in foci:
                buckets[region][focus].add(raw)

    result = {
        region: {focus: sorted(raws) for focus, raws in sorted(buckets[region].items())}
        for region in GLOBAL_REGIONS
    }
    result["Unspecified"] = sorted(unspecified)
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify raw study descriptions into region/focus buckets.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to glassbeam_data.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Path to write region_focus_ontology.json")
    parser.add_argument("--query", default=None, help="Classify a single string instead of processing the CSV")
    parser.add_argument("--modality", default=None, help="Modality code to use with --query (e.g. MR, MG)")
    args = parser.parse_args()

    architecture = load_modality_architecture()

    if args.query is not None:
        if args.modality and not attribute_applies(args.modality, "region", architecture):
            print(f"Unspecified  <-  {args.query!r} (modality={args.modality} excluded from region/focus by modality_model_architecture.json)")
            return
        result = classify_region_focus(args.query)
        if not result:
            print(f"Unspecified  <-  {args.query!r}")
        else:
            for region, foci in sorted(result.items()):
                print(f"{region:<18} {sorted(foci)}  <-  {args.query!r}")
        return

    df = pd.read_csv(args.csv)
    result = build_region_focus(df, architecture)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    unique_count = df["study_desc_raw"].dropna().nunique()
    print(f"Classified {unique_count} unique study descriptions (multi-label) -> {out_path}")
    for region in GLOBAL_REGIONS:
        total = sum(len(v) for v in result[region].values())
        print(f"  {region:<18} {total:>5}  ({len(result[region])} foci)")
    print(f"  {'Unspecified':<18} {len(result['Unspecified']):>5}")


if __name__ == "__main__":
    main()
