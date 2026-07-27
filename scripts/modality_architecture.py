"""
modality_architecture.py
--------------------------
Loads data/modality_model_architecture.json -- the routing table that says
which extraction attributes (region, focus, laterality, contrast, ...)
apply to which imaging modality -- so the extractor scripts can enforce it
directly instead of relying on modality-specific text happening to be
silent for attributes that don't apply.

Usage:
    from modality_architecture import load_modality_architecture, attribute_applies
    architecture = load_modality_architecture()
    attribute_applies("NM", "laterality", architecture)  # False
"""

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "modality_model_architecture.json"


def load_modality_architecture(path=DEFAULT_PATH):
    """Load the modality -> [attributes] routing table."""
    return json.loads(Path(path).read_text())


def attribute_applies(modality, attribute, architecture):
    """Whether `attribute` (e.g. "contrast", "laterality", "region") is listed for `modality`."""
    return attribute in architecture.get(modality, [])
