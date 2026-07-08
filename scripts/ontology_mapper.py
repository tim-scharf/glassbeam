"""
ontology_mapper.py
------------------
Defines empirical ontology mappings for imaging modalities based on glassbeam_data analysis.
Maps modality codes to relevant extractable attributes and global anatomical regions.

Usage:
    from ontology_mapper import MODALITY_ONTOLOGY, GLOBAL_REGIONS, get_modality_config
    config = get_modality_config('MR')
"""

# ── Global Anatomical Regions ─────────────────────────────────────────────────
# Based on LOINC Radiology Playbook analysis:
# 9 canonical anatomical regions covering 96.9% of region mentions
# Spine added as hierarchical parent of cervical/thoracic/lumbar subdivisions

GLOBAL_REGIONS = [
    'Head',
    'Neck',
    'Chest',
    'Abdomen',
    'Pelvis',
    'Spine',
    'Upper extremity',
    'Lower extremity',
    'Whole Body',
]

REGION_ALIASES = {
    'brain': 'Head',
    'skull': 'Head',
    'face': 'Head',
    'orbit': 'Head',
    'shoulder': 'Upper extremity',
    'elbow': 'Upper extremity',
    'wrist': 'Upper extremity',
    'hand': 'Upper extremity',
    'finger': 'Upper extremity',
    'arm': 'Upper extremity',
    'hip': 'Lower extremity',
    'knee': 'Lower extremity',
    'ankle': 'Lower extremity',
    'foot': 'Lower extremity',
    'leg': 'Lower extremity',
    'femur': 'Lower extremity',
    'tibia': 'Lower extremity',
    'fibula': 'Lower extremity',
    'spine': 'Spine',
    'cervical': 'Spine',
    'thoracic': 'Spine',
    'lumbar': 'Spine',
    'sacrum': 'Pelvis',
    'liver': 'Abdomen',
    'kidney': 'Abdomen',
    'spleen': 'Abdomen',
    'pancreas': 'Abdomen',
    'stomach': 'Abdomen',
    'colon': 'Abdomen',
    'heart': 'Chest',
    'lung': 'Chest',
}

# ── Modality Ontology Mappings ─────────────────────────────────────────────────
# Based on analysis of glassbeam_data.csv (9 modalities, 6,140 unique studies)

MODALITY_ONTOLOGY = {
    'MR': {
        'name': 'Magnetic Resonance Imaging',
        'total_records': 1850,
        'region_presence': 81.5,
        'extract': ['region', 'anatomical_focus', 'contrast', 'laterality', 'sequence'],
        'contrast_keywords': ['WITH', 'WITHOUT', 'IV CONTRAST', 'AND WITH', 'THEN WITH'],
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'CT': {
        'name': 'Computed Tomography',
        'total_records': 1479,
        'region_presence': 75.8,
        'extract': ['region', 'anatomical_focus', 'contrast', 'laterality', 'protocol'],
        'contrast_keywords': ['WITH', 'WITHOUT', 'IV CONTRAST', 'CONTRAST'],
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'DX': {
        'name': 'Digital X-ray',
        'total_records': 579,
        'region_presence': 77.5,
        'extract': ['region', 'laterality', 'view_count'],
        'view_pattern': r'(\d+)\s+(?:OR\s+)?(\d+)?\s+VIEWS?',
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'CR': {
        'name': 'Computed Radiography',
        'total_records': 513,
        'region_presence': 71.2,
        'extract': ['region', 'laterality', 'view_count'],
        'view_pattern': r'(\d+)\s+(?:OR\s+)?(\d+)?\s+VIEWS?',
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'RF': {
        'name': 'Radiofluoroscopy',
        'total_records': 213,
        'region_presence': 59.2,
        'extract': ['region', 'anatomical_focus', 'laterality', 'procedure_type'],
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'US': {
        'name': 'Ultrasound',
        'total_records': 878,
        'region_presence': 50.0,
        'extract': ['region', 'anatomical_focus', 'laterality', 'scan_type'],
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'XA': {
        'name': 'X-ray Angiography',
        'total_records': 212,
        'region_presence': 43.9,
        'extract': ['region', 'anatomical_focus', 'laterality', 'vessel_type'],
        'laterality_keywords': ['LEFT', 'RIGHT', 'BILATERAL'],
    },
    'NM': {
        'name': 'Nuclear Medicine',
        'total_records': 144,
        'region_presence': 40.3,
        'extract': ['anatomical_focus', 'organ_system', 'imaging_type'],
        'imaging_types': ['SPECT', 'PET', 'WHOLE BODY', 'SCAN'],
    },
    'MG': {
        'name': 'Mammography',
        'total_records': 272,
        'region_presence': 37.9,
        'extract': ['breast_laterality', 'procedure_type', 'special_features'],
        'procedure_types': ['SCREENING', 'DIAGNOSTIC', 'BIOPSY', 'TOMOGRAPHY'],
        'breast_laterality_keywords': ['BILATERAL', 'UNILATERAL', 'LEFT', 'RIGHT'],
        'note': 'Region is always breast; do not extract general region',
    },
}

# ── Extraction Configuration ───────────────────────────────────────────────────

MULTI_LABEL_EXTRACTION = {
    'anatomical_structures': {
        'description': 'Multi-label anatomical structure extraction (region + focus)',
        'applies_to': ['MR', 'CT', 'DX', 'CR', 'RF', 'US', 'XA', 'NM'],
        'skip': ['MG'],
        'regions': GLOBAL_REGIONS,
        'aliases': REGION_ALIASES,
    },
    'laterality': {
        'description': 'Body side specification',
        'applies_to': ['MR', 'CT', 'DX', 'CR', 'RF', 'US', 'XA'],
        'skip': ['MG', 'NM'],
        'values': ['LEFT', 'RIGHT', 'BILATERAL', 'UNILATERAL'],
    },
    'contrast': {
        'description': 'Imaging agent presence',
        'applies_to': ['MR', 'CT', 'RF', 'US'],
        'skip': ['DX', 'CR', 'MG', 'NM', 'XA'],
        'values': ['WITH', 'WITHOUT', 'IV CONTRAST'],
    },
    'breast_detail': {
        'description': 'Mammography-specific attributes',
        'applies_to': ['MG'],
        'skip': ['MR', 'CT', 'DX', 'CR', 'RF', 'US', 'XA', 'NM'],
        'attributes': ['laterality', 'procedure_type', 'implants'],
    },
}


# ── Utility Functions ──────────────────────────────────────────────────────────

def get_modality_config(modality_code):
    """Get extraction configuration for a specific modality."""
    if modality_code not in MODALITY_ONTOLOGY:
        raise ValueError(f"Unknown modality: {modality_code}")
    return MODALITY_ONTOLOGY[modality_code]


def get_extraction_attributes(modality_code):
    """Get list of attributes to extract for a modality."""
    config = get_modality_config(modality_code)
    return config.get('extract', [])


def applies_to_modality(extraction_type, modality_code):
    """Check if an extraction type applies to a modality."""
    if extraction_type not in MULTI_LABEL_EXTRACTION:
        raise ValueError(f"Unknown extraction type: {extraction_type}")

    config = MULTI_LABEL_EXTRACTION[extraction_type]
    if modality_code in config.get('skip', []):
        return False
    if modality_code in config.get('applies_to', []):
        return True
    return False


def list_modalities():
    """List all modalities with basic info."""
    for code, config in sorted(MODALITY_ONTOLOGY.items()):
        print(f"{code:<5} {config['name']:<30} {config['total_records']:>6,} records ({config['region_presence']:>5.1f}% have region)")


if __name__ == '__main__':
    print("Modality Ontology Configuration\n")
    list_modalities()
    print(f"\nGlobal Regions: {len(GLOBAL_REGIONS)}")
    for region in GLOBAL_REGIONS:
        print(f"  - {region}")
