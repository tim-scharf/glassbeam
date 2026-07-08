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


def build_focus_to_region_map():
    """
    Build focus→region mapping from LOINC playbook.
    Maps all imaging foci to 9 global regions.
    """
    FOCUS_TO_REGION = {
        'Head': [
            'Brain', 'Brain.temporal', 'Brain stem', 'Brain veins',
            'Skull', 'Skull.base', 'Skull vertex to mid-thigh',
            'Skull base to mid-thigh', 'Skull base to mid-thigh.bone',
            'Face', 'Facial bones', 'Orbit', 'Orbit veins', 'Orbit vessels',
            'Eye', 'Sinuses', 'Sella turcica', 'Temporal bone', 'Mastoid',
            'Pituitary', 'Salivary gland', 'Submandibular gland', 'Parotid gland',
            'Mandible', 'Maxilla', 'Maxillofacial region', 'Teeth', 'Teeth.maxilla', 'Teeth.mandible',
            'Nasal bones', 'Nasopharynx', 'Oropharynx', 'Pharynx', 'Hypopharynx',
            'Larynx', 'Internal auditory canal', 'Optic foramen',
            'Lacrimal duct', 'Carotid artery', 'Carotid artery.external', 'Carotid artery.internal',
            'Carotid artery.cervical', 'Carotid artery.common', 'Carotid artery.extracranial',
            'Carotid artery.intracranial', 'Carotid arteries', 'Carotid vessels',
            'Cerebral artery', 'Cerebral arteries', 'Cerebral artery internal', 'Cerebral cisterns',
            'Cerebral sinuses', 'Cerebral vein', 'Cerebral vessels', 'Head vessels', 'Head arteries',
            'Head veins', 'Head artery', 'Intracranial vessel', 'Circle of Willis',
            'Vertebral artery', 'Vertebral vessels',
        ],
        'Neck': [
            'Neck', 'Neck vessels', 'Neck veins', 'Neck artery', 'Neck vessel',
            'Thyroid gland', 'Thymus gland', 'Larynx', 'Trachea',
            'Esophagus', 'Esophagus.cervical',
            'Internal jugular vein', 'Jugular vein',
            'Subclavian artery', 'Subclavian vessels', 'Subclavian vein',
            'Brachiocephalic artery',
        ],
        'Chest': [
            'Chest', 'Chest vessels', 'Chest veins', 'Chest wall',
            'Heart', 'Heart.ventricle', 'Endomyocardium', 'Pericardial space',
            'Aorta', 'Aorta.thoracic', 'Aorta.abdominal', 'Aorta.abdominal.infrarenal',
            'Aortic arch', 'Aortic valve', 'Aortic root',
            'Coronary arteries', 'Pulmonary arteries', 'Pulmonary system', 'Pulmonary veins',
            'Lung', 'Lung parenchyma',
            'Bronchial artery',
            'Ribs', 'Ribs.anterior', 'Ribs.posterior', 'Ribs.upper', 'Ribs.lower',
            'Ribs.upper.anterior', 'Ribs.upper.posterior', 'Ribs.lower.anterior', 'Ribs.lower.posterior',
            'Sternum', 'Clavicle', 'Acromioclavicular joint', 'Scapula',
            'Pleura', 'Pleural space', 'Mediastinum', 'Mediastinum.superior',
            'Diaphragm', 'Internal thoracic artery',
            'Thoracic inlet vessels', 'Thoracic artery',
            'Popliteal space', 'Axilla', 'Brachial plexus',
        ],
        'Abdomen': [
            'Abdomen', 'Abdominal vessels', 'Abdominal veins', 'Abdominal arteries',
            'Abdominal lymphatic vessels', 'Abdominal wall',
            'Retroperitoneum', 'Retroperitoneal',
            'Liver', 'Hepatic artery', 'Hepatic vein', 'Hepatic veins',
            'Hepatic vessels', 'Intrahepatic portal system',
            'Kidney', 'Kidney cortex', 'Renal artery', 'Renal arteries', 'Renal vein',
            'Renal vein', 'Renal Vessels', 'Renal artery', 'Renal vessel',
            'Adrenal gland', 'Adrenal artery', 'Adrenal vein', 'Adrenal vessels',
            'Pancreas', 'Pancreatic duct', 'Pancreatic artery',
            'Spleen', 'Splenic artery', 'Splenic vein',
            'Stomach', 'Gastric artery', 'Gastrointestinal tract', 'Gastrointestinal tract.upper',
            'Small bowel', 'Ileum', 'Duodenum', 'Jejunum',
            'Colon', 'Rectum', 'Appendix',
            'Gallbladder', 'Biliary ducts', 'Biliary duct.common',
            'Portal vein',
            'Urinary bladder', 'Urinary bladder arteries',
            'Ureter',
            'Peritoneum', 'Peritoneal space', 'Subphrenic space', 'Perirenal space',
            'Perirectal region',
            'Vessel', 'Vessels', 'Vein', 'Artery',
            'Superior mesenteric artery', 'Superior mesenteric vein', 'Superior mesenteric vessels',
            'Inferior mesenteric artery', 'Inferior mesenteric vein',
            'Celiac artery', 'Celiac plexus', 'Celiac vessels',
            'Mesenteric artery', 'Mesenteric vein', 'Mesenteric vessels', 'Mesenteric arteries',
            'Vena cava.inferior', 'Vena cava.superior',
            'Iliac artery', 'Iliac artery.internal', 'Iliac vessels', 'Iliac graft',
            'Pelvis vessels', 'Pelvis arteries', 'Pelvis veins', 'Pelvis bones',
            'Groin',
            'Muscle', 'Soft tissue', 'Tissue',
        ],
        'Pelvis': [
            'Pelvis', 'Pelvis vessels', 'Pelvis veins', 'Pelvis bones',
            'Coccyx', 'Symphysis pubis',
            'Prostate', 'Seminal vesicle', 'Testicle', 'Testicular vessels',
            'Penis', 'Penis.soft tissue', 'Penis vessels',
            'Scrotum',
            'Uterus', 'Uterine artery', 'Fallopian tubes', 'Fallopian tube',
            'Ovary', 'Ovarian vessels',
            'Urethra',
            'Pelvic lymphatic vessels',
            'Vas deferens',
            'Epididymis',
            'Anus',
        ],
        'Spine': [
            'Spine', 'Spine vessels', 'Spine vessel', 'Spine veins', 'Spine epidural space',
            'Spine vertebra', 'Spine facet joint',
            'Intervertebral disc',
            'Spinal artery', 'Spinal cord', 'Spinal cavity', 'Spinal veins',
            'Spine.cervical', 'Spine.cervical.axis', 'Spine.cervical.odontoid',
            'Spine.cervical intervertebral disc', 'Spine.cervical facet joint',
            'Spine.cervical vessels', 'Spine.cervical epidural space',
            'Spine.cervicothoracic junction',
            'Spine.thoracic', 'Spine.thoracic.axis', 'Spine.thoracic vessels',
            'Spine.thoracic facet joint', 'Spine.thoracic epidural space',
            'Spine.thoracic intercostal nerve', 'Spine.thoracolumbar junction',
            'Spine.thoracolumbar',
            'Spine.lumbar', 'Spine.lumbar vessels', 'Spine.lumbosacral junction',
            'Spine.lumbar facet joint', 'Spine.lumbar epidural space',
            'Spine.lumbar space', 'Spine.lumbar intervertebral disc',
            'Spine.lumbar intercostal arteries',
            'Sacrum', 'Sacrum epidural space', 'Sacroiliac joint',
            'Lumbosacral plexus',
        ],
        'Upper extremity': [
            'Shoulder', 'Shoulder.glenohumeral joint', 'Shoulder vessels',
            'Arm', 'Upper arm', 'Upper extremity', 'Upper extremity vessels',
            'Upper extremity veins', 'Upper extremity arteries', 'Upper extremity vein',
            'Upper extremity vessel', 'Upper extremity joint',
            'Elbow', 'Olecranon',
            'Forearm', 'Forearm vessels',
            'Wrist', 'Wrist.scaphoid', 'Wrist.hamate', 'Wrist.pisiform', 'Wrist vessels',
            'Hand', 'Hand vessels',
            'Finger', 'Finger.second', 'Finger.third', 'Finger.fourth', 'Finger.fifth', 'Thumb',
            'Humerus', 'Humerus.bicipital groove',
            'Radius', 'Ulna',
            'Carpal tunnel', 'Trapezium', 'Trapezoid', 'Triquetrum',
            'Axilla',
            'Brachial artery', 'Brachial plexus',
        ],
        'Lower extremity': [
            'Hip', 'Hip vessels',
            'Leg', 'Lower leg', 'Lower leg vessels',
            'Lower extremity', 'Lower extremity vessels', 'Lower extremity veins',
            'Lower extremity arteries', 'Lower extremity vein', 'Lower extremity artery',
            'Lower extremity vein', 'Lower extremity joint', 'Lower extremity arteries',
            'Thigh', 'Thigh.soft tissue', 'Thigh vessels',
            'Knee', 'Knee vessels',
            'Ankle', 'Ankle vessels', 'Ankle arteries',
            'Foot', 'Foot.sesamoid bones', 'Foot.cuneiform bones', 'Foot.subtalar joint',
            'Foot vessels', 'Foot joint',
            'Toe', 'Toe.second', 'Toe.third', 'Toe.fourth', 'Toe.fifth', 'Great toe', 'Toes',
            'Femur', 'Femoral artery', 'Femoral vein', 'Femoral vessels', 'Femoral vessel',
            'Tibia', 'Fibula',
            'Patella', 'Calcaneus',
            'Acetabulum',
            'Popliteal artery', 'Popliteal vein',
            'Tibioperoneal arteries', 'Tibioperoneal vessels',
            'Forefoot', 'Midfoot', 'Hindfoot',
            'Pedal lymphatic vessels',
            'Iliac artery', 'Iliac artery.internal',
        ],
        'Whole Body': [
            'Whole Body', 'Skull base to mid-thigh', 'Skeletal system',
            'Skeletal system.axial', 'Skeletal system.peripheral',
            'Bone', 'Bones', 'Bones.long',
            'Tendon', 'Tendon or ligament',
            'Ligament',
            'Joint', 'Joint.major', 'Joint.intermediate', 'Joint.small',
            'Vessel', 'Vessels', 'Vein', 'Veins',
            'Artery', 'Arteries',
            'Nerve', 'Nerve root', 'Nerves.cranial',
            'Lymph node', 'Lymphatic vessels', 'Lymphatic vessel',
            'Peripheral nerve', 'Peripheral veins', 'Peripheral arteries', 'Peripheral artery',
            'Peripheral vessels',
            'Musculoskeletal tissue',
            'Placenta',
            'Genitourinary tract',
            'Three vessels', 'Two vessels',
            'Airway', 'Tube', 'Catheter', 'Stent', 'Fistula', 'AV fistula', 'AV shunt', 'Shunt',
            'Pseudoaneurysm',
        ],
    }
    return FOCUS_TO_REGION


def generate_focus_region_report(csv_path, output_path):
    """
    Generate focus→region mapping report from LOINC playbook.
    Outputs mapped and unmapped foci to output file.
    """
    import csv
    from pathlib import Path

    # Load all foci from playbook
    foci = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['PartTypeName'] == 'Rad.Anatomic Location.Imaging Focus':
                foci.add(row['PartName'])

    # Get mapping
    focus_to_region = build_focus_to_region_map()

    # Find mapped/unmapped
    mapped_foci = set()
    for region, focus_list in focus_to_region.items():
        mapped_foci.update(focus_list)

    unmapped = sorted(foci - mapped_foci)

    # Write report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Focus → Region Mapping Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total unique imaging foci: {len(foci)}\n")
        f.write(f"Mapped: {len(mapped_foci)} ({len(mapped_foci)/len(foci)*100:.1f}%)\n")
        f.write(f"Unmapped: {len(unmapped)} ({len(unmapped)/len(foci)*100:.1f}%)\n\n")

        f.write("MAPPED FOCI BY REGION\n")
        f.write("-" * 80 + "\n\n")

        for region in GLOBAL_REGIONS:
            foci_list = focus_to_region.get(region, [])
            f.write(f"\n{region.upper()} ({len(foci_list)} foci)\n")
            f.write("-" * 40 + "\n")
            for focus in sorted(foci_list):
                f.write(f"  - {focus}\n")

        if unmapped:
            f.write("\n\nUNMAPPED FOCI\n")
            f.write("-" * 80 + "\n")
            f.write(f"({len(unmapped)} foci)\n\n")
            for focus in unmapped:
                f.write(f"  - {focus}\n")

    return output_file, len(mapped_foci), len(unmapped)


if __name__ == '__main__':
    import sys
    from pathlib import Path

    print("Modality Ontology Configuration\n")
    list_modalities()
    print(f"\nGlobal Regions: {len(GLOBAL_REGIONS)}")
    for region in GLOBAL_REGIONS:
        print(f"  - {region}")

    # Generate focus→region mapping report
    print("\n\nGenerating focus→region mapping report...")
    script_dir = Path(__file__).parent.parent
    csv_path = script_dir / 'data' / 'LoincRsnaRadiologyPlaybook.csv'
    output_path = script_dir / 'output' / 'focus_region_mapping.txt'

    if csv_path.exists():
        out_file, mapped_count, unmapped_count = generate_focus_region_report(str(csv_path), str(output_path))
        print(f"✓ Mapped {mapped_count} foci, {unmapped_count} unmapped")
        print(f"✓ Report saved to {out_file}")
    else:
        print(f"ERROR: CSV not found: {csv_path}")
