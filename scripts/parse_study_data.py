"""
parse_study_data.py
-------------------
Parse DICOM and HL7 study data from Excel files in Larger_Dataset folder.
Combines data from multiple study sources and filters by modality frequency.

Usage:
    python3 scripts/parse_study_data.py
    python3 scripts/parse_study_data.py --input-dir data/Larger_Dataset --output output/glassbeam_data.csv
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def load_excel_files(dataset_dir):
    """Load all Excel files from dataset directory recursively."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    xlsx_files = list(dataset_path.rglob("*.xlsx"))
    if not xlsx_files:
        print(f"ERROR: No .xlsx files found in {dataset_path}")
        sys.exit(1)

    print(f"Found {len(xlsx_files)} Excel files")
    return xlsx_files


def extract_customer_name(file_path):
    """Extract customer name from file path (last part before .xlsx)."""
    name = file_path.name
    return name.split('.')[0]


def read_and_combine_files(xlsx_files):
    """Read all Excel files and combine into single DataFrame."""
    dfs = []

    for idx, file_path in enumerate(xlsx_files, 1):
        customer = extract_customer_name(file_path)
        try:
            print(f"  [{idx}/{len(xlsx_files)}] Reading {customer}...", end='', flush=True)
            start = time.time()
            df = pd.read_excel(file_path, engine='calamine')
            df['customer'] = customer
            dfs.append(df)
            elapsed = time.time() - start
            print(f" ✓ {len(df):,} records ({elapsed:.2f}s)")
        except Exception as e:
            print(f" ✗ SKIP: {e}")
            continue

    if not dfs:
        print("ERROR: No files were successfully read")
        sys.exit(1)

    print(f"Combining {len(dfs)} files...")
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def process_data(df):
    """Create DICOM and HL7 views, combine, filter, and deduplicate."""

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Check for required columns
    required_cols = {'dicom_modality', 'customer'}
    available_cols = set(df.columns)

    if not required_cols.issubset(available_cols):
        print(f"ERROR: Missing required columns. Available: {sorted(available_cols)}")
        sys.exit(1)

    # DICOM view
    dicom_cols = ['dicom_modality', 'customer']
    dicom_study_col = 'dicom_studydesc' if 'dicom_studydesc' in df.columns else None

    if dicom_study_col:
        dicom = df[[dicom_cols[0], dicom_study_col, dicom_cols[1]]].copy()
        dicom.columns = ['modality', 'study_desc_raw', 'customer']
        dicom['type'] = 'dicom'
    else:
        dicom = None

    # HL7 view
    hl7_study_col = 'hl7_studydesc' if 'hl7_studydesc' in df.columns else None

    if hl7_study_col:
        hl7 = df[[dicom_cols[0], hl7_study_col, dicom_cols[1]]].copy()
        hl7.columns = ['modality', 'study_desc_raw', 'customer']
        hl7['type'] = 'hl7'
    else:
        hl7 = None

    # Combine views
    views = [v for v in [dicom, hl7] if v is not None]
    if not views:
        print("ERROR: No DICOM or HL7 study columns found")
        sys.exit(1)

    combined = pd.concat(views, ignore_index=True)

    # Remove NAs
    combined = combined.dropna(subset=['modality'])

    # Remove duplicates by study description
    combined = combined.drop_duplicates(subset=['study_desc_raw'])

    return combined


def filter_by_modality(df, min_count=100, exclude=None):
    """Keep only modalities with >min_count records, excluding specified ones."""
    if exclude is None:
        exclude = {'SR', 'OT'}

    mod_counts = df['modality'].value_counts()
    keep_mods = mod_counts[mod_counts > min_count].index.tolist()
    keep_mods = [m for m in keep_mods if m not in exclude]

    print(f"\nModality summary (before filter):")
    print(f"  Total modalities: {len(mod_counts)}")
    print(f"  Keeping {len(keep_mods)} with >100 records (excluding {exclude})")
    print(f"\n  Top modalities:")
    for mod, count in mod_counts.head(15).items():
        status = "✓" if mod in keep_mods else "✗"
        print(f"    {status} {mod:<15} {count:>6,} records")

    filtered = df[df['modality'].isin(keep_mods)].copy()
    return filtered


def main():
    parser = argparse.ArgumentParser(description='Parse study data from Excel files.')
    parser.add_argument('--input-dir', '-i', help='Input directory with Excel files (default: data/Larger_Dataset)')
    parser.add_argument('--output', '-o', help='Output CSV path (default: output/glassbeam_data.csv)')
    parser.add_argument('--min-count', type=int, default=100, help='Minimum modality count to keep (default: 100)')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.parent
    input_dir = Path(args.input_dir) if args.input_dir else script_dir / 'data' / 'Larger_Dataset'
    output_path = Path(args.output) if args.output else script_dir / 'output' / 'glassbeam_data.csv'

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Excel files from {input_dir}...")
    xlsx_files = load_excel_files(input_dir)

    print(f"\nReading {len(xlsx_files)} files...")
    combined_df = read_and_combine_files(xlsx_files)
    print(f"Combined: {len(combined_df):,} rows × {len(combined_df.columns)} columns")

    print(f"\nProcessing data...")
    processed_df = process_data(combined_df)
    print(f"After deduplication: {len(processed_df):,} unique studies")

    print(f"\nFiltering by modality (min {args.min_count} records)...")
    filtered_df = filter_by_modality(processed_df, min_count=args.min_count)
    print(f"Final dataset: {len(filtered_df):,} rows")

    print(f"\nWriting to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Done! ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
