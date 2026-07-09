"""
generate_triplet_data.py
------------------------
Generate triplet loss training data for joint region-focus extraction.

Triplets: (anchor: study_desc_raw, positive: {region, focus}, negative: {region, focus})

Usage:
    python3 scripts/generate_triplet_data.py
"""

import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))

from ontology_mapper import build_focus_to_region_map, GLOBAL_REGIONS


def load_focus_region_mapping():
    """Load focus→region mapping from ontology."""
    focus_to_region = build_focus_to_region_map()

    # Invert to create focus→region lookup
    focus_lookup = {}
    for region, foci_list in focus_to_region.items():
        for focus in foci_list:
            focus_lookup[focus] = region

    return focus_to_region, focus_lookup


def create_mapping_dataframe():
    """Create dataframe: region, focus."""
    focus_to_region, focus_lookup = load_focus_region_mapping()

    rows = []
    for focus, region in sorted(focus_lookup.items()):
        rows.append({'region': region, 'focus': focus})

    df = pd.DataFrame(rows)
    return df


def create_region_sets(df):
    """
    Create a set for each region containing the region name + all its foci.

    Example: Head region set = {"Head", "Brain", "Brain stem", "Skull", ...}
    """
    region_sets = {}
    for region in df['region'].unique():
        foci = set(df[df['region'] == region]['focus'].tolist())
        region_sets[region] = {region} | foci  # region name + all foci
    return region_sets


def triplet_generator(df, batch_size=32, hard_ratio=0.0, region_groups=None):
    """
    Generate triplets: (query, positive, negative)

    For each triplet:
    - Positive set: random region set {region, focus1, focus2, ...}
    - Query, Positive: 2 random elements from positive set (same region = similar)
    - Negative: 1 random element
      - With prob hard_ratio: from SAME group (harder negative)
      - Else: from DIFFERENT group (easy negative)

    region_groups: dict mapping group_name -> [region1, region2, ...]
    """
    import random

    region_sets = create_region_sets(df)
    region_list = list(region_sets.keys())
    all_regions_flat = {r: list(s) for r, s in region_sets.items()}

    # Build region to group mapping if provided
    region_to_group = {}
    if region_groups:
        for group_name, regions in region_groups.items():
            for region in regions:
                region_to_group[region] = group_name

    while True:
        batch = []

        for _ in range(batch_size):
            # Sample positive region
            pos_region = random.choice(region_list)
            pos_set_list = all_regions_flat[pos_region]

            # Sample 2 different elements from positive set
            if len(pos_set_list) >= 2:
                query, positive = random.sample(pos_set_list, 2)
            else:
                query = positive = random.choice(pos_set_list)

            # Sample negative
            if region_groups and random.random() < hard_ratio:
                # Hard negative: from same group
                pos_group = region_to_group.get(pos_region)
                if pos_group:
                    same_group_regions = [r for r in region_groups[pos_group] if r != pos_region]
                    if same_group_regions:
                        neg_region = random.choice(same_group_regions)
                    else:
                        # Fallback to any other region
                        neg_region = random.choice([r for r in region_list if r != pos_region])
                else:
                    neg_region = random.choice([r for r in region_list if r != pos_region])
            else:
                # Easy negative: from different group
                if region_groups:
                    pos_group = region_to_group.get(pos_region)
                    other_group_regions = [r for r in region_list if region_to_group.get(r) != pos_group]
                    if other_group_regions:
                        neg_region = random.choice(other_group_regions)
                    else:
                        neg_region = random.choice([r for r in region_list if r != pos_region])
                else:
                    neg_region = random.choice([r for r in region_list if r != pos_region])

            negative = random.choice(all_regions_flat[neg_region])

            batch.append((query, positive, negative))

        yield batch


def save_triplets(num_triplets=10000, output_path='output/triplet_data.csv', hard_ratio=0.0, region_groups=None):
    """Generate and save triplets to CSV."""
    import csv

    df = create_mapping_dataframe()
    gen = triplet_generator(df, batch_size=32, hard_ratio=hard_ratio, region_groups=region_groups)

    hard_str = f" (hard_ratio={hard_ratio})" if hard_ratio > 0 else ""
    print(f"Generating {num_triplets:,} triplets{hard_str}...")

    triplets = []
    batches_needed = (num_triplets + 31) // 32

    for i in range(batches_needed):
        batch = next(gen)
        triplets.extend(batch)
        if (i + 1) % 50 == 0:
            print(f"  Generated {len(triplets):,} triplets...")

    triplets = triplets[:num_triplets]

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'positive', 'negative'])
        writer.writerows(triplets)

    print(f"\n✓ Saved {len(triplets):,} triplets to {output_file}")
    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate triplet data with optional hard negatives')
    parser.add_argument('--num', type=int, default=100000, help='Number of triplets to generate')
    parser.add_argument('--hard-ratio', type=float, default=0.0, help='Ratio of hard negatives (0-1)')
    parser.add_argument('--output', type=str, default='output/triplet_data.csv', help='Output CSV path')
    args = parser.parse_args()

    print("Loading focus→region ontology mapping...\n")

    df = create_mapping_dataframe()

    print(f"Created focus→region dataframe: {len(df)} rows × {len(df.columns)} columns\n")

    # Define region groups for hard negative mining
    region_groups = {
        'upper': ['Head', 'Neck', 'Chest'],
        'core': ['Abdomen', 'Pelvis', 'Spine'],
        'limbs': ['Upper extremity', 'Lower extremity'],
        'all': ['Whole Body'],
    }

    print("Region groups:")
    for group_name, regions in region_groups.items():
        print(f"  {group_name}: {regions}")
    print()

    print("Testing triplet generator:\n")
    gen = triplet_generator(df, batch_size=5, hard_ratio=args.hard_ratio, region_groups=region_groups)
    batch = next(gen)

    print(f"{'Query':<20} {'Positive':<30} {'Negative':<30}")
    print("-" * 80)
    for query, positive, negative in batch:
        print(f"{query:<20} {positive:<30} {negative:<30}")

    print(f"\n\nSaving {args.num:,} triplets to {args.output}...")
    save_triplets(
        num_triplets=args.num,
        output_path=args.output,
        hard_ratio=args.hard_ratio,
        region_groups=region_groups
    )


if __name__ == '__main__':
    main()
