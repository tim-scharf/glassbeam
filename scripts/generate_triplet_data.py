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


def triplet_generator(df, batch_size=32):
    """
    Generate triplets: (query, positive, negative)

    For each triplet:
    - Positive set: random region set {region, focus1, focus2, ...}
    - Query, Positive: 2 random elements from positive set (same region = similar)
    - Negative: 1 random element from any other region set (different region = dissimilar)
    """
    import random

    region_sets = create_region_sets(df)
    region_list = list(region_sets.keys())
    # Flatten all elements from non-positive regions for fast negative sampling
    all_regions_flat = {r: list(s) for r, s in region_sets.items()}

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
                # Fallback if region has <2 elements
                query = positive = random.choice(pos_set_list)

            # Sample negative from any other region
            other_regions = [r for r in region_list if r != pos_region]
            neg_region = random.choice(other_regions)
            negative = random.choice(all_regions_flat[neg_region])

            batch.append((query, positive, negative))

        yield batch


def save_triplets(num_triplets=10000, output_path='output/triplet_data.csv'):
    """Generate and save triplets to CSV."""
    import csv

    df = create_mapping_dataframe()
    gen = triplet_generator(df, batch_size=32)

    print(f"Generating {num_triplets:,} triplets...")

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
    print("Loading focus→region ontology mapping...\n")

    df = create_mapping_dataframe()

    print(f"Created focus→region dataframe: {len(df)} rows × {len(df.columns)} columns\n")

    print("Testing triplet generator:\n")
    gen = triplet_generator(df, batch_size=5)
    batch = next(gen)

    print(f"{'Query':<10} {'Positive':<40} {'Negative':<40}")
    print("-" * 90)
    for query, positive, negative in batch:
        print(f"{query:<10} {positive:<40} {negative:<40}")

    print("\n\nSaving 10k triplets to CSV...")
    save_triplets(num_triplets=10000)


if __name__ == '__main__':
    main()
