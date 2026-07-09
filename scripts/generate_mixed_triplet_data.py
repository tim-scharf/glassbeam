"""
generate_mixed_triplet_data.py
------------------------------
Generate triplet data with mixed strategy:
- Query: focus (e.g., "Brain")
- Positive: region (e.g., "Head")
- Negative: 50% focus from same region, 50% nearby region name

Usage:
    python3 scripts/generate_mixed_triplet_data.py --num 100000 --output output/triplet_data_mixed.csv
"""

import argparse
import csv
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ontology_mapper import build_focus_to_region_map


def load_focus_region_mapping():
    """Load focus→region mapping from ontology."""
    region_to_foci = build_focus_to_region_map()

    # Invert to create focus→region lookup
    focus_to_region = {}
    for region, foci_list in region_to_foci.items():
        for focus in foci_list:
            focus_to_region[focus] = region

    return focus_to_region, region_to_foci


def define_region_groups():
    """Define which regions are 'nearby' to each other."""
    return {
        'upper': ['Head', 'Neck', 'Chest'],
        'core': ['Abdomen', 'Pelvis', 'Spine'],
        'limbs': ['Upper extremity', 'Lower extremity'],
        'all': ['Whole Body'],
    }


def get_nearby_regions(region, region_groups):
    """Get regions in the same group as the given region."""
    for group_name, regions in region_groups.items():
        if region in regions:
            return [r for r in regions if r != region]
    return []


def generate_mixed_triplets(focus_to_region, region_to_foci, region_groups, num_triplets=100000):
    """
    Generate triplets with mixed strategy:
    - Query: focus
    - Positive: region name
    - Negative: 50% focus from same region, 50% nearby region name
    """
    triplets = []

    regions = list(region_to_foci.keys())
    foci_list = list(focus_to_region.keys())

    print(f"Generating {num_triplets:,} mixed triplets...")

    for i in range(num_triplets):
        # Sample a random focus as query
        query_focus = random.choice(foci_list)
        query_region = focus_to_region[query_focus]

        # Positive is the region name
        positive = query_region

        # Negative: 50/50 between intra-region focus and nearby region
        if random.random() < 0.5:
            # Hard negative: another focus from same region
            same_region_foci = [f for f in region_to_foci[query_region] if f != query_focus]
            if same_region_foci:
                negative = random.choice(same_region_foci)
            else:
                # Fallback: nearby region
                nearby = get_nearby_regions(query_region, region_groups)
                negative = random.choice(nearby) if nearby else random.choice([r for r in regions if r != query_region])
        else:
            # Softer negative: nearby region name
            nearby = get_nearby_regions(query_region, region_groups)
            if nearby:
                negative = random.choice(nearby)
            else:
                # Fallback: any other region
                negative = random.choice([r for r in regions if r != query_region])

        triplets.append((query_focus, positive, negative))

        if (i + 1) % 10000 == 0:
            print(f"  Generated {len(triplets):,} triplets...")

    return triplets


def main():
    parser = argparse.ArgumentParser(description='Generate mixed triplet data')
    parser.add_argument('--num', type=int, default=100000, help='Number of triplets')
    parser.add_argument('--output', type=str, default='output/triplet_data_mixed.csv', help='Output CSV path')
    args = parser.parse_args()

    print("Loading focus→region mapping...\n")
    focus_to_region, region_to_foci = load_focus_region_mapping()
    region_groups = define_region_groups()

    print(f"Total foci: {len(focus_to_region)}")
    print(f"Total regions: {len(region_to_foci)}\n")

    print("Region groups:")
    for group_name, regions in region_groups.items():
        print(f"  {group_name}: {regions}")
    print()

    # Generate triplets
    triplets = generate_mixed_triplets(focus_to_region, region_to_foci, region_groups, args.num)

    # Save to CSV
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'positive', 'negative'])
        writer.writerows(triplets)

    print(f"\n✓ Saved {len(triplets):,} triplets to {output_file}")

    # Show sample triplets
    print("\nSample triplets:")
    print(f"{'Query':<25} {'Positive':<20} {'Negative':<25}")
    print("-" * 70)
    for i in range(min(5, len(triplets))):
        q, p, n = triplets[i]
        print(f"{q:<25} {p:<20} {n:<25}")


if __name__ == '__main__':
    main()
