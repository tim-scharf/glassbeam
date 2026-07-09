"""
generate_stratified_triplets.py
-------------------------------
Generate triplet data with stratified region sampling:
- Equal probability per region (stratified)
- Query: random focus from selected region
- Positive: region name
- Negative: random focus from different region

Usage:
    python3 scripts/generate_stratified_triplets.py --num 100000
"""

import argparse
import csv
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ontology_mapper import build_focus_to_region_map


def main():
    parser = argparse.ArgumentParser(description='Generate stratified triplet data')
    parser.add_argument('--num', type=int, default=100000, help='Number of triplets')
    parser.add_argument('--output', type=str, default='output/triplet_data_stratified.csv', help='Output CSV path')
    args = parser.parse_args()

    print("Loading focus→region mapping...\n")
    region_to_foci = build_focus_to_region_map()

    regions = list(region_to_foci.keys())
    num_regions = len(regions)

    print(f"Total regions: {num_regions}")
    print(f"Total foci: {sum(len(f) for f in region_to_foci.values())}\n")

    print(f"Generating {args.num:,} stratified triplets...\n")

    triplets = []
    for i in range(args.num):
        # Stratified: equal probability per region
        query_region = random.choice(regions)
        query_foci = region_to_foci[query_region]

        # Within region: random focus
        query = random.choice(query_foci)

        # Positive: region name
        positive = query_region

        # Negative: random region name (different from query region)
        other_regions = [r for r in regions if r != query_region]
        negative = random.choice(other_regions)

        triplets.append((query, positive, negative))

        if (i + 1) % 10000 == 0:
            print(f"  Generated {len(triplets):,} triplets...")

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
    for i in range(min(10, len(triplets))):
        q, p, n = triplets[i]
        print(f"{q:<25} {p:<20} {n:<25}")


if __name__ == '__main__':
    main()
