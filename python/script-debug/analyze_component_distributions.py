#!/usr/bin/env python3
"""
Analyze distributions of Jaccard similarities for direct vs transitive edges
across all components from component analysis output.
"""

import json
import sys
from collections import defaultdict


def analyze_distributions(component_file):
    """Extract all direct and transitive Jaccard scores."""
    direct_jaccards = []
    transitive_jaccards = []

    component_stats = {
        'total_components': 0,
        'fully_connected': 0,
        'has_transitive': 0,
        'total_direct_edges': 0,
        'total_transitive_edges': 0,
    }

    with open(component_file) as f:
        for line in f:
            comp = json.loads(line)
            component_stats['total_components'] += 1

            if comp['is_fully_connected']:
                component_stats['fully_connected'] += 1

            if comp['num_transitive_edges'] > 0:
                component_stats['has_transitive'] += 1

            # Collect direct edge Jaccards
            for edge in comp.get('direct_edges', []):
                direct_jaccards.append(edge['jaccard'])
                component_stats['total_direct_edges'] += 1

            # Collect transitive edge Jaccards
            for edge in comp.get('transitive_edges', []):
                transitive_jaccards.append(edge['jaccard'])
                component_stats['total_transitive_edges'] += 1

    return direct_jaccards, transitive_jaccards, component_stats


def compute_histogram(values, bins=20, min_val=0.0, max_val=1.0):
    """Compute histogram with fixed bins."""
    bin_width = (max_val - min_val) / bins
    hist = [0] * bins

    for val in values:
        if val < min_val or val > max_val:
            continue
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        hist[bin_idx] += 1

    return hist, bin_width


def print_histogram(values, title, bins=20):
    """Print ASCII histogram."""
    if not values:
        print(f"\n{title}: NO DATA")
        return

    print(f"\n{title}")
    print(f"Count: {len(values)}")
    print(f"Mean:  {sum(values)/len(values):.4f}")
    print(f"Min:   {min(values):.4f}")
    print(f"Max:   {max(values):.4f}")

    # Compute percentiles
    sorted_vals = sorted(values)
    p10 = sorted_vals[int(len(sorted_vals) * 0.10)]
    p25 = sorted_vals[int(len(sorted_vals) * 0.25)]
    p50 = sorted_vals[int(len(sorted_vals) * 0.50)]
    p75 = sorted_vals[int(len(sorted_vals) * 0.75)]
    p90 = sorted_vals[int(len(sorted_vals) * 0.90)]

    print(f"P10:   {p10:.4f}")
    print(f"P25:   {p25:.4f}")
    print(f"P50:   {p50:.4f}")
    print(f"P75:   {p75:.4f}")
    print(f"P90:   {p90:.4f}")

    # Compute histogram
    hist, bin_width = compute_histogram(values, bins=bins)
    max_count = max(hist) if hist else 1

    print(f"\nDistribution (bin width: {bin_width:.3f}):")
    print("-" * 70)

    for i, count in enumerate(hist):
        bin_start = i * bin_width
        bin_end = (i + 1) * bin_width

        # Scale bar to 50 characters max
        bar_length = int(50 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_length

        print(f"{bin_start:.2f}-{bin_end:.2f} | {count:5d} | {bar}")


def print_summary_stats(component_stats):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("COMPONENT ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total Components:           {component_stats['total_components']}")
    print(f"Fully Connected:            {component_stats['fully_connected']} ({100*component_stats['fully_connected']/component_stats['total_components']:.1f}%)")
    print(f"With Transitive Edges:      {component_stats['has_transitive']} ({100*component_stats['has_transitive']/component_stats['total_components']:.1f}%)")
    print(f"Total Direct Edges:         {component_stats['total_direct_edges']}")
    print(f"Total Transitive Edges:     {component_stats['total_transitive_edges']}")

    total_edges = component_stats['total_direct_edges'] + component_stats['total_transitive_edges']
    if total_edges > 0:
        print(f"Direct Edge Rate:           {100*component_stats['total_direct_edges']/total_edges:.1f}%")


def compare_distributions(direct_jaccards, transitive_jaccards):
    """Compare direct vs transitive distributions."""
    print("\n" + "=" * 70)
    print("COMPARISON: Direct vs Transitive Edges")
    print("=" * 70)

    if direct_jaccards and transitive_jaccards:
        direct_mean = sum(direct_jaccards) / len(direct_jaccards)
        trans_mean = sum(transitive_jaccards) / len(transitive_jaccards)

        print(f"\nMean Jaccard:")
        print(f"  Direct:      {direct_mean:.4f}")
        print(f"  Transitive:  {trans_mean:.4f}")
        print(f"  Difference:  {direct_mean - trans_mean:.4f}")

        # Count edges below threshold
        threshold = 0.7
        direct_below = sum(1 for j in direct_jaccards if j < threshold)
        trans_below = sum(1 for j in transitive_jaccards if j < threshold)

        print(f"\nBelow {threshold} threshold:")
        print(f"  Direct:      {direct_below}/{len(direct_jaccards)} ({100*direct_below/len(direct_jaccards):.1f}%)")
        print(f"  Transitive:  {trans_below}/{len(transitive_jaccards)} ({100*trans_below/len(transitive_jaccards):.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_component_distributions.py <component_analysis_results.jsonl>")
        sys.exit(1)

    component_file = sys.argv[1]

    print("Analyzing component distributions...")
    direct_jaccards, transitive_jaccards, component_stats = analyze_distributions(component_file)

    print_summary_stats(component_stats)
    print_histogram(direct_jaccards, "DIRECT EDGES (MinHash Band Matches)", bins=20)
    print_histogram(transitive_jaccards, "TRANSITIVE EDGES (Union-Find Only)", bins=20)
    compare_distributions(direct_jaccards, transitive_jaccards)


if __name__ == '__main__':
    main()
