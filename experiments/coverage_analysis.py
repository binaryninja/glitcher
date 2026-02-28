#!/usr/bin/env python3
"""
Coverage Gap Analysis Script

Compares brute-force vocabulary scan results against entropy mining results
to quantify the coverage gap. Computes statistical tests and generates plots.

Usage:
    python experiments/coverage_analysis.py \
        --scan-results data/vocab_scan_results.json \
        --mining-results data/mining_results.json \
        [--output data/coverage_gap_analysis.json] \
        [--plots-dir plots/]
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_scan_results(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_mining_results(path):
    """Load mining results from glitch_tokens.json format."""
    with open(path) as f:
        data = json.load(f)
    return data


def compute_coverage_gap(scan_data, mining_data, asr_threshold=0.5):
    """Compute set differences between scan and mining results."""
    # Extract confirmed glitch token IDs from scan
    scan_glitch_ids = set()
    scan_all = {}
    for entry in scan_data.get("results", []):
        scan_all[entry["token_id"]] = entry
        if entry.get("is_glitch", False) or entry.get("asr", 0) >= asr_threshold:
            scan_glitch_ids.add(entry["token_id"])

    # Extract mining token IDs
    mining_ids = set(mining_data.get("glitch_token_ids", []))

    # Set operations
    only_scan = scan_glitch_ids - mining_ids
    only_mining = mining_ids - scan_glitch_ids
    both = scan_glitch_ids & mining_ids
    # Tokens in mining but not in sample (expected since sample is partial)
    mining_not_sampled = mining_ids - set(scan_all.keys())
    mining_sampled = mining_ids & set(scan_all.keys())

    return {
        "scan_glitch_count": len(scan_glitch_ids),
        "mining_glitch_count": len(mining_ids),
        "found_by_scan_only": sorted(only_scan),
        "found_by_mining_only": sorted(only_mining),
        "found_by_both": sorted(both),
        "mining_not_in_sample": sorted(mining_not_sampled),
        "mining_sampled": sorted(mining_sampled),
        "scan_glitch_ids": sorted(scan_glitch_ids),
        "scan_all": scan_all,
    }


def analyze_l2_norms(gap_data, scan_data):
    """Analyze L2 norm distributions for missed vs found tokens."""
    scan_all = gap_data["scan_all"]
    norm_mean = scan_data.get("l2_norm_mean", 0)
    norm_std = scan_data.get("l2_norm_std", 1)

    # Norms of tokens found only by scan (missed by mining)
    missed_norms = []
    for tid in gap_data["found_by_scan_only"]:
        if tid in scan_all:
            missed_norms.append(scan_all[tid]["l2_norm"])

    # Norms of tokens found by both
    found_norms = []
    for tid in gap_data["found_by_both"]:
        if tid in scan_all:
            found_norms.append(scan_all[tid]["l2_norm"])

    # All scan glitch norms
    all_glitch_norms = []
    for tid in gap_data["scan_glitch_ids"]:
        if tid in scan_all:
            all_glitch_norms.append(scan_all[tid]["l2_norm"])

    # Within 1 sigma of mean
    within_1sigma = sum(1 for n in missed_norms if abs(n - norm_mean) <= norm_std)

    # KS test
    ks_stat = None
    ks_pvalue = None
    if missed_norms and found_norms:
        try:
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(missed_norms, found_norms)
        except ImportError:
            pass

    return {
        "missed_norms": missed_norms,
        "found_norms": found_norms,
        "all_glitch_norms": all_glitch_norms,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "missed_within_1sigma": within_1sigma,
        "missed_total": len(missed_norms),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
    }


def analyze_categories(gap_data, scan_data):
    """Analyze Unicode category and behavioral distributions."""
    scan_all = gap_data["scan_all"]

    # Unicode categories for missed tokens
    missed_ucats = Counter()
    for tid in gap_data["found_by_scan_only"]:
        if tid in scan_all:
            missed_ucats[scan_all[tid].get("unicode_category", "unknown")] += 1

    # Unicode categories for found tokens
    found_ucats = Counter()
    for tid in gap_data["found_by_both"]:
        if tid in scan_all:
            found_ucats[scan_all[tid].get("unicode_category", "unknown")] += 1

    # ASR distribution for all scan glitches
    asr_values = []
    for tid in gap_data["scan_glitch_ids"]:
        if tid in scan_all:
            asr_values.append(scan_all[tid].get("asr", 0))

    intermediate_asr = [a for a in asr_values if 0.5 <= a < 0.95]
    full_asr = [a for a in asr_values if a >= 0.95]

    return {
        "missed_unicode_categories": dict(missed_ucats),
        "found_unicode_categories": dict(found_ucats),
        "asr_values": asr_values,
        "intermediate_asr_count": len(intermediate_asr),
        "full_asr_count": len(full_asr),
        "intermediate_asr_fraction": len(intermediate_asr) / len(asr_values) if asr_values else 0,
    }


def generate_plots(norm_analysis, category_analysis, gap_data, asr_before=None, asr_after=None, plots_dir="plots"):
    """Generate all required plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: L2 norm comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if norm_analysis["missed_norms"]:
        ax.hist(norm_analysis["missed_norms"], bins=30, alpha=0.6, label="Missed by mining", color="red")
    if norm_analysis["found_norms"]:
        ax.hist(norm_analysis["found_norms"], bins=30, alpha=0.6, label="Found by mining", color="blue")
    ax.axvline(norm_analysis["norm_mean"], color="black", linestyle="--", label=f"Vocab mean ({norm_analysis['norm_mean']:.3f})")
    ax.axvline(norm_analysis["norm_mean"] - norm_analysis["norm_std"], color="gray", linestyle=":", alpha=0.5)
    ax.axvline(norm_analysis["norm_mean"] + norm_analysis["norm_std"], color="gray", linestyle=":", alpha=0.5, label="+/- 1 sigma")
    ax.set_xlabel("L2 Embedding Norm")
    ax.set_ylabel("Count")
    ax.set_title("L2 Norm Distribution: Mining-Found vs Missed Tokens")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "l2_norm_comparison.png"), dpi=150)
    plt.close()

    # --- Plot 2: ASR distribution (before/after if available) ---
    fig, axes = plt.subplots(1, 2 if (asr_before and asr_after) else 1, figsize=(12 if (asr_before and asr_after) else 8, 5))
    if asr_before and asr_after:
        ax1, ax2 = axes
        ax1.hist(asr_before, bins=20, range=(0, 1), alpha=0.7, color="orange", edgecolor="black")
        ax1.set_title("ASR Distribution (Before Patch)")
        ax1.set_xlabel("ASR")
        ax1.set_ylabel("Count")
        ax2.hist(asr_after, bins=20, range=(0, 1), alpha=0.7, color="green", edgecolor="black")
        ax2.set_title("ASR Distribution (After Patch)")
        ax2.set_xlabel("ASR")
        ax2.set_ylabel("Count")
    else:
        ax = axes if not isinstance(axes, np.ndarray) else axes[0]
        asr_vals = category_analysis.get("asr_values", [])
        if asr_vals:
            ax.hist(asr_vals, bins=20, range=(0, 1), alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title("ASR Distribution (Scan Results)")
        ax.set_xlabel("ASR")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "asr_distribution_before_after.png"), dpi=150)
    plt.close()

    # --- Plot 3: Coverage Venn-style bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Scan Only\n(Missed by Mining)", "Both Methods", "Mining Only\n(Not in Sample)"]
    values = [
        len(gap_data["found_by_scan_only"]),
        len(gap_data["found_by_both"]),
        len(gap_data["mining_not_in_sample"]),
    ]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of Glitch Tokens")
    ax.set_title("Coverage Comparison: Vocab Scan vs Entropy Mining")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "coverage_venn.png"), dpi=150)
    plt.close()

    # --- Plot 4: Category breakdown ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    missed_cats = category_analysis.get("missed_unicode_categories", {})
    found_cats = category_analysis.get("found_unicode_categories", {})
    if missed_cats:
        ax1.bar(missed_cats.keys(), missed_cats.values(), color="#e74c3c", edgecolor="black")
        ax1.set_title("Unicode Categories: Missed by Mining")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Count")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    if found_cats:
        ax2.bar(found_cats.keys(), found_cats.values(), color="#2ecc71", edgecolor="black")
        ax2.set_title("Unicode Categories: Found by Mining")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Count")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_breakdown.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {plots_dir}/")


def compute_binomial_plausibility(asr_values, num_attempts, true_asr=0.95):
    """
    Compute the probability of observing zero intermediate ASR values
    under a binomial model.
    """
    from math import comb
    # P(all N attempts succeed) + P(all N attempts fail)
    p_all_succeed = true_asr ** num_attempts
    p_all_fail = (1 - true_asr) ** num_attempts
    p_extreme = p_all_succeed + p_all_fail

    # For K tokens, P(all tokens have extreme ASR)
    n_tokens = len(asr_values)
    intermediate = sum(1 for a in asr_values if 0.0 < a < 1.0)
    p_no_intermediate = p_extreme ** n_tokens if n_tokens > 0 else 1.0

    return {
        "p_extreme_single": p_extreme,
        "p_no_intermediate_all_tokens": p_no_intermediate,
        "n_tokens": n_tokens,
        "n_intermediate": intermediate,
        "num_attempts": num_attempts,
        "assumed_true_asr": true_asr,
    }


def main():
    parser = argparse.ArgumentParser(description="Coverage Gap Analysis")
    parser.add_argument("--scan-results", required=True, help="Path to vocab scan results JSON")
    parser.add_argument("--mining-results", required=True, help="Path to mining results JSON (glitch_tokens.json format)")
    parser.add_argument("--asr-before", default=None, help="Path to ASR audit JSON (before patch)")
    parser.add_argument("--asr-after", default=None, help="Path to ASR audit JSON (after patch)")
    parser.add_argument("--output", default="data/coverage_gap_analysis.json")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--asr-threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("Loading scan results...")
    scan_data = load_scan_results(args.scan_results)
    print(f"  {len(scan_data.get('results', []))} tokens in scan")

    print("Loading mining results...")
    mining_data = load_mining_results(args.mining_results)
    mining_ids = mining_data.get("glitch_token_ids", [])
    print(f"  {len(set(mining_ids))} unique mining token IDs")

    print("Computing coverage gap...")
    gap_data = compute_coverage_gap(scan_data, mining_data, args.asr_threshold)
    print(f"  Scan glitches: {gap_data['scan_glitch_count']}")
    print(f"  Mining glitches: {gap_data['mining_glitch_count']}")
    print(f"  Found by scan only: {len(gap_data['found_by_scan_only'])}")
    print(f"  Found by both: {len(gap_data['found_by_both'])}")
    print(f"  Mining not in sample: {len(gap_data['mining_not_in_sample'])}")

    print("Analyzing L2 norms...")
    norm_analysis = analyze_l2_norms(gap_data, scan_data)
    if norm_analysis["missed_total"] > 0:
        print(f"  Missed tokens within 1 sigma: {norm_analysis['missed_within_1sigma']}/{norm_analysis['missed_total']}")
    if norm_analysis["ks_pvalue"] is not None:
        print(f"  KS test: statistic={norm_analysis['ks_statistic']:.4f}, p={norm_analysis['ks_pvalue']:.4e}")

    print("Analyzing categories...")
    category_analysis = analyze_categories(gap_data, scan_data)
    print(f"  Intermediate ASR tokens: {category_analysis['intermediate_asr_count']}")
    print(f"  Full ASR tokens: {category_analysis['full_asr_count']}")

    # Load before/after ASR data if available
    asr_before_vals = None
    asr_after_vals = None
    if args.asr_before:
        with open(args.asr_before) as f:
            asr_before_data = json.load(f)
        asr_before_vals = [t["asr"] for t in asr_before_data.get("tokens", [])]
    if args.asr_after:
        with open(args.asr_after) as f:
            asr_after_data = json.load(f)
        asr_after_vals = [t["asr"] for t in asr_after_data.get("tokens", [])]

    # Binomial plausibility check
    num_attempts = scan_data.get("num_attempts", 5)
    binomial = compute_binomial_plausibility(category_analysis["asr_values"], num_attempts)

    print("\nGenerating plots...")
    generate_plots(norm_analysis, category_analysis, gap_data, asr_before_vals, asr_after_vals, args.plots_dir)

    # Coverage gap as percentage increase
    if gap_data["scan_glitch_count"] > 0:
        coverage_gap_pct = len(gap_data["found_by_scan_only"]) / gap_data["scan_glitch_count"] * 100
    else:
        coverage_gap_pct = 0

    # Save analysis
    analysis = {
        "scan_results_path": args.scan_results,
        "mining_results_path": args.mining_results,
        "asr_threshold": args.asr_threshold,
        "coverage_gap": {
            "scan_glitch_count": gap_data["scan_glitch_count"],
            "mining_glitch_count": gap_data["mining_glitch_count"],
            "found_by_scan_only_count": len(gap_data["found_by_scan_only"]),
            "found_by_scan_only_ids": gap_data["found_by_scan_only"],
            "found_by_both_count": len(gap_data["found_by_both"]),
            "found_by_both_ids": gap_data["found_by_both"],
            "mining_not_in_sample_count": len(gap_data["mining_not_in_sample"]),
            "coverage_gap_pct": coverage_gap_pct,
        },
        "l2_norm_analysis": {
            "missed_within_1sigma": norm_analysis["missed_within_1sigma"],
            "missed_total": norm_analysis["missed_total"],
            "missed_within_1sigma_pct": (norm_analysis["missed_within_1sigma"] / norm_analysis["missed_total"] * 100)
                if norm_analysis["missed_total"] > 0 else 0,
            "ks_statistic": norm_analysis["ks_statistic"],
            "ks_pvalue": norm_analysis["ks_pvalue"],
            "vocab_l2_mean": norm_analysis["norm_mean"],
            "vocab_l2_std": norm_analysis["norm_std"],
        },
        "category_analysis": {
            "missed_unicode_categories": category_analysis["missed_unicode_categories"],
            "found_unicode_categories": category_analysis["found_unicode_categories"],
            "intermediate_asr_count": category_analysis["intermediate_asr_count"],
            "intermediate_asr_fraction": category_analysis["intermediate_asr_fraction"],
        },
        "binomial_plausibility": binomial,
        "plots": {
            "l2_norm_comparison": os.path.join(args.plots_dir, "l2_norm_comparison.png"),
            "asr_distribution": os.path.join(args.plots_dir, "asr_distribution_before_after.png"),
            "coverage_venn": os.path.join(args.plots_dir, "coverage_venn.png"),
            "category_breakdown": os.path.join(args.plots_dir, "category_breakdown.png"),
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to {args.output}")


if __name__ == "__main__":
    main()
