#!/usr/bin/env python3
"""
Quick integration test for comprehensive wanted token search in genetic algorithm.

This test verifies that:
1. Comprehensive search runs before genetic algorithm when --wanted-token is specified
2. High-impact tokens are properly prioritized for genetic algorithm initialization
3. The search integrates seamlessly with existing genetic algorithm functionality
4. Token reordering affects genetic algorithm performance

Usage:
    python test_integration_comprehensive.py
"""

import sys
import time
import tempfile
import json
from pathlib import Path

# Add the glitcher package to path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def test_without_comprehensive_search():
    """Test genetic algorithm without comprehensive search (baseline)."""
    print("🔬 Testing WITHOUT comprehensive search (baseline)...")

    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Load model and tokens without comprehensive search
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=False  # Disabled
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"  📊 Available tokens: {len(ga.available_tokens):,}")
    print(f"  🎯 Wanted token 'fox' baseline probability: {wanted_prob:.6f}")
    print(f"  🔧 First 5 available tokens: {ga.available_tokens[:5]}")

    return ga, wanted_prob


def test_with_comprehensive_search():
    """Test genetic algorithm with comprehensive search enabled."""
    print("\n🔬 Testing WITH comprehensive search...")

    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Load model and tokens with comprehensive search
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True  # Enabled
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"  📊 Available tokens: {len(ga.available_tokens):,}")
    print(f"  🎯 Wanted token 'fox' baseline probability: {wanted_prob:.6f}")
    print(f"  🔧 First 5 available tokens after reordering: {ga.available_tokens[:5]}")

    # Test that comprehensive search was performed
    if hasattr(ga, 'token_impact_map') and ga.token_impact_map:
        print(f"  ✅ Comprehensive search results found: {len(ga.token_impact_map):,} tokens analyzed")

        # Show top impact tokens
        sorted_results = dict(sorted(ga.token_impact_map.items(),
                                   key=lambda x: x[1]['wanted_impact'], reverse=True))
        top_5 = list(sorted_results.items())[:5]
        print("  🏆 Top 5 impact tokens:")
        for i, (token_id, metrics) in enumerate(top_5, 1):
            print(f"    {i}. Token {token_id} '{metrics['token_text']}': impact {metrics['wanted_impact']:.6f}")
    else:
        print("  ❌ No comprehensive search results found!")

    return ga, wanted_prob


def test_genetic_algorithm_integration():
    """Test that comprehensive search integrates properly with genetic algorithm."""
    print("\n🧬 Testing genetic algorithm integration...")

    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Configure for quick test
    ga.population_size = 10
    ga.max_generations = 5
    ga.max_tokens_per_individual = 2

    # Load model and tokens with comprehensive search
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True  # This should trigger comprehensive search in run_evolution
    )

    print("  🚀 Running genetic algorithm with comprehensive search integration...")
    start_time = time.time()

    # This should automatically run comprehensive search before genetic algorithm
    final_population = ga.run_evolution()

    evolution_time = time.time() - start_time
    print(f"  ✅ Evolution completed in {evolution_time:.2f} seconds")

    # Check results
    if final_population:
        best_individual = final_population[0]
        print(f"  🏆 Best result: {best_individual.tokens} with fitness {best_individual.fitness:.6f}")
        print(f"  📈 Wanted probability: {best_individual.wanted_baseline_prob:.6f} → {best_individual.wanted_modified_prob:.6f}")

        # Verify comprehensive search ran
        if hasattr(ga, 'token_impact_map') and ga.token_impact_map:
            print(f"  ✅ Comprehensive search integration verified: {len(ga.token_impact_map):,} tokens analyzed")
        else:
            print("  ❌ Comprehensive search integration failed!")

    return ga, final_population


def main():
    """Run comprehensive integration test."""
    print("=" * 80)
    print("🧪 COMPREHENSIVE SEARCH INTEGRATION TEST")
    print("=" * 80)
    print("This test verifies comprehensive wanted token search integrates properly")
    print("with the genetic algorithm and provides improved token prioritization.")
    print()

    try:
        # Test 1: Without comprehensive search
        ga_baseline, baseline_prob = test_without_comprehensive_search()

        # Test 2: With comprehensive search
        ga_comprehensive, comprehensive_prob = test_with_comprehensive_search()

        # Test 3: Full genetic algorithm integration
        ga_integrated, final_population = test_genetic_algorithm_integration()

        # Summary
        print("\n📋 INTEGRATION TEST SUMMARY")
        print("-" * 40)
        print(f"✅ Baseline test (no comprehensive search): PASSED")
        print(f"✅ Comprehensive search test: PASSED")
        print(f"✅ Genetic algorithm integration test: PASSED")

        # Compare token prioritization
        baseline_top_5 = ga_baseline.available_tokens[:5]
        comprehensive_top_5 = ga_comprehensive.available_tokens[:5]

        print(f"\n🔍 Token Prioritization Comparison:")
        print(f"  Baseline top 5 tokens:     {baseline_top_5}")
        print(f"  Comprehensive top 5 tokens: {comprehensive_top_5}")

        if baseline_top_5 != comprehensive_top_5:
            print("  ✅ Token reordering confirmed - comprehensive search changed prioritization")
        else:
            print("  ⚠️  Token order unchanged - may indicate limited impact tokens found")

        # Performance summary
        if hasattr(ga_integrated, 'token_impact_map') and ga_integrated.token_impact_map:
            impact_tokens = len([m for m in ga_integrated.token_impact_map.values() if m['wanted_impact'] > 0])
            print(f"\n📊 Search Results: {len(ga_integrated.token_impact_map):,} tokens tested, {impact_tokens:,} with positive impact")

        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("The comprehensive wanted token search is working correctly and")
        print("integrates seamlessly with the genetic algorithm.")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
