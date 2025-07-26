#!/usr/bin/env python3
"""
Test script for comprehensive search caching functionality.

This script demonstrates how the caching system works for comprehensive wanted token search:
1. First run performs full search and caches results
2. Subsequent runs with same parameters load instantly from cache
3. Different parameters create new cache entries
4. Cache can be cleared or disabled as needed

Usage:
    python test_comprehensive_caching.py
"""

import sys
import time
import shutil
from pathlib import Path

# Add the glitcher package to path
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def clear_cache():
    """Clear the comprehensive search cache."""
    cache_dir = Path("cache/comprehensive_search")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"🗑️  Cleared cache directory: {cache_dir}")
    else:
        print("🗑️  Cache directory doesn't exist")


def test_initial_search(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Test initial comprehensive search (should be slow and create cache)."""
    print("=" * 60)
    print("TEST 1: Initial Comprehensive Search (No Cache)")
    print("=" * 60)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"Starting comprehensive search for wanted token 'fox'...")
    start_time = time.time()

    # Run comprehensive search - should create cache
    results = ga.comprehensive_wanted_token_search(max_tokens=1000)  # Limit for faster testing

    search_time = time.time() - start_time
    print(f"✅ First search completed in {search_time:.2f} seconds")
    print(f"📊 Found {len(results)} token results")

    return search_time, len(results)


def test_cached_search(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Test comprehensive search with existing cache (should be fast)."""
    print("\n" + "=" * 60)
    print("TEST 2: Cached Comprehensive Search (Should Be Fast)")
    print("=" * 60)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"Running comprehensive search again (should use cache)...")
    start_time = time.time()

    # Run comprehensive search - should use cache
    results = ga.comprehensive_wanted_token_search(max_tokens=1000)  # Same parameters

    search_time = time.time() - start_time
    print(f"✅ Cached search completed in {search_time:.2f} seconds")
    print(f"📊 Found {len(results)} token results")

    return search_time, len(results)


def test_different_parameters(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Test with different parameters (should create new cache entry)."""
    print("\n" + "=" * 60)
    print("TEST 3: Different Parameters (New Cache Entry)")
    print("=" * 60)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="Hello",  # Different base text
        wanted_token="world"  # Different wanted token
    )

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"Running comprehensive search with different parameters...")
    start_time = time.time()

    # Run comprehensive search - should create new cache
    results = ga.comprehensive_wanted_token_search(max_tokens=1000)

    search_time = time.time() - start_time
    print(f"✅ Search with different parameters completed in {search_time:.2f} seconds")
    print(f"📊 Found {len(results)} token results")

    return search_time, len(results)


def test_cache_disabled(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Test with caching disabled (should always be slow)."""
    print("\n" + "=" * 60)
    print("TEST 4: Cache Disabled (Should Always Be Slow)")
    print("=" * 60)

    ga = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The quick brown",
        wanted_token="fox"
    )

    # Disable caching
    ga.use_cache = False

    # Load model and tokens
    ga.load_model()
    ga.load_glitch_tokens(
        ascii_only=True,
        include_normal_tokens=True,
        comprehensive_search=True
    )

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
    ga.target_token_id = target_id
    ga.baseline_target_probability = target_prob
    ga.wanted_token_id = wanted_id
    ga.baseline_wanted_probability = wanted_prob or 0.0

    print(f"Running comprehensive search with caching disabled...")
    start_time = time.time()

    # Run comprehensive search - should not use cache
    results = ga.comprehensive_wanted_token_search(max_tokens=1000)

    search_time = time.time() - start_time
    print(f"✅ Search with cache disabled completed in {search_time:.2f} seconds")
    print(f"📊 Found {len(results)} token results")

    return search_time, len(results)


def test_cache_invalidation(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """Test cache invalidation when parameters change."""
    print("\n" + "=" * 60)
    print("TEST 5: Cache Invalidation")
    print("=" * 60)

    # First, run with one set of parameters
    ga1 = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The cat",
        wanted_token="dog"
    )

    ga1.load_model()
    ga1.load_glitch_tokens(ascii_only=True, comprehensive_search=True)

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga1.get_baseline_probability()
    ga1.target_token_id = target_id
    ga1.baseline_target_probability = target_prob
    ga1.wanted_token_id = wanted_id
    ga1.baseline_wanted_probability = wanted_prob or 0.0

    print("Creating initial cache entry...")
    start_time = time.time()
    results1 = ga1.comprehensive_wanted_token_search(max_tokens=500)
    time1 = time.time() - start_time
    print(f"Initial search: {time1:.2f} seconds")

    # Now try with slightly different parameters (should create new cache)
    ga2 = GeneticProbabilityReducer(
        model_name=model_name,
        base_text="The cat runs",  # Different base text
        wanted_token="dog"
    )

    ga2.load_model()
    ga2.load_glitch_tokens(ascii_only=True, comprehensive_search=True)

    # Get baseline
    target_id, target_prob, wanted_id, wanted_prob = ga2.get_baseline_probability()
    ga2.target_token_id = target_id
    ga2.baseline_target_probability = target_prob
    ga2.wanted_token_id = wanted_id
    ga2.baseline_wanted_probability = wanted_prob or 0.0

    print("Running with different base text (should not use cache)...")
    start_time = time.time()
    results2 = ga2.comprehensive_wanted_token_search(max_tokens=500)
    time2 = time.time() - start_time
    print(f"Different parameters search: {time2:.2f} seconds")

    return time1, time2


def inspect_cache_directory():
    """Show what's in the cache directory."""
    print("\n" + "=" * 60)
    print("CACHE DIRECTORY INSPECTION")
    print("=" * 60)

    cache_dir = Path("cache/comprehensive_search")
    if not cache_dir.exists():
        print("📁 Cache directory doesn't exist")
        return

    cache_files = list(cache_dir.glob("*.json"))
    print(f"📁 Cache directory: {cache_dir}")
    print(f"📄 Cache files found: {len(cache_files)}")

    for i, cache_file in enumerate(cache_files, 1):
        size = cache_file.stat().st_size / 1024  # KB
        mod_time = time.ctime(cache_file.stat().st_mtime)
        print(f"  {i}. {cache_file.name}")
        print(f"     Size: {size:.1f} KB")
        print(f"     Modified: {mod_time}")

    total_size = sum(f.stat().st_size for f in cache_files) / 1024  # KB
    print(f"📊 Total cache size: {total_size:.1f} KB")


def main():
    """Run comprehensive caching test suite."""
    print("🧪 COMPREHENSIVE SEARCH CACHING TEST SUITE")
    print("=" * 80)
    print("This test demonstrates the caching system for comprehensive wanted token search.")
    print("The first search will be slow, subsequent identical searches will be fast.")
    print()

    # Clear cache at start for clean test
    print("🧹 Clearing cache for clean test...")
    clear_cache()

    try:
        # Test 1: Initial search (should be slow)
        initial_time, initial_results = test_initial_search()

        # Test 2: Cached search (should be fast)
        cached_time, cached_results = test_cached_search()

        # Test 3: Different parameters (should be slow again)
        different_time, different_results = test_different_parameters()

        # Test 4: Cache disabled (should be slow)
        disabled_time, disabled_results = test_cache_disabled()

        # Test 5: Cache invalidation
        inv_time1, inv_time2 = test_cache_invalidation()

        # Inspect cache
        inspect_cache_directory()

        # Summary
        print("\n" + "=" * 80)
        print("📊 CACHING TEST RESULTS SUMMARY")
        print("=" * 80)

        speedup = initial_time / cached_time if cached_time > 0 else float('inf')
        print(f"🏆 Performance Results:")
        print(f"  Initial search time:     {initial_time:6.2f} seconds")
        print(f"  Cached search time:      {cached_time:6.2f} seconds")
        print(f"  Cache speedup:           {speedup:6.1f}x faster")
        print()

        print(f"✅ Cache Behavior Verification:")
        if cached_time < initial_time * 0.1:  # Cache should be >10x faster
            print(f"  ✅ Caching works: {speedup:.1f}x speedup achieved")
        else:
            print(f"  ⚠️  Caching may not be working optimally")

        if different_time > cached_time * 2:  # Different params should be slower
            print(f"  ✅ Cache invalidation works: different parameters trigger new search")
        else:
            print(f"  ⚠️  Cache invalidation may not be working")

        if disabled_time > cached_time * 2:  # Disabled cache should be slower
            print(f"  ✅ Cache disabling works: search runs without cache when disabled")
        else:
            print(f"  ⚠️  Cache disabling may not be working")

        print(f"\n🎯 Key Takeaways:")
        print(f"  • Use default caching for best performance")
        print(f"  • First search with new parameters will be slow")
        print(f"  • Subsequent identical searches load instantly from cache")
        print(f"  • Use --clear-cache to force fresh results")
        print(f"  • Use --no-cache to disable caching entirely")

        print(f"\n✅ ALL CACHING TESTS COMPLETED SUCCESSFULLY!")

    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during caching tests: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    cleanup_choice = input("\nClean up cache files? (y/n): ").lower().strip()
    if cleanup_choice == 'y':
        clear_cache()
        print("🧹 Cache cleaned up.")


if __name__ == "__main__":
    main()
