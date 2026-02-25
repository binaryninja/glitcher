#!/usr/bin/env python3
"""
Test Script for Multi-Objective Genetic Algorithm

This script tests the new multi-objective genetic algorithm functionality
including wanted token support and normal vocabulary token integration.

Author: Claude
Date: 2024
"""

import json
import logging
import tempfile
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_glitch_tokens():
    """Create a small test file with glitch tokens for testing."""
    test_tokens = [
        {"token_id": 128001, "token_text": "<|begin_of_text|>"},
        {"token_id": 128009, "token_text": "<|eot_id|>"},
        {"token_id": 89472, "token_text": "SolidColorBrush"},
        {"token_id": 127438, "token_text": "ライブラリ"},
        {"token_id": 85069, "token_text": "HeaderCode"}
    ]

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(test_tokens, temp_file, indent=2)
    temp_file.close()

    return temp_file.name


def test_basic_import():
    """Test that we can import the genetic module."""
    print("🧪 Testing basic imports...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer, Individual
        print("✅ Successfully imported genetic modules")
        return True
    except ImportError as e:
        print(f"❌ Failed to import genetic modules: {e}")
        return False


def test_class_instantiation():
    """Test that we can create instances of the genetic algorithm classes."""
    print("\n🧪 Testing class instantiation...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer, Individual

        # Test Individual
        individual = Individual(tokens=[1, 2, 3])
        assert individual.tokens == [1, 2, 3]
        assert individual.fitness == 0.0
        print("✅ Individual class works")

        # Test GeneticProbabilityReducer - single objective
        ga_single = GeneticProbabilityReducer(
            model_name="test_model",
            base_text="test text",
            target_token="test"
        )
        assert ga_single.target_token == "test"
        assert ga_single.wanted_token is None
        print("✅ Single-objective GeneticProbabilityReducer works")

        # Test GeneticProbabilityReducer - multi-objective
        ga_multi = GeneticProbabilityReducer(
            model_name="test_model",
            base_text="test text",
            target_token="reduce_this",
            wanted_token="increase_this"
        )
        assert ga_multi.target_token == "reduce_this"
        assert ga_multi.wanted_token == "increase_this"
        print("✅ Multi-objective GeneticProbabilityReducer works")

        return True
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        return False


def test_token_loading():
    """Test token loading functionality."""
    print("\n🧪 Testing token loading...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer

        # Create test glitch tokens file
        token_file = create_test_glitch_tokens()

        # Create GA instance
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="test text"
        )

        # Load model first (needed for tokenizer)
        print("Loading model...")
        ga.load_model()

        # Test 1: Load only glitch tokens
        print("Testing glitch tokens only...")
        ga.load_glitch_tokens(
            token_file=token_file,
            ascii_only=True,
            include_normal_tokens=False
        )
        assert len(ga.glitch_tokens) > 0
        assert len(ga.ascii_tokens) == 0
        print(f"✅ Loaded {len(ga.glitch_tokens)} glitch tokens")

        # Test 2: Load glitch + normal tokens
        print("Testing glitch + normal tokens...")
        ga.load_glitch_tokens(
            token_file=token_file,
            ascii_only=True,
            include_normal_tokens=True
        )
        assert len(ga.glitch_tokens) > 0
        assert len(ga.ascii_tokens) > 0
        assert len(ga.available_tokens) > len(ga.glitch_tokens)
        print(f"✅ Loaded {len(ga.glitch_tokens)} glitch + {len(ga.ascii_tokens)} normal tokens")

        # Test 3: Load only normal tokens
        print("Testing normal tokens only...")
        ga.load_glitch_tokens(
            token_file=None,
            ascii_only=True,
            include_normal_tokens=True
        )
        assert len(ga.glitch_tokens) == 0
        assert len(ga.ascii_tokens) > 0
        print(f"✅ Loaded {len(ga.ascii_tokens)} normal tokens only")

        # Cleanup
        os.unlink(token_file)

        return True
    except Exception as e:
        print(f"❌ Token loading failed: {e}")
        return False


def test_baseline_probability():
    """Test baseline probability calculation."""
    print("\n🧪 Testing baseline probability calculation...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer

        # Create GA instance
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="The quick brown",
            target_token="fox",
            wanted_token="dog"
        )

        # Load model
        print("Loading model...")
        ga.load_model()

        # Test baseline calculation
        print("Calculating baseline probabilities...")
        target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()

        assert target_id is not None
        assert target_prob > 0.0
        assert wanted_id is not None
        assert wanted_prob >= 0.0

        print(f"✅ Target token '{ga.target_token}' (ID: {target_id}) baseline prob: {target_prob:.4f}")
        print(f"✅ Wanted token '{ga.wanted_token}' (ID: {wanted_id}) baseline prob: {wanted_prob:.4f}")

        return True
    except Exception as e:
        print(f"❌ Baseline probability calculation failed: {e}")
        return False


def test_fitness_evaluation():
    """Test fitness evaluation for multi-objective."""
    print("\n🧪 Testing fitness evaluation...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer, Individual

        # Create test glitch tokens file
        token_file = create_test_glitch_tokens()

        # Create GA instance
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="The weather is",
            target_token="sunny",
            wanted_token="rainy"
        )

        # Load model and tokens
        print("Loading model and tokens...")
        ga.load_model()
        ga.load_glitch_tokens(
            token_file=token_file,
            ascii_only=True,
            include_normal_tokens=True
        )

        # Get baseline
        target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
        ga.target_token_id = target_id
        ga.baseline_target_probability = target_prob
        ga.wanted_token_id = wanted_id
        ga.baseline_wanted_probability = wanted_prob or 0.0

        # Create test individual
        test_tokens = ga.available_tokens[:3] if len(ga.available_tokens) >= 3 else ga.available_tokens
        individual = Individual(tokens=test_tokens)

        # Evaluate fitness
        print("Evaluating fitness...")
        fitness = ga.evaluate_fitness(individual)

        assert individual.fitness == fitness
        assert hasattr(individual, 'target_reduction')
        assert hasattr(individual, 'wanted_increase')
        assert hasattr(individual, 'baseline_prob')
        assert hasattr(individual, 'modified_prob')

        print(f"✅ Fitness evaluation successful:")
        print(f"   Combined fitness: {fitness:.4f}")
        print(f"   Target reduction: {individual.target_reduction:.4f}")
        print(f"   Wanted increase: {individual.wanted_increase:.4f}")

        # Cleanup
        os.unlink(token_file)

        return True
    except Exception as e:
        print(f"❌ Fitness evaluation failed: {e}")
        return False


def test_population_creation():
    """Test population creation with baseline seeding."""
    print("\n🧪 Testing population creation...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer

        # Create test glitch tokens file
        token_file = create_test_glitch_tokens()

        # Create GA instance
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="Hello world",
            target_token="there"
        )

        # Configure for quick test
        ga.population_size = 10
        ga.max_tokens_per_individual = 2

        # Load model and tokens
        print("Loading model and tokens...")
        ga.load_model()
        ga.load_glitch_tokens(
            token_file=token_file,
            ascii_only=True,
            include_normal_tokens=True
        )

        # Get baseline and run token impact analysis
        target_id, target_prob, wanted_id, wanted_prob = ga.get_baseline_probability()
        ga.target_token_id = target_id
        ga.baseline_target_probability = target_prob
        ga.wanted_token_id = wanted_id
        ga.baseline_wanted_probability = wanted_prob or 0.0

        print("Running baseline token impact analysis...")
        ga.baseline_token_impacts(max_tokens=50)  # Quick test

        # Create initial population
        print("Creating initial population...")
        population = ga.create_initial_population()

        assert len(population) == ga.population_size
        assert all(len(ind.tokens) <= ga.max_tokens_per_individual for ind in population)
        assert all(len(ind.tokens) > 0 for ind in population)

        print(f"✅ Created population of {len(population)} individuals")
        print(f"   Token counts: {[len(ind.tokens) for ind in population[:5]]}")

        # Cleanup
        os.unlink(token_file)

        return True
    except Exception as e:
        print(f"❌ Population creation failed: {e}")
        return False


def test_short_evolution():
    """Test a short evolution run."""
    print("\n🧪 Testing short evolution run...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer

        # Create test glitch tokens file
        token_file = create_test_glitch_tokens()

        # Create GA instance with minimal parameters
        ga = GeneticProbabilityReducer(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            base_text="Today is",
            target_token="Monday",
            wanted_token="Friday"
        )

        # Quick configuration
        ga.population_size = 8
        ga.max_generations = 5
        ga.max_tokens_per_individual = 2
        ga.early_stopping_threshold = 0.5  # Lower threshold for quick test

        # Load model and tokens
        print("Loading model and tokens...")
        ga.load_model()
        ga.load_glitch_tokens(
            token_file=token_file,
            ascii_only=True,
            include_normal_tokens=True
        )

        # Run short evolution
        print("Running short evolution...")
        final_population = ga.run_evolution()

        assert len(final_population) > 0
        assert all(ind.fitness >= 0 for ind in final_population)

        # Check that we have valid results
        best_individual = final_population[0]
        assert best_individual.fitness > 0
        assert hasattr(best_individual, 'target_reduction')
        assert hasattr(best_individual, 'wanted_increase')

        print(f"✅ Evolution completed successfully:")
        print(f"   Best fitness: {best_individual.fitness:.4f}")
        print(f"   Best tokens: {best_individual.tokens}")

        # Test results display
        print("\nTesting results display...")
        ga.display_results(final_population, top_n=3)

        # Test results saving
        print("Testing results saving...")
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        output_file.close()
        ga.save_results(final_population, output_file.name)

        # Verify saved file
        with open(output_file.name, 'r') as f:
            saved_results = json.load(f)

        assert 'model_name' in saved_results
        assert 'target_token_id' in saved_results
        assert 'wanted_token_id' in saved_results
        assert 'results' in saved_results
        assert len(saved_results['results']) > 0

        print("✅ Results saved and verified")

        # Cleanup
        os.unlink(token_file)
        os.unlink(output_file.name)

        return True
    except Exception as e:
        print(f"❌ Short evolution failed: {e}")
        return False


def test_cli_integration():
    """Test CLI integration."""
    print("\n🧪 Testing CLI integration...")

    try:
        import subprocess
        import sys

        # Test help command
        result = subprocess.run([
            sys.executable, "-m", "glitcher.cli", "genetic", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"❌ CLI help failed: {result.stderr}")
            return False

        # Check for new parameters
        help_text = result.stdout
        assert "--wanted-token" in help_text
        assert "--include-normal-tokens" in help_text

        print("✅ CLI integration verified - new parameters available")
        return True
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("🔬 Multi-Objective Genetic Algorithm Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Import", test_basic_import),
        ("Class Instantiation", test_class_instantiation),
        ("Token Loading", test_token_loading),
        ("Baseline Probability", test_baseline_probability),
        ("Fitness Evaluation", test_fitness_evaluation),
        ("Population Creation", test_population_creation),
        ("Short Evolution", test_short_evolution),
        ("CLI Integration", test_cli_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)

            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")

        except Exception as e:
            print(f"\n💥 {test_name} CRASHED: {e}")

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print('='*60)

    if passed == total:
        print("🎉 All tests passed! Multi-objective genetic algorithm is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
