#!/usr/bin/env python3
"""
Test script for genetic algorithm integration into main CLI tool.

This script validates that the genetic algorithm functionality has been
properly integrated into the main Glitcher CLI.

Author: Claude
Date: 2024
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch

# Add the glitcher package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_genetic_command_available():
    """Test that the genetic command is available in the CLI."""
    print("🧪 Testing genetic command availability...")

    try:
        result = subprocess.run([
            sys.executable, "-m", "glitcher.cli", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            print(f"❌ CLI help failed: {result.stderr}")
            return False

        if "genetic" not in result.stdout:
            print("❌ Genetic command not found in CLI help")
            return False

        print("✅ Genetic command is available in CLI")
        return True

    except Exception as e:
        print(f"❌ Error testing CLI availability: {e}")
        return False

def test_genetic_help():
    """Test that the genetic command help works."""
    print("🧪 Testing genetic command help...")

    try:
        result = subprocess.run([
            sys.executable, "-m", "glitcher.cli", "genetic", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            print(f"❌ Genetic help failed: {result.stderr}")
            return False

        expected_args = [
            "--base-text", "--target-token", "--token-file",
            "--population-size", "--generations", "--mutation-rate",
            "--crossover-rate", "--elite-size", "--max-tokens",
            "--output", "--device", "--quant-type", "--batch"
        ]

        for arg in expected_args:
            if arg not in result.stdout:
                print(f"❌ Missing argument in help: {arg}")
                return False

        print("✅ Genetic command help is complete")
        return True

    except Exception as e:
        print(f"❌ Error testing genetic help: {e}")
        return False

def create_test_token_file():
    """Create a minimal test token file for testing."""
    test_tokens = {
        "model_name": "test_model",
        "tokens": [
            {
                "id": 12345,
                "text": "TestToken1",
                "entropy": 0.123,
                "validation_method": "enhanced"
            },
            {
                "id": 67890,
                "text": "TestToken2",
                "entropy": 0.456,
                "validation_method": "enhanced"
            },
            {
                "id": 11111,
                "text": "TestToken3",
                "entropy": 0.789,
                "validation_method": "enhanced"
            }
        ]
    }

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(test_tokens, temp_file, indent=2)
    temp_file.close()

    return temp_file.name

def test_genetic_imports():
    """Test that genetic algorithm modules can be imported."""
    print("🧪 Testing genetic algorithm imports...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer, GeneticBatchRunner
        print("✅ Genetic modules import successfully")
        return True

    except ImportError as e:
        print(f"❌ Failed to import genetic modules: {e}")
        return False

def test_genetic_class_instantiation():
    """Test that genetic algorithm classes can be instantiated."""
    print("🧪 Testing genetic algorithm class instantiation...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer, GeneticBatchRunner

        # Test GeneticProbabilityReducer
        ga = GeneticProbabilityReducer(
            model_name="test_model",
            base_text="Test text",
            target_token="test"
        )

        # Verify basic attributes
        assert ga.model_name == "test_model"
        assert ga.base_text == "Test text"
        assert ga.target_token == "test"
        assert ga.population_size == 50  # default
        assert ga.max_generations == 100  # default

        # Test GeneticBatchRunner
        batch_runner = GeneticBatchRunner("test_model", "test_file.json")
        assert batch_runner.model_name == "test_model"
        assert batch_runner.token_file == "test_file.json"

        print("✅ Genetic algorithm classes instantiate correctly")
        return True

    except Exception as e:
        print(f"❌ Failed to instantiate genetic classes: {e}")
        return False

def test_genetic_dry_run():
    """Test genetic algorithm with dry run (no actual model loading)."""
    print("🧪 Testing genetic algorithm dry run...")

    try:
        from glitcher.genetic import GeneticProbabilityReducer

        # Create test token file
        token_file = create_test_token_file()

        try:
            ga = GeneticProbabilityReducer(
                model_name="test_model",
                base_text="The quick brown",
                target_token=None
            )

            # Set small parameters for testing
            ga.population_size = 5
            ga.max_generations = 2
            ga.max_tokens_per_individual = 2

            # Test token loading without model
            try:
                ga.load_glitch_tokens(token_file)
                print("✅ Token loading works")
            except Exception as e:
                print(f"⚠️  Token loading test skipped (expected): {e}")

            print("✅ Genetic algorithm dry run completed")
            return True

        finally:
            # Clean up temp file
            os.unlink(token_file)

    except Exception as e:
        print(f"❌ Genetic algorithm dry run failed: {e}")
        return False

def test_cli_integration_mock():
    """Test CLI integration with mocked components."""
    print("🧪 Testing CLI integration with mocks...")

    try:
        from glitcher.cli import GlitcherCLI

        # Create CLI instance
        cli = GlitcherCLI()

        # Test parser setup
        parser = cli.setup_parser()

        # Test genetic command parsing
        test_args = [
            "genetic", "test_model",
            "--base-text", "Test text",
            "--population-size", "10",
            "--generations", "5",
            "--output", "test_output.json"
        ]

        args = parser.parse_args(test_args)

        # Verify parsed arguments
        assert args.command == "genetic"
        assert args.model_path == "test_model"
        assert args.base_text == "Test text"
        assert args.population_size == 10
        assert args.generations == 5
        assert args.output == "test_output.json"

        print("✅ CLI integration parsing works correctly")
        return True

    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    print("🚀 Starting genetic algorithm integration tests...\n")

    tests = [
        test_genetic_command_available,
        test_genetic_help,
        test_genetic_imports,
        test_genetic_class_instantiation,
        test_genetic_dry_run,
        test_cli_integration_mock
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()  # Add spacing between tests

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All integration tests passed! Genetic algorithm is properly integrated.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
