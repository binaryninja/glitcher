#!/usr/bin/env python3
"""
Test script for GUI integration with genetic algorithm.

This script validates that the GUI animation functionality has been
properly integrated into the Glitcher genetic algorithm system.

Author: Claude
Date: 2024
"""

import os
import sys
import time
import subprocess
import tempfile
import json
from unittest.mock import Mock, patch

# Add the glitcher package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_matplotlib_availability():
    """Test if matplotlib is available for GUI functionality."""
    print("🧪 Testing matplotlib availability...")

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        print(f"✅ matplotlib {matplotlib.__version__} is available")
        return True
    except ImportError:
        print("❌ matplotlib is not available")
        print("   Install with: pip install matplotlib")
        return False

def test_gui_animator_import():
    """Test that GUI animator classes can be imported."""
    print("🧪 Testing GUI animator imports...")

    try:
        from glitcher.genetic import RealTimeGeneticAnimator, GeneticAnimationCallback
        print("✅ GUI animator classes import successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GUI animator classes: {e}")
        return False

def test_gui_animator_instantiation():
    """Test that GUI animator can be instantiated."""
    print("🧪 Testing GUI animator instantiation...")

    try:
        # Mock matplotlib to avoid opening windows during testing
        with patch('matplotlib.pyplot.ion'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.pause'):

            from glitcher.genetic import RealTimeGeneticAnimator, GeneticAnimationCallback

            # Test animator creation
            animator = RealTimeGeneticAnimator(
                base_text="Test text",
                target_token_text="test",
                target_token_id=12345,
                baseline_probability=0.5,
                max_generations=100
            )

            # Test basic attributes
            assert animator.base_text == "Test text"
            assert animator.target_token_text == "test"
            assert animator.target_token_id == 12345
            assert animator.baseline_probability == 0.5
            assert animator.max_generations == 100

            # Test callback creation
            callback = GeneticAnimationCallback(animator)
            assert callback.animator == animator

            print("✅ GUI animator instantiates correctly")
            return True

    except Exception as e:
        print(f"❌ Failed to instantiate GUI animator: {e}")
        return False

def test_gui_flag_in_cli():
    """Test that the --gui flag is available in the CLI."""
    print("🧪 Testing --gui flag in CLI...")

    try:
        result = subprocess.run([
            sys.executable, "-m", "glitcher.cli", "genetic", "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode != 0:
            print(f"❌ CLI help failed: {result.stderr}")
            return False

        if "--gui" not in result.stdout:
            print("❌ --gui flag not found in genetic command help")
            return False

        if "Show real-time GUI animation" not in result.stdout:
            print("❌ GUI flag description not found")
            return False

        print("✅ --gui flag is available in CLI")
        return True

    except Exception as e:
        print(f"❌ Error testing CLI GUI flag: {e}")
        return False

def test_callback_system():
    """Test the callback system between genetic algorithm and GUI."""
    print("🧪 Testing callback system...")

    try:
        with patch('matplotlib.pyplot.ion'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.pause'):

            from glitcher.genetic import RealTimeGeneticAnimator, GeneticAnimationCallback
            from glitcher.genetic.reducer import Individual

            # Create mock animator and callback
            animator = RealTimeGeneticAnimator(
                base_text="Test text",
                max_generations=10
            )
            callback = GeneticAnimationCallback(animator)

            # Test evolution start callback
            callback.on_evolution_start(
                baseline_prob=0.8,
                target_token_id=12345,
                target_token_text="test"
            )

            # Verify data was updated
            assert animator.baseline_probability == 0.8
            assert animator.target_token_id == 12345
            assert animator.target_token_text == "test"

            # Test generation complete callback
            mock_individual = Individual(tokens=[1, 2, 3])
            mock_individual.fitness = 0.6

            callback.on_generation_complete(
                generation=5,
                best_individual=mock_individual,
                avg_fitness=0.4,
                current_probability=0.3
            )

            # Verify data was updated
            assert animator.current_generation == 5
            assert animator.current_best_fitness == 0.6
            assert animator.current_avg_fitness == 0.4
            assert animator.current_probability == 0.3
            assert animator.current_best_tokens == [1, 2, 3]

            # Test evolution complete callback
            final_population = [mock_individual]
            callback.on_evolution_complete(final_population, 10)

            assert animator.is_complete == True

            print("✅ Callback system works correctly")
            return True

    except Exception as e:
        print(f"❌ Callback system test failed: {e}")
        return False

def test_data_update_system():
    """Test the data update system in the animator."""
    print("🧪 Testing data update system...")

    try:
        with patch('matplotlib.pyplot.ion'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.pause'), \
             patch('matplotlib.pyplot.draw'), \
             patch('matplotlib.pyplot.fignum_exists', return_value=True):

            from glitcher.genetic import RealTimeGeneticAnimator

            animator = RealTimeGeneticAnimator(
                base_text="Test text",
                max_generations=50
            )

            # Test multiple data updates
            test_data = [
                (0, 0.1, 0.05, [100, 200], ["token1", "token2"], 0.9),
                (5, 0.3, 0.15, [100, 300], ["token1", "token3"], 0.7),
                (10, 0.5, 0.25, [400, 300], ["token4", "token3"], 0.5),
            ]

            for gen, best_fit, avg_fit, tokens, texts, prob in test_data:
                animator.update_data(
                    generation=gen,
                    best_fitness=best_fit,
                    avg_fitness=avg_fit,
                    best_tokens=tokens,
                    token_texts=texts,
                    current_probability=prob
                )

            # Verify final state
            assert animator.current_generation == 10
            assert animator.current_best_fitness == 0.5
            assert animator.current_avg_fitness == 0.25
            assert animator.current_best_tokens == [400, 300]
            assert animator.current_token_texts == ["token4", "token3"]
            assert animator.current_probability == 0.5

            # Verify data storage
            assert len(animator.generations) == 3
            assert len(animator.best_fitness) == 3
            assert len(animator.avg_fitness) == 3

            print("✅ Data update system works correctly")
            return True

    except Exception as e:
        print(f"❌ Data update system test failed: {e}")
        return False

def test_cli_integration_with_gui():
    """Test CLI integration with GUI components."""
    print("🧪 Testing CLI integration with GUI...")

    try:
        from glitcher.cli import GlitcherCLI

        # Create CLI instance
        cli = GlitcherCLI()

        # Test parser setup with GUI flag
        parser = cli.setup_parser()

        # Test genetic command parsing with GUI flag
        test_args = [
            "genetic", "test_model",
            "--base-text", "Test text",
            "--population-size", "10",
            "--generations", "5",
            "--gui",
            "--output", "test_output.json"
        ]

        args = parser.parse_args(test_args)

        # Verify parsed arguments
        assert args.command == "genetic"
        assert args.model_path == "test_model"
        assert args.base_text == "Test text"
        assert args.population_size == 10
        assert args.generations == 5
        assert args.gui == True
        assert args.output == "test_output.json"

        print("✅ CLI integration with GUI works correctly")
        return True

    except Exception as e:
        print(f"❌ CLI integration with GUI test failed: {e}")
        return False

def test_error_handling():
    """Test error handling when matplotlib is not available."""
    print("🧪 Testing error handling for missing matplotlib...")

    try:
        # Test by creating animator with matplotlib temporarily disabled
        # We'll patch the MATPLOTLIB_AVAILABLE flag instead of hiding modules
        from glitcher.genetic import gui_animator

        # Save original state
        original_available = gui_animator.MATPLOTLIB_AVAILABLE
        original_plt = gui_animator.plt

        try:
            # Temporarily disable matplotlib
            gui_animator.MATPLOTLIB_AVAILABLE = False
            gui_animator.plt = None

            from glitcher.genetic.gui_animator import RealTimeGeneticAnimator

            # This should raise ImportError
            try:
                animator = RealTimeGeneticAnimator("test", max_generations=10)
                print("❌ Expected ImportError was not raised")
                result = False
            except ImportError as e:
                if "matplotlib is required" in str(e):
                    print("✅ Proper ImportError raised when matplotlib unavailable")
                    result = True
                else:
                    print(f"❌ Unexpected ImportError: {e}")
                    result = False

        finally:
            # Restore original state
            gui_animator.MATPLOTLIB_AVAILABLE = original_available
            gui_animator.plt = original_plt

        return result

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all GUI integration tests."""
    print("🚀 Starting GUI integration tests...\n")

    tests = [
        test_matplotlib_availability,
        test_gui_animator_import,
        test_gui_animator_instantiation,
        test_gui_flag_in_cli,
        test_callback_system,
        test_data_update_system,
        test_cli_integration_with_gui,
        test_error_handling
    ]

    passed = 0
    failed = 0
    skipped = 0

    # Check if matplotlib is available first
    matplotlib_available = False
    try:
        import matplotlib
        matplotlib_available = True
    except ImportError:
        pass

    for test in tests:
        try:
            # Skip matplotlib-dependent tests if not available
            if not matplotlib_available and test.__name__ in [
                'test_gui_animator_instantiation',
                'test_callback_system',
                'test_data_update_system'
            ]:
                print(f"⏭️  Skipping {test.__name__} (matplotlib not available)")
                skipped += 1
                continue

            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()  # Add spacing between tests

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("🎉 All available GUI integration tests passed!")
        if skipped > 0:
            print("💡 Install matplotlib to enable full GUI testing: pip install matplotlib")
        return True
    else:
        print("⚠️  Some tests failed. Please check the GUI integration.")
        return False

if __name__ == "__main__":
    success = run_all_tests()

    print("\n" + "=" * 60)
    print("GUI Integration Summary:")
    print("- GUI animation support has been added to genetic algorithm")
    print("- Use --gui flag with 'glitcher genetic' command to enable")
    print("- Real-time visualization shows fitness evolution and token combinations")
    print("- Requires matplotlib: pip install matplotlib")
    print("=" * 60)

    sys.exit(0 if success else 1)
