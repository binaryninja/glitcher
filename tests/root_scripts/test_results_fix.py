#!/usr/bin/env python3
"""
Test Results Fix

This script tests the fix for the results handling issue where the
run_evolution method returns a list but the GUI expects a dictionary.

Tests:
1. GUICallback properly converts list results to dictionary format
2. _display_final_results handles the formatted results correctly
3. No more AttributeError: 'list' object has no attribute 'get'
"""

import tkinter as tk
import sys
import os
import json
from pathlib import Path

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def create_mock_individual(tokens, fitness, prob_before=None, prob_after=None):
    """Create a mock individual for testing"""
    class MockIndividual:
        def __init__(self, tokens, fitness, prob_before=None, prob_after=None):
            self.tokens = tokens
            self.fitness = fitness
            if prob_before is not None:
                self.target_prob_before = prob_before
                self.target_prob_after = prob_after or prob_before * (1 - fitness)
            if prob_after is not None and 'wanted' in str(prob_after):
                self.wanted_prob_before = 0.001
                self.wanted_prob_after = 0.1

    return MockIndividual(tokens, fitness, prob_before, prob_after)

def test_gui_callback_conversion():
    """Test that GUICallback properly converts list results to dictionary"""
    print("🧪 Testing GUICallback results conversion...")

    try:
        from glitcher.genetic.gui_controller import GUICallback, GeneticControllerGUI, GeneticConfig

        # Create a minimal GUI controller for testing
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing

        gui = GeneticControllerGUI(root)
        gui.config.generations = 10

        # Create GUI callback
        callback = GUICallback(gui)

        # Create mock results as list (what run_evolution returns)
        mock_population = [
            create_mock_individual([12345, 67890], 0.8567, 0.1234, 0.0567),
            create_mock_individual([11111, 22222], 0.7234, 0.1234, 0.0789),
            create_mock_individual([33333, 44444], 0.6543, 0.1234, 0.0891)
        ]

        # Test the conversion by capturing what gets passed to _on_evolution_complete
        captured_results = None

        def mock_on_evolution_complete(results):
            nonlocal captured_results
            captured_results = results

        # Replace the method temporarily
        original_method = gui._on_evolution_complete
        gui._on_evolution_complete = mock_on_evolution_complete

        # Call the callback with list results
        callback.on_evolution_complete(mock_population)

        # Process any pending GUI events
        root.update()

        # Check that results were converted to dictionary format
        assert captured_results is not None, "Results should have been captured"
        assert isinstance(captured_results, dict), f"Results should be dict, got {type(captured_results)}"
        assert 'best_individual' in captured_results, "Results should have best_individual key"
        assert 'final_population' in captured_results, "Results should have final_population key"
        assert 'generations_completed' in captured_results, "Results should have generations_completed key"
        assert 'best_fitness' in captured_results, "Results should have best_fitness key"

        # Check that best individual is correct
        best_individual = captured_results['best_individual']
        assert best_individual.fitness == 0.8567, f"Best individual should have highest fitness, got {best_individual.fitness}"

        print("✓ GUICallback correctly converts list to dictionary")

        # Restore original method
        gui._on_evolution_complete = original_method
        root.destroy()

        return True

    except Exception as e:
        print(f"✗ GUICallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_final_results():
    """Test that _display_final_results handles formatted results correctly"""
    print("🧪 Testing _display_final_results with formatted dictionary...")

    try:
        from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig

        # Create GUI controller
        root = tk.Tk()
        root.withdraw()  # Hide the window for testing

        gui = GeneticControllerGUI(root)

        # Set up some config
        gui.config.model_name = "test-model"
        gui.config.base_text = "The quick brown"
        gui.config.wanted_token = "fox"
        gui.config.generations = 10

        # Create formatted results (what GUICallback produces)
        best_individual = create_mock_individual([12345, 67890], 0.8567, 0.1234, 0.0567)
        formatted_results = {
            'best_individual': best_individual,
            'final_population': [best_individual],
            'generations_completed': 10,
            'best_fitness': 0.8567
        }

        # Test display method
        gui._display_final_results(formatted_results)

        # Check that results text was populated
        results_text = gui.results_text.get(1.0, tk.END)
        assert "EVOLUTION RESULTS" in results_text, "Results should contain header"
        assert "Best Individual:" in results_text, "Results should show best individual"
        assert "Token IDs: [12345, 67890]" in results_text, "Results should show token IDs"
        assert "Fitness Score: 0.856700" in results_text, "Results should show fitness score"

        print("✓ _display_final_results correctly handles dictionary format")

        root.destroy()
        return True

    except Exception as e:
        print(f"✗ _display_final_results test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases for results handling"""
    print("🧪 Testing edge cases...")

    try:
        from glitcher.genetic.gui_controller import GeneticControllerGUI

        root = tk.Tk()
        root.withdraw()

        gui = GeneticControllerGUI(root)

        # Test with empty results
        gui._display_final_results(None)
        results_text = gui.results_text.get(1.0, tk.END)
        assert "No results available" in results_text, "Should handle None results"

        # Test with empty dictionary
        gui._display_final_results({})
        results_text = gui.results_text.get(1.0, tk.END)
        assert "EVOLUTION RESULTS" in results_text, "Should handle empty dict"

        # Test with no best individual
        empty_results = {
            'best_individual': None,
            'final_population': [],
            'generations_completed': 0,
            'best_fitness': 0.0
        }
        gui._display_final_results(empty_results)
        results_text = gui.results_text.get(1.0, tk.END)
        assert "No best individual found" in results_text, "Should handle no best individual"

        print("✓ Edge cases handled correctly")

        root.destroy()
        return True

    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 Testing Results Fix")
    print("=" * 50)
    print("Testing the fix for AttributeError: 'list' object has no attribute 'get'")
    print()

    tests = [
        ("GUICallback Conversion", test_gui_callback_conversion),
        ("Display Final Results", test_display_final_results),
        ("Edge Cases", test_edge_cases)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! The results fix is working correctly.")
        print()
        print("✓ GUICallback now properly converts list results to dictionary")
        print("✓ _display_final_results handles formatted results without errors")
        print("✓ No more AttributeError when evolution completes")
        print()
        print("The GUI should now complete evolution runs without crashing!")
    else:
        print("❌ Some tests failed. The fix may need additional work.")

if __name__ == "__main__":
    main()
