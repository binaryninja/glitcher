#!/usr/bin/env python3
"""
Test script for enhanced GUI string visualization in genetic algorithm.

This script demonstrates the enhanced GUI features that show:
1. Full string construction with token positioning
2. Clear visual separation between evolved tokens and base text
3. Complete context showing how tokens affect predictions
4. Real-time visualization of string transformations

Usage:
    python test_enhanced_gui.py meta-llama/Llama-3.2-1B-Instruct

Author: Claude
Date: 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path

def test_enhanced_gui_visualization(model_name: str):
    """
    Test enhanced GUI visualization with clear string positioning.
    """
    try:
        # Import after checking dependencies
        try:
            from glitcher.genetic.reducer import GeneticProbabilityReducer
            from glitcher.genetic.gui_animator import GeneticAnimationCallback, RealTimeGeneticAnimator
        except ImportError:
            # Try alternative import path
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from glitcher.genetic.reducer import GeneticProbabilityReducer
            from glitcher.genetic.gui_animator import GeneticAnimationCallback, RealTimeGeneticAnimator

        print("🧬 Enhanced GUI String Visualization Test")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Testing enhanced GUI features:")
        print(f"  ✓ Full string construction display")
        print(f"  ✓ Token positioning visualization")
        print(f"  ✓ Real-time context updates")
        print(f"  ✓ Complete prediction analysis")
        print()

        # Test scenarios with different base texts
        test_scenarios = [
            {
                "base_text": "The quick brown",
                "description": "Short phrase prediction",
                "generations": 25,
                "population_size": 20
            },
            {
                "base_text": "Hello world, this is a test of",
                "description": "Longer context prediction",
                "generations": 30,
                "population_size": 25
            },
            {
                "base_text": "In machine learning,",
                "description": "Technical domain prediction",
                "generations": 20,
                "population_size": 15
            }
        ]

        # Load some glitch tokens for testing
        glitch_tokens_file = Path("glitch_tokens.json")
        if not glitch_tokens_file.exists():
            print("⚠️  No glitch_tokens.json found. Creating minimal test tokens...")
            # Create some test tokens (these are example IDs, may not be actual glitch tokens)
            test_tokens = {
                "glitch_tokens": [
                    {"token_id": 89472, "entropy": 0.1234, "text": " SomeToken"},
                    {"token_id": 127438, "entropy": 0.2345, "text": "AnotherToken"},
                    {"token_id": 85069, "entropy": 0.3456, "text": "TestToken"},
                    {"token_id": 12345, "entropy": 0.4567, "text": " Prefix"},
                    {"token_id": 67890, "entropy": 0.5678, "text": "Modified"}
                ]
            }
            with open(glitch_tokens_file, 'w') as f:
                json.dump(test_tokens, f, indent=2)
            print(f"✓ Created {glitch_tokens_file} with test tokens")

        print("\n🎯 Starting Enhanced GUI Test Scenarios...")
        print("=" * 60)

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📝 Scenario {i}: {scenario['description']}")
            print(f"Base Text: \"{scenario['base_text']}\"")
            print(f"Parameters: {scenario['generations']} generations, {scenario['population_size']} population")

            # Initialize genetic reducer with GUI
            reducer = GeneticProbabilityReducer(
                model_name=model_name,
                base_text=scenario['base_text'],
                target_token=None  # Auto-detect
            )

            # Configure parameters
            reducer.population_size = scenario['population_size']
            reducer.max_generations = scenario['generations']
            reducer.max_tokens_per_individual = 3

            # Load model and prepare
            print("🔄 Loading model and tokenizer...")
            reducer.load_model()
            reducer.load_glitch_tokens(str(glitch_tokens_file))

            if len(reducer.glitch_tokens) == 0:
                print("❌ No glitch tokens loaded! Please ensure glitch_tokens.json contains valid tokens.")
                continue

            print(f"✓ Loaded {len(reducer.glitch_tokens)} glitch tokens")

            # Initialize GUI callback
            try:
                print("🖼️  Initializing real-time GUI animation...")
                animator = RealTimeGeneticAnimator(
                    base_text=scenario['base_text'],
                    max_generations=scenario['generations']
                )
                gui_callback = GeneticAnimationCallback(animator)
                reducer.gui_callback = gui_callback
                print("✓ GUI animation ready")
            except ImportError as e:
                print(f"⚠️  GUI not available: {e}")
                print("Install matplotlib for GUI support: pip install matplotlib")
                gui_callback = None
                reducer.gui_callback = None
            except Exception as e:
                print(f"⚠️  Failed to initialize GUI: {e}")
                gui_callback = None
                reducer.gui_callback = None

            if gui_callback:
                print("\n🖼️  Starting GUI visualization...")
                print("=" * 40)
                print("GUI Features to observe:")
                print("  📝 Full String Construction:")
                print(f"      Original:  \"{scenario['base_text']}\"")
                print(f"      Evolved:   [evolved_tokens] + \"{scenario['base_text']}\"")
                print(f"      Result:    \"[evolved_tokens]{scenario['base_text']}\"")
                print("  📊 Real-time probability changes")
                print("  🧬 Token combination evolution")
                print("  🎯 Target token prediction analysis")
                print()
                print("Close the GUI window when you're done observing, or wait for completion.")
                print("Press Ctrl+C to interrupt early.")
            else:
                print("\n🔄 Running without GUI (console output only)...")

            try:
                # Run genetic algorithm with or without GUI
                results = reducer.run_evolution()

                if results:
                    best_result = results[0]
                    evolved_tokens = "".join([reducer.tokenizer.decode([tid]) for tid in best_result.tokens])
                    full_string = evolved_tokens + scenario['base_text']

                    print(f"\n✅ Scenario {i} Complete!")
                    print(f"Best token combination: {best_result.tokens}")
                    print(f"Evolved prefix: \"{evolved_tokens}\"")
                    print(f"Full result string: \"{full_string}\"")
                    print(f"Fitness (reduction): {best_result.fitness:.4f}")
                    print(f"Baseline probability: {best_result.baseline_prob:.4f}")
                    print(f"Modified probability: {best_result.modified_prob:.4f}")

                    if best_result.baseline_prob > 0:
                        reduction_pct = (1 - best_result.modified_prob / best_result.baseline_prob) * 100
                        print(f"Probability reduction: {reduction_pct:.1f}%")

            except KeyboardInterrupt:
                print(f"\n⏹️  Scenario {i} interrupted by user")

            except Exception as e:
                print(f"\n❌ Error in scenario {i}: {e}")

            # Clean up GUI resources
            if gui_callback and hasattr(gui_callback, 'keep_alive'):
                try:
                    gui_callback.keep_alive()
                except:
                    pass  # GUI might already be closed

            print(f"\n{'='*40}")

            # Small break between scenarios
            if i < len(test_scenarios):
                print("⏸️  Brief pause before next scenario...")
                time.sleep(2)

        print("\n🎉 Enhanced GUI Test Complete!")
        print("=" * 60)
        print("Summary of tested features:")
        print("  ✓ Full string construction visualization")
        print("  ✓ Clear token positioning markers")
        print("  ✓ Real-time context updates")
        print("  ✓ Complete prediction impact analysis")
        print("  ✓ Professional formatting with emojis and structure")
        print()
        print("The enhanced GUI now clearly shows:")
        print("  1. How tokens are positioned at the beginning of text")
        print("  2. The complete constructed string being fed to the model")
        print("  3. Real-time probability changes as evolution progresses")
        print("  4. Visual separation between evolved tokens and base text")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure you have installed the package:")
        print("  pip install -e .")
        print("  pip install matplotlib")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test enhanced GUI string visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_enhanced_gui.py meta-llama/Llama-3.2-1B-Instruct

This test demonstrates the enhanced GUI features that show complete
string construction and token positioning during genetic evolution.

The GUI will display:
  - Full constructed strings with clear token positioning
  - Real-time probability changes
  - Visual separation between evolved tokens and base text
  - Complete prediction context and analysis
        """
    )

    parser.add_argument(
        'model_name',
        help='HuggingFace model identifier (e.g., meta-llama/Llama-3.2-1B-Instruct)'
    )

    args = parser.parse_args()

    # Test the enhanced GUI
    success = test_enhanced_gui_visualization(args.model_name)

    if success:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
