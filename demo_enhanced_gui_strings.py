#!/usr/bin/env python3
"""
Demo script showing enhanced GUI string visualization for genetic algorithm.

This script demonstrates the key improvements made to the GUI interface:

BEFORE (Old GUI):
- Only showed token IDs and basic fitness
- No clear indication of where tokens were inserted
- Limited context about string construction
- Basic probability information

AFTER (Enhanced GUI):
- Shows full string construction: [evolved_tokens] + "base_text" = "result"
- Clear visual separation between evolved tokens and base text
- Complete prediction context display
- Real-time string transformation tracking
- Enhanced formatting with emojis and structure

Usage:
    python demo_enhanced_gui_strings.py meta-llama/Llama-3.2-1B-Instruct

Key GUI Enhancements Demonstrated:
1. Full String Construction Display
2. Token Positioning Visualization
3. Real-time Context Updates
4. Complete Prediction Impact Analysis
5. Professional Formatting and Structure

Author: Claude
Date: 2024
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

def create_demo_scenarios() -> List[Dict[str, Any]]:
    """Create demonstration scenarios for enhanced GUI features."""
    return [
        {
            "name": "short_phrase",
            "base_text": "The quick brown",
            "description": "Short phrase - shows clear token insertion",
            "generations": 20,
            "population_size": 15,
            "expected_display": "[evolved_tokens]The quick brown"
        },
        {
            "name": "sentence_completion",
            "base_text": "Hello world, this is a",
            "description": "Sentence completion - demonstrates context impact",
            "generations": 25,
            "population_size": 20,
            "expected_display": "[evolved_tokens]Hello world, this is a"
        },
        {
            "name": "technical_context",
            "base_text": "In machine learning, the most important",
            "description": "Technical domain - shows specialized vocabulary impact",
            "generations": 30,
            "population_size": 25,
            "expected_display": "[evolved_tokens]In machine learning, the most important"
        },
        {
            "name": "long_context",
            "base_text": "Once upon a time, in a galaxy far far away, there lived a",
            "description": "Long context - demonstrates string truncation handling",
            "generations": 15,
            "population_size": 20,
            "expected_display": "[evolved_tokens]Once upon a time, in a galaxy far far away..."
        }
    ]

def print_gui_enhancement_overview():
    """Print overview of GUI enhancements."""
    print("🎨 Enhanced GUI String Visualization Demo")
    print("=" * 60)
    print()
    print("KEY ENHANCEMENTS:")
    print()
    print("1. 📝 FULL STRING CONSTRUCTION DISPLAY")
    print("   Before: Only showed token IDs: [12345, 67890]")
    print("   After:  Shows construction: [evolved_tokens] + \"base_text\" = \"result\"")
    print()
    print("2. 🎯 TOKEN POSITIONING VISUALIZATION")
    print("   Before: Unclear where tokens were inserted")
    print("   After:  Clear visual: \"[TokenA][TokenB]The quick brown fox\"")
    print()
    print("3. 📊 REAL-TIME CONTEXT UPDATES")
    print("   Before: Static baseline information")
    print("   After:  Live context: \"Input: [prefix]text → target_token\"")
    print()
    print("4. 🧬 COMPLETE PREDICTION ANALYSIS")
    print("   Before: Basic probability numbers")
    print("   After:  Full analysis: baseline → current → reduction%")
    print()
    print("5. ✨ PROFESSIONAL FORMATTING")
    print("   Before: Plain text displays")
    print("   After:  Emojis, structure, monospace fonts, color coding")
    print()

def demonstrate_string_construction_logic():
    """Demonstrate the string construction logic used in the genetic algorithm."""
    print("🔧 STRING CONSTRUCTION LOGIC")
    print("=" * 50)
    print()
    print("The genetic algorithm works by:")
    print("1. Taking a base text: \"The quick brown\"")
    print("2. Evolving glitch tokens: [TokenA, TokenB]")
    print("3. Constructing input: TokenA + TokenB + \"The quick brown\"")
    print("4. Feeding to model: \"[decoded_tokens]The quick brown\"")
    print("5. Measuring probability change of next token")
    print()

    # Example construction
    examples = [
        {
            "base": "Hello world",
            "tokens": ["<special>", "██"],
            "result": "<special>██Hello world"
        },
        {
            "base": "The cat sat",
            "tokens": ["[PREFIX]", "★"],
            "result": "[PREFIX]★The cat sat"
        },
        {
            "base": "Machine learning is",
            "tokens": ["∅", "<|endoftext|>"],
            "result": "∅<|endoftext|>Machine learning is"
        }
    ]

    print("CONSTRUCTION EXAMPLES:")
    print("-" * 30)
    for i, ex in enumerate(examples, 1):
        print(f"{i}. Base: \"{ex['base']}\"")
        print(f"   Evolved tokens: {ex['tokens']}")
        print(f"   Final input: \"{ex['result']}\"")
        print()

def show_gui_panels_layout():
    """Show the layout and content of each GUI panel."""
    print("🖼️  GUI PANEL LAYOUT")
    print("=" * 50)
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                  INFO PANEL (Top)                      │")
    print("│  🎯 Target: \"fox\" (ID: 21831)                          │")
    print("│  📝 Input: \"[evolved]The quick brown\" → \"fox\"           │")
    print("│  📊 Baseline: 0.1234 | Current: 0.0567 | -54.1%       │")
    print("└─────────────────────────────────────────────────────────┘")
    print("┌─────────────────────┬───────────────────────────────────┐")
    print("│   FITNESS CHART     │        STATISTICS PANEL          │")
    print("│   (Left Middle)     │        (Right Middle)            │")
    print("│                     │  Generation: 15/50               │")
    print("│   [Fitness Graph    │  Best Fitness: 0.6789            │")
    print("│    showing evolution│  Avg Fitness: 0.4321             │")
    print("│    over time]       │  Progress: 30%                   │")
    print("└─────────────────────┴───────────────────────────────────┘")
    print("┌─────────────────────┬───────────────────────────────────┐")
    print("│   TOKEN DISPLAY     │    EVOLUTION ANALYSIS             │")
    print("│   (Left Bottom)     │    (Right Bottom)                 │")
    print("│                     │                                   │")
    print("│ Token IDs: [12, 34] │ 🧬 STRING CONSTRUCTION:           │")
    print("│ Decoded: ['█', '∅'] │   Original: \"The quick brown\"     │")
    print("│                     │   Evolved: [█∅] + \"The quick...\"  │")
    print("│ Full String:        │   Result: \"█∅The quick brown\"     │")
    print("│ [█∅] + \"The quick\" │                                   │")
    print("│ = \"█∅The quick\"     │ 📊 PREDICTION IMPACT:             │")
    print("│                     │   Target: \"fox\" (ID: 21831)      │")
    print("│ Fitness: 0.6789     │   Baseline → Current: -54.1%     │")
    print("└─────────────────────┴───────────────────────────────────┘")
    print()

def run_demonstration(model_name: str, scenario_name: str = None):
    """Run the actual GUI demonstration."""
    try:
        # Try to import required modules
        try:
            from glitcher.genetic.reducer import GeneticProbabilityReducer
            from glitcher.genetic.gui_animator import GeneticAnimationCallback, RealTimeGeneticAnimator
        except ImportError as e:
            print(f"❌ Cannot import required modules: {e}")
            print("Please ensure you have:")
            print("  1. Installed the package: pip install -e .")
            print("  2. Installed matplotlib: pip install matplotlib")
            return False

        scenarios = create_demo_scenarios()

        # Filter to specific scenario if requested
        if scenario_name:
            scenarios = [s for s in scenarios if s['name'] == scenario_name]
            if not scenarios:
                print(f"❌ Scenario '{scenario_name}' not found!")
                print("Available scenarios:", [s['name'] for s in create_demo_scenarios()])
                return False

        print(f"\n🚀 Running Enhanced GUI Demonstration")
        print(f"Model: {model_name}")
        print(f"Scenarios: {len(scenarios)}")
        print()

        for i, scenario in enumerate(scenarios, 1):
            print(f"📋 Scenario {i}/{len(scenarios)}: {scenario['description']}")
            print(f"   Base text: \"{scenario['base_text']}\"")
            print(f"   Expected display: \"{scenario['expected_display']}\"")
            print(f"   Parameters: {scenario['generations']} gen, {scenario['population_size']} pop")

            try:
                # Initialize genetic reducer
                print("   🔄 Loading model and initializing...")

                # Setup GUI
                animator = RealTimeGeneticAnimator(
                    base_text=scenario['base_text'],
                    max_generations=scenario['generations']
                )
                gui_callback = GeneticAnimationCallback(animator)

                # Create reducer with GUI
                ga = GeneticProbabilityReducer(
                    model_name=model_name,
                    base_text=scenario['base_text'],
                    target_token=None,
                    gui_callback=gui_callback
                )

                # Configure parameters
                ga.population_size = scenario['population_size']
                ga.max_generations = scenario['generations']
                ga.max_tokens_per_individual = 3

                # Load model and tokens
                ga.load_model()
                ga.load_glitch_tokens("glitch_tokens.json")

                print(f"   ✅ Ready! Loaded {len(ga.glitch_tokens)} glitch tokens")
                print()
                print("   🖼️  GUI FEATURES TO OBSERVE:")
                print("      📝 Full string construction in Token Display panel")
                print("      🎯 Complete prediction context in Info panel")
                print("      🧬 Real-time string evolution in Analysis panel")
                print("      📊 Live probability changes in Statistics panel")
                print()
                print("   ⏯️  Starting evolution... (close GUI window when done)")

                # Run evolution
                results = ga.run_evolution()

                if results:
                    best = results[0]
                    evolved_text = "".join([ga.tokenizer.decode([tid]) for tid in best.tokens])
                    full_string = evolved_text + scenario['base_text']
                    reduction = ((best.baseline_prob - best.modified_prob) / best.baseline_prob * 100) if best.baseline_prob > 0 else 0

                    print(f"   ✅ Scenario {i} completed!")
                    print(f"      Best tokens: {best.tokens}")
                    print(f"      Evolved prefix: \"{evolved_text}\"")
                    print(f"      Full string: \"{full_string}\"")
                    print(f"      Probability reduction: {reduction:.1f}%")
                    print()

                # Keep GUI alive for viewing
                if gui_callback:
                    print("   💡 GUI is live - close window to continue to next scenario")
                    try:
                        gui_callback.keep_alive(duration=10)  # 10 second timeout
                    except KeyboardInterrupt:
                        print("   ⏸️  Interrupted by user")
                        break

            except KeyboardInterrupt:
                print(f"   ⏸️  Scenario {i} interrupted")
                break
            except Exception as e:
                print(f"   ❌ Error in scenario {i}: {e}")
                continue

            print("   " + "─" * 50)

        print("\n🎉 Enhanced GUI Demonstration Complete!")
        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demo enhanced GUI string visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DEMONSTRATION SCENARIOS:

1. short_phrase: "The quick brown"
   - Shows basic token insertion and string construction

2. sentence_completion: "Hello world, this is a"
   - Demonstrates context impact on predictions

3. technical_context: "In machine learning, the most important"
   - Shows specialized vocabulary effects

4. long_context: "Once upon a time, in a galaxy far far away..."
   - Demonstrates string truncation handling

EXAMPLES:
  # Run all scenarios
  python demo_enhanced_gui_strings.py meta-llama/Llama-3.2-1B-Instruct

  # Run specific scenario
  python demo_enhanced_gui_strings.py meta-llama/Llama-3.2-1B-Instruct --scenario short_phrase
        """
    )

    parser.add_argument(
        'model_name',
        help='HuggingFace model identifier'
    )

    parser.add_argument(
        '--scenario',
        help='Run specific scenario only',
        choices=['short_phrase', 'sentence_completion', 'technical_context', 'long_context']
    )

    parser.add_argument(
        '--overview-only',
        action='store_true',
        help='Show overview and examples without running actual GUI'
    )

    args = parser.parse_args()

    # Show overview
    print_gui_enhancement_overview()
    demonstrate_string_construction_logic()
    show_gui_panels_layout()

    if args.overview_only:
        print("📖 Overview complete! Use without --overview-only to run actual GUI demo.")
        return

    # Run demonstration
    print("🎬 Starting live GUI demonstration...")
    print("=" * 60)

    success = run_demonstration(args.model_name, args.scenario)

    if success:
        print()
        print("✅ DEMO SUMMARY")
        print("=" * 40)
        print("Enhanced GUI now shows:")
        print("  ✓ Full string construction with visual token positioning")
        print("  ✓ Real-time context updates during evolution")
        print("  ✓ Complete prediction impact analysis")
        print("  ✓ Professional formatting with clear structure")
        print("  ✓ Enhanced user experience with emojis and color coding")
        print()
        print("The GUI clearly displays WHERE tokens are inserted and")
        print("HOW they affect the complete input string to the model!")

    else:
        print("❌ Demo encountered issues. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
