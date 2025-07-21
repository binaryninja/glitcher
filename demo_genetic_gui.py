#!/usr/bin/env python3
"""
GUI Demo Script for Genetic Algorithm Animation

This script demonstrates the real-time GUI animation functionality
for the genetic algorithm without requiring a full model download.
It uses simulated data to show how the GUI works during evolution.

Usage:
    python demo_genetic_gui.py

Author: Claude
Date: 2024
"""

import time
import random
import threading
from typing import List
import os
import sys

# Add the glitcher package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simulate_genetic_evolution():
    """
    Simulate a genetic algorithm evolution with realistic data progression.

    This function demonstrates how the GUI animation works by feeding it
    simulated data that mimics a real genetic algorithm run.
    """
    print("🚀 Starting Genetic Algorithm GUI Demo")
    print("=" * 50)

    try:
        # Import GUI components
        from glitcher.genetic import RealTimeGeneticAnimator, GeneticAnimationCallback
        from glitcher.genetic.reducer import Individual

        print("✅ GUI components loaded successfully")

        # Demo parameters
        base_text = "The quick brown"
        target_token_text = "fox"
        target_token_id = 39935
        baseline_prob = 0.9487
        max_generations = 100
        population_size = 50

        # Create animator
        print("🎬 Initializing real-time animation...")
        animator = RealTimeGeneticAnimator(
            base_text=base_text,
            target_token_text=target_token_text,
            target_token_id=target_token_id,
            baseline_probability=baseline_prob,
            max_generations=max_generations
        )

        # Create callback
        callback = GeneticAnimationCallback(animator)

        # Start evolution
        print("🧬 Starting simulated evolution...")
        callback.on_evolution_start(
            baseline_prob=baseline_prob,
            target_token_id=target_token_id,
            target_token_text=target_token_text
        )

        # Simulate realistic evolution data
        evolution_stages = [
            # (gen_range, fitness_range, tokens_pool, improvement_rate)
            (range(0, 20), (0.001, 0.05), [[127865], [102853], [114540]], 0.002),
            (range(20, 40), (0.05, 0.15), [[102853, 127896], [114540, 127865]], 0.005),
            (range(40, 60), (0.15, 0.35), [[102853, 127896, 114540], [125654, 127896]], 0.008),
            (range(60, 80), (0.35, 0.55), [[118508, 127896, 114540], [125654, 104516]], 0.006),
            (range(80, 95), (0.55, 0.75), [[126357, 104516, 118508], [118508, 118508, 114540]], 0.004),
            (range(95, 100), (0.75, 0.793), [[126357, 104516, 118508]], 0.001),
        ]

        # Token text mappings for display
        token_texts = {
            127865: "' предназнач'",
            102853: "'ılmış'",
            114540: "'ávající'",
            127896: "'מיועד'",
            125654: "'bestämm'",
            118508: "'ám'",
            104516: "'zem'",
            126357: "'предназнач'"
        }

        print("📊 Evolution stages:")
        for i, (gen_range, fit_range, tokens, rate) in enumerate(evolution_stages):
            print(f"  Stage {i+1}: Gen {gen_range.start}-{gen_range.stop-1}, "
                  f"Fitness {fit_range[0]:.3f}-{fit_range[1]:.3f}")

        print("\n🎮 Starting animation (close window to stop)...")
        print("    Watch the real-time evolution of:")
        print("    - Fitness scores over generations")
        print("    - Current best token combinations")
        print("    - Probability reduction statistics")
        print("    - Progress tracking")

        # Run evolution simulation
        current_fitness = 0.0
        current_tokens = [127865]

        for stage_idx, (gen_range, fitness_range, tokens_pool, improvement_rate) in enumerate(evolution_stages):
            print(f"\n📈 Stage {stage_idx + 1}: Generations {gen_range.start}-{gen_range.stop-1}")

            for generation in gen_range:
                # Simulate fitness improvement with some randomness
                target_fitness = fitness_range[0] + (fitness_range[1] - fitness_range[0]) * \
                               ((generation - gen_range.start) / len(gen_range))

                # Add some noise but generally improve
                noise = random.uniform(-0.02, 0.04)
                current_fitness = min(max(target_fitness + noise, current_fitness), fitness_range[1])

                # Occasionally change token combination
                if random.random() < 0.1 or generation == gen_range.start:
                    current_tokens = random.choice(tokens_pool).copy()

                # Calculate average fitness (slightly lower than best)
                avg_fitness = current_fitness * random.uniform(0.3, 0.8)

                # Calculate current probability
                current_prob = baseline_prob * (1 - current_fitness)

                # Get token texts
                current_token_texts = [token_texts.get(t, f"Token{t}") for t in current_tokens]

                # Create mock individual
                individual = Individual(tokens=current_tokens)
                individual.fitness = current_fitness

                # Update GUI
                callback.on_generation_complete(
                    generation=generation,
                    best_individual=individual,
                    avg_fitness=avg_fitness,
                    current_probability=current_prob
                )

                # Print progress
                if generation % 10 == 0:
                    reduction = (1 - current_prob / baseline_prob) * 100
                    print(f"    Gen {generation:3d}: Fitness={current_fitness:.4f}, "
                          f"Reduction={reduction:.1f}%, Tokens={current_tokens}")

                # Simulation delay - adjust for faster/slower demo
                time.sleep(0.1)  # 100ms per generation

        # Mark evolution as complete
        final_population = [Individual(tokens=current_tokens)]
        final_population[0].fitness = current_fitness

        final_reduction = (1 - current_prob / baseline_prob) * 100
        completion_message = f"Final: {current_fitness:.4f} fitness, {final_reduction:.1f}% reduction"

        callback.on_evolution_complete(final_population, max_generations)

        print(f"\n🏆 Evolution completed!")
        print(f"    Final fitness: {current_fitness:.4f}")
        print(f"    Final tokens: {current_tokens}")
        print(f"    Final probability: {current_prob:.4f}")
        print(f"    Total reduction: {final_reduction:.1f}%")
        print(f"    Token meanings: {[token_texts.get(t, f'Token{t}') for t in current_tokens]}")

        # Keep GUI alive for viewing
        print("\n🖼️  GUI is now live - you can examine the results!")
        print("    Close the animation window when done viewing.")

        try:
            callback.keep_alive(duration=None)  # Keep alive until closed
        except KeyboardInterrupt:
            print("⏹️  Demo stopped by user.")

    except ImportError as e:
        print(f"❌ GUI components not available: {e}")
        print("   Install matplotlib: pip install matplotlib")
        return False

    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def run_cli_demo():
    """Demonstrate the CLI integration with --gui flag."""
    print("\n" + "=" * 50)
    print("🖥️  CLI Integration Demo")
    print("=" * 50)

    print("The GUI is now integrated into the main CLI!")
    print("Here's how to use it with real models:")
    print()
    print("🧬 Basic usage:")
    print("   glitcher genetic meta-llama/Llama-3.2-1B-Instruct --gui")
    print()
    print("🎛️  With custom parameters:")
    print("   glitcher genetic meta-llama/Llama-3.2-1B-Instruct \\")
    print("     --gui \\")
    print("     --base-text 'Hello world' \\")
    print("     --generations 50 \\")
    print("     --population-size 30")
    print()
    print("🔬 Batch experiments with GUI:")
    print("   glitcher genetic meta-llama/Llama-3.2-1B-Instruct \\")
    print("     --gui \\")
    print("     --batch \\")
    print("     --token-file glitch_tokens.json")
    print()
    print("💡 Features shown in GUI:")
    print("   • Real-time fitness evolution graphs")
    print("   • Current best token combinations")
    print("   • Probability reduction statistics")
    print("   • Generation progress tracking")
    print("   • Token ID to text decoding")

def main():
    """Main demo function."""
    print("🧬 Genetic Algorithm GUI Animation Demo")
    print("=" * 60)
    print()
    print("This demo shows the new real-time GUI animation feature")
    print("that has been integrated into the Glitcher genetic algorithm.")
    print()
    print("The demo will:")
    print("• Simulate a realistic genetic algorithm evolution")
    print("• Show live updates of fitness and token combinations")
    print("• Demonstrate all GUI features without requiring model download")
    print()

    # Check matplotlib availability
    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__} is available")
    except ImportError:
        print("❌ matplotlib is required for GUI demo")
        print("   Install with: pip install matplotlib")
        return False

    print()
    input("📱 Press Enter to start the GUI demo...")

    # Run the simulation
    success = simulate_genetic_evolution()

    if success:
        # Show CLI integration info
        run_cli_demo()
        print("\n🎉 Demo completed successfully!")
        print("   The GUI is now ready for use with real genetic algorithms.")
    else:
        print("\n⚠️  Demo encountered issues.")
        print("   Please check the error messages above.")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
