#!/usr/bin/env python3
"""
Debug test to verify fitness calculation and reduction percentage bug.

This script runs a short genetic algorithm test and examines the final population
to understand why reduction percentages are being calculated incorrectly.
"""

import sys
from pathlib import Path

# Add the parent directory to sys.path to import glitcher modules
sys.path.insert(0, str(Path(__file__).parent))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def debug_fitness_calculation():
    """Debug fitness calculation issues."""

    print("🔍 DEBUGGING FITNESS CALCULATION")
    print("=" * 50)

    # Create genetic algorithm instance
    ga = GeneticProbabilityReducer(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        base_text="The quick brown"
    )

    # Configure for quick test
    ga.population_size = 10
    ga.max_generations = 20
    ga.adaptive_mutation = True
    ga.initial_mutation_rate = 0.8
    ga.final_mutation_rate = 0.1
    ga.current_mutation_rate = 0.8
    ga.max_tokens_per_individual = 3

    # Load model and tokens
    print("📁 Loading model and tokens...")
    ga.load_model()
    ga.load_glitch_tokens("email_extraction_all_glitches.json", ascii_only=True)

    print(f"✓ Loaded {len(ga.glitch_tokens)} glitch tokens")
    print(f"✓ Baseline probability: {ga.baseline_probability:.4f}")
    print(f"✓ Target token ID: {ga.target_token_id}")

    # Run short evolution
    print("\n🧬 Running short evolution...")
    final_population = ga.run_evolution()

    print(f"\n📊 FINAL POPULATION ANALYSIS")
    print("=" * 40)
    print(f"Population size: {len(final_population)}")

    # Examine each individual in detail
    valid_individuals = []
    for i, individual in enumerate(final_population):
        print(f"\nIndividual {i+1}:")
        print(f"  Tokens: {individual.tokens}")
        print(f"  Has fitness: {hasattr(individual, 'fitness')}")

        if hasattr(individual, 'fitness'):
            print(f"  Fitness: {individual.fitness:.6f}")
            print(f"  Has baseline_prob: {hasattr(individual, 'baseline_prob')}")
            print(f"  Has modified_prob: {hasattr(individual, 'modified_prob')}")

            if hasattr(individual, 'baseline_prob') and hasattr(individual, 'modified_prob'):
                baseline = individual.baseline_prob
                modified = individual.modified_prob
                fitness = individual.fitness

                # Verify fitness calculation
                expected_fitness = baseline - modified
                fitness_matches = abs(fitness - expected_fitness) < 1e-6

                # Calculate reduction percentage
                reduction_pct = ((baseline - modified) / baseline) * 100

                print(f"  Baseline prob: {baseline:.6f}")
                print(f"  Modified prob: {modified:.6f}")
                print(f"  Expected fitness: {expected_fitness:.6f}")
                print(f"  Fitness matches: {fitness_matches}")
                print(f"  Reduction %: {reduction_pct:.2f}%")

                if individual.fitness > -1.0:
                    valid_individuals.append(individual)
            else:
                print(f"  ⚠️ Missing baseline_prob or modified_prob")
        else:
            print(f"  ❌ No fitness attribute")

    print(f"\n📈 SUMMARY")
    print("=" * 20)
    print(f"Valid individuals: {len(valid_individuals)}")

    if valid_individuals:
        # Find best individual
        best_individual = max(valid_individuals, key=lambda x: x.fitness)
        best_fitness = best_individual.fitness

        if hasattr(best_individual, 'baseline_prob') and hasattr(best_individual, 'modified_prob'):
            baseline = best_individual.baseline_prob
            modified = best_individual.modified_prob
            correct_reduction_pct = ((baseline - modified) / baseline) * 100

            print(f"Best individual:")
            print(f"  Tokens: {best_individual.tokens}")
            print(f"  Token texts: {[ga.tokenizer.decode([tid]).strip() for tid in best_individual.tokens]}")
            print(f"  Fitness: {best_fitness:.6f}")
            print(f"  Baseline prob: {baseline:.6f}")
            print(f"  Modified prob: {modified:.6f}")
            print(f"  Correct reduction %: {correct_reduction_pct:.2f}%")

            # Show the wrong calculation that was being used
            wrong_reduction_pct = (best_fitness / baseline) * 100
            print(f"  Wrong calculation would give: {wrong_reduction_pct:.2f}%")

            print(f"\n💡 CONCLUSION:")
            if correct_reduction_pct > 70:
                print(f"✅ Algorithm is working! {correct_reduction_pct:.1f}% reduction achieved")
            else:
                print(f"⚠️ Partial success: {correct_reduction_pct:.1f}% reduction")

            if abs(wrong_reduction_pct) < 1e-3:
                print(f"❌ Bug confirmed: Wrong calculation gives {wrong_reduction_pct:.1f}%")

        else:
            print(f"❌ Best individual missing probability attributes")

        # Test manual evaluation of best individual
        print(f"\n🔬 MANUAL EVALUATION TEST")
        print("=" * 30)

        # Re-evaluate the best individual manually
        original_fitness = best_individual.fitness
        manual_fitness = ga.evaluate_fitness(best_individual)

        print(f"Original fitness: {original_fitness:.6f}")
        print(f"Manual fitness: {manual_fitness:.6f}")
        print(f"Fitness consistent: {abs(original_fitness - manual_fitness) < 1e-6}")

    else:
        print(f"❌ No valid individuals found!")
        print(f"This indicates a serious issue with the fitness evaluation.")


if __name__ == "__main__":
    debug_fitness_calculation()
