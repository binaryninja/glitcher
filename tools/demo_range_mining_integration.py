#!/usr/bin/env python3
"""
Demo script showing the new integrated range mining functionality.

This script demonstrates how range mining has been integrated into the main glitcher CLI,
showing the new commands and their usage patterns.
"""

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_command(description, command):
    """Print a formatted command example."""
    print(f"\n📌 {description}")
    print(f"   Command: {command}")

def main():
    print("🔧 Glitcher Range Mining Integration Demo")
    print("=" * 60)
    print("Range mining functionality has been successfully integrated into the main glitcher CLI!")
    print("You can now use the --mode parameter to access different mining strategies.")

    print_section("1. Basic Range Mining Commands")

    print_command(
        "Standard entropy-based mining (default behavior)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50"
    )

    print_command(
        "Custom token ID range mining",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 0 --range-end 1000 --sample-rate 0.1"
    )

    print_command(
        "Unicode range mining (systematic Unicode block exploration)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --sample-rate 0.05 --max-tokens-per-range 50"
    )

    print_command(
        "Special token range mining (vocabulary artifacts)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode special --sample-rate 0.2 --max-tokens-per-range 100"
    )

    print_section("2. Enhanced Validation with Range Mining")

    print_command(
        "Range mining with enhanced validation and multiple attempts",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 128000 --range-end 128256 --sample-rate 1.0 --num-attempts 3"
    )

    print_command(
        "High-confidence Unicode mining with strict ASR threshold",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --sample-rate 0.1 --asr-threshold 0.8 --num-attempts 5"
    )

    print_command(
        "Special range mining with custom validation parameters",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode special --sample-rate 0.3 --validation-tokens 100 --asr-threshold 0.7"
    )

    print_section("3. Targeted Range Mining Examples")

    print_command(
        "Reserved special tokens (common glitch location)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 128000 --range-end 128256 --sample-rate 1.0"
    )

    print_command(
        "Early vocabulary (often contains artifacts)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 0 --range-end 1000 --sample-rate 0.5"
    )

    print_command(
        "High token IDs (rare and specialized tokens)",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 100000 --range-end 128000 --sample-rate 0.05"
    )

    print_section("4. Output and Result Management")

    print_command(
        "Custom output files for different mining modes",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --output unicode_glitches.json"
    )

    print_command(
        "Compare different mining approaches",
        "glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode special --output special_tokens.json && glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode range --range-start 0 --range-end 5000 --output early_vocab.json"
    )

    print_section("5. Mining Mode Parameters")

    print("\n📋 Available Mining Modes:")
    print("   • entropy  - Standard entropy-based mining (default)")
    print("   • range    - Custom token ID range (requires --range-start and --range-end)")
    print("   • unicode  - Unicode character block exploration")
    print("   • special  - Special token and artifact ranges")

    print("\n📋 Range Mining Specific Parameters:")
    print("   • --sample-rate          - Fraction of tokens to test (0.0-1.0, default: 0.1)")
    print("   • --max-tokens-per-range - Maximum tokens per range (default: 100)")
    print("   • --range-start          - Starting token ID (range mode only)")
    print("   • --range-end            - Ending token ID (range mode only)")

    print("\n📋 Enhanced Validation Parameters (work with all modes):")
    print("   • --num-attempts    - Multiple validation attempts for non-deterministic models")
    print("   • --asr-threshold   - Attack Success Rate threshold (0.0-1.0, default: 0.5)")
    print("   • --validation-tokens - Max tokens to generate in validation (default: 50)")

    print("\n📋 Device Usage Best Practices:")
    print("   • Default: --device cuda (optimal performance)")
    print("   • Alternative: --device auto (automatic GPU detection)")
    print("   • Testing only: --device cpu (slow, only for debugging)")
    print("   • Multi-GPU: --device auto (automatic device mapping)")

    print_section("6. Backward Compatibility")

    print("\n✅ The integration maintains full backward compatibility:")
    print("   • Existing glitcher mine commands work unchanged")
    print("   • Default behavior is unchanged (entropy mode)")
    print("   • All existing parameters and options are preserved")
    print("   • Legacy range_mining.py script is still available")

    print_section("7. Migration from Legacy Scripts")

    print("\n🔄 Old vs New Command Patterns:")
    print("\nOLD (standalone script):")
    print("   python range_mining.py model --range-start 0 --range-end 1000 --sample-rate 0.1")
    print("\nNEW (integrated CLI):")
    print("   glitcher mine model --mode range --range-start 0 --range-end 1000 --sample-rate 0.1")

    print("\nOLD (Unicode ranges):")
    print("   python range_mining.py model --unicode-ranges --sample-rate 0.05")
    print("\nNEW (integrated CLI):")
    print("   glitcher mine model --mode unicode --sample-rate 0.05")

    print("\nOLD (special ranges):")
    print("   python range_mining.py model --special-ranges --sample-rate 0.2")
    print("\nNEW (integrated CLI):")
    print("   glitcher mine model --mode special --sample-rate 0.2")

    print_section("8. Quick Start Examples")

    print("\n🚀 Try these commands to get started:")

    print_command(
        "Quick test with small range (fast)",
        "glitcher mine gpt2 --mode range --range-start 0 --range-end 100 --sample-rate 1.0"
    )

    print_command(
        "Unicode exploration (comprehensive)",
        "glitcher mine gpt2 --mode unicode --sample-rate 0.1 --max-tokens-per-range 20"
    )

    print_command(
        "High-confidence special token mining",
        "glitcher mine gpt2 --mode special --sample-rate 0.3 --asr-threshold 0.8 --num-attempts 3"
    )

    print("\n" + "="*60)
    print("✨ Integration Complete! Range mining is now part of the main glitcher CLI.")
    print("📖 Use 'glitcher mine --help' to see all available options.")
    print("⚡ Note: All examples use GPU by default for optimal performance.")
    print("="*60)

if __name__ == "__main__":
    main()
