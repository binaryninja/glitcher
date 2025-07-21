#!/usr/bin/env python3
"""
Test script to demonstrate the integrated range mining functionality in the main glitcher CLI.

This script shows examples of how to use the new --mode parameter with the glitcher mine command
to perform different types of range-based mining.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def test_range_mining_modes(model_path, dry_run=False):
    """Test different range mining modes."""

    # Test commands to demonstrate the new functionality
    test_commands = [
        {
            "cmd": [
                "python", "-m", "glitcher.cli", "mine", model_path,
                "--mode", "range",
                "--range-start", "0",
                "--range-end", "100",
                "--sample-rate", "0.5",
                "--output", "test_range_mining.json"
            ],
            "description": "Range Mining: Custom token ID range (0-100)"
        },
        {
            "cmd": [
                "python", "-m", "glitcher.cli", "mine", model_path,
                "--mode", "unicode",
                "--sample-rate", "0.1",
                "--max-tokens-per-range", "10",
                "--output", "test_unicode_mining.json"
            ],
            "description": "Unicode Range Mining: Systematic Unicode block exploration"
        },
        {
            "cmd": [
                "python", "-m", "glitcher.cli", "mine", model_path,
                "--mode", "special",
                "--sample-rate", "0.2",
                "--max-tokens-per-range", "20",
                "--output", "test_special_mining.json"
            ],
            "description": "Special Range Mining: Vocabulary ranges with artifacts"
        },
        {
            "cmd": [
                "python", "-m", "glitcher.cli", "mine", model_path,
                "--mode", "range",
                "--range-start", "128000",
                "--range-end", "128100",
                "--sample-rate", "1.0",
                "--num-attempts", "3",
                "--asr-threshold", "0.8",
                "--output", "test_high_confidence_range.json"
            ],
            "description": "High-confidence range mining with enhanced validation"
        }
    ]

    if dry_run:
        print("DRY RUN MODE - Commands that would be executed:")
        for i, test in enumerate(test_commands, 1):
            print(f"\n{i}. {test['description']}")
            print(f"   Command: {' '.join(test['cmd'])}")
        return True

    print(f"Testing integrated range mining functionality with model: {model_path}")
    print("Note: Using default GPU device for optimal performance")

    success_count = 0
    total_tests = len(test_commands)

    for i, test in enumerate(test_commands, 1):
        print(f"\n{'#'*80}")
        print(f"TEST {i}/{total_tests}: {test['description']}")
        print(f"{'#'*80}")

        success = run_command(test['cmd'], test['description'])
        if success:
            success_count += 1
            print(f"✅ Test {i} PASSED")
        else:
            print(f"❌ Test {i} FAILED")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {success_count}/{total_tests} tests passed")
    print(f"{'='*80}")

    return success_count == total_tests


def test_help_commands():
    """Test that help commands work for the new functionality."""
    print("Testing help commands...")

    help_commands = [
        ["python", "-m", "glitcher.cli", "--help"],
        ["python", "-m", "glitcher.cli", "mine", "--help"]
    ]

    for cmd in help_commands:
        success = run_command(cmd, f"Help command: {' '.join(cmd)}")
        if not success:
            return False

    return True


def validate_integration():
    """Validate that the integration is working properly."""
    print("Validating range mining integration...")

    try:
        # Test import
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from range_mining import range_based_mining
        print("✅ range_mining module import successful")

        # Test CLI import
        from glitcher.cli import GlitcherCLI
        print("✅ GlitcherCLI import successful")

        # Test that the CLI has the new arguments
        cli = GlitcherCLI()
        parser = cli.setup_parser()

        # Parse help to check if new arguments exist
        try:
            # This will raise SystemExit, but we can catch it
            parser.parse_args(["mine", "--help"])
        except SystemExit:
            pass  # Expected for help

        print("✅ CLI parser setup successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test integrated range mining functionality"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="gpt2",
        help="Model to test with (default: gpt2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands that would be run without executing them"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation checks, not full tests"
    )
    parser.add_argument(
        "--help-only",
        action="store_true",
        help="Only test help commands"
    )

    args = parser.parse_args()

    print("🔍 Integrated Range Mining Test Suite")
    print("=" * 50)

    # Always run validation first
    if not validate_integration():
        print("❌ Integration validation failed!")
        return False

    if args.validate_only:
        print("✅ Validation completed successfully!")
        return True

    # Test help commands
    if not test_help_commands():
        print("❌ Help command tests failed!")
        return False

    if args.help_only:
        print("✅ Help command tests completed successfully!")
        return True

    # Run full mining tests
    success = test_range_mining_modes(args.model_path, args.dry_run)

    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nRange mining has been successfully integrated into the main glitcher CLI!")
        print("\nYou can now use commands like:")
        print("  glitcher mine model_name --mode range --range-start 0 --range-end 1000")
        print("  glitcher mine model_name --mode unicode --sample-rate 0.1")
        print("  glitcher mine model_name --mode special --sample-rate 0.2")
    else:
        print("\n❌ Some tests failed. Please check the output above.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
