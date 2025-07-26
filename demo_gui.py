#!/usr/bin/env python3
"""
Demo script for Glitcher Genetic Algorithm GUI

This script demonstrates how to launch and use the graphical user interface
for the Glitcher genetic algorithm evolution system.
"""

import os
import sys
import json
import tkinter as tk
from pathlib import Path

def create_sample_config():
    """Create a sample configuration file for testing"""
    config = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "base_text": "The quick brown",
        "target_token": "",
        "wanted_token": "fox",
        "token_file": "glitch_tokens.json",

        "population_size": 30,
        "generations": 50,
        "max_tokens": 3,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "elite_size": 5,
        "early_stopping_threshold": 0.95,

        "asr_threshold": 0.5,
        "num_attempts": 3,
        "validation_tokens": 50,

        "ascii_only": True,
        "enhanced_validation": True,
        "comprehensive_search": True,
        "include_normal_tokens": False,
        "baseline_seeding": True,
        "sequence_diversity": True,
        "exact_token_count": True,
        "enable_shuffle_mutation": False,

        "baseline_seeding_ratio": 0.7,
        "sequence_diversity_ratio": 0.3,

        "output_file": "genetic_results.json",
        "baseline_output": "token_impact_baseline.json",
        "show_gui_animation": True
    }

    config_path = "sample_gui_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created sample configuration: {config_path}")
    return config_path

def create_sample_token_file():
    """Create a sample glitch token file for testing"""
    # Sample glitch tokens (these are examples, real tokens would come from mining)
    sample_tokens = [
        {"token_id": 89472, "token_text": "SomeGlitchToken", "entropy": 0.1234},
        {"token_id": 127438, "token_text": "AnotherToken", "entropy": 0.2345},
        {"token_id": 85069, "token_text": "ThirdToken", "entropy": 0.3456},
        {"token_id": 92847, "token_text": "FourthToken", "entropy": 0.4567},
        {"token_id": 118923, "token_text": "FifthToken", "entropy": 0.5678}
    ]

    token_file = "glitch_tokens.json"
    if not os.path.exists(token_file):
        with open(token_file, 'w') as f:
            json.dump(sample_tokens, f, indent=2)
        print(f"✅ Created sample token file: {token_file}")
    else:
        print(f"📁 Using existing token file: {token_file}")

    return token_file

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []

    try:
        import torch
        print("✅ PyTorch available")
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers
        print("✅ Transformers available")
    except ImportError:
        missing_deps.append("transformers")

    try:
        import matplotlib
        print("✅ Matplotlib available")
    except ImportError:
        print("⚠️  Matplotlib not available (GUI animation will be disabled)")

    try:
        import tkinter
        print("✅ Tkinter available")
    except ImportError:
        missing_deps.append("tkinter")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install torch transformers matplotlib")
        return False

    return True

def demo_gui_launch():
    """Launch the GUI with sample configuration"""
    print("🚀 Glitcher GUI Demo")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running the demo.")
        return False

    try:
        # Create sample files
        print("\n📁 Setting up demo files...")
        token_file = create_sample_token_file()
        config_file = create_sample_config()

        # Import GUI components
        print("\n📦 Loading GUI components...")
        from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig

        # Create root window
        print("🎨 Initializing GUI...")
        root = tk.Tk()
        root.title("Glitcher Genetic Algorithm Demo")
        root.geometry("1200x800")
        root.minsize(800, 600)

        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Create application
        app = GeneticControllerGUI(root)

        # Load sample configuration
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)

            for key, value in config_dict.items():
                if hasattr(app.config, key):
                    setattr(app.config, key, value)

            app.update_gui_from_config()
            print(f"✅ Loaded sample configuration")
        except Exception as e:
            print(f"⚠️  Failed to load configuration: {e}")

        # Setup cleanup
        def on_closing():
            if app.is_running:
                response = tk.messagebox.askyesno(
                    "Quit",
                    "Evolution is running. Stop and quit?"
                )
                if response:
                    app.should_stop = True
                    if app.evolution_thread and app.evolution_thread.is_alive():
                        app.evolution_thread.join(timeout=2.0)
                    root.destroy()
            else:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Show usage instructions
        print("\n✅ GUI Demo Ready!")
        print("\n📋 Demo Instructions:")
        print("   1. The GUI is pre-configured with sample settings")
        print("   2. Go to the 'Configuration' tab to review/modify parameters")
        print("   3. Use the 'Control' tab to start the evolution")
        print("   4. Monitor progress in the 'Progress' tab")
        print("   5. View results in the 'Results' tab")
        print("\n🎯 Demo Features:")
        print("   • Sample model: meta-llama/Llama-3.2-1B-Instruct")
        print("   • Target text: 'The quick brown'")
        print("   • Wanted token: 'fox'")
        print("   • Comprehensive search enabled")
        print("   • GUI animation enabled")
        print("   • Sample glitch tokens provided")
        print("\n⚠️  Note: This demo uses sample glitch tokens.")
        print("   For real experiments, mine tokens first with:")
        print("   glitcher mine meta-llama/Llama-3.2-1B-Instruct")
        print()

        # Start GUI
        root.mainloop()

        print("👋 Demo completed!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the glitcher package is properly installed:")
        print("   pip install -e .")
        return False

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_cli_gui():
    """Demonstrate launching GUI via CLI"""
    print("\n🔧 CLI GUI Launch Demo")
    print("=" * 30)
    print("You can also launch the GUI via command line:")
    print("   glitcher gui")
    print("   glitcher gui --config sample_gui_config.json")
    print()
    print("Or directly:")
    print("   python -m glitcher.gui_launcher")
    print()

def show_help():
    """Show help information"""
    help_text = """
Glitcher GUI Demo Script

USAGE:
    python demo_gui.py [--help] [--cli-demo]

OPTIONS:
    --help      Show this help message
    --cli-demo  Show CLI launch examples only

DESCRIPTION:
    This script demonstrates the Glitcher genetic algorithm GUI interface.
    It creates sample configuration files and launches the GUI with
    pre-configured settings for testing.

FEATURES DEMONSTRATED:
    • Interactive parameter configuration
    • Real-time evolution control (start/pause/stop)
    • Live progress monitoring
    • Comprehensive search capabilities
    • Configuration save/load
    • GUI animation integration
    • Results analysis and export

REQUIREMENTS:
    • Python 3.8+
    • torch
    • transformers
    • matplotlib (for GUI animation)
    • tkinter (usually included with Python)

FILES CREATED:
    • sample_gui_config.json (sample configuration)
    • glitch_tokens.json (sample token file, if not exists)

NEXT STEPS:
    1. Run this demo to familiarize yourself with the GUI
    2. Mine real glitch tokens: glitcher mine <model_name>
    3. Use the GUI with real tokens for actual experiments
"""
    print(help_text)

def main():
    """Main demo function"""
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
            return 0
        elif sys.argv[1] == '--cli-demo':
            demo_cli_gui()
            return 0

    try:
        success = demo_gui_launch()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
