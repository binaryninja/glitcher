#!/usr/bin/env python3
"""
Demo: Integrated GUI for Glitcher Genetic Algorithm

This script demonstrates the new integrated GUI features including:
- Real-time matplotlib fitness evolution plots embedded in Tkinter
- Enhanced progress visualization with detailed token information
- Integrated context display showing input strings and predictions
- Proper callback integration for smooth progress updates
- Thread-safe GUI updates during evolution
- Start/Pause/Stop controls with proper evolution management

New Features Demonstrated:
1. Embedded matplotlib plots in the GUI (no separate windows)
2. Detailed best individual display with token breakdown
3. Context strings showing how tokens modify the input
4. Real-time fitness evolution graphs
5. Enhanced log display with evolution progress
6. Proper GUI callback integration for smooth updates
7. Thread-safe progress updates without conflicts

Usage:
    python demo_integrated_gui.py
"""

import tkinter as tk
import sys
import os
import json
from pathlib import Path

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def create_demo_config():
    """Create a demo configuration file"""
    config = {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "base_text": "The quick brown",
        "target_token": "",
        "wanted_token": "fox",
        "token_file": "demo_glitch_tokens.json",

        "population_size": 30,
        "generations": 50,
        "max_tokens": 3,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "elite_size": 5,
        "early_stopping_threshold": 0.999,

        "asr_threshold": 0.5,
        "num_attempts": 3,
        "validation_tokens": 50,

        "ascii_only": True,
        "enhanced_validation": True,
        "comprehensive_search": False,
        "include_normal_tokens": False,
        "baseline_seeding": True,
        "sequence_diversity": True,
        "exact_token_count": True,
        "enable_shuffle_mutation": False,

        "baseline_seeding_ratio": 0.7,
        "sequence_diversity_ratio": 0.3,

        "output_file": "demo_genetic_results.json",
        "baseline_output": "demo_token_impact_baseline.json",
        "show_gui_animation": False
    }

    config_file = Path("demo_integrated_gui_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_file)

def create_demo_tokens():
    """Create a demo glitch tokens file"""
    demo_tokens = {
        "glitch_tokens": [
            {"id": 12345, "text": "Red", "entropy": 0.1234, "target_prob": 0.001},
            {"id": 67890, "text": "Quick", "entropy": 0.2345, "target_prob": 0.002},
            {"id": 11111, "text": "Fast", "entropy": 0.3456, "target_prob": 0.003},
            {"id": 22222, "text": "Swift", "entropy": 0.4567, "target_prob": 0.004},
            {"id": 33333, "text": "Speed", "entropy": 0.5678, "target_prob": 0.005},
            {"id": 44444, "text": "Rapid", "entropy": 0.6789, "target_prob": 0.006},
            {"id": 55555, "text": "Fleet", "entropy": 0.7890, "target_prob": 0.007},
            {"id": 66666, "text": "Agile", "entropy": 0.8901, "target_prob": 0.008},
            {"id": 77777, "text": "Nimble", "entropy": 0.9012, "target_prob": 0.009},
            {"id": 88888, "text": "Brisk", "entropy": 0.1023, "target_prob": 0.010}
        ]
    }

    token_file = Path("demo_glitch_tokens.json")
    with open(token_file, 'w') as f:
        json.dump(demo_tokens, f, indent=2)

    return str(token_file)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []

    try:
        import tkinter
        print("✓ tkinter available")
    except ImportError:
        missing_deps.append("tkinter")

    try:
        import matplotlib
        print("✓ matplotlib available - embedded plots will work")
    except ImportError:
        print("⚠️  matplotlib not available - fallback display will be used")
        print("   Install with: pip install matplotlib")

    try:
        from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig
        print("✓ Glitcher GUI components available")
    except ImportError as e:
        print(f"✗ Glitcher GUI components not available: {e}")
        missing_deps.append("glitcher.genetic.gui_controller")

    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        return False

    return True

def setup_demo_files():
    """Set up demo configuration and token files"""
    print("📁 Setting up demo files...")

    config_file = create_demo_config()
    print(f"✓ Created demo configuration: {config_file}")

    token_file = create_demo_tokens()
    print(f"✓ Created demo token file: {token_file}")

    return config_file, token_file

def launch_integrated_gui(config_file):
    """Launch the integrated GUI with demo configuration"""
    from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig

    print("🚀 Launching Integrated GUI...")

    # Create main window
    root = tk.Tk()

    # Create GUI controller
    gui = GeneticControllerGUI(root)

    # Load demo configuration
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # Update config object
        for key, value in config_dict.items():
            if hasattr(gui.config, key):
                setattr(gui.config, key, value)

        gui.update_gui_from_config()
        print("✓ Demo configuration loaded")
    except Exception as e:
        print(f"⚠️  Could not load config file: {e}")
        # Use default configuration
        gui.config.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        gui.config.base_text = "The quick brown"
        gui.config.wanted_token = "fox"
        gui.config.generations = 30
        gui.config.population_size = 20
        gui.update_gui_from_config()

    # Show instructions
    instructions = """
🎉 Welcome to the Integrated Glitcher GUI!

NEW FEATURES IN THIS VERSION:
✨ Real-time matplotlib fitness plots embedded in the GUI
✨ Enhanced progress visualization with detailed token info
✨ Integrated context display showing input string construction
✨ Proper callback integration for smooth progress updates
✨ Thread-safe GUI updates during evolution
✨ No more separate animation windows!

HOW TO USE:
1. 📝 Configuration Tab: Adjust evolution parameters
2. 🎮 Control Tab: Start/Pause/Stop evolution with buttons
3. 📊 Progress Tab: Watch real-time fitness evolution + detailed info
4. 📋 Results Tab: View final results and save data

KEY IMPROVEMENTS:
• Fitness evolution graph updates in real-time (no separate window)
• Best individual display shows token breakdown and decoded text
• Context display shows complete input strings fed to the model
• Progress updates are smooth and don't conflict with GUI
• Enhanced log messages track evolution progress
• All visualization integrated into single window

DEMO SCENARIO:
• Model: Llama-3.2-1B-Instruct
• Task: Evolve tokens to increase probability of "fox"
• Base text: "The quick brown" → "The quick brown fox"
• Watch the fitness plot show probability increases over generations!

Ready to start? Click the Start Evolution button in the Control tab!
    """

    gui.log_message("=== INTEGRATED GUI DEMO ===")
    for line in instructions.strip().split('\n'):
        gui.log_message(line)

    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')

    print("\n🎯 GUI launched successfully!")
    print("👀 Check the Progress tab to see the new integrated visualization")
    print("🎮 Use the Control tab to start evolution")
    print("📝 Adjust parameters in the Configuration tab")
    print("\n💡 The main improvement: everything is now in one integrated window!")

    # Start the GUI main loop
    root.mainloop()

    return True

def cleanup_demo_files():
    """Clean up demo files"""
    files_to_clean = [
        "demo_integrated_gui_config.json",
        "demo_glitch_tokens.json",
        "demo_genetic_results.json",
        "demo_token_impact_baseline.json"
    ]

    cleaned = []
    for file_path in files_to_clean:
        if Path(file_path).exists():
            try:
                Path(file_path).unlink()
                cleaned.append(file_path)
            except:
                pass

    if cleaned:
        print(f"🧹 Cleaned up demo files: {', '.join(cleaned)}")

def main():
    """Main demo function"""
    print("🎬 Glitcher Integrated GUI Demo")
    print("=" * 50)
    print("Demonstrating the new integrated GUI with embedded matplotlib plots")
    print()

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot run demo due to missing dependencies")
        return

    print()

    # Setup demo files
    try:
        config_file, token_file = setup_demo_files()
    except Exception as e:
        print(f"❌ Failed to setup demo files: {e}")
        return

    print()

    try:
        # Launch the integrated GUI
        success = launch_integrated_gui(config_file)

        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up demo files
        cleanup_demo_files()

if __name__ == "__main__":
    main()
