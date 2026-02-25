#!/usr/bin/env python3
"""
GUI Launcher for Glitcher Genetic Algorithm

This script provides a standalone launcher for the Glitcher genetic algorithm
graphical user interface, allowing interactive control of evolution experiments.
"""

import sys
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []

    try:
        import torch  # noqa: F401
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers  # noqa: F401
    except ImportError:
        missing_deps.append("transformers")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing_deps.append("matplotlib (optional, for GUI animation)")

    return missing_deps

def main():
    """Main launcher function"""
    print("🚀 Glitcher Genetic Algorithm GUI Launcher")
    print("=" * 50)

    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("⚠️  Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies with:")
        print("   pip install torch transformers matplotlib")

        # Ask if user wants to continue anyway
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting...")
            return 1

    try:
        # Import the GUI controller
        print("📦 Loading GUI components...")
        from glitcher.genetic.gui_controller import GeneticControllerGUI

        # Create and configure root window
        print("🎨 Initializing GUI...")
        root = tk.Tk()

        # Set window properties
        root.title("Glitcher Genetic Algorithm Controller")
        root.geometry("1200x800")
        root.minsize(800, 600)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        # Create the application
        app = GeneticControllerGUI(root)

        # Setup cleanup handler
        def on_closing():
            if app.is_running:
                if messagebox.askyesno("Quit", "Evolution is running. Stop and quit?"):
                    app.should_stop = True
                    # Give the evolution thread a moment to stop
                    if app.evolution_thread and app.evolution_thread.is_alive():
                        app.evolution_thread.join(timeout=2.0)
                    root.destroy()
            else:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        print("✅ GUI ready! Starting application...")
        print("\n📋 Usage Instructions:")
        print("   1. Configure parameters in the 'Configuration' tab")
        print("   2. Use the 'Control' tab to start/stop evolution")
        print("   3. Monitor progress in the 'Progress' tab")
        print("   4. View results in the 'Results' tab")
        print("\n🎯 Features:")
        print("   • Real-time parameter adjustment")
        print("   • Start/pause/stop controls")
        print("   • Live progress monitoring")
        print("   • Comprehensive search support")
        print("   • Configuration save/load")
        print("   • GUI animation integration")
        print()

        # Start the GUI main loop
        root.mainloop()

        print("👋 GUI closed. Goodbye!")
        return 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the glitcher package is properly installed.")
        print("Try: pip install -e .")
        return 1

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def show_help():
    """Show help information"""
    help_text = """
Glitcher Genetic Algorithm GUI Launcher

USAGE:
    python -m glitcher.gui_launcher
    python glitcher/gui_launcher.py

DESCRIPTION:
    Launches a graphical user interface for controlling and monitoring
    genetic algorithm evolution experiments with glitch tokens.

FEATURES:
    • Interactive parameter configuration
    • Real-time evolution control (start/pause/stop)
    • Live progress monitoring and visualization
    • Comprehensive search capabilities
    • Configuration management (save/load)
    • Results analysis and export
    • Integration with GUI animation

REQUIREMENTS:
    • Python 3.8+
    • torch
    • transformers
    • tkinter (usually included with Python)
    • matplotlib (optional, for GUI animation)

INSTALLATION:
    pip install torch transformers matplotlib
    pip install -e .  # Install glitcher package

For command-line usage, use:
    glitcher genetic --help
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_help()
        sys.exit(0)

    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
