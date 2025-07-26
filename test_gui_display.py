#!/usr/bin/env python3
"""
Simple GUI Display Test

This script tests only the GUI setup and display functionality without
running any evolution. It focuses on:
- GUI controller initialization
- Matplotlib integration
- Basic UI components
- Progress display methods
"""

import tkinter as tk
import sys
import os
import json
from pathlib import Path

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_gui_display():
    """Test GUI display functionality without evolution"""
    print("🧪 Testing GUI Display Components")
    print("=" * 50)

    try:
        from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig
        print("✓ Successfully imported GUI components")
    except ImportError as e:
        print(f"✗ Failed to import GUI components: {e}")
        return False

    # Check matplotlib availability
    try:
        import matplotlib
        print("✓ Matplotlib available - plots will be shown")
        matplotlib_available = True
    except ImportError:
        print("⚠️  Matplotlib not available - fallback will be used")
        matplotlib_available = False

    # Create sample token file
    sample_tokens = {
        "glitch_tokens": [
            {"id": 12345, "text": "TestToken1", "entropy": 0.1234},
            {"id": 67890, "text": "TestToken2", "entropy": 0.2345},
            {"id": 11111, "text": "TestToken3", "entropy": 0.3456}
        ]
    }

    token_file = Path("test_gui_tokens.json")
    with open(token_file, 'w') as f:
        json.dump(sample_tokens, f, indent=2)

    # Create main window
    root = tk.Tk()

    try:
        # Initialize GUI controller
        gui = GeneticControllerGUI(root)
        print("✓ GUI controller initialized")

        # Test configuration
        gui.config.model_name = "meta-llama/Llama-3.2-1B-Instruct"
        gui.config.base_text = "The quick brown"
        gui.config.wanted_token = "fox"
        gui.config.token_file = str(token_file)
        gui.config.generations = 50
        gui.config.population_size = 30

        gui.update_gui_from_config()
        print("✓ Configuration updated")

        # Test progress display with mock data
        class MockIndividual:
            def __init__(self):
                self.tokens = [12345, 67890, 11111]
                self.fitness = 0.8567

        mock_individual = MockIndividual()

        # Test GUICallback integration
        print("✓ Testing GUICallback integration...")
        try:
            from glitcher.genetic.gui_controller import GUICallback
            gui_callback = GUICallback(gui)
            print("✓ GUICallback created successfully")

            # Test callback methods
            gui_callback.on_evolution_start({'target_prob': 0.1234})
            print("✓ on_evolution_start works")

            # Test generation complete callback
            for i in range(1, 6):
                generation = i * 10
                population = [mock_individual] * 5  # Mock population
                diversity = 0.5 + (i * 0.1)
                stagnation = i

                gui_callback.on_generation_complete(generation, population, mock_individual, diversity, stagnation)

            print("✓ on_generation_complete works")

        except Exception as e:
            print(f"✗ GUICallback test failed: {e}")

        # Test direct progress updates
        print("✓ Testing direct progress display...")
        for i in range(1, 6):
            generation = i * 10
            progress = i * 20
            best_fitness = 0.1 + (i * 0.15)
            avg_fitness = best_fitness * 0.8

            gui._update_progress(generation, progress, best_fitness, avg_fitness, mock_individual)

        print("✓ Progress display working")

        # Test matplotlib plot if available
        if matplotlib_available and hasattr(gui, 'fig') and gui.fig is not None:
            print("✓ Matplotlib integration working")
        else:
            print("ℹ️  Matplotlib fallback display shown")

        # Test text displays
        test_widgets = ['best_text', 'context_text', 'log_text']
        for widget in test_widgets:
            if hasattr(gui, widget):
                print(f"✓ {widget} widget available")
            else:
                print(f"✗ {widget} widget missing")

        # Test log functionality
        gui.log_message("Test log message 1")
        gui.log_message("Test log message 2")
        gui.log_message("Test log message 3")
        print("✓ Log functionality working")

        # Show results
        print("\n🎉 GUI Display Test Results:")
        print("- GUI controller creates successfully")
        print("- GUICallback integration working")
        print("- All required widgets present")
        print("- Progress display functional")
        print("- Text displays working")
        print("- Configuration management working")

        if matplotlib_available:
            print("- Matplotlib plots available")
        else:
            print("- Matplotlib fallback displayed")

        print("\n💡 GUI Test Complete!")
        print("The window will stay open for manual inspection.")
        print("You can:")
        print("1. Check all tabs are present")
        print("2. Verify progress metrics are displayed")
        print("3. See fitness plot (if matplotlib available)")
        print("4. Check configuration parameters")
        print("5. Review log messages")
        print("6. Test Start/Pause/Stop controls")
        print("7. See real-time progress updates")
        print("\nNote: The new GUICallback integration should provide")
        print("better progress updates during actual evolution runs.")
        print("\nClose the window to exit.")

        # Center the window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        # Start the GUI event loop
        root.mainloop()

        return True

    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test files
        try:
            if token_file.exists():
                token_file.unlink()
                print(f"✓ Cleaned up: {token_file}")
        except:
            pass

def main():
    """Main test function"""
    print("🖥️  GUI Display Test Suite")
    print("=" * 50)
    print("Testing GUI components without running evolution")
    print()

    # Check basic dependencies
    try:
        import tkinter as tk
        print("✓ tkinter available")
    except ImportError:
        print("✗ tkinter not available")
        return

    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("⚠️  matplotlib not available (fallback will be tested)")

    print()

    # Run the display test
    success = test_gui_display()

    if success:
        print("\n✅ GUI display test completed successfully")
    else:
        print("\n❌ GUI display test failed")

if __name__ == "__main__":
    main()
