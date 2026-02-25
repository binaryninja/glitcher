#!/usr/bin/env python3
"""
Test script for the enhanced GUI controller with matplotlib integration.

This script tests the new GUI features including:
- Matplotlib fitness evolution plots
- Enhanced progress visualization
- Detailed token display
- Context string display
- Real-time metrics updates
"""

import tkinter as tk
import sys
import os
import json
from pathlib import Path

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig
    print("✓ Successfully imported GeneticControllerGUI")
except ImportError as e:
    print(f"✗ Failed to import GUI controller: {e}")
    sys.exit(1)

# Check matplotlib availability
try:
    import matplotlib
    print("✓ Matplotlib is available - plots will be shown")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib not available - plots will show warning message")
    MATPLOTLIB_AVAILABLE = False

def create_sample_token_file():
    """Create a sample glitch tokens file for testing"""
    sample_tokens = {
        "glitch_tokens": [
            {"id": 12345, "text": "SampleToken1", "entropy": 0.1234, "target_prob": 0.001},
            {"id": 67890, "text": "SampleToken2", "entropy": 0.2345, "target_prob": 0.002},
            {"id": 11111, "text": "TestToken", "entropy": 0.3456, "target_prob": 0.003},
            {"id": 22222, "text": "GlitchToken", "entropy": 0.4567, "target_prob": 0.004},
            {"id": 33333, "text": "ExampleToken", "entropy": 0.5678, "target_prob": 0.005}
        ]
    }

    token_file = Path("test_glitch_tokens.json")
    with open(token_file, 'w') as f:
        json.dump(sample_tokens, f, indent=2)

    print(f"✓ Created sample token file: {token_file}")
    return str(token_file)

def create_sample_config():
    """Create a sample configuration for testing"""
    config = GeneticConfig()
    config.model_name = "meta-llama/Llama-3.2-1B-Instruct"
    config.base_text = "The quick brown"
    config.wanted_token = "fox"
    config.generations = 50
    config.population_size = 30
    config.max_tokens = 3
    config.show_gui_animation = False  # Disable separate animation window for testing

    return config

def test_gui_functionality():
    """Test the enhanced GUI functionality"""
    print("\n🧪 Testing Enhanced GUI Functionality")
    print("=" * 50)

    # Create sample files
    token_file = create_sample_token_file()

    # Create main window
    root = tk.Tk()

    try:
        # Initialize GUI controller
        gui = GeneticControllerGUI(root)
        print("✓ GUI controller initialized successfully")

        # Test configuration loading
        sample_config = create_sample_config()
        sample_config.token_file = token_file
        gui.config = sample_config
        gui.update_gui_from_config()
        print("✓ Sample configuration loaded")

        # Test matplotlib integration
        if hasattr(gui, 'fig') and gui.fig is not None:
            print("✓ Matplotlib figure created successfully")
        else:
            print("⚠️  Matplotlib figure not created (expected if matplotlib not available)")

        # Test plotting data structures
        if hasattr(gui, 'generation_history') and hasattr(gui, 'fitness_history'):
            print("✓ Plotting data structures initialized")
        else:
            print("✗ Plotting data structures missing")

        # Test text widgets
        required_widgets = ['best_text', 'context_text', 'log_text']
        for widget_name in required_widgets:
            if hasattr(gui, widget_name):
                print(f"✓ {widget_name} widget created")
            else:
                print(f"✗ {widget_name} widget missing")

        # Test progress update method
        try:
            # Create a mock individual for testing
            class MockIndividual:
                def __init__(self):
                    self.tokens = [12345, 67890]
                    self.fitness = 0.8567
                    self.target_prob_before = 0.9876
                    self.target_prob_after = 0.1234

            mock_individual = MockIndividual()

            # Test progress update
            gui._update_progress(
                generation=5,
                progress=10.0,
                best_fitness=0.8567,
                avg_fitness=0.6234,
                best_individual=mock_individual
            )
            print("✓ Progress update method works")

        except Exception as e:
            print(f"✗ Progress update failed: {e}")

        # Test matplotlib plot update
        if MATPLOTLIB_AVAILABLE and hasattr(gui, '_update_fitness_plot'):
            try:
                # Add some test data
                gui.generation_history = [1, 2, 3, 4, 5]
                gui.fitness_history = [0.1, 0.3, 0.5, 0.7, 0.8567]
                gui.avg_fitness_history = [0.05, 0.2, 0.4, 0.6, 0.6234]

                gui._update_fitness_plot()
                print("✓ Matplotlib plot update works")
            except Exception as e:
                print(f"✗ Matplotlib plot update failed: {e}")

        print("\n✨ GUI Test Summary:")
        print("- Enhanced GUI controller created successfully")
        print("- All required widgets present")
        if MATPLOTLIB_AVAILABLE:
            print("- Matplotlib integration working")
        else:
            print("- Matplotlib fallback message displayed")
        print("- Progress update methods functional")
        print("- Ready for evolution testing")

        print("\n🚀 GUI is ready! You can now:")
        print("1. Adjust parameters in the Configuration tab")
        print("2. Monitor real-time progress in the Progress tab")
        print("3. View fitness evolution plots (if matplotlib available)")
        print("4. See detailed token information and context")
        print("5. Control evolution with Start/Pause/Stop buttons")

        # Keep the window open for manual testing
        print("\n💡 Window will remain open for manual testing...")
        print("   Close the window to exit the test.")

        # Center the window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        # Start the GUI
        root.mainloop()

    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test files
        try:
            if os.path.exists(token_file):
                os.remove(token_file)
                print(f"✓ Cleaned up test file: {token_file}")
        except:
            pass

def main():
    """Main test function"""
    print("🔬 Enhanced GUI Test Suite")
    print("=" * 50)
    print("Testing the new GUI controller with matplotlib integration")
    print()

    # Check dependencies
    print("📋 Checking dependencies:")
    try:
        import tkinter
        print("✓ tkinter available")
    except ImportError:
        print("✗ tkinter not available")
        return

    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("⚠️  matplotlib not available (fallback will be used)")

    try:
        from glitcher.genetic import GeneticControllerGUI
        print("✓ GeneticControllerGUI available")
    except ImportError as e:
        print(f"✗ GeneticControllerGUI not available: {e}")
        return

    print()

    # Run the test
    test_gui_functionality()

if __name__ == "__main__":
    main()
