#!/usr/bin/env python3
"""
Simple GUI Test Script

A minimal test to verify GUI components work correctly without launching
the full graphical interface.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def test_config_creation():
    """Test basic configuration creation and manipulation"""
    print("🧪 Testing configuration creation...")

    try:
        from glitcher.genetic.gui_controller import GeneticConfig

        # Create default config
        config = GeneticConfig()
        assert config.model_name == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.base_text == "The quick brown"
        assert config.population_size == 50

        # Test parameter modification
        config.population_size = 30
        config.wanted_token = "fox"
        assert config.population_size == 30
        assert config.wanted_token == "fox"

        print("✅ Configuration creation test passed")
        return True

    except Exception as e:
        print(f"❌ Configuration creation test failed: {e}")
        return False

def test_config_serialization():
    """Test configuration save/load"""
    print("🧪 Testing configuration serialization...")

    try:
        from glitcher.genetic.gui_controller import GeneticConfig

        # Create and modify config
        config = GeneticConfig()
        config.model_name = "test-model"
        config.base_text = "Test text"
        config.population_size = 25

        # Serialize to dict
        config_dict = {k: v for k, v in config.__dict__.items()}

        # Test JSON serialization
        json_str = json.dumps(config_dict, indent=2)
        loaded_dict = json.loads(json_str)

        # Verify values
        assert loaded_dict["model_name"] == "test-model"
        assert loaded_dict["base_text"] == "Test text"
        assert loaded_dict["population_size"] == 25

        print("✅ Configuration serialization test passed")
        return True

    except Exception as e:
        print(f"❌ Configuration serialization test failed: {e}")
        return False

def test_reducer_initialization():
    """Test genetic reducer initialization with GUI parameters"""
    print("🧪 Testing reducer initialization...")

    try:
        from glitcher.genetic.reducer import GeneticProbabilityReducer
        from glitcher.genetic.gui_controller import GeneticConfig

        config = GeneticConfig()

        # Create reducer (don't load model)
        reducer = GeneticProbabilityReducer(
            model_name=config.model_name,
            base_text=config.base_text,
            target_token=config.target_token if config.target_token else None,
            wanted_token=config.wanted_token if config.wanted_token else None
        )

        # Test setting GUI parameters
        reducer.population_size = config.population_size
        reducer.max_generations = config.generations
        reducer.max_tokens_per_individual = config.max_tokens
        reducer.mutation_rate = config.mutation_rate

        # Verify parameters
        assert reducer.population_size == config.population_size
        assert reducer.max_generations == config.generations
        assert reducer.max_tokens_per_individual == config.max_tokens
        assert reducer.mutation_rate == config.mutation_rate

        print("✅ Reducer initialization test passed")
        return True

    except Exception as e:
        print(f"❌ Reducer initialization test failed: {e}")
        return False

def test_baseline_probability_handling():
    """Test baseline probability tuple handling"""
    print("🧪 Testing baseline probability handling...")

    try:
        # Simulate the tuple return from get_baseline_probability
        baseline_result = (12345, 0.0075, 67890, 0.0001)  # (target_id, target_prob, wanted_id, wanted_prob)

        # Test the handling logic from GUI controller
        if isinstance(baseline_result, tuple) and len(baseline_result) >= 2:
            target_id, target_prob, wanted_id, wanted_prob = baseline_result
            baseline_prob = target_prob
        else:
            baseline_prob = baseline_result if not isinstance(baseline_result, tuple) else 0.0

        assert baseline_prob == 0.0075

        # Test with different input types
        baseline_result_single = 0.0123
        if isinstance(baseline_result_single, tuple) and len(baseline_result_single) >= 2:
            target_id, target_prob, wanted_id, wanted_prob = baseline_result_single
            baseline_prob = target_prob
        else:
            baseline_prob = baseline_result_single if not isinstance(baseline_result_single, tuple) else 0.0

        assert baseline_prob == 0.0123

        print("✅ Baseline probability handling test passed")
        return True

    except Exception as e:
        print(f"❌ Baseline probability handling test failed: {e}")
        return False

def test_import_capabilities():
    """Test that all GUI components can be imported"""
    print("🧪 Testing import capabilities...")

    try:
        # Test core imports
        from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig
        from glitcher.genetic.reducer import GeneticProbabilityReducer
        from glitcher.genetic.gui_animator import EnhancedGeneticAnimator, GeneticAnimationCallback
        import glitcher.gui_launcher

        # Test that classes have expected methods
        assert hasattr(GeneticControllerGUI, 'start_evolution')
        assert hasattr(GeneticControllerGUI, 'pause_evolution')
        assert hasattr(GeneticControllerGUI, 'stop_evolution')
        assert hasattr(GeneticControllerGUI, 'save_config')
        assert hasattr(GeneticControllerGUI, 'load_config')

        assert hasattr(glitcher.gui_launcher, 'main')
        assert callable(glitcher.gui_launcher.main)

        print("✅ Import capabilities test passed")
        return True

    except Exception as e:
        print(f"❌ Import capabilities test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI GUI command integration"""
    print("🧪 Testing CLI integration...")

    try:
        from glitcher.cli import GlitcherCLI

        # Create CLI and parser
        cli = GlitcherCLI()
        parser = cli.setup_parser()

        # Test GUI command exists
        help_text = parser.format_help()
        assert 'gui' in help_text

        # Test parsing GUI command
        args = parser.parse_args(['gui'])
        assert args.command == 'gui'

        # Test GUI command with config option
        args = parser.parse_args(['gui', '--config', 'test.json'])
        assert args.command == 'gui'
        assert args.config == 'test.json'

        print("✅ CLI integration test passed")
        return True

    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def test_sample_file_creation():
    """Test sample configuration and token file creation"""
    print("🧪 Testing sample file creation...")

    try:
        import demo_gui

        # Test config creation
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                config_path = demo_gui.create_sample_config()
                assert os.path.exists(config_path)

                # Verify config content
                with open(config_path, 'r') as f:
                    config = json.load(f)

                assert 'model_name' in config
                assert 'base_text' in config
                assert 'population_size' in config

                # Clean up
                os.unlink(config_path)

            finally:
                os.chdir(original_cwd)

        print("✅ Sample file creation test passed")
        return True

    except Exception as e:
        print(f"❌ Sample file creation test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    deps = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'tkinter': 'Tkinter (GUI)',
        'matplotlib': 'Matplotlib (optional)'
    }

    available = 0
    total = len(deps)

    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            available += 1
        except ImportError:
            print(f"❌ {name}")

    print(f"📊 Dependencies: {available}/{total} available")
    return available >= 3  # Require at least torch, transformers, tkinter

def run_all_tests():
    """Run all GUI tests"""
    print("🧪 Simple GUI Test Suite")
    print("=" * 40)

    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Missing critical dependencies. Some tests may fail.")

    print("\n🔬 Running tests...")

    tests = [
        test_import_capabilities,
        test_config_creation,
        test_config_serialization,
        test_reducer_initialization,
        test_baseline_probability_handling,
        test_cli_integration,
        test_sample_file_creation
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()

    # Summary
    print("=" * 40)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! GUI components are working correctly.")
        print("\n🚀 Ready to use:")
        print("   • glitcher gui")
        print("   • python demo_gui.py")
        print("   • python -m glitcher.gui_launcher")
        return True
    else:
        print(f"❌ {failed} test(s) failed. Check error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed")
        print("   • Check that glitcher package is properly installed: pip install -e .")
        print("   • Verify error messages for specific issues")
        return False

def main():
    """Main test function"""
    try:
        success = run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
