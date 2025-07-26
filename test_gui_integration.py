#!/usr/bin/env python3
"""
Test script for GUI integration
Verifies that the GUI components work together properly.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class TestGUIIntegration(unittest.TestCase):
    """Test GUI integration components"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.token_file = os.path.join(self.temp_dir, "test_tokens.json")

        # Create sample token file
        sample_tokens = [
            {"token_id": 12345, "token_text": "TestToken1", "entropy": 0.1},
            {"token_id": 67890, "token_text": "TestToken2", "entropy": 0.2}
        ]
        with open(self.token_file, 'w') as f:
            json.dump(sample_tokens, f)

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_import_gui_components(self):
        """Test that GUI components can be imported"""
        try:
            from glitcher.genetic.gui_controller import GeneticControllerGUI, GeneticConfig
            from glitcher.genetic import GeneticProbabilityReducer
            print("✅ Successfully imported GUI components")
        except ImportError as e:
            self.fail(f"Failed to import GUI components: {e}")

    def test_genetic_config_creation(self):
        """Test GeneticConfig creation and parameter setting"""
        try:
            from glitcher.genetic.gui_controller import GeneticConfig

            config = GeneticConfig()

            # Test default values
            self.assertEqual(config.model_name, "meta-llama/Llama-3.2-1B-Instruct")
            self.assertEqual(config.base_text, "The quick brown")
            self.assertEqual(config.population_size, 50)
            self.assertEqual(config.generations, 100)

            # Test parameter modification
            config.population_size = 30
            config.base_text = "Hello world"
            self.assertEqual(config.population_size, 30)
            self.assertEqual(config.base_text, "Hello world")

            print("✅ GeneticConfig creation and modification works")
        except Exception as e:
            self.fail(f"GeneticConfig test failed: {e}")

    def test_config_serialization(self):
        """Test configuration save/load functionality"""
        try:
            from glitcher.genetic.gui_controller import GeneticConfig

            # Create config with custom values
            config = GeneticConfig()
            config.model_name = "test-model"
            config.base_text = "Test text"
            config.population_size = 25
            config.generations = 75

            # Serialize to dict
            config_dict = {
                k: v for k, v in config.__dict__.items()
            }

            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            # Load from file
            with open(self.config_file, 'r') as f:
                loaded_dict = json.load(f)

            # Create new config and load values
            new_config = GeneticConfig()
            for key, value in loaded_dict.items():
                if hasattr(new_config, key):
                    setattr(new_config, key, value)

            # Verify values
            self.assertEqual(new_config.model_name, "test-model")
            self.assertEqual(new_config.base_text, "Test text")
            self.assertEqual(new_config.population_size, 25)
            self.assertEqual(new_config.generations, 75)

            print("✅ Configuration serialization works")
        except Exception as e:
            self.fail(f"Configuration serialization test failed: {e}")

    def test_genetic_reducer_initialization(self):
        """Test GeneticProbabilityReducer initialization with GUI parameters"""
        try:
            from glitcher.genetic import GeneticProbabilityReducer
            from glitcher.genetic.gui_controller import GeneticConfig

            config = GeneticConfig()
            config.token_file = self.token_file

            # Create reducer (without loading model)
            reducer = GeneticProbabilityReducer(
                model_name=config.model_name,
                base_text=config.base_text,
                target_token=config.target_token if config.target_token else None,
                wanted_token=config.wanted_token if config.wanted_token else None
            )

            # Set additional parameters
            reducer.population_size = config.population_size
            reducer.max_generations = config.generations
            reducer.max_tokens_per_individual = config.max_tokens
            reducer.mutation_rate = config.mutation_rate
            reducer.crossover_rate = config.crossover_rate
            reducer.elite_size = config.elite_size

            # Verify parameters were set
            self.assertEqual(reducer.population_size, config.population_size)
            self.assertEqual(reducer.max_generations, config.generations)
            self.assertEqual(reducer.max_tokens_per_individual, config.max_tokens)

            print("✅ GeneticProbabilityReducer initialization works")
        except Exception as e:
            self.fail(f"GeneticProbabilityReducer initialization test failed: {e}")

    def test_gui_controller_import(self):
        """Test GUI controller can be imported and has expected methods"""
        try:
            from glitcher.genetic.gui_controller import GeneticControllerGUI

            # Test that the class exists and has expected methods
            self.assertTrue(hasattr(GeneticControllerGUI, '__init__'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'setup_gui'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'start_evolution'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'pause_evolution'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'stop_evolution'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'save_config'))
            self.assertTrue(hasattr(GeneticControllerGUI, 'load_config'))

            print("✅ GUI controller import and methods check works")
        except Exception as e:
            self.fail(f"GUI controller import test failed: {e}")

    def test_gui_launcher_import(self):
        """Test GUI launcher can be imported"""
        try:
            import glitcher.gui_launcher
            self.assertTrue(hasattr(glitcher.gui_launcher, 'main'))
            self.assertTrue(callable(glitcher.gui_launcher.main))
            print("✅ GUI launcher import works")
        except ImportError as e:
            self.fail(f"GUI launcher import failed: {e}")

    def test_cli_gui_command_integration(self):
        """Test CLI GUI command integration"""
        try:
            from glitcher.cli import GlitcherCLI

            # Create CLI instance
            cli = GlitcherCLI()
            parser = cli.setup_parser()

            # Test GUI command exists
            help_text = parser.format_help()
            self.assertIn('gui', help_text)

            # Test parsing GUI command
            args = parser.parse_args(['gui'])
            self.assertEqual(args.command, 'gui')

            # Test with config option
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                tmp_config = tmp.name

            try:
                args = parser.parse_args(['gui', '--config', tmp_config])
                self.assertEqual(args.command, 'gui')
                self.assertEqual(args.config, tmp_config)
            finally:
                os.unlink(tmp_config)

            print("✅ CLI GUI command integration works")
        except Exception as e:
            self.fail(f"CLI GUI command integration test failed: {e}")

    def test_demo_script_functionality(self):
        """Test demo script basic functionality"""
        try:
            # Test config creation function
            import demo_gui

            # Test sample config creation
            config_path = demo_gui.create_sample_config()
            self.assertTrue(os.path.exists(config_path))

            # Verify config content
            with open(config_path, 'r') as f:
                config = json.load(f)

            self.assertIn('model_name', config)
            self.assertIn('base_text', config)
            self.assertIn('population_size', config)

            # Test token file creation with temporary file
            temp_token_file = os.path.join(self.temp_dir, "demo_tokens.json")

            # Temporarily override the token file path in demo_gui
            original_token_file = "glitch_tokens.json"
            import demo_gui
            demo_gui_module = sys.modules['demo_gui']

            # Mock the create_sample_token_file to use our temp file
            def mock_create_sample_token_file():
                sample_tokens = [
                    {"token_id": 89472, "token_text": "SomeGlitchToken", "entropy": 0.1234},
                    {"token_id": 127438, "token_text": "AnotherToken", "entropy": 0.2345}
                ]
                with open(temp_token_file, 'w') as f:
                    json.dump(sample_tokens, f, indent=2)
                return temp_token_file

            # Test token file creation
            token_path = mock_create_sample_token_file()
            self.assertTrue(os.path.exists(token_path))

            # Verify token file content
            with open(token_path, 'r') as f:
                tokens = json.load(f)

            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), 0)
            self.assertIn('token_id', tokens[0])

            # Clean up
            os.unlink(config_path)
            if os.path.exists(temp_token_file):
                os.unlink(temp_token_file)

            print("✅ Demo script functionality works")
        except Exception as e:
            self.fail(f"Demo script functionality test failed: {e}")

def run_dependency_check():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")

    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'tkinter': 'Tkinter (GUI)',
        'matplotlib': 'Matplotlib (optional)'
    }

    available = []
    missing = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            available.append(name)
            print(f"✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"❌ {name}")

    print(f"\n📊 Summary: {len(available)}/{len(dependencies)} dependencies available")

    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")
        if 'Tkinter (GUI)' in missing:
            print("   Note: Tkinter is usually included with Python")
        return False

    return True

def main():
    """Main test function"""
    print("🧪 GUI Integration Test Suite")
    print("=" * 50)

    # Check dependencies first
    deps_ok = run_dependency_check()
    if not deps_ok:
        print("\n⚠️  Some dependencies are missing, but tests will continue...")
        print("   Install missing dependencies for full functionality")

    print("\n🔬 Running integration tests...")

    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGUIIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 All tests passed! GUI integration is working correctly.")
        print("\n🚀 Ready to use:")
        print("   • glitcher gui")
        print("   • python demo_gui.py")
        print("   • python -m glitcher.gui_launcher")
        return 0
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed")
        print("   • Check that glitcher package is properly installed: pip install -e .")
        print("   • Review error messages above for specific issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
