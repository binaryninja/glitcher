# Glitcher GUI Interface

A comprehensive graphical user interface for controlling and monitoring genetic algorithm evolution experiments with glitch tokens.

## Overview

The Glitcher GUI provides an intuitive interface for:
- **Interactive Parameter Configuration**: Adjust all evolution parameters via a tabbed interface
- **Real-time Process Control**: Start, pause, resume, and stop evolution with live monitoring
- **Progress Visualization**: Watch evolution progress with real-time metrics and logs
- **Configuration Management**: Save and load experimental configurations
- **Results Analysis**: View detailed results and export findings
- **Animation Integration**: Optional real-time visualization of genetic algorithm evolution

## Quick Start

### Installation

```bash
# Install core dependencies
pip install torch transformers matplotlib

# Install glitcher package
pip install -e .

# Verify installation
python test_gui_integration.py
```

### Launch Options

```bash
# Option 1: Via CLI command
glitcher gui

# Option 2: With pre-loaded configuration
glitcher gui --config my_config.json

# Option 3: Direct module launch
python -m glitcher.gui_launcher

# Option 4: Interactive demo
python demo_gui.py
```

## Interface Guide

### Tab 1: Configuration

**Model Configuration**
- **Model Name**: HuggingFace model identifier (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
- **Token File**: Path to JSON file containing glitch tokens
- **Browse Button**: File picker for easy token file selection

**Text Configuration**
- **Base Text**: Text to test probability reduction on (e.g., "The quick brown")
- **Target Token**: Specific token to reduce probability for (auto-detected if empty)
- **Wanted Token**: Token to increase probability for (optional)

**Evolution Parameters**
- **Population Size**: Number of individuals per generation (10-200)
- **Generations**: Maximum evolution generations (10-1000)
- **Max Tokens**: Maximum tokens per combination (1-10)
- **Mutation Rate**: Probability of mutation (0.0-1.0)
- **Crossover Rate**: Probability of crossover (0.0-1.0)
- **Elite Size**: Number of top individuals preserved (1-20)
- **Early Stopping**: Stop when reduction threshold reached (0.0-1.0)

**Validation Parameters**
- **ASR Threshold**: Attack Success Rate threshold (0.0-1.0)
- **Num Attempts**: Validation attempts per token (1-10)
- **Validation Tokens**: Tokens used in validation (10-1000)

**Mode Flags**
- **ASCII Only**: Filter to ASCII-only tokens for cleaner results
- **Enhanced Validation**: Use advanced validation methods
- **Comprehensive Search**: Thorough vocabulary exploration for wanted tokens
- **Include Normal Tokens**: Include standard vocabulary tokens
- **Baseline Seeding**: Use token impact analysis for population initialization
- **Sequence Diversity**: Enable sequence-aware diversity injection
- **Exact Token Count**: Use exact token count vs. variable count
- **GUI Animation**: Show real-time evolution visualization

**Advanced Parameters**
- **Baseline Seeding Ratio**: Fraction of population seeded with baseline guidance (0.0-1.0)
- **Sequence Diversity Ratio**: Fraction using sequence-aware strategies (0.0-1.0)

### Tab 2: Control

**Status Display**
- Shows current evolution state: Ready, Running, Paused, Complete, Error

**Control Buttons**
- **Start Evolution**: Begin genetic algorithm with current configuration
- **Pause/Resume**: Pause evolution (can be resumed)
- **Stop**: Halt evolution permanently (confirms before stopping)

**Progress Tracking**
- **Progress Bar**: Visual progress through generations
- **Generation Counter**: Current generation / total generations
- **Progress Percentage**: Completion percentage

**Current Best Display**
- **Token IDs**: Best individual's token combination
- **Decoded Tokens**: Human-readable token text
- **Fitness Score**: Current best fitness (0.0-1.0)
- **Probability Reduction**: Percentage reduction achieved

### Tab 3: Progress

**Real-time Metrics**
- **Generation**: Current generation number
- **Best Fitness**: Highest fitness score achieved
- **Average Fitness**: Population average fitness
- **Reduction**: Probability reduction percentage

**Evolution Log**
- Timestamped activity log
- Error messages and warnings
- Process status updates
- Performance metrics and milestones

### Tab 4: Results

**Final Results Display**
- Complete evolution summary
- Best individual analysis
- Token combination details
- Performance statistics
- Configuration used

**Results Management**
- **Save Results**: Export results to file
- **Clear Results**: Reset results display

## Configuration Management

### Saving Configurations

1. Configure parameters in the Configuration tab
2. Click "Save Configuration"
3. Choose filename and location
4. Configuration saved as JSON file

### Loading Configurations

1. Click "Load Configuration" 
2. Select previously saved JSON file
3. Parameters automatically populate GUI
4. Review and adjust as needed

### Configuration File Format

```json
{
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "base_text": "The quick brown",
  "target_token": "",
  "wanted_token": "fox",
  "token_file": "glitch_tokens.json",
  
  "population_size": 50,
  "generations": 100,
  "max_tokens": 3,
  "mutation_rate": 0.1,
  "crossover_rate": 0.7,
  "elite_size": 5,
  "early_stopping_threshold": 0.999,
  
  "asr_threshold": 0.5,
  "num_attempts": 3,
  "validation_tokens": 50,
  
  "ascii_only": true,
  "enhanced_validation": true,
  "comprehensive_search": false,
  "include_normal_tokens": false,
  "baseline_seeding": true,
  "sequence_diversity": true,
  "exact_token_count": true,
  "enable_shuffle_mutation": false,
  
  "baseline_seeding_ratio": 0.7,
  "sequence_diversity_ratio": 0.3,
  
  "output_file": "genetic_results.json",
  "baseline_output": "token_impact_baseline.json",
  "show_gui_animation": true
}
```

## Workflow Examples

### Basic Experiment

1. **Setup**: Launch GUI with `glitcher gui`
2. **Configure**: Set model name, base text, token file path
3. **Adjust**: Set population size (30), generations (50) for quick test
4. **Enable**: Turn on ASCII filtering for cleaner results
5. **Start**: Click "Start Evolution" in Control tab
6. **Monitor**: Watch progress in Progress tab
7. **Analyze**: Review results in Results tab when complete

### Advanced Research

1. **Load Config**: Start with saved research configuration
2. **Enable Features**: Turn on comprehensive search, baseline seeding
3. **Set Parameters**: Higher population (100), more generations (200)
4. **Target Analysis**: Specify wanted token for directed search
5. **Monitor**: Use GUI animation for real-time visualization
6. **Export**: Save configuration and results for reproducibility

### Parameter Tuning

1. **Baseline Run**: Start with default parameters
2. **Analyze**: Review results and evolution log
3. **Adjust**: Modify mutation rate, crossover rate based on performance
4. **Test**: Run shorter experiments (30 generations) to test changes
5. **Optimize**: Gradually increase population and generations
6. **Compare**: Save configurations for different parameter sets

## Best Practices

### Configuration Management
- **Descriptive Names**: Use clear, descriptive configuration filenames
- **Version Control**: Keep numbered versions of configurations (v1, v2, etc.)
- **Documentation**: Add notes about configuration purpose and results
- **Backup**: Maintain copies of successful configurations

### Performance Optimization
- **Start Small**: Begin with smaller populations (20-30) and fewer generations (30-50)
- **Scale Gradually**: Increase parameters based on initial results
- **ASCII Filtering**: Enable for faster execution and cleaner results
- **Monitor Resources**: Watch system memory and GPU usage during evolution

### Process Control
- **Save Frequently**: Export results at key milestones
- **Pause Strategically**: Use pause to check intermediate results
- **Plan Stopping**: Know when to stop based on convergence or time limits
- **Log Review**: Check evolution log for warnings or performance issues

### Experimental Design
- **Control Variables**: Change one parameter at a time
- **Replicate**: Run multiple trials with same configuration
- **Compare**: Use consistent metrics across experiments
- **Document**: Record observations and insights

## Troubleshooting

### Common Issues

**GUI Won't Start**
- Verify tkinter installation (usually included with Python)
- Check that glitcher package is properly installed: `pip install -e .`
- Ensure all dependencies are available: `python test_gui_integration.py`

**Evolution Fails to Start**
- Verify model name is correct and accessible
- Check token file exists and is properly formatted
- Ensure sufficient GPU/CPU memory for model
- Review evolution log for specific error messages

**Slow Performance**
- Reduce population size and generation count
- Enable ASCII filtering to reduce token set size
- Check system resources (RAM, GPU memory)
- Consider using smaller model for testing

**Animation Issues**
- Ensure matplotlib is installed: `pip install matplotlib`
- Try disabling GUI animation if causing problems
- Check that display/X11 forwarding works (if using SSH)

**Configuration Problems**
- Verify JSON syntax in configuration files
- Check file paths are correct and accessible
- Ensure parameter values are within valid ranges
- Use "Reset to Defaults" if configuration is corrupted

### Getting Help

**Diagnostic Steps**
1. Run integration test: `python test_gui_integration.py`
2. Check dependencies: All should show ✅
3. Review error messages in evolution log
4. Test with demo configuration: `python demo_gui.py`

**Debug Mode**
- Use smaller test parameters first
- Enable detailed logging in evolution log
- Test individual components (model loading, token loading)
- Compare with command-line equivalent

## Advanced Features

### GUI Animation Integration

When enabled, shows real-time visualization of:
- Fitness evolution over generations
- Token combination development
- Population statistics
- String construction analysis

### Comprehensive Search

For wanted token optimization:
- Tests entire model vocabulary systematically
- Uses intelligent batching for performance
- Provides early stopping when sufficient tokens found
- Caches results for subsequent runs

### Baseline-Guided Seeding

Intelligent population initialization:
- Analyzes individual token impacts
- Seeds population with high-impact combinations
- Balances exploitation with exploration
- Significantly improves convergence speed

## Integration with Command Line

The GUI complements command-line usage:

```bash
# Mine tokens first
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50

# Use GUI for evolution experiments
glitcher gui --config experiment_config.json

# Compare with command-line genetic algorithm
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --generations 50
```

## System Requirements

**Minimum**
- Python 3.8+
- 4GB RAM
- CPU-only execution supported

**Recommended**
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM
- Fast storage (SSD)

**Dependencies**
- torch, transformers (core functionality)
- tkinter (GUI, usually included)
- matplotlib (animation, optional)
- accelerate, bitsandbytes (model optimization, optional)

## Future Enhancements

Planned features:
- **Multi-experiment Management**: Run and compare multiple experiments
- **Advanced Visualization**: Enhanced plots and analysis charts
- **Export Options**: PDF reports, CSV data export
- **Remote Monitoring**: Web-based interface for remote experiments
- **Experiment Templates**: Pre-configured setups for common use cases