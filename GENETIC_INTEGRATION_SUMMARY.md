# Genetic Algorithm Integration Summary

## Overview

The genetic algorithm functionality has been successfully integrated into the main Glitcher CLI tool. Previously, genetic algorithm features were only available as standalone scripts in the `examples/` directory. Now they are fully integrated as a core command in the main CLI interface.

## Changes Made

### 1. Module Structure Reorganization

**Before:**
```
glitcher/
├── examples/
│   ├── genetic_probability_reducer.py
│   └── genetic_batch_runner.py
└── glitcher/
    └── cli.py (no genetic functionality)
```

**After:**
```
glitcher/
├── examples/
│   ├── genetic_probability_reducer.py (legacy)
│   └── genetic_batch_runner.py (legacy)
└── glitcher/
    ├── genetic/
    │   ├── __init__.py
    │   ├── reducer.py
    │   └── batch_runner.py
    └── cli.py (with genetic command)
```

### 2. New CLI Command

Added `genetic` command to the main CLI with full argument support:

```bash
glitcher genetic model_path [options]
```

### 3. Integration Benefits

- **Unified Interface**: All Glitcher functionality now accessible through single CLI
- **Consistent UX**: Same argument patterns as other commands (--device, --quant-type, etc.)
- **Better Maintainability**: Centralized codebase instead of scattered examples
- **Improved Documentation**: Integrated help system with `--help`

## New Command Usage

### Basic Genetic Algorithm

```bash
# Simple genetic algorithm run
glitcher genetic meta-llama/Llama-3.2-1B-Instruct --base-text "The quick brown" --generations 50 --population-size 30
```

### Advanced Configuration

```bash
# Custom parameters
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --base-text "Hello world" \
  --generations 100 \
  --population-size 50 \
  --mutation-rate 0.15 \
  --crossover-rate 0.8 \
  --max-tokens 5 \
  --elite-size 10
```

### Batch Experiments

```bash
# Run batch experiments across multiple scenarios
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --batch \
  --token-file glitch_tokens.json \
  --generations 30 \
  --population-size 25
```

### Target-Specific Breeding

```bash
# Target specific token
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --base-text "The quick brown" \
  --target-token "fox" \
  --generations 50
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | str | Required | HuggingFace model identifier |
| `--base-text` | str | "The quick brown" | Base text for probability reduction testing |
| `--target-token` | str | None | Specific token to target (auto-detected if not provided) |
| `--token-file` | str | "glitch_tokens.json" | JSON file containing glitch tokens |
| `--population-size` | int | 50 | Population size for genetic algorithm |
| `--generations` | int | 100 | Maximum number of generations |
| `--mutation-rate` | float | 0.1 | Mutation rate (0.0-1.0) |
| `--crossover-rate` | float | 0.7 | Crossover rate (0.0-1.0) |
| `--elite-size` | int | 5 | Elite size for genetic algorithm |
| `--max-tokens` | int | 3 | Maximum tokens per individual combination |
| `--output` | str | "genetic_results.json" | Output file for results |
| `--device` | str | "cuda" | Device to use |
| `--quant-type` | str | "bfloat16" | Quantization type |
| `--batch` | flag | False | Run batch experiments across multiple scenarios |

## Output Format

### Single Experiment Results

The genetic algorithm produces JSON output with evolved token combinations:

```json
{
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "base_text": "The quick brown",
  "target_token_id": 4419,
  "target_token_text": "fox",
  "baseline_probability": 0.123456,
  "ga_parameters": {
    "population_size": 50,
    "max_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
    "max_tokens_per_individual": 3
  },
  "results": [
    {
      "tokens": [12345, 67890],
      "token_texts": ["Token1", "Token2"],
      "fitness": 0.856432,
      "baseline_prob": 0.123456,
      "modified_prob": 0.017789,
      "reduction_percentage": 85.64
    }
  ]
}
```

### Console Output

```
🧬 Starting genetic algorithm for breeding glitch token combinations...
Model: meta-llama/Llama-3.2-1B-Instruct
Base text: 'The quick brown'
Population size: 50
Generations: 100

🏆 Top Results:
1. Tokens: [12345, 67890] (['Token1', 'Token2'])
   Fitness: 0.856432
   Probability reduction: 85.64%

2. Tokens: [11111, 22222, 33333] (['TokenA', 'TokenB', 'TokenC'])
   Fitness: 0.823451
   Probability reduction: 82.35%
```

## Technical Implementation

### Core Classes

1. **`GeneticProbabilityReducer`**: Main genetic algorithm engine
   - Evolves token combinations to maximize probability reduction
   - Supports 1-N token combinations (configurable via `--max-tokens`)
   - Uses selection, crossover, and mutation operators

2. **`GeneticBatchRunner`**: Batch experiment runner
   - Runs multiple genetic algorithm scenarios
   - Analyzes patterns across experiments
   - Generates comprehensive reports

3. **`Individual`**: Population member representation
   - Contains token combination and fitness score
   - Tracks probability reduction metrics

### Key Features

- **Configurable Population Size**: Adjust exploration vs. speed tradeoffs
- **Multi-Token Combinations**: Evolve combinations of 1-N tokens
- **Elitism**: Preserve best individuals across generations
- **Adaptive Mutation**: Configurable mutation and crossover rates
- **Target Flexibility**: Auto-detect or manually specify target tokens

## Integration Testing

Comprehensive integration tests were implemented and all passed:

- ✅ CLI command availability
- ✅ Help system functionality
- ✅ Module imports
- ✅ Class instantiation
- ✅ Token loading
- ✅ Argument parsing

## Backward Compatibility

The original standalone scripts remain in `examples/` for backward compatibility, but are now marked as legacy:

```bash
# Legacy usage (still works)
python examples/genetic_probability_reducer.py model_name --base-text "text"

# New integrated usage (recommended)
glitcher genetic model_name --base-text "text"
```

## Documentation Updates

- Updated `CLAUDE.md` with new genetic algorithm commands
- Added comprehensive parameter documentation
- Provided usage examples and recommended settings
- Marked legacy commands as integrated

## Recommended Usage Patterns

### Quick Exploration
```bash
glitcher genetic model_name --generations 30 --population-size 20
```

### Thorough Search
```bash
glitcher genetic model_name --generations 100 --population-size 50
```

### Large Combinations
```bash
glitcher genetic model_name --max-tokens 5 --generations 75
```

### Batch Analysis
```bash
glitcher genetic model_name --batch --generations 50
```

## Benefits of Integration

1. **Unified Workflow**: All Glitcher functionality in one command
2. **Consistent Interface**: Same patterns as other commands
3. **Better Discoverability**: `glitcher --help` shows all capabilities
4. **Improved Maintainability**: Single codebase to maintain
5. **Enhanced Documentation**: Integrated help system
6. **Professional UX**: Consistent argument naming and behavior

## Future Enhancements

The integrated genetic algorithm provides a foundation for future enhancements:

- Multi-objective optimization
- Advanced selection strategies
- Population diversity metrics
- Real-time visualization
- Distributed evolution across multiple models

## Conclusion

The genetic algorithm integration successfully modernizes Glitcher's architecture by moving from scattered example scripts to a unified, professional CLI tool. This enhancement improves usability, maintainability, and sets the stage for future development.