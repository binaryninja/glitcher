# Genetic Algorithm for Glitch Token Breeding

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Understanding Results](#understanding-results)
- [Configuration Options](#configuration-options)
- [Theory and Methodology](#theory-and-methodology)
- [Batch Experiments](#batch-experiments)
- [Visualization](#visualization)
- [Research Findings](#research-findings)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

The Genetic Algorithm system for glitch token breeding is a sophisticated tool that automatically evolves combinations of glitch tokens to maximize probability reduction in language models. Unlike manual testing approaches, this system uses evolutionary computation to systematically discover the most effective token combinations.

### Key Features

- **Evolutionary Optimization**: Uses genetic algorithms to evolve token combinations over generations
- **Dynamic Token Limits**: Configurable combination sizes (1-N tokens per individual)
- **Batch Processing**: Run experiments across multiple scenarios simultaneously
- **Rich Visualization**: Comprehensive plots and analysis tools
- **Pattern Discovery**: Automatic identification of effective token types and combinations
- **Extensible Architecture**: Easy to add new scenarios and analysis methods

### What Makes This Unique

Traditional approaches test individual tokens or small sets manually. Our genetic algorithm:
1. **Explores millions of combinations** efficiently through evolutionary search
2. **Discovers synergistic effects** between tokens that wouldn't be obvious individually
3. **Adapts to different contexts** by evolving scenario-specific solutions
4. **Provides scientific insights** into token interaction patterns

## Quick Start

```bash
# Basic run with default settings
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct

# Test with weak prediction (more vulnerable)
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --base-text "The weather is" \
    --generations 50 \
    --population-size 40

# Run batch experiments across multiple scenarios
python examples/genetic_batch_runner.py meta-llama/Llama-3.2-1B-Instruct \
    --use-predefined \
    --output-dir results

# Create visualizations
python examples/visualize_genetic_results.py results/genetic_batch_results_*.json \
    --output-dir plots
```

## Installation

### Prerequisites

```bash
pip install torch transformers matplotlib seaborn pandas numpy tqdm
```

### Files Required

- `genetic_probability_reducer.py` - Main genetic algorithm implementation
- `genetic_batch_runner.py` - Batch experiment runner
- `visualize_genetic_results.py` - Visualization tools
- `email_llams321.json` - Glitch tokens database

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for faster model inference
- **RAM**: 8GB+ for 1B parameter models, 16GB+ for larger models
- **Storage**: 5GB+ for model weights and result storage

## Basic Usage

### Single Experiment

```bash
python examples/genetic_probability_reducer.py MODEL_NAME [OPTIONS]
```

**Key Parameters:**
- `MODEL_NAME`: HuggingFace model identifier
- `--base-text`: Text to test probability reduction on
- `--target-token`: Specific token to target (auto-detected if omitted)
- `--generations`: Number of evolution cycles (default: 100)
- `--population-size`: Number of individuals per generation (default: 50)
- `--max-tokens`: Maximum tokens per combination (default: 3)

**Examples:**

```bash
# Test strong prediction resistance
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --base-text "The quick brown" \
    --generations 100 \
    --population-size 50 \
    --max-tokens 3

# Test weak prediction vulnerability
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --base-text "I think that" \
    --generations 50 \
    --population-size 30 \
    --max-tokens 2

# Target specific token
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --base-text "2 + 2 =" \
    --target-token "4" \
    --output math_results.json
```

### Token Combination Size Configuration

The `--max-tokens` parameter controls the maximum number of tokens per combination:

```bash
# Single tokens only
python examples/genetic_probability_reducer.py MODEL_NAME --max-tokens 1

# Up to 2 tokens per combination
python examples/genetic_probability_reducer.py MODEL_NAME --max-tokens 2

# Complex combinations (up to 5 tokens)
python examples/genetic_probability_reducer.py MODEL_NAME --max-tokens 5

# Very complex combinations (up to 10 tokens) - may have diminishing returns
python examples/genetic_probability_reducer.py MODEL_NAME --max-tokens 10
```

**Recommendations:**
- **1-2 tokens**: Fast, good for initial exploration
- **3-4 tokens**: Sweet spot for most applications
- **5+ tokens**: Slower, may find complex synergies but diminishing returns

## Advanced Features

### Custom Genetic Algorithm Parameters

```bash
python examples/genetic_probability_reducer.py MODEL_NAME \
    --population-size 100 \
    --generations 200 \
    --mutation-rate 0.15 \
    --crossover-rate 0.8 \
    --elite-size 10 \
    --max-tokens 4
```

**Parameter Tuning Guidelines:**

- **Population Size**: 30-100 individuals
  - Smaller: Faster, may miss optimal solutions
  - Larger: More thorough exploration, slower
  
- **Generations**: 50-200 cycles
  - Fewer: Faster, may not fully converge
  - More: Better optimization, diminishing returns after convergence
  
- **Mutation Rate**: 0.05-0.2
  - Lower: More exploitation of current solutions
  - Higher: More exploration of new combinations
  
- **Crossover Rate**: 0.6-0.9
  - Lower: More independence between generations
  - Higher: More mixing of successful combinations
  
- **Elite Size**: 5-15 individuals
  - Smaller: More diversity, slower convergence
  - Larger: Faster convergence, less diversity

### Custom Token Sets

```bash
# Use specific token file
python examples/genetic_probability_reducer.py MODEL_NAME \
    --token-file custom_tokens.json

# Token file format options:
# 1. Simple list: [12345, 67890, 11111]
# 2. Classification format: {"classifications": [{"token_id": 12345}, ...]}
# 3. Dictionary format: {"tokens": [{"id": 12345}, ...]}
```

## Understanding Results

### Console Output

```
=== Top 10 Probability Reducers ===
Base text: 'The quick brown'
Target token: 'fox' (ID: 39935)
Baseline probability: 0.9487

 1. Tokens: [127896, 113983, 106920] → ["'ıldığında'", "'ışları'", "'rodní'"]
    Fitness: 0.1001 (10.6% reduction)
    Probability: 0.9487 → 0.8486

 2. Tokens: [120454, 124292] → ["'lıkları'", "'илася'"]
    Fitness: 0.0856 (9.0% reduction)
    Probability: 0.9487 → 0.8631
```

### Interpreting Results

**Fitness Score**:
- **> 0.05**: Excellent reducer (>5% probability reduction)
- **0.02-0.05**: Good reducer (2-5% reduction)
- **0.01-0.02**: Moderate reducer (1-2% reduction)
- **< 0.01**: Weak reducer (<1% reduction)
- **< 0**: Probability increaser (counterproductive)

**Reduction Percentage**:
- **> 90%**: Extremely effective (typically weak predictions)
- **50-90%**: Very effective
- **10-50%**: Moderately effective
- **1-10%**: Weakly effective (typically strong predictions)
- **< 1%**: Minimally effective

### JSON Output Format

```json
{
  "model_name": "meta-llama/Llama-3.2-1B-Instruct",
  "base_text": "The quick brown",
  "target_token_id": 39935,
  "target_token_text": "fox",
  "baseline_probability": 0.9487,
  "ga_parameters": {
    "population_size": 50,
    "max_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
    "max_tokens_per_individual": 3
  },
  "results": [
    {
      "tokens": [127896, 113983, 106920],
      "token_texts": ["'ıldığında'", "'ışları'", "'rodní'"],
      "fitness": 0.1001,
      "baseline_prob": 0.9487,
      "modified_prob": 0.8486,
      "reduction_percentage": 10.55
    }
  ]
}
```

## Configuration Options

### Genetic Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `population_size` | 50 | 10-200 | Number of individuals per generation |
| `max_generations` | 100 | 10-500 | Maximum evolution cycles |
| `mutation_rate` | 0.1 | 0.01-0.5 | Probability of mutation |
| `crossover_rate` | 0.7 | 0.1-0.9 | Probability of crossover |
| `elite_size` | 5 | 1-20 | Top individuals preserved |
| `max_tokens_per_individual` | 3 | 1-10 | Maximum tokens per combination |

### Optimization Strategies

**For Strong Predictions** (>80% baseline probability):
```bash
--population-size 100 --generations 200 --mutation-rate 0.15 --max-tokens 4
```

**For Weak Predictions** (<30% baseline probability):
```bash
--population-size 30 --generations 50 --mutation-rate 0.1 --max-tokens 2
```

**For Exploration** (finding diverse solutions):
```bash
--population-size 80 --mutation-rate 0.2 --elite-size 3 --max-tokens 5
```

**For Exploitation** (refining known good solutions):
```bash
--population-size 40 --mutation-rate 0.05 --elite-size 15 --max-tokens 3
```

## Theory and Methodology

### Genetic Algorithm Components

1. **Individual Representation**
   - Each individual = list of 1-N token IDs
   - Fitness = probability reduction achieved
   - Genes = individual token IDs

2. **Selection Method**
   - Tournament selection (tournament size = 3)
   - Probabilistic selection based on fitness
   - Preserves diversity while favoring fitness

3. **Crossover Operations**
   - Combine tokens from two parents
   - Create diverse offspring combinations
   - Respect max_tokens_per_individual limit

4. **Mutation Operations**
   - Replace: Change random token to new token
   - Add: Insert new token (if under limit)
   - Remove: Delete random token (if above 1)

5. **Fitness Evaluation**
   - Calculate baseline probability
   - Insert token combination at text beginning
   - Measure probability reduction
   - Fitness = baseline_prob - modified_prob

### Evolutionary Dynamics

**Generation 0**: Random initialization from glitch token pool

**Generations 1-N**: 
1. Evaluate fitness for all individuals
2. Select parents via tournament selection
3. Create offspring through crossover
4. Apply mutations to offspring
5. Preserve elite individuals
6. Replace population with new generation

**Convergence**: Population stabilizes around optimal solutions

### Token Interaction Theory

**Morphological Disruption**: Tokens containing partial morphemes from agglutinative languages (Turkish, Finnish) disrupt linguistic parsing.

**Cross-linguistic Interference**: Mixing tokens from different language families creates orthographic conflicts.

**Attention Fragmentation**: Unusual token sequences fragment attention patterns, reducing prediction confidence.

## Batch Experiments

### Running Batch Experiments

```bash
python examples/genetic_batch_runner.py MODEL_NAME [OPTIONS]
```

**Options:**
- `--use-predefined`: Use built-in scenario set
- `--scenarios-file`: Custom scenario JSON file
- `--output-dir`: Results directory
- `--max-tokens`: Maximum tokens per combination

### Predefined Scenarios

1. **strong_prediction_fox**: "The quick brown" → "fox" (94.87% baseline)
2. **weak_prediction_weather**: "The weather is" → "getting" (7.89% baseline)
3. **weak_prediction_think**: "I think that" → "'s" (28.96% baseline)
4. **coding_context**: "def function_name(" → "phrase" (7.77% baseline)
5. **conversation_start**: "Hello, how are" → "you" (98.63% baseline)
6. **question_start**: "What is the" → "value" (6.51% baseline)
7. **math_context**: "2 + 2 =" → " " (38.89% baseline)
8. **incomplete_sentence**: "The cat sat on the" → "mat" (53.81% baseline)

### Custom Scenarios

Create `scenarios.json`:
```json
[
  {
    "name": "my_scenario",
    "base_text": "Once upon a time",
    "target_token": "there",
    "ga_params": {
      "population_size": 60,
      "max_generations": 80,
      "max_tokens_per_individual": 4
    }
  },
  {
    "name": "another_scenario",
    "base_text": "The capital of France is",
    "ga_params": {
      "max_tokens_per_individual": 2
    }
  }
]
```

Use with:
```bash
python examples/genetic_batch_runner.py MODEL_NAME --scenarios-file scenarios.json
```

### Batch Results Analysis

Results include:
- **Cross-scenario effectiveness**: Which tokens work across multiple contexts
- **Scenario difficulty ranking**: Which contexts are most/least vulnerable
- **Token frequency analysis**: Most commonly selected tokens
- **Combination pattern discovery**: Effective multi-token patterns

## Visualization

### Generating Visualizations

```bash
python examples/visualize_genetic_results.py RESULTS_FILE [OPTIONS]
```

**Options:**
- `--output-dir`: Save plots to directory
- `--dpi`: Resolution for saved plots (default: 300)

### Visualization Types

1. **Scenario Difficulty Analysis**
   - Baseline probability vs reduction success
   - Difficulty score ranking
   - Solution count distribution

2. **Token Effectiveness**
   - Individual token frequency and impact
   - Cross-scenario effectiveness ratios
   - Average reduction percentages

3. **Combination Patterns**
   - Most effective token combinations
   - Combination length distribution
   - Frequency vs effectiveness correlations

4. **Experiment Overview**
   - Result distributions across scenarios
   - Baseline probability impact analysis
   - Fitness score distributions

5. **Token-Scenario Heatmap**
   - Effectiveness matrix of tokens vs scenarios
   - Cross-scenario pattern identification
   - Specialized vs universal token detection

### Interpreting Visualizations

**Scenario Difficulty Plot**: Lower baseline probability = higher vulnerability

**Token Effectiveness Plot**: High frequency + high reduction = reliable token

**Combination Patterns**: Look for synergistic effects where combinations outperform individual tokens

**Heatmap**: Diagonal patterns indicate scenario-specific tokens, horizontal patterns indicate universal tokens

## Research Findings

### Key Discoveries

1. **Prediction Strength Vulnerability Gap**
   - Strong predictions (>90%): 1-10% reduction possible
   - Weak predictions (<30%): 50-99% reduction possible
   - Medium predictions (30-90%): 10-50% reduction possible

2. **Most Effective Token Types**
   - Turkish morphological suffixes: `'lıkları'`, `'ıldığında'`
   - Slavic language fragments: `'rodní'`, `'ослож'`
   - Corrupted text: `' CLIIIK'`, malformed unicode
   - Incomplete morphemes: Partial word endings

3. **Combination Synergies**
   - Token repetition amplifies effects
   - Cross-linguistic mixing creates powerful disruption
   - 2-3 tokens per combination optimal
   - Morphological diversity within combinations effective

4. **Context Dependencies**
   - Conversational contexts most resistant
   - Question contexts most vulnerable
   - Code completion contexts highly vulnerable
   - Mathematical contexts moderately resistant

### Statistical Patterns

- **Success Rate**: 60-90% of final population shows positive fitness
- **Convergence Time**: 10-20 generations for most scenarios
- **Optimal Population**: 30-50 individuals for efficiency
- **Token Effectiveness**: Top 20% of tokens account for 80% of reduction

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Solution: Reduce population size or use CPU
python examples/genetic_probability_reducer.py MODEL_NAME \
    --population-size 20 \
    --generations 30

# Force CPU usage
CUDA_VISIBLE_DEVICES="" python examples/genetic_probability_reducer.py MODEL_NAME
```

**No Positive Fitness Results**:
```bash
# Solution: Try different base text or increase generations
python examples/genetic_probability_reducer.py MODEL_NAME \
    --base-text "Different text" \
    --generations 200 \
    --population-size 80
```

**Slow Performance**:
```bash
# Solution: Reduce parameters or use smaller model
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --population-size 25 \
    --generations 50
```

**Token File Loading Errors**:
```bash
# Check file format - should be one of:
# 1. {"classifications": [{"token_id": 123}, ...]}
# 2. [123, 456, 789, ...]
# 3. {"tokens": [{"id": 123}, ...]}
```

### Performance Optimization

**GPU Memory Management**:
```bash
# Monitor GPU usage
nvidia-smi

# Use gradient checkpointing for large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Batch Processing**:
```bash
# Use smaller batch sizes for large experiments
python examples/genetic_batch_runner.py MODEL_NAME \
    --population-size 25 \
    --generations 40
```

### Debugging Tips

1. **Check Model Loading**: Verify model path and permissions
2. **Token File Validation**: Ensure correct JSON format
3. **Baseline Probability**: Very high (>99%) or very low (<1%) baselines may be difficult to modify
4. **Generation Monitoring**: Watch for convergence patterns in logs
5. **Memory Usage**: Monitor RAM/GPU usage during execution

## API Reference

### GeneticProbabilityReducer Class

```python
from genetic_probability_reducer import GeneticProbabilityReducer

# Initialize
analyzer = GeneticProbabilityReducer(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    base_text="The quick brown",
    target_token=None  # Auto-detect
)

# Configure parameters
analyzer.population_size = 50
analyzer.max_generations = 100
analyzer.mutation_rate = 0.1
analyzer.crossover_rate = 0.7
analyzer.elite_size = 5
analyzer.max_tokens_per_individual = 3

# Load resources
analyzer.load_model()
analyzer.load_glitch_tokens("email_llams321.json")

# Run evolution
final_population = analyzer.run_evolution()

# Display results
analyzer.display_results(final_population, top_n=10)

# Save results
analyzer.save_results(final_population, "results.json")
```

### Key Methods

**`load_model()`**: Load language model and tokenizer

**`load_glitch_tokens(file_path)`**: Load glitch tokens from JSON file

**`get_baseline_probability()`**: Calculate baseline prediction probability

**`run_evolution()`**: Execute genetic algorithm evolution

**`display_results(population, top_n)`**: Show top N results

**`save_results(population, file_path)`**: Save results to JSON

### Individual Class

```python
from genetic_probability_reducer import Individual

# Create individual
individual = Individual(
    tokens=[12345, 67890],
    fitness=0.0,
    baseline_prob=0.0,
    modified_prob=0.0
)

# Access properties
print(individual.tokens)  # [12345, 67890]
print(individual.fitness)  # 0.0
```

### Batch Runner Integration

```python
from genetic_batch_runner import GeneticBatchRunner

# Initialize batch runner
runner = GeneticBatchRunner(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    token_file="email_llams321.json"
)

# Add scenarios
runner.add_scenario("test1", "The quick brown")
runner.add_scenario("test2", "The weather is")

# Run experiments
results = runner.run_all_experiments()

# Analyze patterns
analysis = runner.analyze_token_patterns()

# Generate report
report = runner.generate_report()

# Save results
runner.save_results("output_dir")
```

---

## Future Enhancements

### Planned Features

1. **Multi-objective Optimization**: Balance reduction vs text coherence
2. **Adaptive Parameters**: Self-tuning GA parameters based on performance
3. **Parallel Evolution**: Multiple population islands for better exploration
4. **Transfer Learning**: Apply successful patterns across different models
5. **Real-time Monitoring**: Live evolution progress visualization
6. **Ensemble Methods**: Combine results from multiple GA runs

### Research Directions

1. **Theoretical Analysis**: Mathematical modeling of token interactions
2. **Scalability Studies**: Performance with larger vocabularies and models
3. **Cross-model Generalization**: Effectiveness patterns across architectures
4. **Temporal Dynamics**: How token effectiveness changes during training
5. **Defense Mechanisms**: Developing countermeasures against discovered patterns

---

For additional support, examples, or contributions, please refer to the main Glitcher repository documentation.