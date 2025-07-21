# Genetic Algorithm Integration Guide

## Overview

The Genetic Algorithm system integrates seamlessly with the broader Glitcher ecosystem, providing an advanced automated approach to discovering effective glitch token combinations. This guide explains how the genetic algorithm components work with existing Glitcher tools and workflows.

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Glitch Token  │───▶│  Genetic Algo    │───▶│  Visualization  │
│    Mining       │    │    Breeding      │    │   & Analysis    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Token Database  │◀───│  Enhanced        │───▶│  Research       │
│ (JSON files)    │    │  Validation      │    │  Reports        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Workflow Integration

### 1. Token Discovery → Genetic Breeding Pipeline

**Step 1: Initial Token Discovery**
```bash
# Standard glitch token mining
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 100 --batch-size 8

# Range-based mining for comprehensive coverage
glitcher mine meta-llama/Llama-3.2-1B-Instruct --mode unicode --sample-rate 0.1
```

**Step 2: Genetic Evolution**
```bash
# Use discovered tokens for genetic breeding
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --token-file glitch_tokens.json \
    --base-text "Your target text" \
    --max-tokens 3
```

**Step 3: Validation & Analysis**
```bash
# Validate discovered combinations with enhanced validation
glitcher test meta-llama/Llama-3.2-1B-Instruct \
    --token-ids 12345,67890,11111 \
    --enhanced --num-attempts 5
```

### 2. Complementary Approaches

| Method | Best For | Output | Integration Point |
|--------|----------|---------|-------------------|
| **Standard Mining** | Individual token discovery | Single effective tokens | Input to genetic algorithm |
| **Range Mining** | Systematic exploration | Tokens by category | Filtered input set |
| **Genetic Algorithm** | Combination optimization | Multi-token synergies | Enhanced effectiveness |
| **Enhanced Validation** | Reliability testing | ASR-based confidence | Result verification |

## Data Flow and File Formats

### Input Data Sources

**1. Standard Mining Output** (`email_llams321.json`):
```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "tokens_classified": 464,
  "classifications": [
    {
      "token_id": 116552,
      "token": "uj\u00edc\u00edm",
      "test_results": [...]
    }
  ]
}
```

**2. Range Mining Output** (various formats):
```json
{
  "range_results": {
    "Unicode Block: Latin Extended-A": {
      "total_tokens": 127,
      "glitch_tokens": [12345, 67890],
      "success_rate": 0.039
    }
  }
}
```

**3. Custom Token Lists**:
```json
[12345, 67890, 11111, 22222]
```

### Output Integration

**Genetic Algorithm Results** → **Enhanced Validation**:
```bash
# Extract top combinations from genetic results
python -c "
import json
with open('genetic_results.json') as f:
    data = json.load(f)
top_tokens = []
for result in data['results'][:5]:
    top_tokens.extend(result['tokens'])
print(','.join(map(str, set(top_tokens))))
" > top_combo_tokens.txt

# Validate with enhanced validation
glitcher test meta-llama/Llama-3.2-1B-Instruct \
    --token-ids $(cat top_combo_tokens.txt) \
    --enhanced --num-attempts 10
```

## Use Case Decision Matrix

### When to Use Genetic Algorithm

**✅ Ideal Scenarios:**
- Need to find multi-token synergies
- Have computational resources for evolution
- Want systematic exploration of combinations
- Research focused on token interaction patterns
- Need reproducible, scientific methodology

**❌ Not Recommended:**
- Quick individual token testing
- Limited computational resources
- Only need basic glitch detection
- Real-time or interactive usage

### When to Use Standard Mining

**✅ Ideal Scenarios:**
- Initial token discovery
- Quick individual token assessment
- Building token databases
- High-throughput screening
- Production vulnerability scanning

### When to Use Range Mining

**✅ Ideal Scenarios:**
- Systematic vocabulary exploration
- Understanding token distribution patterns
- Academic research on model vocabularies
- Comprehensive coverage requirements

## Integration Examples

### Example 1: Comprehensive Vulnerability Assessment

```bash
# Phase 1: Discover individual tokens
echo "Phase 1: Individual token discovery"
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --num-iterations 200 \
    --output-dir phase1_tokens

# Phase 2: Range-based systematic exploration
echo "Phase 2: Systematic exploration"
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
    --mode unicode \
    --sample-rate 0.05 \
    --output-dir phase2_ranges

# Phase 3: Genetic combination optimization
echo "Phase 3: Combination breeding"
python examples/genetic_batch_runner.py meta-llama/Llama-3.2-1B-Instruct \
    --token-file phase1_tokens/glitch_tokens.json \
    --use-predefined \
    --max-tokens 4 \
    --output-dir phase3_genetic

# Phase 4: Enhanced validation of top results
echo "Phase 4: Validation"
# Extract and validate top combinations
python -c "
import json, sys
with open('phase3_genetic/genetic_batch_results_*.json') as f:
    data = json.load(f)
top_combos = []
for exp in data['experiments']:
    for result in exp.get('top_results', [])[:3]:
        top_combos.append(','.join(map(str, result['tokens'])))
for combo in top_combos[:10]:
    print(f'glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids {combo} --enhanced')
"
```

### Example 2: Research Pipeline

```bash
# Discover baseline token set
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 100

# Run genetic experiments across contexts
python examples/genetic_batch_runner.py meta-llama/Llama-3.2-1B-Instruct \
    --use-predefined \
    --generations 100 \
    --population-size 50 \
    --max-tokens 3

# Generate comprehensive analysis
python examples/visualize_genetic_results.py genetic_batch_results/*.json \
    --output-dir research_plots

# Create research report
echo "Generating research findings..."
python -c "
from genetic_batch_runner import GeneticBatchRunner
runner = GeneticBatchRunner('meta-llama/Llama-3.2-1B-Instruct', 'email_llams321.json')
# Load previous results and generate report
# (implementation would load existing results)
"
```

### Example 3: Production Security Testing

```bash
# Quick discovery for immediate threats
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 50 --quick

# Test specific application contexts
cat > production_scenarios.json << EOF
[
    {"name": "user_input", "base_text": "User said: "},
    {"name": "search_query", "base_text": "Search for: "},
    {"name": "chat_completion", "base_text": "Assistant: "}
]
EOF

# Run targeted genetic analysis
python examples/genetic_batch_runner.py meta-llama/Llama-3.2-1B-Instruct \
    --scenarios-file production_scenarios.json \
    --generations 30 \
    --population-size 25 \
    --max-tokens 2

# Extract high-risk combinations for monitoring
python -c "
import json
with open('genetic_batch_results/*.json') as f:
    data = json.load(f)
high_risk = []
for exp in data['experiments']:
    for result in exp.get('top_results', []):
        if result['reduction_percentage'] > 50:
            high_risk.append(result['tokens'])
print('High-risk token combinations for monitoring:')
for combo in high_risk:
    print(f'  {combo}')
"
```

## Result Cross-Validation

### Genetic Results → Standard Validation

```bash
# Test genetic discoveries with standard validation
python -c "
import json
with open('genetic_results.json') as f:
    data = json.load(f)
for result in data['results'][:10]:
    tokens = ','.join(map(str, result['tokens']))
    print(f'glitcher test meta-llama/Llama-3.2-1B-Instruct --token-ids {tokens}')
"
```

### Cross-Model Validation

```bash
# Test genetic discoveries across multiple models
MODELS=("meta-llama/Llama-3.2-1B-Instruct" "microsoft/DialoGPT-medium")

for model in "${MODELS[@]}"; do
    echo "Testing genetic results on $model"
    python examples/genetic_probability_reducer.py "$model" \
        --base-text "Same test text" \
        --token-file genetic_discovered_tokens.json \
        --generations 20
done
```

## Performance Optimization in Ecosystem

### Memory Management

```bash
# For large-scale integration workflows
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Sequential processing for memory efficiency
for scenario in scenario1 scenario2 scenario3; do
    python examples/genetic_probability_reducer.py MODEL \
        --base-text "$scenario" \
        --generations 50 \
        --population-size 30
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()"
done
```

### Parallel Processing

```bash
# Run multiple scenarios in parallel (if sufficient resources)
cat scenarios.txt | xargs -P 4 -I {} python examples/genetic_probability_reducer.py MODEL --base-text "{}"
```

## Data Standards and Compatibility

### Token ID Consistency

All Glitcher tools use consistent token ID formats:
- **Integer IDs**: Direct tokenizer vocabulary indices
- **JSON Format**: Standardized across all tools
- **Unicode Handling**: Proper encoding/decoding everywhere

### Result Format Standardization

```json
{
    "metadata": {
        "model_name": "string",
        "timestamp": "ISO 8601",
        "tool_version": "string",
        "method": "genetic_algorithm|standard_mining|range_mining"
    },
    "results": [
        {
            "tokens": [int],
            "effectiveness_metrics": {},
            "validation_data": {}
        }
    ]
}
```

## Troubleshooting Integration

### Common Integration Issues

**1. Token File Format Mismatches**
```bash
# Convert between formats
python -c "
import json
# Convert classifications format to simple list
with open('email_llams321.json') as f:
    data = json.load(f)
token_ids = [t['token_id'] for t in data['classifications']]
with open('simple_tokens.json', 'w') as f:
    json.dump(token_ids, f)
"
```

**2. Model Loading Conflicts**
```bash
# Ensure consistent model loading across tools
export TRANSFORMERS_CACHE=/path/to/shared/cache
export HF_HOME=/path/to/shared/cache
```

**3. Memory Issues in Pipelines**
```bash
# Use memory-efficient sequential processing
python -c "
import gc, torch
# Between tool invocations
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"
```

## Advanced Integration Patterns

### 1. Feedback Loop Integration

```bash
# Genetic results inform next mining iteration
python examples/genetic_probability_reducer.py MODEL --base-text "text1" > results1.json
# Extract effective token patterns
python analyze_genetic_patterns.py results1.json > patterns.json
# Use patterns to guide next mining
glitcher mine MODEL --pattern-file patterns.json
```

### 2. Multi-Model Consensus

```bash
# Test genetic discoveries across model family
for model in llama-1b llama-3b llama-7b; do
    python examples/genetic_probability_reducer.py "$model" \
        --token-file consensus_tokens.json \
        --output "consensus_${model}.json"
done
# Analyze cross-model effectiveness
python analyze_consensus.py consensus_*.json
```

### 3. Continuous Monitoring Integration

```bash
# Set up monitoring for genetic discoveries
crontab -e
# Add: 0 2 * * * /path/to/genetic_monitor.sh

# genetic_monitor.sh
#!/bin/bash
python examples/genetic_probability_reducer.py meta-llama/Llama-3.2-1B-Instruct \
    --base-text "Current production prompt" \
    --generations 20 \
    --output "daily_genetic_$(date +%Y%m%d).json"
# Alert if new high-impact combinations found
python check_genetic_alerts.py "daily_genetic_$(date +%Y%m%d).json"
```

## Future Integration Roadmap

### Planned Integrations

1. **Web Interface**: Browser-based genetic algorithm control
2. **API Endpoints**: RESTful API for genetic breeding
3. **Database Integration**: Persistent storage of genetic discoveries
4. **Real-time Monitoring**: Live genetic algorithm results
5. **Multi-model Orchestration**: Automated cross-model validation

### Research Integrations

1. **Academic Datasets**: Integration with research benchmarks
2. **Defense Mechanisms**: Integration with robustness training
3. **Explainability Tools**: Integration with interpretation frameworks
4. **Scaling Studies**: Integration with distributed computing platforms

---

This integration guide ensures that the genetic algorithm system works seamlessly with existing Glitcher tools while opening new possibilities for advanced token analysis and discovery workflows.