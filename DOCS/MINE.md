# Glitch Token Mining Guide

## Overview

Glitch token mining is the core functionality of Glitcher that automatically discovers tokens exhibiting anomalous behavior in language models. The mining process combines entropy-based exploration with sophisticated validation techniques to identify tokens that cause models to generate unexpected or repetitive outputs.

## How Mining Works

### The Mining Algorithm

The mining process follows a sophisticated iterative approach that creates a guided exploration path through the token embedding space:

1. **Initialization**: Start with the token having the smallest L2 norm in the embedding space (typically a simple, common token)
2. **Entropy Analysis**: Calculate the entropy of the model's next-token predictions for the current token
3. **Gradient-Based Exploration**: Use gradients to approximate entropy changes for nearby tokens
4. **Candidate Selection**: Select the top `k` nearest tokens with highest predicted entropy
5. **Batch Evaluation**: Test `batch_size` most promising candidates for glitch behavior
6. **Validation**: Apply either standard or enhanced validation to confirm glitch behavior
7. **Iteration**: Move to the highest-entropy token from the batch and repeat

### Understanding Iterations

Each iteration represents one step in a guided walk through the token embedding space. When you specify `--num-iterations 500`, you're not randomly testing 500 tokens - instead, you're creating a 500-step exploration path where each step builds intelligently on the previous one.

**What happens in each iteration:**
- **Current Position**: Start from the current token position in embedding space
- **Local Exploration**: Find the `k` nearest neighbor tokens that haven't been tested yet (default k=32)
- **Entropy Prediction**: Use gradient information to predict which neighbors will increase entropy
- **Batch Testing**: Evaluate the `batch_size` most promising candidates (default 8 tokens)
- **Path Selection**: Calculate actual entropy for each candidate and move to the highest-entropy token
- **Token Exclusion**: ALL tested tokens in the batch are permanently excluded from future iterations
- **Next Step**: This highest-entropy token becomes the starting point for the next iteration
- **Permanent Exclusion**: All tokens tested in the current batch are permanently excluded from future iterations

**Why this approach is effective:**
- **Clustering Effect**: Tokens with similar embeddings often have similar behavioral properties
- **Entropy Correlation**: Higher entropy regions tend to contain more problematic tokens
- **Efficient Exploration**: Avoids random sampling by following entropy gradients toward interesting regions
- **No Revisiting**: Once tested, tokens are permanently excluded, ensuring efficient coverage
- **Diminishing Search Space**: As iterations progress, the algorithm is forced to explore new regions
- **No Retesting**: Once tokens are evaluated, they're excluded to ensure comprehensive coverage without redundancy

**Example 5-iteration sequence:**
```
Iteration 1: Start at "the" → test 8 neighbors → exclude all 8 → move to highest entropy "uncommon"
Iteration 2: Start at "uncommon" → test 8 new neighbors → exclude all 8 → move to "SpecialToken"
Iteration 3: Start at "SpecialToken" → test 8 new neighbors → exclude all 8 → move to "GlitchToken"
Iteration 4: Start at "GlitchToken" → test 8 new neighbors → exclude all 8 → move to "SimilarGlitch"
Iteration 5: Start at "SimilarGlitch" → test 8 new neighbors → exclude all 8 → move to "AnotherGlitch"
```

This creates a path through embedding space that systematically explores different regions while permanently excluding tested tokens, ensuring comprehensive coverage without redundancy.

### Detection Criteria

A token is considered a potential glitch if:
- Its probability of being predicted is below a very low threshold (< 0.00001 for Llama 3.2)
- It's not the model's top predicted token for the given context
- It doesn't match known false-positive patterns (brackets, underscores, etc.)

### Validation Methods

#### Enhanced Validation (Default)
- **Multi-token generation**: Generates up to `max_tokens` tokens and searches for the target token
- **Multiple contexts**: Tests token in 3 different prompt formats for robustness
- **ASR-based classification**: Uses Attack Success Rate to determine glitch status
- **Multiple attempts**: Supports non-deterministic validation with configurable attempts

#### Standard Validation (Legacy)
- **Single-response validation**: Tests if the model outputs the exact target token
- **Strict criteria**: Requires perfect reproduction in specific contexts
- **Faster but less accurate**: Quicker execution but higher false positive rates

## Command Reference

### Basic Mining Command

```bash
glitcher mine meta-llama/Llama-3.2-1B-Instruct [OPTIONS]
```

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-iterations` | 50 | Number of exploration steps through embedding space |
| `--batch-size` | 8 | Number of candidate tokens to evaluate per iteration |
| `--k` | 32 | Number of nearest neighbors to consider when exploring from current position |
| `--output` | glitch_tokens.json | Output file for discovered glitch tokens |

### Enhanced Validation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enhanced-validation` | True | Use enhanced validation (enabled by default) |
| `--disable-enhanced-validation` | False | Disable enhanced validation, use standard method |
| `--validation-tokens` | 50 | Maximum tokens to generate during validation |
| `--num-attempts` | 1 | Number of validation attempts per token |
| `--asr-threshold` | 0.5 | ASR threshold for considering token a glitch (0.0-1.0) |

### Progress and Resume Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save-interval` | 5 | Save progress every N iterations |
| `--resume` | False | Resume from previous progress file |
| `--progress-file` | glitch_progress.json | File to save/load progress |

### Performance Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | cuda | Device to use (cuda/cpu) |
| `--quant-type` | bfloat16 | Quantization type (bfloat16/float16/int8/int4) |

## ASR (Attack Success Rate) System

### Understanding ASR

ASR measures the percentage of validation attempts where a token exhibits glitch behavior:

```
ASR = (Glitch attempts) / (Total attempts) × 100%
```

### ASR Thresholds

| Threshold | Use Case | Description |
|-----------|----------|-------------|
| **1.0 (100%)** | Research/Academic | Only tokens that ALWAYS show glitch behavior |
| **0.8 (80%)** | High Confidence | Very consistent problematic tokens |
| **0.5 (50%)** | Balanced (Default) | Tokens that are more often glitchy than not |
| **0.3 (30%)** | Broad Detection | Tokens that occasionally show glitch behavior |
| **0.0 (0%)** | Maximum Coverage | Any token showing glitch behavior at least once |

### ASR Examples

**Token with 3/5 glitch attempts (ASR = 60%)**:
- `--asr-threshold 0.5`: Classified as GLITCH ✓
- `--asr-threshold 0.7`: Classified as normal ✗
- `--asr-threshold 1.0`: Classified as normal ✗

## Usage Examples

### Quick Discovery
```bash
# Fast exploration with 20 steps through embedding space
glitcher mine meta-llama/Llama-3.2-1B-Instruct --num-iterations 20
```

### Comprehensive Mining
```bash
# Thorough exploration with 100 steps and enhanced validation
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 100 \
  --validation-tokens 100 \
  --num-attempts 3 \
  --batch-size 8 \
  --k 32
```

### Deep Exploration
```bash
# Extensive 500-step exploration for comprehensive token discovery
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 500 \
  --validation-tokens 500 \
  --num-attempts 10 \
  --batch-size 8 \
  --k 32
```

### High-Confidence Detection
```bash
# Strict ASR threshold for research use
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --asr-threshold 0.8 \
  --num-attempts 5 \
  --validation-tokens 100
```

### Broad Screening
```bash
# Permissive detection for comprehensive discovery
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --asr-threshold 0.3 \
  --num-attempts 10 \
  --validation-tokens 50
```

### Legacy Mining
```bash
# Use standard validation for speed
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 50 \
  --disable-enhanced-validation \
  --batch-size 16
```

### Resume Interrupted Mining
```bash
# Resume from previous session
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 100 \
  --resume
```

## Search Strategy Tradeoffs

### Minimal vs Comprehensive Search

The mining algorithm's effectiveness depends heavily on balancing exploration depth with computational resources. Understanding these tradeoffs helps you choose the right parameters for your specific use case.

### Key Parameters for Search Strategy

| Parameter | Minimal Search | Comprehensive Search | Impact |
|-----------|---------------|---------------------|---------|
| `--num-iterations` | 10-50 | 200-1000 | Exploration depth through embedding space |
| `--batch-size` | 16-32 | 4-8 | Tokens tested per iteration (higher = broader but shallower) |
| `--k` | 16-32 | 64-128 | Neighborhood size (higher = more diverse candidates) |
| `--num-attempts` | 1 | 3-10 | Validation reliability (higher = more accurate ASR) |
| `--asr-threshold` | 0.3-0.5 | 0.7-1.0 | Classification strictness (higher = fewer false positives) |

### Minimal Search Strategy

**Best for**: Quick exploration, resource-constrained environments, initial discovery

```bash
# Fast minimal search - broad but shallow exploration
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 20 \
  --batch-size 16 \
  --k 32 \
  --num-attempts 1 \
  --asr-threshold 0.3 \
  --validation-tokens 25
```

**Characteristics**:
- **Speed**: Very fast execution (minutes)
- **Coverage**: Broad but shallow exploration
- **Quality**: Higher false positive rate
- **Discovery**: Good for finding obvious glitch tokens
- **Resource Usage**: Low memory and compute requirements

**When to use**:
- Initial exploration of a new model
- Limited computational resources
- Quick proof-of-concept validation
- Screening for obvious problematic tokens

### Comprehensive Search Strategy

**Best for**: Research, production deployment, thorough analysis

```bash
# Comprehensive deep search - thorough exploration
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 500 \
  --batch-size 4 \
  --k 64 \
  --num-attempts 5 \
  --asr-threshold 0.8 \
  --validation-tokens 100
```

**Characteristics**:
- **Speed**: Slow execution (hours)
- **Coverage**: Deep, systematic exploration
- **Quality**: Low false positive rate, high confidence
- **Discovery**: Finds subtle and edge-case glitch tokens
- **Resource Usage**: High memory and compute requirements

**When to use**:
- Production model validation
- Academic research requiring high confidence
- Comprehensive security assessment
- Building authoritative glitch token databases

### Balanced Search Strategy

**Best for**: Most practical applications, development workflows

```bash
# Balanced search - good compromise
glitcher mine meta-llama/Llama-3.2-1B-Instruct \
  --num-iterations 100 \
  --batch-size 8 \
  --k 48 \
  --num-attempts 3 \
  --asr-threshold 0.6 \
  --validation-tokens 50
```

**Characteristics**:
- **Speed**: Moderate execution time (30-60 minutes)
- **Coverage**: Good balance of breadth and depth
- **Quality**: Reasonable false positive rate
- **Discovery**: Finds most important glitch tokens
- **Resource Usage**: Manageable resource requirements

### Search Strategy Comparison

| Metric | Minimal | Balanced | Comprehensive |
|--------|---------|----------|---------------|
| **Execution Time** | 5-15 min | 30-60 min | 2-8 hours |
| **Tokens Evaluated** | ~500 | ~800 | ~2000 |
| **Memory Usage** | Low | Medium | High |
| **False Positive Rate** | High (20-40%) | Medium (10-20%) | Low (2-10%) |
| **Glitch Discovery** | Obvious cases | Most important | Comprehensive |
| **Recommended Use** | Initial exploration | Development | Production |

### Parameter Interaction Effects

#### Iterations vs Batch Size
- **High iterations + Small batch**: Deep, focused exploration
- **Low iterations + Large batch**: Broad, shallow coverage
- **Optimal balance**: Usually iterations = 10-20 × batch_size

#### K-value vs Search Quality
- **Low k (16-32)**: Faster but may miss distant glitch clusters
- **High k (64-128)**: Slower but better coverage of embedding space
- **Diminishing returns**: k > 128 rarely improves results significantly

#### ASR Threshold vs Discovery Rate
- **Low threshold (0.3)**: More discoveries but higher false positives
- **High threshold (0.8)**: Fewer discoveries but higher confidence
- **Context matters**: Non-deterministic models need lower thresholds

### Real-World Workflow Examples

#### Development Phase
```bash
# Quick discovery for development
glitcher mine model \
  --num-iterations 30 \
  --batch-size 12 \
  --asr-threshold 0.4 \
  --num-attempts 1
```

#### Testing Phase
```bash
# Balanced validation for testing
glitcher mine model \
  --num-iterations 100 \
  --batch-size 8 \
  --asr-threshold 0.6 \
  --num-attempts 3
```

#### Production Validation
```bash
# Comprehensive validation for production
glitcher mine model \
  --num-iterations 500 \
  --batch-size 4 \
  --asr-threshold 0.8 \
  --num-attempts 5 \
  --validation-tokens 100
```

#### Research Analysis
```bash
# Deep analysis for research
glitcher mine model \
  --num-iterations 1000 \
  --batch-size 2 \
  --k 128 \
  --asr-threshold 1.0 \
  --num-attempts 10 \
  --validation-tokens 200
```

### Progressive Search Strategy

For unknown models, consider a progressive approach:

1. **Phase 1: Quick Survey**
```bash
glitcher mine model --num-iterations 20 --batch-size 16 --asr-threshold 0.3
```

2. **Phase 2: Targeted Analysis**
```bash
# Use results from Phase 1 to guide parameters
glitcher mine model --num-iterations 100 --batch-size 8 --asr-threshold 0.6
```

3. **Phase 3: Deep Validation**
```bash
glitcher mine model --num-iterations 300 --batch-size 4 --asr-threshold 0.8
```

This approach balances discovery speed with thoroughness, allowing you to adjust strategy based on initial findings.

### Mathematical Analysis of Search Coverage

Understanding the mathematical relationship between parameters helps optimize search efficiency:

#### Search Space Coverage

**Total tokens evaluated over iterations:**
```
Total = Σ(batch_size) for each iteration
Maximum possible = num_iterations × batch_size
```

**Effective search space reduction:**
```
Available tokens at iteration i = Total_tokens - (i × batch_size)
Coverage rate = Tokens_evaluated / Total_vocabulary_size
```

#### Efficiency Metrics

**Discovery Efficiency:**
```
Efficiency = Glitch_tokens_found / Total_tokens_evaluated
Optimal range: 0.01-0.05 (1-5% of evaluated tokens are glitches)
```

**Exploration Diversity:**
```
Diversity = Unique_embedding_regions_visited / Total_iterations
Higher k-values increase diversity but reduce iteration speed
```

#### Parameter Optimization Guidelines

**For vocabulary size V and target coverage C:**
- **Minimal search**: `num_iterations = V × C / (batch_size × 4)`
- **Comprehensive search**: `num_iterations = V × C / (batch_size × 1.5)`

**Example for Llama 3.2 (V ≈ 128,000):**
- **5% coverage minimal**: 200 iterations × 16 batch = 3,200 tokens (2.5% coverage)
- **10% coverage comprehensive**: 800 iterations × 4 batch = 3,200 tokens (2.5% coverage)

The comprehensive approach evaluates the same number of tokens but explores more diverse regions due to smaller batch sizes and deeper iteration paths.

## Output and Logging

### Progress Display

During mining, you'll see real-time progress:

```
Mining glitch tokens: 45%|████▌     | 23/50 [02:34<02:51, 1.57it/s]
✓ Validated glitch token: 'SpecialToken' (ID: 12345, asr: 75.00%, entropy: 0.1234, target_prob: 0.001234, top_prob: 0.567890, method: enhanced)
✗ False positive: 'NormalToken' (ID: 67890, asr: 33.33%, failed enhanced validation)
```

### Output Files

#### Results File (glitch_tokens.json)
```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "glitch_tokens": ["SpecialToken", "AnotherGlitch"],
  "glitch_token_ids": [12345, 67890],
  "total_iterations": 50,
  "runtime_seconds": 1234.56
}
```

#### Progress File (glitch_progress.json)
```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "num_iterations": 50,
  "iterations_completed": 23,
  "total_tokens_checked": 184,
  "glitch_tokens": ["SpecialToken"],
  "glitch_token_ids": [12345],
  "start_time": 1752425609.123,
  "last_saved": 1752425920.456
}
```

#### Detailed Log File (glitch_mining_log_[timestamp].jsonl)
Contains comprehensive event logs including:
- Mining configuration and model info
- Token verification details with probabilities
- Validation results with ASR data
- Template and formatting information

## Performance Optimization

### Memory Management
- **Batch size**: Reduce `--batch-size` if encountering OOM errors
- **Validation tokens**: Lower `--validation-tokens` to reduce memory usage
- **Quantization**: Use `int8` or `int4` for memory-constrained environments

### Speed Optimization
- **Disable enhanced validation**: Use `--disable-enhanced-validation` for faster mining
- **Reduce attempts**: Lower `--num-attempts` for quicker validation
- **Increase batch size**: Higher `--batch-size` for better GPU utilization

### Accuracy vs Speed Trade-offs

| Configuration | Speed | Accuracy | Use Case |
|---------------|-------|----------|----------|
| Enhanced + Multiple attempts | Slow | High | Research/Production |
| Enhanced + Single attempt | Medium | Good | General discovery |
| Standard validation | Fast | Lower | Quick exploration |

## Model-Specific Considerations

### Llama 3.2 Models
- **Optimized thresholds**: Automatic probability threshold adjustment
- **Chat template handling**: Built-in support for Llama 3.2 chat format
- **Memory efficiency**: Optimized for 1B and 3B parameter models

### Supported Models
- meta-llama/Llama-3.1-1B-Instruct
- meta-llama/Llama-3.2-1B-Instruct  
- meta-llama/Llama-3.2-3B-Instruct

## Best Practices

### For Research Use
```bash
glitcher mine model_path \
  --num-iterations 100 \
  --asr-threshold 0.8 \
  --num-attempts 5 \
  --validation-tokens 100 \
  --batch-size 4
```

### For Production Screening
```bash
glitcher mine model_path \
  --num-iterations 50 \
  --asr-threshold 0.5 \
  --num-attempts 3 \
  --validation-tokens 50 \
  --batch-size 8
```

### For Quick Exploration
```bash
glitcher mine model_path \
  --num-iterations 20 \
  --asr-threshold 0.3 \
  --num-attempts 1 \
  --validation-tokens 25 \
  --batch-size 16
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce memory usage
glitcher mine model_path \
  --batch-size 2 \
  --validation-tokens 25 \
  --quant-type int8
```

#### Slow Performance
```bash
# Optimize for speed
glitcher mine model_path \
  --disable-enhanced-validation \
  --batch-size 16 \
  --num-iterations 20
```

#### Too Many False Positives
```bash
# Increase ASR threshold
glitcher mine model_path \
  --asr-threshold 0.8 \
  --num-attempts 5
```

#### Missing Known Glitch Tokens
```bash
# Lower ASR threshold and increase attempts
glitcher mine model_path \
  --asr-threshold 0.3 \
  --num-attempts 10
```

#### Inconsistent Results
```bash
# Increase validation attempts for stability
glitcher mine model_path \
  --num-attempts 5 \
  --asr-threshold 0.5
```

### Search Strategy Optimization

#### Poor Discovery Rate (Few Glitch Tokens Found)
**Symptoms**: Very few or no glitch tokens discovered after mining
**Solutions**:
```bash
# Increase exploration breadth
glitcher mine model_path \
  --num-iterations 200 \
  --batch-size 4 \
  --k 64 \
  --asr-threshold 0.3

# Try different starting regions
glitcher mine model_path \
  --num-iterations 100 \
  --batch-size 8 \
  --k 48 \
  --asr-threshold 0.4
```

#### High False Positive Rate
**Symptoms**: Many discovered tokens fail validation or seem normal
**Solutions**:
```bash
# Increase validation rigor
glitcher mine model_path \
  --asr-threshold 0.8 \
  --num-attempts 5 \
  --validation-tokens 100

# Use stricter detection criteria
glitcher mine model_path \
  --asr-threshold 1.0 \
  --num-attempts 10
```

#### Inefficient Resource Usage
**Symptoms**: Long runtime with poor token discovery efficiency
**Solutions**:
```bash
# Optimize for better discovery rate
glitcher mine model_path \
  --num-iterations 50 \
  --batch-size 16 \
  --k 32 \
  --num-attempts 1

# Progressive refinement approach
# Phase 1: Broad discovery
glitcher mine model_path --num-iterations 30 --batch-size 16 --asr-threshold 0.3
# Phase 2: Targeted validation  
glitcher mine model_path --num-iterations 100 --batch-size 8 --asr-threshold 0.7
```

#### Search Space Exhaustion
**Symptoms**: Later iterations find very few candidates
**Solutions**:
```bash
# Increase neighborhood size
glitcher mine model_path \
  --k 128 \
  --num-iterations 200

# Reduce batch size for deeper exploration
glitcher mine model_path \
  --batch-size 2 \
  --num-iterations 400
```

#### Inconsistent Results Across Runs
**Symptoms**: Different glitch tokens found in repeated mining runs
**Solutions**:
```bash
# Increase validation attempts for stability
glitcher mine model_path \
  --num-attempts 5 \
  --asr-threshold 0.7

# Use deterministic settings
glitcher mine model_path \
  --num-attempts 1 \
  --asr-threshold 1.0 \
  --disable-enhanced-validation
```

### Debug Commands

#### Verbose Logging
```bash
# Enable detailed logging
glitcher mine model_path --num-iterations 5 --batch-size 2
```

#### Test Specific Region
```bash
# Use fewer iterations to test specific token regions
glitcher mine model_path --num-iterations 10 --batch-size 4
```

#### Analyze Search Efficiency
```bash
# Monitor discovery rate with small batches
glitcher mine model_path \
  --num-iterations 20 \
  --batch-size 4 \
  --k 32 \
  --num-attempts 1

# Compare different strategies
glitcher mine model_path --num-iterations 50 --batch-size 16  # Broad
glitcher mine model_path --num-iterations 200 --batch-size 4  # Deep
```

## Understanding the Algorithm

### Entropy-Based Exploration

The mining algorithm uses entropy as a measure of model uncertainty. Tokens that lead to high entropy in the model's predictions are more likely to be problematic, as they indicate situations where the model is "confused" about what to generate next.

### Gradient-Guided Search

Instead of randomly sampling tokens, the algorithm uses gradients of the entropy with respect to token embeddings to predict which nearby tokens will lead to even higher entropy. This makes the search much more efficient than brute force approaches.

### Embedding Space Navigation

The algorithm creates an intelligent exploration path through the token embedding space:

1. **Starting Point**: Begin with a token having minimal L2 norm (typically a simple, common token like "the" or "a")
2. **Local Neighborhood**: Find the k-nearest neighbors in the normalized embedding space around the current token (excluding already tested tokens)
3. **Entropy Prediction**: Use gradient information to predict which neighbors will lead to higher entropy
4. **Candidate Evaluation**: Test the most promising candidates and calculate their actual entropy values
5. **Path Selection**: Move to the token with the highest actual entropy, which becomes the new starting point
6. **Token Exclusion**: All tested tokens are permanently removed from the available pool
6. **Permanent Exclusion**: All tested candidates are permanently excluded from future iterations

**Why this navigation strategy works:**
- **Entropy Gradients**: Gradients point toward directions of increasing entropy, guiding the search efficiently
- **Locality Principle**: Tokens with similar embeddings often exhibit similar behavioral properties
- **Progressive Discovery**: Each step builds on previous knowledge, creating a coherent exploration path
- **Forced Diversification**: Excluding tested tokens forces the algorithm to explore new regions
- **Diminishing Returns**: As good regions are exhausted, the algorithm naturally moves to previously unexplored areas
- **Systematic Coverage**: Tested tokens are permanently excluded, ensuring no redundant evaluation

**Iteration Progression Example:**
```
Step 1: "the" (entropy: 0.23) → test 8 neighbors → exclude all 8 → move to "The" (entropy: 0.31)
Step 2: "The" (entropy: 0.31) → test 8 new neighbors → exclude all 8 → move to "THE" (entropy: 0.45)
Step 3: "THE" (entropy: 0.45) → test 8 new neighbors → exclude all 8 → move to "THEHE" (entropy: 0.72)
Step 4: "THEHE" (entropy: 0.72) → test 8 new neighbors → exclude all 8 → move to "glitchtoken" (entropy: 1.23)
...continuing toward higher entropy regions with shrinking candidate pool
```

This creates a one-way exploration path that systematically covers different regions of the embedding space, with each iteration permanently removing tested tokens and forcing discovery of new areas.

### Validation Pipeline

Each potential glitch token goes through a multi-stage validation:
1. **Probability filtering**: Eliminates tokens with high self-prediction probability
2. **Pattern filtering**: Removes known false-positive patterns
3. **Enhanced validation**: Tests token behavior in multiple contexts
4. **ASR calculation**: Determines if the token meets the glitch threshold

## Advanced Usage

### Custom ASR Thresholds by Use Case

#### Academic Research (High Confidence)
```bash
--asr-threshold 1.0 --num-attempts 10
```

#### Security Assessment (Balanced)
```bash
--asr-threshold 0.6 --num-attempts 5
```

#### Comprehensive Screening (Broad)
```bash
--asr-threshold 0.2 --num-attempts 10
```

### Long-Running Mining Sessions
```bash
# For extensive 500-step exploration with regular checkpoints
# This creates a long exploration path through embedding space
glitcher mine model_path \
  --num-iterations 500 \
  --save-interval 10 \
  --batch-size 8 \
  --resume
```

### Understanding Large Iteration Counts
When using high iteration counts (e.g., 500), you're creating an extensive exploration path:
- **Early iterations (1-50)**: Explore common token neighborhoods, establish baseline entropy
- **Mid iterations (51-200)**: Discover moderately problematic tokens, build toward higher entropy regions
- **Late iterations (201-500)**: Explore remaining untested regions, discover edge cases as search space shrinks

Each iteration processes `batch_size` tokens, so 500 iterations with batch_size=8 evaluates up to 4,000 tokens total (500 × 8), but the actual number depends on the available token pool. Since tested tokens are permanently excluded, later iterations may have fewer candidates available.

### Memory-Constrained Environments
```bash
# Optimize for limited GPU memory
glitcher mine model_path \
  --batch-size 2 \
  --validation-tokens 20 \
  --quant-type int4 \
  --num-attempts 1
```

## Integration with Other Tools

After mining, use discovered tokens with other Glitcher tools:

```bash
# Test discovered tokens
glitcher test model_path --token-file glitch_tokens.json --enhanced

# Compare validation methods
glitcher compare model_path --token-file glitch_tokens.json

# Extract domain-specific behavior
glitcher domain model_path --token-file glitch_tokens.json

# Interactive testing
glitcher chat model_path 12345 --max-size 50
```

## Future Enhancements

The mining system continues to evolve with planned improvements:
- Adaptive ASR thresholds based on model behavior
- Parallel validation for improved performance  
- Advanced embedding space exploration strategies
- Integration with domain-specific validation methods
- Real-time mining performance optimization