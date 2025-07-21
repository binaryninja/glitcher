# Glitch Token Impact Analyzer

A comprehensive tool for analyzing how glitch tokens affect language model predictions and attention patterns when inserted into text sequences.

## Overview

The Glitch Token Impact Analyzer tests how different glitch tokens influence model behavior when inserted before "The quick brown". It provides detailed metrics on probability distribution changes, attention pattern shifts, and prediction impacts.

## What It Does

1. **Loads glitch tokens** from classification data (`email_llams321.json`)
2. **Establishes baseline** predictions for "The quick brown" 
3. **Tests each glitch token** by inserting it before the base text
4. **Measures impact** using multiple statistical metrics
5. **Ranks tokens** by their influence strength
6. **Analyzes attention patterns** to understand why impacts occur

## Key Features

### Impact Metrics
- **Manhattan Distance**: Total variation between probability distributions
- **Hellinger Distance**: Symmetric measure of distribution similarity  
- **JS Divergence**: Jensen-Shannon divergence for distribution comparison
- **Entropy Change**: Difference in prediction uncertainty
- **Attention Shifts**: How attention patterns change for each token

### Analysis Capabilities
- Probability distribution comparisons
- Top prediction changes
- Attention pattern visualization
- Token-by-token attention shift analysis
- Ranking by impact strength

## Usage

```bash
# Basic usage
python glitch_impact_analyzer.py

# The tool will automatically:
# 1. Load the Llama-3.2-1B-Instruct model
# 2. Load glitch tokens from ../email_llams321.json  
# 3. Test impact of each token on "The quick brown"
# 4. Display ranked results and detailed analysis
```

## Understanding the Results

### Impact Summary Table

```
Rank Token                Manhattan  Hellinger  Entropy Δ    Token Δ  Prob Δ
------------------------------------------------------------------------------------------
1    'useRalative'        0.099670   0.144165   0.000000     ✗        +0.049805
2    ' CLIIIK'            0.099670   0.143921   0.000000     ✗        +0.049805
3    'espoň'              0.099548   0.143311   0.000000     ✗        +0.049805
```

**Columns Explained:**
- **Rank**: Impact strength ranking (1 = highest impact)
- **Token**: The glitch token being tested
- **Manhattan**: Total variation distance (higher = more impact)
- **Hellinger**: Symmetric distribution distance
- **Entropy Δ**: Change in prediction uncertainty
- **Token Δ**: Whether top predicted token changed (✓/✗)
- **Prob Δ**: Change in top token probability

### Detailed Analysis Output

For each top-impact token, the tool shows:

#### Impact Metrics
```
• Manhattan Distance: 0.099670
• Hellinger Distance: 0.144165  
• JS Divergence: inf
• KL Divergence: inf
• Entropy Change: +0.000000
• Top Token Changed: No
```

#### Prediction Changes
```
Baseline: ' fox' (0.948730)
With Glitch: ' fox' (0.998535)
Probability Change: +0.049805
```

#### Top 5 Predictions Comparison
```
Rank Baseline             Prob       With Glitch          Prob
----------------------------------------------------------------------
1    ' fox'               0.948730   ' fox'               0.998535
2    ' cow'               0.010704   ' Fox'               0.000284
3    ' bear'              0.002502   ' squirrel'          0.000102
```

#### Attention Pattern Analysis
```
Attention shifts for original tokens:
Token                Baseline     With Glitch  Change       Visual
---------------------------------------------------------------------------
'<|begin_of_text|>'  0.739258     0.002562     -0.736696 ↓
'The'                0.018509     0.015335     -0.003174 ↓
' quick'             0.068054     0.043060     -0.024994 ↓
' brown'             0.174316     0.166748     -0.007568 ↓

Glitch token attention: 0.002562
```

## Key Insights from Results

### 1. Probability Concentration Effect
Most glitch tokens **increase** the model's confidence in predicting "fox":
- Baseline: 94.87% confidence
- With glitch: 99.85% confidence
- **Effect**: Glitch tokens make the model more certain, not less

### 2. Attention Redistribution  
Glitch tokens cause dramatic attention shifts:
- **Massive decrease** in attention to `<|begin_of_text|>` (73.9% → 0.3%)
- **Small decreases** in attention to content words
- **Minimal attention** to the glitch token itself (~0.3%)

### 3. Distribution Flattening
While the top prediction stays the same, the probability mass redistributes:
- Top choice becomes more dominant
- Alternative options get compressed
- Overall distribution becomes more peaked

### 4. Ranking Insights
Tokens ranked by **Manhattan distance** (total variation):
- `'useRalative'` has highest impact (0.099670)
- Most impactful tokens are foreign language words
- Programming artifacts like `' CLIIIK'` also rank high

## Technical Details

### Distance Metrics Used

1. **Manhattan Distance**: `Σ|p_i - q_i|`
   - Measures total variation between distributions
   - Range: [0, 2], where 2 = completely different distributions

2. **Hellinger Distance**: `√(Σ(√p_i - √q_i)²)/√2`  
   - Symmetric, bounded metric
   - Range: [0, 1], where 1 = no overlap

3. **Jensen-Shannon Divergence**: `0.5 * (KL(p||m) + KL(q||m))`
   - Symmetric version of KL divergence
   - Where m = 0.5 * (p + q)

### Model Configuration
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Attention**: `eager` implementation (required for attention outputs)
- **Precision**: `float16` for memory efficiency

## Files Required

- `../email_llams321.json`: Glitch token classification data
- Model access to `meta-llama/Llama-3.2-1B-Instruct` (or GPT-2 fallback)

## Example Use Cases

1. **Research**: Understanding how anomalous tokens affect model behavior
2. **Safety**: Identifying tokens that significantly alter model outputs  
3. **Debugging**: Finding tokens that cause unexpected prediction changes
4. **Analysis**: Studying attention mechanism responses to out-of-distribution inputs

## Limitations

- Tests only insertion before "The quick brown" 
- Limited to first 20 tokens from classification file
- Requires GPU for reasonable performance
- Some distance metrics may show `inf` for extreme distributions

## Future Enhancements

- Test different insertion positions
- Compare multiple base phrases
- Add visualization graphs
- Support batch processing of custom token lists
- Include semantic similarity analysis