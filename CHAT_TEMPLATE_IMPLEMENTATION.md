# Chat Template Implementation for Genetic Algorithm

## Summary

This document describes the implementation of proper chat template support in the genetic algorithm for glitch token breeding. The changes ensure that instruct-tuned models are used correctly with their intended conversation formatting instead of raw text completion.

## Problem Identified

### Original Issue
The genetic algorithm was using instruct-tuned models (like `meta-llama/Llama-3.2-1B-Instruct`) incorrectly by:

1. **No Chat Template Usage**: Sending raw text directly to instruct models instead of using proper conversation formatting
2. **Context Mismatch**: Testing text completion on models trained for conversational interaction
3. **Inconsistent Framework**: The `TransformersProvider` had chat template support, but the genetic algorithm bypassed it entirely

### Impact
- Instruct models received improper input format: `"({glitch_tokens}): The quick brown"`
- Instead of expected format: Properly formatted conversation with system prompt and user message
- This led to suboptimal probability measurements and reduced effectiveness

## Solution Implemented

### Key Changes

#### 1. **Chat Template Detection and Setup**
```python
def _setup_chat_template(self):
    """Set up chat template and system prompt for instruct models."""
    # Check if this is an instruct model
    model_name_lower = self.model_name.lower()
    self.is_instruct_model = any(keyword in model_name_lower for keyword in
                               ['instruct', 'chat', 'assistant', 'conv'])

    if self.is_instruct_model:
        # Create system prompt for continuation task
        self.system_prompt = (
            "You are a text completion assistant. Your task is to continue sequences of text. "
            "When the user provides text, you should predict and output the most likely next word(s) "
            "that would naturally follow in the sequence. Respond with only the continuation, "
            "without any additional explanation or formatting."
        )
```

#### 2. **Input Formatting Based on Model Type**
```python
def _format_input_for_model(self, modified_text: str) -> str:
    """Format input text according to model type (instruct vs base model)."""
    if not self.is_instruct_model or not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
        # For base models or models without chat templates, use direct text
        return modified_text

    # For instruct models, format as conversation
    messages = []
    if self.system_prompt:
        messages.append({"role": "system", "content": self.system_prompt})

    messages.append({"role": "user", "content": modified_text})

    try:
        # Use the model's chat template
        formatted_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_input
    except Exception as e:
        self.logger.warning(f"Failed to apply chat template: {e}, falling back to direct text")
        return modified_text
```

#### 3. **Proper Glitch Token Integration**
For instruct models:
- **Before**: `"({glitch_tokens}): The quick brown"`
- **After**: User message contains `"{glitch_tokens} The quick brown"`

#### 4. **Correct Probability Measurement**
- Probabilities are measured at the last position (assistant response start)
- This is correct for both instruct and base models
- The formatting ensures we're measuring what the assistant would say next

## Technical Details

### Model Type Detection
```python
self.is_instruct_model = any(keyword in model_name_lower for keyword in
                           ['instruct', 'chat', 'assistant', 'conv'])
```

### System Prompt Design
The system prompt is specifically crafted to guide instruct models to perform text continuation:
```
"You are a text completion assistant. Your task is to continue sequences of text. 
When the user provides text, you should predict and output the most likely next word(s) 
that would naturally follow in the sequence. Respond with only the continuation, 
without any additional explanation or formatting."
```

### Chat Template Application
For Llama-3.2-1B-Instruct, the formatted input becomes:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2025

You are a text completion assistant. Your task is to continue sequences of text. When the user provides text, you should predict and output the most likely next word(s) that would naturally follow in the sequence. Respond with only the continuation, without any additional explanation or formatting.<|eot_id|><|start_header_id|>user<|end_header_id|>

The quick brown<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

## Results and Improvements

### Effectiveness Comparison
Testing with `meta-llama/Llama-3.2-1B-Instruct` on "The quick brown" → "fox":

| Metric | Old Method (Raw Text) | New Method (Chat Template) | Improvement |
|--------|----------------------|---------------------------|-------------|
| Baseline "fox" probability | 0.000184 (0.02%) | 0.910156 (91.02%) | **4,943x higher** |
| With glitch tokens | 0.000127 (0.01%) | 0.117798 (11.78%) | **925x higher** |
| Probability reduction | 30.87% | 87.06% | **+56.18% points** |

### Key Findings

1. **Massive Baseline Improvement**: Chat template format produces dramatically higher baseline probabilities for the expected "fox" completion
2. **Better Glitch Token Effectiveness**: Glitch tokens work much more effectively in the proper conversational context
3. **Significant Reduction Improvement**: Overall probability reduction improved by over 56 percentage points

### System Prompt Impact
Testing different system prompts shows the importance of task-specific guidance:

| System Prompt | "fox" Probability |
|--------------|-------------------|
| None (raw text) | 0.02% |
| Generic assistant | 0.00% |
| **Text completion assistant** | **91.02%** |
| Creative writing assistant | 7.24% |

## Testing and Validation

### Test Scripts Created
1. **`test_chat_template_fix.py`**: Basic functionality testing
2. **`demo_chat_template_fix.py`**: Comprehensive comparison demonstration

### Validation Results
```bash
# Test basic functionality
python test_chat_template_fix.py meta-llama/Llama-3.2-1B-Instruct

# Run comprehensive demonstration
python demo_chat_template_fix.py meta-llama/Llama-3.2-1B-Instruct

# Test with genetic algorithm
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --base-text "The quick brown" --target-token "fox" \
  --generations 5 --population-size 10 \
  --token-file email_extraction_all_glitches.json
```

## Backward Compatibility

### Base Models
- Models without "instruct", "chat", "assistant", or "conv" in their names continue using raw text
- No change in behavior for base language models

### Fallback Mechanism
- If chat template application fails, system falls back to raw text
- Ensures robustness across different model types and configurations

### Legacy Support
- All existing functionality preserved
- No breaking changes to CLI interface or API

## Usage Examples

### Basic Genetic Algorithm
```bash
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --base-text "The quick brown" \
  --target-token "fox" \
  --generations 50 \
  --population-size 30
```

### Baseline Analysis
```bash
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --baseline-only \
  --base-text "Hello world" \
  --target-token "today"
```

### With ASCII Filtering
```bash
glitcher genetic meta-llama/Llama-3.2-1B-Instruct \
  --ascii-only \
  --base-text "The quick brown" \
  --target-token "fox"
```

## Future Enhancements

### Potential Improvements
1. **Custom System Prompts**: Allow users to specify custom system prompts for specific tasks
2. **Multi-turn Conversations**: Support for multi-turn conversation contexts
3. **Provider Integration**: Better integration with the existing multi-provider framework
4. **Template Validation**: More robust chat template detection and validation

### Model Support
- Currently tested with Llama-3.2-1B-Instruct
- Should work with other instruct models using standard chat templates
- Future testing planned for Mistral, GPT-style, and other model families

## Conclusion

The chat template implementation represents a significant improvement in how the genetic algorithm interfaces with instruct-tuned language models. By using proper conversation formatting and task-specific system prompts, we achieve:

- **56+ percentage point improvement** in probability reduction effectiveness
- **Proper model usage** according to training methodology
- **Maintained compatibility** with base models
- **Robust fallback mechanisms** for edge cases

This change ensures that the glitcher framework correctly leverages the capabilities of modern instruct-tuned models while maintaining support for traditional language models.