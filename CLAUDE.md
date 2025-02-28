# GlitchMiner Development Guide

## Installation and Setup
```bash
# Install package in development mode
pip install -e .
pip install accelerate  # Required for loading models with device_map
```

## Common Commands
```bash
# Run glitch token mining
glitcher mine meta-llama/Llama-3.2-1B --num-iterations 50 --batch-size 8 --k 32

# Test specific token IDs
glitcher test meta-llama/Llama-3.2-1B --token-ids 89472,127438,85069

# Chat test with a specific token
glitcher chat meta-llama/Llama-3.2-1B 89472 --max-size 20

# Run validation tests
glitcher validate meta-llama/Llama-3.2-1B --output-dir validation_results

# Test known glitch tokens
python test_known_glitches.py meta-llama/Llama-3.2-1B --token-file glitch_tokens.json

# Run token repetition test
python token_repetition_test.py meta-llama/Llama-3.2-1B
```

## Code Style Guidelines
- Follow PEP 8 conventions for Python code
- Import order: standard library, third-party packages, local modules
- Type hints required for function parameters and return values
- Naming: snake_case for functions/variables, PascalCase for classes
- Include bilingual docstrings (English and Chinese)
- Wrap code at 100 characters per line
- Use explicit error handling with informative messages
- Maintain clean module structure with clear separation of concerns