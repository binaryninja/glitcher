# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Instructions

- Never use emojis in code
- When committing changes, ensure that the commit message is clear and concise. You have access to gh cli tools
- Always use ssh for pushing changes

## Setup

```bash
pip install -e .
pip install accelerate bitsandbytes matplotlib
```

For dev tooling: `pip install -e ".[dev]"`

## Commands

### CLI entry points (defined in setup.py)

```bash
glitcher          # Main CLI (glitcher.cli:main)
glitch-scan       # Scan and validate (glitcher.scan_and_validate:main)
glitch-classify   # Classify glitch tokens (glitcher.classify_glitches:main)
```

### Key subcommands

```bash
glitcher mine MODEL --num-iterations 50 --batch-size 8 --k 32
glitcher test MODEL --token-ids 89472,127438 --enhanced --num-attempts 3
glitcher compare MODEL --token-ids 89472,127438 --num-attempts 5
glitcher genetic MODEL --base-text "The quick brown" --generations 50
glitcher genetic MODEL --wanted-token "fox" --comprehensive-search --gui
glitcher genetic MODEL --baseline-only --ascii-only
glitcher classify MODEL --token-file glitch_tokens.json
glitcher chat MODEL 89472 --max-size 20
glitcher domain MODEL --token-ids 89472,127438
glitcher gui
```

Use `glitcher <subcommand> --help` for full parameter documentation.

### Running tests

Tests are standalone scripts (not pytest-compatible). Run them directly with `python`:

```bash
# Unit tests (require GPU + model)
python tests/unit/test_genetic_probabilities.py meta-llama/Llama-3.2-1B-Instruct
python tests/unit/test_glitch_error_detection.py
python tests/unit/test_next_token_probability.py meta-llama/Llama-3.2-1B-Instruct
python tests/unit/test_proper_context.py meta-llama/Llama-3.2-1B-Instruct

# Integration tests (require GPU + model)
python tests/integration/test_enhanced_classification.py
python tests/integration/test_genetic_integration.py
python tests/integration/test_report_generator.py
python tests/integration/test_reports.py
```

Note: many test files also exist in the project root (e.g., `test_comprehensive_search.py`) and under `poc/` - these are standalone scripts run directly with `python`.

### Linting

```bash
flake8 glitcher/
black glitcher/ --check
black glitcher/  # auto-format
```

## Architecture

### Core package: `glitcher/`

```
glitcher/
  cli.py                  # Main CLI with subcommands (mine, test, genetic, gui, etc.)
  model.py                # Core mining engine: entropy calculation, embedding search,
                          #   token validation, chat template management, Harmony support
  enhanced_validation.py  # Multi-attempt ASR validation via sequence generation
  classify_glitches.py    # Token classifier (email/domain extraction, behavioral tests)
  genetic/
    reducer.py            # GeneticProbabilityReducer - evolutionary token combination search
    batch_runner.py       # Batch experiments across multiple scenarios
    gui_animator.py       # Real-time matplotlib visualization
    gui_controller.py     # Full tkinter GUI with tabs (config/control/progress/results)
  classification/
    glitch_classifier.py  # Modular classifier with pluggable test categories
    base_classifier.py    # Abstract base for custom classifiers
    types.py              # Shared type definitions
  tests/
    email_tests.py        # Email extraction test module
    domain_tests.py       # Domain extraction test module
  utils/
    json_utils.py         # JSON parsing helpers
    validation_utils.py   # Validation helpers
```

### Multi-provider framework: `poc/providers/`

Provider registry pattern for testing across different LLM APIs:

```
poc/providers/
  base.py                   # BaseProvider abstract class
  openai_provider.py        # OpenAI / ChatGPT
  anthropic_provider.py     # Claude
  mistral.py                # Mistral
  openrouter_provider.py    # OpenRouter
  transformers_provider.py  # Local HuggingFace models (int4/int8/fp16 quantization)
  lambda_ai.py              # Lambda Labs
```

Usage: `from poc.providers import get_provider, list_available_providers`

### Data flow

1. **Mining** (`glitcher mine`): Low L2 norm embeddings -> entropy-guided gradient search -> multi-prompt validation -> `glitch_tokens.json`
2. **Classification** (`glitch-classify`): glitch_tokens.json -> email/domain/behavioral tests -> classification results
3. **Genetic optimization** (`glitcher genetic`): glitch_tokens.json -> baseline token impact analysis -> population seeding -> evolutionary search for token combinations that maximize probability reduction
4. **Comprehensive search** (`--comprehensive-search`): Full vocabulary scan for wanted token optimization, results cached in `cache/comprehensive_search/`

### Key concepts

- **Glitch tokens**: Anomalous tokens with low prediction probability that cause unexpected model behavior (hallucinations, loops, safety bypasses)
- **ASR (Attack Success Rate)**: Percentage of validation attempts where a token exhibits glitch behavior. Default threshold: 0.5. Range: 0.0 (any glitch) to 1.0 (always glitches)
- **Enhanced validation**: Generates 30-100+ tokens and checks for target token appearance across multiple attempts, reducing false positives from ~30-50% to ~5-15%
- **Harmony support**: Structured generation for gpt-oss models with channel-based output parsing (analysis/commentary/final) and reasoning effort levels

### Key output files

- `glitch_tokens.json` - Discovered glitch tokens with metadata (token_id, token_text, l2_norm, entropy, asr, method)
- `glitch_progress.json` - Resumable mining progress
- `genetic_results.json` - Evolved token combinations
- `token_impact_baseline.json` - Individual token effectiveness rankings

## Code Style

- PEP 8, 100 char line width
- Type hints on function parameters and return values
- Bilingual docstrings (English and Chinese)
- Import order: stdlib, third-party, local
- snake_case for functions/variables, PascalCase for classes
