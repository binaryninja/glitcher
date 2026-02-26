# Appendix B: Reproducibility Checklist

This directory contains scripts and templates for ensuring reproducibility of
glitcher experiments. Each item below maps to a concrete artifact or tool.

## Checklist

| # | Item | Artifact |
|---|------|----------|
| 1 | Record model + tokenizer identifiers and exact versions | `collect_environment.py` auto-captures; `reproducibility_config.json` template |
| 2 | Log decoding settings (temperature/top-p/max tokens/stop sequences) | `reproducibility_config.json` > `decoding` section |
| 3 | Fix random seeds where supported; record non-deterministic components | `collect_environment.py` sets seeds; `reproducibility_config.json` > `seeds` |
| 4 | Keep a stable set of neutral probe templates | `probe_templates.py` (frozen copy of the 3 verification prompts) |
| 5 | Include a random control sample of non-candidate tokens | `control_tokens.json` |
| 6 | Document hardware/runtime differences (GPU vs CPU, kernels, quantization) | `collect_environment.py` auto-captures hardware info |
| 7 | Record quantization level (int4/int8/fp16/bf16) when using local models | `reproducibility_config.json` > `hardware.quantization` |
| 8 | Document ASR threshold and number of validation attempts | `reproducibility_config.json` > `validation` section |
| 9 | Redact sensitive tokens in public reports; keep raw artifacts private | `redact.py` utility |
| 10 | Include genetic algorithm parameters if using combination search | `reproducibility_config.json` > `genetic_algorithm` section |

## Quick Start

```bash
# 1. Collect environment snapshot before running experiments
python APPENDIX-B/collect_environment.py --model meta-llama/Llama-3.2-1B-Instruct \
    --quant-type bfloat16 \
    --output APPENDIX-B/snapshot.json

# 2. Verify a previous snapshot matches current environment
python APPENDIX-B/verify_reproducibility.py APPENDIX-B/snapshot.json

# 3. Redact sensitive tokens from a results file
python APPENDIX-B/redact.py glitch_tokens.json --output glitch_tokens_redacted.json

# 4. View frozen probe templates
python APPENDIX-B/probe_templates.py --list
```

## File Inventory

| File | Purpose |
|------|---------|
| `README.md` | This checklist and usage guide |
| `reproducibility_config.json` | Template with all parameter defaults |
| `collect_environment.py` | Auto-collects model, hardware, and library versions |
| `verify_reproducibility.py` | Compares current env against a saved snapshot |
| `probe_templates.py` | Frozen copy of the 3 neutral verification prompts |
| `control_tokens.json` | Standard non-glitch control tokens for baseline testing |
| `redact.py` | Redacts sensitive token text from JSON result files |
