# Repository Structure Guide

This document describes the organized structure of the Glitcher repository after reorganization.

## Overview

The repository has been restructured to improve organization, maintainability, and developer experience. All files are now logically grouped by function and type.

## Directory Structure

```
glitcher/
├── README.md                 # Main project README
├── CLAUDE.md                 # Claude-specific development guide
├── setup.py                  # Package setup configuration
├── 
├── DOCS/                     # All documentation
├── glitcher/                 # Main source code package
├── tests/                    # Organized test suites
├── scripts/                  # Utility and standalone scripts
├── outputs/                  # All generated outputs and results
├── examples/                 # Example code and demonstrations
├── poc/                      # Proof of concept implementations
├── tools/                    # Development tools
├── assets/                   # Static assets (images, etc.)
└── json_backup/              # Historical JSON backups
```

## Detailed Directory Descriptions

### `DOCS/` - Documentation
Contains all Markdown documentation files organized by topic:

- **`ASR_IMPLEMENTATION.md`** - Attack Success Rate implementation details
- **`CLASSIFICATION_IMPROVEMENTS.md`** - Token classification enhancements
- **`DEVELOPERS_GUIDE.md`** - Comprehensive development guide
- **`DOMAIN_REPORT_README.md`** - Domain extraction reporting
- **`EMAIL_EXTRACTION_TESTING.md`** - Email extraction testing documentation
- **`ENHANCED_CLASSIFICATION_GUIDE.md`** - Enhanced classification features
- **`ENHANCED_MINING_README.md`** - Enhanced mining algorithm documentation
- **`GENETIC_ALGORITHM_GUIDE.md`** - Genetic algorithm usage and theory
- **`GENETIC_INTEGRATION.md`** - Genetic algorithm integration details
- **`GENETIC_INTEGRATION_SUMMARY.md`** - Summary of genetic integration
- **`GUI_ENHANCEMENTS.md`** - GUI enhancement documentation
- **`GUI_INTEGRATION_SUMMARY.md`** - GUI integration summary
- **`INSTRUCT_MODEL_CLEANUP.md`** - Instruct model cleanup procedures
- **`MINE.md`** - Mining algorithm documentation
- **`MIXTRAL_INTEGRATION.md`** - Mixtral model integration
- **`MODULAR_README.md`** - Modular architecture documentation
- **`PROVIDER_AGNOSTIC_CHANGES.md`** - Provider-agnostic implementation
- **`README.md`** - Documentation index
- **`REFACTORING_SUMMARY.md`** - Code refactoring summary
- **`REPORT_GENERATOR_SUMMARY.md`** - Report generation features
- **`REPORT_OPTIMIZATION_GUIDE.md`** - Report optimization techniques
- **`VALIDATION_SYSTEM_GUIDE.md`** - Validation system documentation

### `tests/` - Test Organization
Organized test suites by category:

#### `tests/unit/` - Unit Tests
- `test_asr_return.py` - ASR return value testing
- `test_chat_template.py` - Chat template testing
- `test_genetic_probabilities.py` - Genetic probability calculations
- `test_glitch_error_detection.py` - Error detection testing
- `test_next_token_probability.py` - Next token probability testing
- `test_proper_context.py` - Context handling testing

#### `tests/integration/` - Integration Tests
- `test_domain_extraction.py` - Domain extraction integration
- `test_email_classification.py` - Email classification integration
- `test_email_extraction.py` - Email extraction integration
- `test_enhanced_classification.py` - Enhanced classification integration
- `test_enhanced_mining.py` - Enhanced mining integration
- `test_enhanced_validation.py` - Enhanced validation integration
- `test_genetic_integration.py` - Genetic algorithm integration
- `test_gui_integration.py` - GUI integration testing
- `test_integrated_range_mining.py` - Range mining integration
- `test_json_extraction.py` - JSON extraction integration
- `test_mixtral_finetune.py` - Mixtral fine-tuning integration
- `test_mixtral_integration.py` - Mixtral model integration
- `test_report_generator.py` - Report generation integration
- `test_reports.py` - Report system integration

#### `tests/demos/` - Demo and Visualization Tests
- `demo_enhanced_classification.py` - Enhanced classification demo
- `demo_enhanced_gui_strings.py` - GUI string enhancement demo
- `demo_genetic_gui.py` - Genetic algorithm GUI demo
- `demo_report_comparison.py` - Report comparison demo
- `test_enhanced_gui.py` - Enhanced GUI testing
- `validation_explanation_demo.py` - Validation explanation demo

### `scripts/` - Utility Scripts
Standalone scripts and utilities:

- **Mining Scripts:**
  - `pattern_mining.py` - Pattern-based mining
  - `range_mining.py` - Range-based mining
  
- **Report Generation:**
  - `generate_domain_report.py` - Domain report generation
  - `generate_domain_report_optimized.py` - Optimized domain reports
  - `generate_enhanced_report.py` - Enhanced report generation
  
- **Data Processing:**
  - `consolidate_results.py` - Result consolidation
  - `find_low_norm_tokens.py` - Low norm token discovery
  
- **Validation:**
  - `validate_email_extraction.py` - Email extraction validation
  - `validate_existing_tokens.py` - Token validation
  
- **Analysis:**
  - `investigate_quick_brown.py` - Quick brown fox analysis
  - `run_deep_scan.py` - Deep scanning operations
  
- **Examples:**
  - `example_modular_usage.py` - Modular usage examples

### `outputs/` - Generated Outputs
All generated files organized by type:

#### `outputs/logs/` - Log Files
- `*.jsonl` - JSON line format logs
- `*.log` - Standard log files
- `chat` - Chat session logs

#### `outputs/reports/` - Generated Reports
- `*.html` - HTML report files
- Domain extraction reports
- Email extraction reports
- Classification reports

#### `outputs/json_data/` - JSON Data Files
- Token classification results
- Validation results
- Mining results
- Extraction results
- Test results

#### `outputs/mining_results/` - Mining Outputs
- Range mining results
- Pattern mining results
- Token mining results

#### `outputs/genetic_results/` - Genetic Algorithm Results
- `genetic_batch_results/` - Batch experiment results
- `genetic_visualizations/` - Visualization outputs
- Genetic algorithm evolution data

#### `outputs/demo_results/` - Demo Outputs
- Demonstration results and outputs

### `glitcher/` - Main Source Code
Core package implementation:
- **`__init__.py`** - Package initialization
- **`model.py`** - Model handling
- **`classification/`** - Classification modules
- **`genetic/`** - Genetic algorithm implementation
- **`tests/`** - Package-level tests
- **`utils/`** - Utility modules

### `examples/` - Example Code
Working examples and demonstrations:
- Genetic algorithm examples
- Analysis examples
- Visualization examples
- Impact analysis tools

### `poc/` - Proof of Concept
Experimental implementations:
- Multi-provider support
- Provider-specific implementations
- Experimental features

## Usage Guidelines

### For Developers
1. **Documentation**: Check `DOCS/` for comprehensive guides
2. **Testing**: Use organized test suites in `tests/`
3. **Scripts**: Utilize `scripts/` for standalone operations
4. **Examples**: Reference `examples/` for implementation patterns

### For Researchers
1. **Results**: Find all outputs in `outputs/` organized by type
2. **Analysis**: Use scripts in `scripts/` for data analysis
3. **Experiments**: Check `outputs/genetic_results/` for genetic algorithm experiments
4. **Logs**: Examine `outputs/logs/` for detailed operation logs

### For Users
1. **Getting Started**: Read main `README.md` and `DOCS/README.md`
2. **Examples**: Start with `examples/` and `tests/demos/`
3. **Reports**: View results in `outputs/reports/`
4. **Data**: Access processed data in `outputs/json_data/`

## File Naming Conventions

### Log Files
- `glitch_mining_log_<timestamp>.jsonl` - Mining operation logs
- `range_mining_log_<timestamp>.jsonl` - Range mining logs
- `token_validation_<timestamp>.jsonl` - Validation logs
- `verification_log_<timestamp>.jsonl` - Verification logs

### Report Files
- `*_report_<timestamp>.html` - Generated HTML reports
- `enhanced_*_report.html` - Enhanced report versions
- `optimized_*_report.html` - Optimized report versions

### Data Files
- `*_results.json` - Analysis results
- `*_tokens.json` - Token data files
- `validated_tokens_<timestamp>.json` - Validation results
- `glitch_*.json` - Glitch-related data

## Migration Notes

### From Old Structure
- All `.md` files moved from root to `DOCS/`
- Test files organized by type in `tests/`
- Generated outputs consolidated in `outputs/`
- Utility scripts consolidated in `scripts/`

### Path Updates Needed
If you have scripts or configurations referencing old paths, update them according to this new structure.

## Maintenance

### Adding New Files
- **Documentation**: Add to `DOCS/` with descriptive names
- **Tests**: Place in appropriate `tests/` subdirectory
- **Scripts**: Add to `scripts/` with clear naming
- **Outputs**: Will be generated in `outputs/` automatically

### Cleaning Up
- **Logs**: Regularly archive old logs from `outputs/logs/`
- **Results**: Archive old results from `outputs/`
- **Backups**: Maintain `json_backup/` for historical data

This structure provides better organization, easier navigation, and clearer separation of concerns for the Glitcher project.