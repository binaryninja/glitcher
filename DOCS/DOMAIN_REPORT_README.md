# Domain Extraction Report Generator

A standalone HTML report generator for analyzing glitch token domain extraction test results. This tool creates comprehensive, interactive reports that can be exported to PDF for documentation and analysis.

## Overview

The Domain Extraction Report Generator analyzes JSON results from glitch token domain extraction tests and produces a professional HTML report with:

- 📊 Interactive charts and visualizations
- 📋 Detailed statistics and analysis
- 🏷️ Categorized token lists with issues
- 🎨 Modern, responsive design
- 🖨️ PDF-export ready styling

## Installation

### Requirements

- Python 3.7+
- Click library

Install dependencies:

```bash
pip install click
# or
pip install -r report_requirements.txt
```

## Usage

### Basic Usage

```bash
python generate_domain_report.py input.json output.html
```

### With Browser Auto-Open

```bash
python generate_domain_report.py input.json output.html --open-browser
```

### Example Commands

```bash
# Generate report from domain extraction results
python generate_domain_report.py email_llams321_domain_extraction.json domain_report.html

# Generate and open in browser
python generate_domain_report.py results.json report.html -o

# Test with provided data
python test_report_generator.py
```

## Input Data Format

The generator expects JSON files with the following structure:

```json
{
  "model_path": "meta-llama/Llama-3.2-1B-Instruct",
  "test_type": "domain_extraction",
  "tokens_tested": 1000,
  "tokens_breaking_extraction": 150,
  "tokens_creating_valid_domains_and_breaking": 50,
  "results": [
    {
      "token_id": 12345,
      "token": "example_token",
      "prompt": "Extract domain from: ...",
      "response": "Generated response",
      "response_length": 120,
      "is_corrupted_output": false,
      "issues": ["issue1", "issue2"],
      "creates_valid_domain": true,
      "breaks_domain_extraction": false
    }
  ]
}
```

## Report Features

### Summary Statistics

- Total tokens tested
- Count of tokens that create valid domains
- Count of tokens that break domain extraction
- Count of tokens with both behaviors
- Processing errors

### Visual Analytics

- **Distribution Chart**: Interactive doughnut chart showing token categorization
- **Response Statistics**: Average, minimum, and maximum response lengths
- **Issue Analysis**: Frequency and distribution of different issue types

### Detailed Token Analysis

The report categorizes tokens into:

1. **Valid Domains Only** - Tokens that create valid domains without breaking extraction
2. **Breaks Extraction Only** - Tokens that break extraction without creating valid domains
3. **Both Valid & Breaks** - Tokens that both create domains AND break extraction
4. **Neither** - Tokens that don't create domains or break extraction
5. **Processing Errors** - Tokens that caused processing errors

### Token Display Features

- Safe HTML rendering of special characters
- Unicode character highlighting (e.g., `\u0000`)
- Issue tagging and categorization
- Response length statistics
- Error message display

## Export to PDF

The report is designed for easy PDF export:

1. **Open the HTML file** in Chrome, Edge, or Safari
2. **Print the page** (Ctrl+P / Cmd+P)
3. **Choose "Save as PDF"** as the destination
4. **Enable "Background graphics"** in print settings for best results
5. **Adjust margins** if needed (recommend: Minimum)

### PDF Export Tips

- Use Chrome or Edge for best PDF rendering
- Enable background graphics to preserve colors and styling
- Consider using "More settings" → "Paper size: A4" for consistency
- For large reports, you may want to print specific sections

## Example Output

The generated report includes:

```
Domain Extraction Analysis Report
Model: meta-llama/Llama-3.2-1B-Instruct | Test Type: domain_extraction

Summary Cards:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Total Tokens    │ Valid Domains   │ Break Extract.  │ Both Valid &    │
│      1000       │       250       │       350       │    Breaks: 50   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

Interactive Charts:
• Token distribution pie chart
• Issue frequency analysis

Detailed Sections:
• Issue Categories with sample tokens
• Token listings by category
• Response statistics and analysis
```

## Testing

### Test with Sample Data

```bash
# Run test with your domain extraction results
python test_report_generator.py

# View sample data structure
python test_report_generator.py --sample

# Get help
python test_report_generator.py --help
```

### Example Test Run

```bash
python example_generate_report.py
```

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: input.json`
**Solution**: Ensure the input JSON file exists and the path is correct

**Issue**: `ImportError: No module named 'click'`
**Solution**: Install click with `pip install click`

**Issue**: Empty or malformed report
**Solution**: Verify input JSON structure matches expected format

**Issue**: Charts not displaying
**Solution**: Ensure internet connection for Chart.js CDN, or download for offline use

### Debugging

Enable verbose output:

```bash
# Check data structure
python test_report_generator.py --sample

# Validate JSON format
python -m json.tool input.json
```

## Customization

### Styling

The report uses embedded CSS that can be customized:

- Colors: Modify the CSS color scheme in `generate_html_report()`
- Layout: Adjust grid layouts and responsive breakpoints
- Charts: Customize Chart.js options in the JavaScript section

### Adding Features

To extend the report:

1. Modify `analyze_results()` for new analysis metrics
2. Update `generate_html_report()` for new sections
3. Add JavaScript for interactive features

## File Structure

```
glitcher/
├── generate_domain_report.py      # Main report generator
├── test_report_generator.py       # Test script
├── example_generate_report.py     # Example usage
├── report_requirements.txt        # Dependencies
└── DOMAIN_REPORT_README.md        # This file
```

## Performance

- **Small datasets** (< 1,000 tokens): < 1 second
- **Medium datasets** (1,000-10,000 tokens): 1-5 seconds  
- **Large datasets** (> 10,000 tokens): 5-30 seconds

Report file sizes:
- **Typical report**: 500KB - 2MB
- **Large reports**: 2MB - 10MB

## Integration

### Command Line Integration

```bash
# Generate domain extraction data and report in one pipeline
glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-file tokens.json --output results.json
python generate_domain_report.py results.json report.html --open-browser
```

### Scripting Integration

```python
from generate_domain_report import load_data, analyze_results, generate_html_report

# Load and analyze data
data = load_data('results.json')
analysis = analyze_results(data)
html = generate_html_report(data, analysis)

# Save report
with open('report.html', 'w') as f:
    f.write(html)
```

## License

This tool is part of the Glitcher project. See the main project license for details.

## Support

For issues and questions:

1. Check this README for common solutions
2. Run test scripts to verify setup
3. Check input data format and structure
4. Ensure all dependencies are installed

## Version History

- **v1.0.0**: Initial release with basic HTML report generation
- Features: Token categorization, issue analysis, PDF export support
- Compatible with: Glitcher domain extraction JSON format