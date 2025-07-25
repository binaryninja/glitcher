#!/usr/bin/env python3
"""
Example usage script for the domain extraction report generator.

This demonstrates how to generate HTML reports from domain extraction test results.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_domain_report import main as generate_report
import click

def run_example():
    """Run example report generation."""

    # Example input file (the one provided in the context)
    input_file = "email_llams321_domain_extraction.json"

    # Output file
    output_file = "domain_extraction_report.html"

    print("=== Domain Extraction Report Generator Example ===")
    print()

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Please ensure you have the domain extraction results JSON file in the current directory.")
        print()
        print("You can generate this file using:")
        print("glitcher domain meta-llama/Llama-3.2-1B-Instruct --token-file glitch_tokens.json")
        return

    print(f"📂 Input file: {input_file}")
    print(f"📄 Output file: {output_file}")
    print()

    try:
        # Use click's testing runner to simulate command line
        from click.testing import CliRunner
        runner = CliRunner()

        # Run the report generator
        result = runner.invoke(generate_report, [input_file, output_file, '--open-browser'])

        if result.exit_code == 0:
            print("✅ Report generated successfully!")
            print(f"📊 Open {output_file} in your browser to view the report")
            print()
            print("Features of the report:")
            print("• 📈 Interactive charts showing token distribution")
            print("• 📋 Detailed statistics and analysis")
            print("• 🏷️  Categorized token lists with issues")
            print("• 🎨 Modern, responsive design")
            print("• 🖨️  PDF-export ready styling")
            print()
            print("To export to PDF:")
            print("1. Open the HTML file in Chrome/Edge")
            print("2. Press Ctrl+P (Cmd+P on Mac)")
            print("3. Choose 'Save as PDF' as destination")
            print("4. Select 'More settings' and enable 'Background graphics'")
        else:
            print(f"❌ Error generating report: {result.output}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_example()
