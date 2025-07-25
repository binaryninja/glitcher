#!/usr/bin/env python3
"""
Test script for the domain extraction report generator.

This script tests the report generator using the provided domain extraction data
and creates a comprehensive HTML report.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

def test_report_generation():
    """Test the report generation with the provided data."""

    print("🧪 Testing Domain Extraction Report Generator")
    print("=" * 50)

    # Input and output files
    input_file = "email_llams321_domain_extraction.json"
    output_file = f"domain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Error: Input file '{input_file}' not found")
        print("Please ensure the domain extraction JSON file is in the current directory.")
        return False

    try:
        # Import the report generator
        from generate_domain_report import load_data, analyze_results, generate_html_report

        print(f"📂 Loading data from: {input_file}")
        data = load_data(input_file)

        print(f"✅ Data loaded successfully")
        print(f"   Model: {data.get('model_path', 'Unknown')}")
        print(f"   Test Type: {data.get('test_type', 'Unknown')}")
        print(f"   Results Count: {len(data.get('results', []))}")

        print("🔍 Analyzing results...")
        analysis = analyze_results(data)

        print("📊 Analysis Summary:")
        print(f"   Total Tokens: {analysis['total_tokens']}")
        print(f"   Valid Domains: {analysis['creates_valid_domains']}")
        print(f"   Breaks Extraction: {analysis['breaks_domain_extraction']}")
        print(f"   Both Valid & Breaks: {analysis['both_valid_and_breaks']}")
        print(f"   Neither: {analysis['neither']}")
        print(f"   Errors: {analysis['has_errors']}")
        print(f"   Issue Categories: {len(analysis['issue_categories'])}")

        print("🎨 Generating HTML report...")
        html_content = generate_html_report(data, analysis)

        print(f"💾 Saving report to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Verify file was created and get size
        output_path = Path(output_file)
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"✅ Report generated successfully!")
            print(f"   File size: {file_size:,} bytes")
            print(f"   HTML length: {len(html_content):,} characters")

            # Show top issues
            if analysis['issue_categories']:
                print("\n🏷️  Top Issues Found:")
                for issue, count in list(analysis['issue_categories'].most_common(5)):
                    print(f"   • {issue}: {count} occurrences")

            print(f"\n🌐 To view the report:")
            print(f"   1. Open {output_file} in your web browser")
            print(f"   2. Or run: python -m webbrowser {output_file}")

            print(f"\n📄 To export to PDF:")
            print(f"   1. Open {output_file} in Chrome/Edge")
            print(f"   2. Press Ctrl+P (Cmd+P on Mac)")
            print(f"   3. Choose 'Save as PDF'")
            print(f"   4. Enable 'Background graphics' in More settings")

            return True
        else:
            print("❌ Error: Report file was not created")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure generate_domain_report.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_sample_data():
    """Show a sample of the data structure for debugging."""
    input_file = "email_llams321_domain_extraction.json"

    if not Path(input_file).exists():
        print(f"❌ Cannot show sample data: {input_file} not found")
        return

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        print("\n📋 Sample Data Structure:")
        print("-" * 30)

        # Show overall structure
        print(f"Top-level keys: {list(data.keys())}")

        results = data.get('results', [])
        if results:
            print(f"Total results: {len(results)}")
            print(f"Sample result keys: {list(results[0].keys())}")

            # Show a few sample tokens
            print(f"\n🎯 Sample Tokens:")
            for i, result in enumerate(results[:3]):
                token_id = result.get('token_id', 'Unknown')
                token = result.get('token', 'Unknown')
                creates_domain = result.get('creates_valid_domain', False)
                breaks_extraction = result.get('breaks_domain_extraction', False)

                print(f"   {i+1}. Token ID: {token_id}")
                print(f"      Token: {repr(token)}")
                print(f"      Creates Domain: {creates_domain}")
                print(f"      Breaks Extraction: {breaks_extraction}")
                print()

    except Exception as e:
        print(f"❌ Error reading sample data: {e}")

if __name__ == "__main__":
    print("🚀 Starting Domain Extraction Report Generator Test")
    print()

    # Show command line options
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sample":
            show_sample_data()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python test_report_generator.py          # Generate report")
            print("  python test_report_generator.py --sample # Show sample data")
            print("  python test_report_generator.py --help   # Show this help")
            sys.exit(0)

    # Run the test
    success = test_report_generation()

    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
