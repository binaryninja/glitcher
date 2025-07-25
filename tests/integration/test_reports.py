#!/usr/bin/env python3
"""
Simple test script to verify both report generators work correctly.

This script generates both optimized and enhanced reports and checks for common issues.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def test_report_generator(script_name, input_file, output_file):
    """Test a report generator and return results."""
    print(f"Testing {script_name}...")

    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, script_name, input_file, output_file
        ], capture_output=True, text=True, timeout=60)

        generation_time = time.time() - start_time

        if result.returncode == 0:
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)

                # Basic validation - check if file contains essential elements
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                has_html_structure = all(tag in content for tag in ['<html', '<body', '</html>'])
                has_javascript = 'filterTokens' in content
                has_charts = 'Chart' in content
                has_data = 'reportData' in content

                return {
                    'success': True,
                    'generation_time': generation_time,
                    'file_size': file_size,
                    'file_size_kb': file_size / 1024,
                    'has_html_structure': has_html_structure,
                    'has_javascript': has_javascript,
                    'has_charts': has_charts,
                    'has_data': has_data,
                    'output': result.stdout,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'error': 'Output file was not created',
                    'generation_time': generation_time
                }
        else:
            return {
                'success': False,
                'error': result.stderr,
                'generation_time': generation_time,
                'output': result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Generation timed out after 60 seconds',
            'generation_time': 60
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'generation_time': 0
        }

def print_test_results(name, result):
    """Print formatted test results."""
    print(f"\n{name} Results:")
    print("-" * 40)

    if result['success']:
        print(f"✅ Status: SUCCESS")
        print(f"⏱️  Generation Time: {result['generation_time']:.2f}s")
        print(f"📁 File Size: {result['file_size_kb']:.1f} KB")
        print(f"🏗️  HTML Structure: {'✅' if result['has_html_structure'] else '❌'}")
        print(f"📜 JavaScript: {'✅' if result['has_javascript'] else '❌'}")
        print(f"📊 Charts: {'✅' if result['has_charts'] else '❌'}")
        print(f"🗃️  Data: {'✅' if result['has_data'] else '❌'}")

        # Overall health check
        all_checks = all([
            result['has_html_structure'],
            result['has_javascript'],
            result['has_charts'],
            result['has_data']
        ])
        print(f"🔍 Overall Health: {'✅ HEALTHY' if all_checks else '⚠️  ISSUES DETECTED'}")

    else:
        print(f"❌ Status: FAILED")
        print(f"⏱️  Generation Time: {result.get('generation_time', 0):.2f}s")
        print(f"🚨 Error: {result['error']}")
        if result.get('output'):
            print(f"📄 Output: {result['output']}")

def main():
    """Main test function."""
    print("🧪 REPORT GENERATOR TEST SUITE")
    print("=" * 50)

    # Check if test data exists
    test_files = [
        'email_llams321_email_extraction.json',
        'email_llams321_domain_extraction.json',
        'classified_tokens_email_extraction.json'
    ]

    input_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            input_file = test_file
            break

    if not input_file:
        print("❌ No test data found. Please ensure one of these files exists:")
        for test_file in test_files:
            print(f"   - {test_file}")
        sys.exit(1)

    print(f"📊 Using test data: {input_file}")

    # Load test data info
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results_count = len(data.get('results', []))
        model_path = data.get('model_path', 'Unknown')
        print(f"🤖 Model: {model_path}")
        print(f"🔢 Test Tokens: {results_count:,}")

    except Exception as e:
        print(f"⚠️  Could not read test data details: {e}")

    # Test both generators
    timestamp = int(time.time())

    # Test optimized generator
    optimized_output = f"test_optimized_{timestamp}.html"
    optimized_result = test_report_generator(
        'generate_domain_report_optimized.py',
        input_file,
        optimized_output
    )

    # Test enhanced generator
    enhanced_output = f"test_enhanced_{timestamp}.html"
    enhanced_result = test_report_generator(
        'generate_enhanced_report.py',
        input_file,
        enhanced_output
    )

    # Print results
    print_test_results("OPTIMIZED GENERATOR", optimized_result)
    print_test_results("ENHANCED GENERATOR", enhanced_result)

    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print("=" * 50)

    optimized_success = optimized_result['success']
    enhanced_success = enhanced_result['success']

    if optimized_success and enhanced_success:
        print("🎉 ALL TESTS PASSED!")

        opt_size = optimized_result['file_size_kb']
        enh_size = enhanced_result['file_size_kb']
        size_increase = ((enh_size - opt_size) / opt_size * 100) if opt_size > 0 else 0

        print(f"\n📊 Comparison:")
        print(f"   Optimized: {opt_size:.1f} KB")
        print(f"   Enhanced:  {enh_size:.1f} KB (+{size_increase:.1f}%)")

        # Recommend which to use
        if results_count < 500:
            print(f"\n💡 Recommendation for {results_count} tokens: Either version works well")
        elif results_count < 1000:
            print(f"\n💡 Recommendation for {results_count} tokens: Enhanced version for analysis")
        else:
            print(f"\n💡 Recommendation for {results_count} tokens: Enhanced version strongly recommended")

        print(f"\n🌐 Generated Reports:")
        print(f"   📁 Optimized: file://{Path(optimized_output).absolute()}")
        print(f"   📁 Enhanced:  file://{Path(enhanced_output).absolute()}")

    elif optimized_success:
        print("⚠️  PARTIAL SUCCESS - Optimized generator works, Enhanced has issues")
        print(f"   📁 Working report: file://{Path(optimized_output).absolute()}")

    elif enhanced_success:
        print("⚠️  PARTIAL SUCCESS - Enhanced generator works, Optimized has issues")
        print(f"   📁 Working report: file://{Path(enhanced_output).absolute()}")

    else:
        print("❌ ALL TESTS FAILED - Both generators have issues")
        print("   Check error messages above for troubleshooting")

    # Cleanup option
    print(f"\n🧹 Cleanup:")
    print(f"   To remove test files: rm test_*_{timestamp}.html")

    print("=" * 50)

if __name__ == '__main__':
    main()
