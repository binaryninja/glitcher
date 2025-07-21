#!/usr/bin/env python3
"""
Demo Script: Report Generator Comparison

This script demonstrates the differences between the optimized and enhanced
report generators by generating both versions and providing performance metrics.

Usage:
    python demo_report_comparison.py input.json
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess

def get_file_size(filepath):
    """Get file size in bytes and human-readable format."""
    if not os.path.exists(filepath):
        return 0, "0 B"

    size_bytes = os.path.getsize(filepath)

    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            size_human = f"{size_bytes:.1f} {unit}"
            break
        size_bytes /= 1024.0
    else:
        size_human = f"{size_bytes:.1f} TB"

    return os.path.getsize(filepath), size_human

def load_json_data(filepath):
    """Load and validate JSON data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def run_report_generator(script_path, input_file, output_file):
    """Run a report generator and measure performance."""
    start_time = time.time()

    try:
        result = subprocess.run([
            sys.executable, script_path, input_file, output_file
        ], capture_output=True, text=True, timeout=120)

        end_time = time.time()
        generation_time = end_time - start_time

        if result.returncode == 0:
            file_size, size_human = get_file_size(output_file)
            return {
                'success': True,
                'generation_time': generation_time,
                'file_size_bytes': file_size,
                'file_size_human': size_human,
                'output': result.stdout,
                'error': None
            }
        else:
            return {
                'success': False,
                'generation_time': generation_time,
                'file_size_bytes': 0,
                'file_size_human': '0 B',
                'output': result.stdout,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'generation_time': 120,
            'file_size_bytes': 0,
            'file_size_human': '0 B',
            'output': '',
            'error': 'Generation timed out after 120 seconds'
        }
    except Exception as e:
        return {
            'success': False,
            'generation_time': 0,
            'file_size_bytes': 0,
            'file_size_human': '0 B',
            'output': '',
            'error': str(e)
        }

def analyze_dataset(data):
    """Analyze the input dataset."""
    results = data.get('results', [])

    analysis = {
        'total_tokens': len(results),
        'creates_valid': 0,
        'breaks_extraction': 0,
        'both_behaviors': 0,
        'errors': 0,
        'avg_response_length': 0,
        'max_response_length': 0,
        'unique_issues': set()
    }

    response_lengths = []

    for result in results:
        if 'error' in result:
            analysis['errors'] += 1
            continue

        creates_valid = result.get('creates_valid_domain', False) or result.get('creates_valid_email', False)
        breaks_extraction = result.get('breaks_domain_extraction', False) or result.get('breaks_email_extraction', False)
        response_length = result.get('response_length', 0)
        issues = result.get('issues', [])

        response_lengths.append(response_length)
        analysis['max_response_length'] = max(analysis['max_response_length'], response_length)

        for issue in issues:
            analysis['unique_issues'].add(issue)

        if creates_valid and breaks_extraction:
            analysis['both_behaviors'] += 1
        elif creates_valid:
            analysis['creates_valid'] += 1
        elif breaks_extraction:
            analysis['breaks_extraction'] += 1

    if response_lengths:
        analysis['avg_response_length'] = sum(response_lengths) / len(response_lengths)

    analysis['unique_issues'] = len(analysis['unique_issues'])

    return analysis

def print_header():
    """Print demo header."""
    print("=" * 80)
    print("🚀 DOMAIN EXTRACTION REPORT GENERATOR COMPARISON DEMO")
    print("=" * 80)
    print()

def print_dataset_info(filepath, data, analysis):
    """Print dataset information."""
    print("📊 DATASET ANALYSIS")
    print("-" * 40)
    print(f"📁 Input File: {filepath}")
    print(f"🔢 Total Tokens: {analysis['total_tokens']:,}")
    print(f"✅ Creates Valid: {analysis['creates_valid']:,}")
    print(f"⚠️  Breaks Extraction: {analysis['breaks_extraction']:,}")
    print(f"🔄 Both Behaviors: {analysis['both_behaviors']:,}")
    print(f"❌ Errors: {analysis['errors']:,}")
    print(f"📏 Avg Response Length: {analysis['avg_response_length']:.1f} chars")
    print(f"📈 Max Response Length: {analysis['max_response_length']:,} chars")
    print(f"🏷️  Unique Issues: {analysis['unique_issues']}")
    print(f"🤖 Model: {data.get('model_path', 'Unknown')}")
    print()

def print_comparison_table(optimized_result, enhanced_result):
    """Print comparison table."""
    print("📋 PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<25} {'Optimized':<20} {'Enhanced':<20} {'Difference':<15}")
    print("-" * 80)

    # Generation time
    opt_time = optimized_result['generation_time']
    enh_time = enhanced_result['generation_time']
    time_diff = f"{((enh_time - opt_time) / opt_time * 100):+.1f}%" if opt_time > 0 else "N/A"
    print(f"{'Generation Time':<25} {opt_time:<20.2f}s {enh_time:<20.2f}s {time_diff:<15}")

    # File size
    opt_size = optimized_result['file_size_bytes']
    enh_size = enhanced_result['file_size_bytes']
    size_diff = f"{((enh_size - opt_size) / opt_size * 100):+.1f}%" if opt_size > 0 else "N/A"
    print(f"{'File Size':<25} {optimized_result['file_size_human']:<20} {enhanced_result['file_size_human']:<20} {size_diff:<15}")

    # File size reduction from original (assuming 625KB original)
    original_size = 625 * 1024  # 625KB in bytes
    opt_reduction = ((original_size - opt_size) / original_size * 100) if opt_size > 0 else 0
    enh_reduction = ((original_size - enh_size) / original_size * 100) if enh_size > 0 else 0
    print(f"{'Size Reduction*':<25} {opt_reduction:<20.1f}% {enh_reduction:<20.1f}% {enh_reduction - opt_reduction:+.1f}%")

    print("-" * 80)
    print("* Compared to original 625KB report")
    print()

def print_feature_comparison():
    """Print feature comparison."""
    print("🔧 FEATURE COMPARISON")
    print("-" * 60)
    print(f"{'Feature':<30} {'Optimized':<15} {'Enhanced':<15}")
    print("-" * 60)

    features = [
        ("Pagination", "✅ Basic", "✅ Advanced"),
        ("Search & Filter", "✅ Basic", "✅ Advanced"),
        ("Charts", "✅ 1 Chart", "✅ 4 Charts"),
        ("Export (CSV/JSON)", "❌", "✅"),
        ("Keyboard Shortcuts", "❌", "✅"),
        ("Performance Metrics", "❌", "✅"),
        ("Tabbed Interface", "❌", "✅"),
        ("Token Complexity", "❌", "✅"),
        ("Issue Analytics", "Basic", "✅ Advanced"),
        ("Mobile Responsive", "✅", "✅"),
        ("Print Support", "✅", "✅"),
    ]

    for feature, opt_support, enh_support in features:
        print(f"{feature:<30} {opt_support:<15} {enh_support:<15}")

    print("-" * 60)
    print()

def print_recommendations(analysis):
    """Print usage recommendations."""
    print("💡 RECOMMENDATIONS")
    print("-" * 40)

    if analysis['total_tokens'] < 500:
        print("📊 Dataset Size: Small (<500 tokens)")
        print("✅ Recommended: Either version works well")
        print("🎯 Best Choice: Optimized (smaller file size)")
    elif analysis['total_tokens'] < 1000:
        print("📊 Dataset Size: Medium (500-1000 tokens)")
        print("✅ Recommended: Enhanced for analysis, Optimized for sharing")
        print("🎯 Best Choice: Enhanced (better features)")
    else:
        print("📊 Dataset Size: Large (1000+ tokens)")
        print("✅ Recommended: Enhanced with pagination")
        print("🎯 Best Choice: Enhanced (essential for large datasets)")

    if analysis['unique_issues'] > 10:
        print("🏷️  High Issue Diversity: Enhanced version recommended for analytics")

    if analysis['breaks_extraction'] / analysis['total_tokens'] > 0.5:
        print("⚠️  High Glitch Rate: Enhanced version recommended for detailed analysis")

    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python demo_report_comparison.py input.json")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    print_header()

    # Load and analyze dataset
    print("🔍 Loading and analyzing dataset...")
    data = load_json_data(input_file)
    if not data:
        sys.exit(1)

    analysis = analyze_dataset(data)
    print_dataset_info(input_file, data, analysis)

    # Generate timestamp for unique output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file).stem

    optimized_output = f"{base_name}_optimized_{timestamp}.html"
    enhanced_output = f"{base_name}_enhanced_{timestamp}.html"

    # Check if generator scripts exist
    optimized_script = "generate_domain_report_optimized.py"
    enhanced_script = "generate_enhanced_report.py"

    if not os.path.exists(optimized_script):
        print(f"❌ Optimized generator not found: {optimized_script}")
        sys.exit(1)

    if not os.path.exists(enhanced_script):
        print(f"❌ Enhanced generator not found: {enhanced_script}")
        sys.exit(1)

    # Generate optimized report
    print("⚡ Generating optimized report...")
    optimized_result = run_report_generator(optimized_script, input_file, optimized_output)

    if optimized_result['success']:
        print(f"✅ Optimized report: {optimized_output} ({optimized_result['file_size_human']})")
    else:
        print(f"❌ Optimized report failed: {optimized_result['error']}")

    # Generate enhanced report
    print("🚀 Generating enhanced report...")
    enhanced_result = run_report_generator(enhanced_script, input_file, enhanced_output)

    if enhanced_result['success']:
        print(f"✅ Enhanced report: {enhanced_output} ({enhanced_result['file_size_human']})")
    else:
        print(f"❌ Enhanced report failed: {enhanced_result['error']}")

    print()

    # Show comparison if both succeeded
    if optimized_result['success'] and enhanced_result['success']:
        print_comparison_table(optimized_result, enhanced_result)
        print_feature_comparison()
        print_recommendations(analysis)

        print("🌐 NEXT STEPS")
        print("-" * 40)
        print(f"📁 Open optimized report: file://{Path(optimized_output).absolute()}")
        print(f"📁 Open enhanced report: file://{Path(enhanced_output).absolute()}")
        print("🔍 Compare the user interfaces and features")
        print("📊 Test export functionality in enhanced version")
        print("⌨️  Try keyboard shortcuts (press '?' in enhanced version)")
        print()

        print("✨ Demo completed successfully!")
        print("   Both reports generated for comparison.")
    else:
        print("❌ Demo completed with errors.")
        if not optimized_result['success']:
            print(f"   Optimized generator error: {optimized_result['error']}")
        if not enhanced_result['success']:
            print(f"   Enhanced generator error: {enhanced_result['error']}")

    print("=" * 80)

if __name__ == '__main__':
    main()
