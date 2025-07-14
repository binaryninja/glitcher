#!/usr/bin/env python3
"""
Consolidate and deduplicate results from all mining phases (Phase 6).

This script combines results from all mining phases, removes duplicates,
and creates a comprehensive database of discovered glitch tokens.

Usage:
    python consolidate_results.py --output-dir results/ --final-output master_glitch_database.json
"""

import argparse
import json
import os
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Tuple
import glob


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and parse a JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}


def extract_glitch_tokens_from_file(filepath: str, file_type: str) -> List[Dict[str, Any]]:
    """Extract glitch tokens from a result file based on its type"""
    data = load_json_file(filepath)
    if not data:
        return []

    glitch_tokens = []

    if file_type == 'low_norm':
        # Low norm results - all tokens are potential glitches
        if 'lowest_norm_tokens' in data:
            for token_info in data['lowest_norm_tokens']:
                glitch_tokens.append({
                    'token_id': token_info['token_id'],
                    'token_text': token_info['token_text'],
                    'l2_norm': token_info['l2_norm'],
                    'source': 'low_norm',
                    'source_file': filepath,
                    'confidence': 'high',  # Low norm tokens are highly likely to be glitches
                    'metadata': {
                        'rank': token_info.get('rank', 0),
                        'l2_norm': token_info['l2_norm']
                    }
                })

    elif file_type == 'glitcher_test':
        # Glitcher test results - only confirmed glitches
        if 'results' in data:
            for result in data['results']:
                if result.get('is_glitch', False):
                    glitch_tokens.append({
                        'token_id': result['token_id'],
                        'token_text': result['token'],
                        'source': 'glitcher_test',
                        'source_file': filepath,
                        'confidence': 'verified',
                        'metadata': {}
                    })

    elif file_type == 'pattern_mining':
        # Pattern mining results - only confirmed glitches
        if 'results' in data:
            for result in data['results']:
                if result.get('is_glitch', False):
                    glitch_tokens.append({
                        'token_id': result['token_id'],
                        'token_text': result['token'],
                        'source': 'pattern_mining',
                        'source_file': filepath,
                        'confidence': 'verified',
                        'metadata': {
                            'matching_patterns': result.get('matching_patterns', []),
                            'pattern_count': result.get('pattern_count', 0)
                        }
                    })

    elif file_type == 'range_mining':
        # Range mining results - only confirmed glitches
        if 'glitch_tokens' in data:
            for token_info in data['glitch_tokens']:
                glitch_tokens.append({
                    'token_id': token_info['token_id'],
                    'token_text': token_info['token_text'],
                    'source': 'range_mining',
                    'source_file': filepath,
                    'confidence': 'verified',
                    'metadata': {
                        'range': token_info.get('range', ''),
                        'range_start': token_info.get('range_start', 0),
                        'range_end': token_info.get('range_end', 0)
                    }
                })

    elif file_type == 'glitcher_mine':
        # Glitcher mine results - extract from log file or results
        if 'glitch_tokens' in data:
            for i, token_text in enumerate(data['glitch_tokens']):
                token_id = data['glitch_token_ids'][i] if i < len(data['glitch_token_ids']) else -1
                glitch_tokens.append({
                    'token_id': token_id,
                    'token_text': token_text,
                    'source': 'glitcher_mine',
                    'source_file': filepath,
                    'confidence': 'verified',
                    'metadata': {
                        'iterations': data.get('total_iterations', 0),
                        'entropy_guided': True
                    }
                })

    return glitch_tokens


def auto_detect_file_type(filepath: str) -> str:
    """Auto-detect the type of result file based on filename patterns"""
    filename = os.path.basename(filepath).lower()

    if 'low_norm' in filename:
        return 'low_norm'
    elif 'pattern' in filename:
        return 'pattern_mining'
    elif 'range' in filename:
        return 'range_mining'
    elif 'phase' in filename and 'test' in filename:
        return 'glitcher_test'
    elif 'phase' in filename and 'region' in filename:
        return 'glitcher_mine'
    elif 'test' in filename or 'results' in filename:
        return 'glitcher_test'
    else:
        return 'unknown'


def find_result_files(directory: str) -> List[Tuple[str, str]]:
    """Find all result files in a directory and detect their types"""
    result_files = []

    # Common result file patterns
    patterns = [
        '*.json',
        'phase*.json',
        '*_results.json',
        '*_mining*.json',
        'low_norm*.json',
        'pattern*.json',
        'range*.json'
    ]

    for pattern in patterns:
        for filepath in glob.glob(os.path.join(directory, pattern)):
            if os.path.isfile(filepath):
                file_type = auto_detect_file_type(filepath)
                result_files.append((filepath, file_type))

    return result_files


def consolidate_glitch_tokens(result_files: List[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Consolidate glitch tokens from all result files"""
    all_tokens = []
    stats = {
        'files_processed': 0,
        'files_by_type': defaultdict(int),
        'tokens_by_source': defaultdict(int),
        'total_tokens_found': 0
    }

    print(f"Processing {len(result_files)} result files...")

    for filepath, file_type in result_files:
        print(f"  Processing {os.path.basename(filepath)} (type: {file_type})")

        tokens = extract_glitch_tokens_from_file(filepath, file_type)
        all_tokens.extend(tokens)

        stats['files_processed'] += 1
        stats['files_by_type'][file_type] += 1
        stats['tokens_by_source'][file_type] += len(tokens)
        stats['total_tokens_found'] += len(tokens)

        print(f"    Found {len(tokens)} glitch tokens")

    return all_tokens, stats


def deduplicate_tokens(tokens: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove duplicates and merge information from multiple sources"""
    token_map = {}  # token_id -> merged token info
    dedup_stats = {
        'total_before': len(tokens),
        'total_after': 0,
        'duplicates_removed': 0,
        'multi_source_tokens': 0
    }

    print(f"Deduplicating {len(tokens)} tokens...")

    for token in tokens:
        token_id = token['token_id']

        if token_id in token_map:
            # Merge with existing token
            existing = token_map[token_id]

            # Add source to list if not already present
            if isinstance(existing['source'], str):
                existing['source'] = [existing['source']]
            if token['source'] not in existing['source']:
                existing['source'].append(token['source'])

            # Add source file to list
            if 'source_files' not in existing:
                existing['source_files'] = [existing['source_file']]
            if token['source_file'] not in existing['source_files']:
                existing['source_files'].append(token['source_file'])

            # Merge metadata
            for key, value in token['metadata'].items():
                if key not in existing['metadata']:
                    existing['metadata'][key] = value
                elif key == 'matching_patterns' and isinstance(value, list):
                    # Merge pattern lists
                    if isinstance(existing['metadata'][key], list):
                        existing['metadata'][key] = list(set(existing['metadata'][key] + value))

            # Update confidence level
            confidence_levels = ['high', 'verified', 'medium', 'low']
            current_level = confidence_levels.index(existing['confidence'])
            new_level = confidence_levels.index(token['confidence'])
            if new_level < current_level:  # verified is better than high
                existing['confidence'] = token['confidence']

            dedup_stats['duplicates_removed'] += 1

        else:
            # New token
            token_map[token_id] = token.copy()

    # Convert back to list
    unique_tokens = list(token_map.values())

    # Count multi-source tokens
    for token in unique_tokens:
        if isinstance(token['source'], list) and len(token['source']) > 1:
            dedup_stats['multi_source_tokens'] += 1

    dedup_stats['total_after'] = len(unique_tokens)

    return unique_tokens, dedup_stats


def analyze_token_patterns(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in the discovered glitch tokens"""
    analysis = {
        'total_tokens': len(tokens),
        'by_confidence': Counter(),
        'by_source': Counter(),
        'token_length_distribution': Counter(),
        'unicode_categories': Counter(),
        'pattern_analysis': Counter(),
        'common_prefixes': Counter(),
        'common_suffixes': Counter(),
        'multi_source_tokens': 0
    }

    print("Analyzing token patterns...")

    for token in tokens:
        token_text = token['token_text']

        # Confidence distribution
        analysis['by_confidence'][token['confidence']] += 1

        # Source distribution
        sources = token['source'] if isinstance(token['source'], list) else [token['source']]
        for source in sources:
            analysis['by_source'][source] += 1

        if len(sources) > 1:
            analysis['multi_source_tokens'] += 1

        # Token length
        analysis['token_length_distribution'][len(token_text)] += 1

        # Unicode categories
        has_ascii = any(ord(c) < 128 for c in token_text)
        has_unicode = any(ord(c) >= 128 for c in token_text)
        has_control = any(ord(c) < 32 or ord(c) == 127 for c in token_text)
        has_replacement = '�' in token_text

        if has_replacement:
            analysis['unicode_categories']['replacement_chars'] += 1
        elif has_control:
            analysis['unicode_categories']['control_chars'] += 1
        elif has_ascii and has_unicode:
            analysis['unicode_categories']['mixed_scripts'] += 1
        elif has_unicode:
            analysis['unicode_categories']['unicode_only'] += 1
        elif has_ascii:
            analysis['unicode_categories']['ascii_only'] += 1

        # Pattern analysis
        if any(c.isdigit() for c in token_text):
            analysis['pattern_analysis']['contains_digits'] += 1
        if any(c.isalpha() for c in token_text):
            analysis['pattern_analysis']['contains_letters'] += 1
        if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in token_text):
            analysis['pattern_analysis']['contains_special'] += 1
        if any(c.isspace() for c in token_text):
            analysis['pattern_analysis']['contains_whitespace'] += 1
        if len(set(token_text)) == 1:
            analysis['pattern_analysis']['repeated_chars'] += 1

        # Common prefixes and suffixes
        if len(token_text) >= 2:
            analysis['common_prefixes'][token_text[:2]] += 1
            analysis['common_suffixes'][token_text[-2:]] += 1

    return analysis


def save_consolidated_results(
    tokens: List[Dict[str, Any]],
    stats: Dict[str, Any],
    dedup_stats: Dict[str, Any],
    analysis: Dict[str, Any],
    output_file: str
):
    """Save the consolidated results to a JSON file"""

    # Sort tokens by confidence and then by token_id
    confidence_order = {'verified': 0, 'high': 1, 'medium': 2, 'low': 3}
    tokens.sort(key=lambda x: (confidence_order.get(x['confidence'], 4), x['token_id']))

    consolidated_data = {
        'metadata': {
            'timestamp': time.time(),
            'total_glitch_tokens': len(tokens),
            'consolidation_stats': stats,
            'deduplication_stats': dedup_stats,
            'analysis': analysis
        },
        'glitch_tokens': tokens
    }

    with open(output_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)

    print(f"Consolidated results saved to: {output_file}")


def print_summary(stats: Dict[str, Any], dedup_stats: Dict[str, Any], analysis: Dict[str, Any]):
    """Print a summary of the consolidation results"""
    print("\n" + "="*60)
    print("CONSOLIDATION SUMMARY")
    print("="*60)

    print(f"Files processed: {stats['files_processed']}")
    print(f"Files by type:")
    for file_type, count in stats['files_by_type'].items():
        print(f"  {file_type}: {count}")

    print(f"\nTokens found by source:")
    for source, count in stats['tokens_by_source'].items():
        print(f"  {source}: {count}")

    print(f"\nDeduplication:")
    print(f"  Before: {dedup_stats['total_before']:,}")
    print(f"  After: {dedup_stats['total_after']:,}")
    print(f"  Duplicates removed: {dedup_stats['duplicates_removed']:,}")
    print(f"  Multi-source tokens: {dedup_stats['multi_source_tokens']:,}")

    print(f"\nFinal results:")
    print(f"  Total unique glitch tokens: {analysis['total_tokens']:,}")
    print(f"  By confidence level:")
    for confidence, count in analysis['by_confidence'].most_common():
        print(f"    {confidence}: {count:,}")

    print(f"  By source:")
    for source, count in analysis['by_source'].most_common():
        print(f"    {source}: {count:,}")

    print(f"  Token patterns:")
    print(f"    Unicode replacement chars: {analysis['unicode_categories']['replacement_chars']:,}")
    print(f"    Mixed scripts: {analysis['unicode_categories']['mixed_scripts']:,}")
    print(f"    Contains digits: {analysis['pattern_analysis']['contains_digits']:,}")
    print(f"    Contains special chars: {analysis['pattern_analysis']['contains_special']:,}")
    print(f"    Repeated characters: {analysis['pattern_analysis']['repeated_chars']:,}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate glitch token mining results")
    parser.add_argument("--input-dir", default=".", help="Directory containing result files")
    parser.add_argument("--output", default="consolidated_glitch_tokens.json", help="Output file")
    parser.add_argument("--file-patterns", nargs="+", help="Additional file patterns to search")
    parser.add_argument("--exclude-low-norm", action="store_true", help="Exclude low-norm candidates (unverified)")

    args = parser.parse_args()

    # Find result files
    result_files = find_result_files(args.input_dir)

    # Add custom patterns if provided
    if args.file_patterns:
        for pattern in args.file_patterns:
            for filepath in glob.glob(os.path.join(args.input_dir, pattern)):
                if os.path.isfile(filepath):
                    file_type = auto_detect_file_type(filepath)
                    result_files.append((filepath, file_type))

    # Remove duplicates from file list
    result_files = list(set(result_files))

    # Exclude low-norm files if requested
    if args.exclude_low_norm:
        result_files = [(f, t) for f, t in result_files if t != 'low_norm']

    if not result_files:
        print("No result files found!")
        return

    # Consolidate tokens
    all_tokens, stats = consolidate_glitch_tokens(result_files)

    # Deduplicate
    unique_tokens, dedup_stats = deduplicate_tokens(all_tokens)

    # Analyze patterns
    analysis = analyze_token_patterns(unique_tokens)

    # Save results
    save_consolidated_results(unique_tokens, stats, dedup_stats, analysis, args.output)

    # Print summary
    print_summary(stats, dedup_stats, analysis)


if __name__ == "__main__":
    main()
