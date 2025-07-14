#!/usr/bin/env python3
"""
Pattern-based mining for glitch tokens (Phase 4).

This script identifies tokens that match specific patterns commonly found
in glitch tokens, such as repeated characters, mixed scripts, code artifacts,
and formatting anomalies.

Usage:
    python pattern_mining.py meta-llama/Llama-3.2-1B-Instruct --output phase4_pattern_results.json
"""

import argparse
import json
import re
import time
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from glitcher.model import strictly_glitch_verify, get_template_for_model


def get_glitch_patterns() -> Dict[str, str]:
    """Define regex patterns commonly found in glitch tokens"""
    return {
        'repeated_chars': r'(.)\1{4,}',  # 5+ repeated characters
        'mixed_scripts_cyrillic': r'[\u0400-\u04FF].*[\u0000-\u007F]',  # Cyrillic + ASCII
        'mixed_scripts_arabic': r'[\u0600-\u06FF].*[\u0000-\u007F]',  # Arabic + ASCII
        'mixed_scripts_cjk': r'[\u4E00-\u9FFF].*[\u0000-\u007F]',  # CJK + ASCII
        'code_artifacts': r'(Element|Token|Xpath|Wrapper|Thunk|Callable|Adjustor)',
        'formatting_junk': r'\\[rn]{2,}',  # Multiple \r\n sequences
        'unicode_corruption': r'[\uFFFD\u0000-\u001F\u007F-\u009F]',  # Replacement/control chars
        'progress_bars': r'[▍▎▏▌▋▊▉█]{3,}',  # Progress bar characters
        'box_drawing': r'[┌┐└┘├┤┬┴┼─│]{3,}',  # Box drawing characters
        'malformed_html': r'<[^>]*>[^<]*</[^>]*>',  # Malformed HTML tags
        'data_leakage': r'(Postal|Code|NL|URL|HTTP|API|Config|Debug)',  # Data artifacts
        'programming_typos': r'(use[A-Z][a-z]*|get[A-Z][a-z]*|set[A-Z][a-z]*)[A-Z][a-z]*[A-Z]',  # CamelCase typos
        'reserved_patterns': r'<\|[^|]+\|>',  # Reserved token patterns
        'whitespace_anomalies': r'[\t\r\n]{2,}',  # Multiple whitespace chars
        'currency_mixed': r'[\$€£¥₹₽].*[a-zA-Z]',  # Currency symbols mixed with letters
        'numeric_anomalies': r'\d+[a-zA-Z]+\d+',  # Numbers mixed with letters
        'special_prefixes': r'^[\$_\-\+\*\#\@\%\&]',  # Special character prefixes
        'bracket_anomalies': r'[\[\]{}()]{2,}',  # Multiple brackets
        'dot_sequences': r'\.{3,}',  # Multiple dots
        'punctuation_spam': r'[!?;:,]{2,}',  # Repeated punctuation
        'encoding_artifacts': r'[�]{1,}',  # Replacement characters
        'script_mixing_any': r'[\u0100-\u017F].*[\u0000-\u007F]',  # Latin Extended + ASCII
        'mathematical_symbols': r'[\u2200-\u22FF]{2,}',  # Mathematical operators
        'diacritical_heavy': r'[\u0300-\u036F]{2,}',  # Multiple diacritical marks
        'private_use': r'[\uE000-\uF8FF]',  # Private use area
        'surrogate_pairs': r'[\uD800-\uDFFF]',  # Surrogate pairs (invalid in UTF-8)
        'byte_order_marks': r'[\uFEFF]',  # BOM characters
        'invisible_chars': r'[\u200B-\u200D\u2060\uFEFF]',  # Invisible characters
        'rtl_marks': r'[\u202A-\u202E]',  # Right-to-left marks
        'combining_chars': r'[\u0300-\u036F][\u0300-\u036F]+',  # Multiple combining characters
    }


def find_pattern_candidates(
    tokenizer: AutoTokenizer,
    patterns: Dict[str, str],
    max_tokens: int = 100000,
    min_token_length: int = 1
) -> List[Dict[str, Any]]:
    """Find tokens that match glitch patterns"""
    candidates = []
    vocab_size = min(max_tokens, tokenizer.vocab_size)

    print(f"Scanning {vocab_size} tokens for pattern matches...")

    for token_id in tqdm(range(vocab_size), desc="Pattern scanning"):
        try:
            token = tokenizer.decode([token_id])

            # Skip very short tokens (often normal punctuation)
            if len(token) < min_token_length:
                continue

            # Skip empty tokens
            if not token.strip():
                continue

            # Check each pattern
            matching_patterns = []
            for pattern_name, pattern in patterns.items():
                try:
                    if re.search(pattern, token, re.IGNORECASE):
                        matching_patterns.append(pattern_name)
                except re.error:
                    # Skip invalid regex patterns
                    continue

            # If token matches any pattern, add it as a candidate
            if matching_patterns:
                candidates.append({
                    'token_id': token_id,
                    'token': token,
                    'token_length': len(token),
                    'matching_patterns': matching_patterns,
                    'pattern_count': len(matching_patterns)
                })

        except Exception as e:
            # Skip tokens that can't be decoded
            continue

    return candidates


def analyze_pattern_distribution(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze the distribution of patterns found"""
    pattern_counts = {}

    for candidate in candidates:
        for pattern in candidate['matching_patterns']:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    return pattern_counts


def pattern_based_mining(
    model_path: str,
    max_tokens: int = 100000,
    min_token_length: int = 1,
    max_candidates: int = 1000,
    device: str = "auto",
    output_file: str = "pattern_mining_results.json"
) -> Dict[str, Any]:
    """
    Perform pattern-based mining for glitch tokens

    Args:
        model_path: Path to the model
        max_tokens: Maximum number of tokens to scan
        min_token_length: Minimum token length to consider
        max_candidates: Maximum number of candidates to test
        device: Device to use for model
        output_file: Output file path

    Returns:
        Dictionary with mining results
    """
    print(f"Loading model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model for validation
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    print(f"Model loaded on {model.device}")

    # Get chat template
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Get glitch patterns
    patterns = get_glitch_patterns()
    print(f"Using {len(patterns)} patterns for detection")

    # Find pattern candidates
    candidates = find_pattern_candidates(
        tokenizer, patterns, max_tokens, min_token_length
    )

    print(f"Found {len(candidates)} pattern candidates")

    # Analyze pattern distribution
    pattern_dist = analyze_pattern_distribution(candidates)
    print("\nPattern distribution:")
    for pattern, count in sorted(pattern_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} matches")

    # Sort candidates by pattern count (tokens matching more patterns are more likely to be glitches)
    candidates.sort(key=lambda x: x['pattern_count'], reverse=True)

    # Limit candidates to test
    if len(candidates) > max_candidates:
        print(f"Limiting to top {max_candidates} candidates for validation")
        candidates = candidates[:max_candidates]

    # Create detailed log file
    log_file = f"pattern_mining_log_{int(time.time())}.jsonl"
    with open(log_file, 'w') as f:
        f.write("# Pattern-based mining log\n")
        f.write(json.dumps({
            "event": "start",
            "model": model_path,
            "candidates": len(candidates),
            "patterns": list(patterns.keys())
        }) + "\n")

    # Validate each candidate
    results = []
    glitch_count = 0

    print(f"\nValidating {len(candidates)} candidates...")

    for i, candidate in enumerate(tqdm(candidates, desc="Validating candidates")):
        token_id = candidate['token_id']
        token = candidate['token']

        try:
            # Log candidate being tested
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "testing_candidate",
                    "token_id": token_id,
                    "token": token,
                    "patterns": candidate['matching_patterns'],
                    "index": i + 1,
                    "total": len(candidates)
                }) + "\n")

            # Validate if it's a glitch token
            is_glitch = strictly_glitch_verify(
                model, tokenizer, token_id, chat_template, log_file
            )

            result = {
                "token_id": token_id,
                "token": token,
                "token_length": candidate['token_length'],
                "matching_patterns": candidate['matching_patterns'],
                "pattern_count": candidate['pattern_count'],
                "is_glitch": is_glitch
            }

            results.append(result)

            if is_glitch:
                glitch_count += 1
                print(f"  ✓ Glitch token found: '{token}' (ID: {token_id}) - Patterns: {candidate['matching_patterns']}")

            # Log result
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "candidate_result",
                    "token_id": token_id,
                    "token": token,
                    "is_glitch": is_glitch,
                    "patterns": candidate['matching_patterns']
                }) + "\n")

        except Exception as e:
            print(f"  Error testing token {token_id}: {e}")
            with open(log_file, 'a') as f:
                f.write(json.dumps({
                    "event": "candidate_error",
                    "token_id": token_id,
                    "token": token,
                    "error": str(e)
                }) + "\n")

    # Calculate success rate by pattern
    pattern_success_rates = {}
    for pattern in patterns.keys():
        pattern_candidates = [r for r in results if pattern in r['matching_patterns']]
        pattern_glitches = [r for r in pattern_candidates if r['is_glitch']]

        if pattern_candidates:
            success_rate = len(pattern_glitches) / len(pattern_candidates)
            pattern_success_rates[pattern] = {
                'candidates': len(pattern_candidates),
                'glitches': len(pattern_glitches),
                'success_rate': success_rate
            }

    # Prepare final results
    final_results = {
        "model_path": model_path,
        "timestamp": time.time(),
        "max_tokens_scanned": max_tokens,
        "total_candidates": len(candidates),
        "total_glitch_tokens": glitch_count,
        "success_rate": glitch_count / len(candidates) if candidates else 0,
        "pattern_distribution": pattern_dist,
        "pattern_success_rates": pattern_success_rates,
        "log_file": log_file,
        "results": results
    }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n=== PATTERN MINING RESULTS ===")
    print(f"Total candidates tested: {len(candidates)}")
    print(f"Glitch tokens found: {glitch_count}")
    print(f"Success rate: {glitch_count/len(candidates)*100:.1f}%")
    print(f"Results saved to: {output_file}")
    print(f"Detailed log saved to: {log_file}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Pattern-based mining for glitch tokens")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Maximum tokens to scan")
    parser.add_argument("--min-token-length", type=int, default=1, help="Minimum token length")
    parser.add_argument("--max-candidates", type=int, default=1000, help="Maximum candidates to test")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output", default="pattern_mining_results.json", help="Output file")

    args = parser.parse_args()

    pattern_based_mining(
        model_path=args.model_path,
        max_tokens=args.max_tokens,
        min_token_length=args.min_token_length,
        max_candidates=args.max_candidates,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
