#!/usr/bin/env python3
"""
Range-based systematic mining for glitch tokens.

This script explores different ranges of token IDs systematically to find
glitch tokens that might be clustered in specific vocabulary ranges.

Usage:
    python range_mining.py meta-llama/Llama-3.2-1B-Instruct --range-start 0 --range-end 1000 --sample-rate 0.1
    python range_mining.py meta-llama/Llama-3.2-1B-Instruct --unicode-ranges --sample-rate 0.05
    python range_mining.py meta-llama/Llama-3.2-1B-Instruct --special-ranges --sample-rate 0.2
"""

import argparse
import json
import random
import time
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from glitcher.model import strictly_glitch_verify, get_template_for_model


def get_unicode_ranges() -> List[Tuple[int, int, str]]:
    """Get common Unicode ranges where glitch tokens are likely to appear"""
    return [
        (0, 127, "ASCII Control"),
        (128, 255, "Latin-1 Supplement"),
        (256, 383, "Latin Extended-A"),
        (384, 591, "Latin Extended-B"),
        (880, 1023, "Greek and Coptic"),
        (1024, 1279, "Cyrillic"),
        (1280, 1327, "Cyrillic Supplement"),
        (1536, 1791, "Arabic"),
        (2304, 2431, "Devanagari"),
        (2432, 2559, "Bengali"),
        (2560, 2687, "Gurmukhi"),
        (2688, 2815, "Gujarati"),
        (2816, 2943, "Oriya"),
        (2944, 3071, "Tamil"),
        (3072, 3199, "Telugu"),
        (3200, 3327, "Kannada"),
        (3328, 3455, "Malayalam"),
        (3456, 3583, "Sinhala"),
        (3584, 3711, "Thai"),
        (3712, 3839, "Lao"),
        (3840, 4095, "Tibetan"),
        (4096, 4255, "Myanmar"),
        (4352, 4607, "Hangul Jamo"),
        (4608, 4991, "Ethiopic"),
        (5024, 5119, "Cherokee"),
        (5120, 5759, "Unified Canadian Aboriginal Syllabics"),
        (5760, 5791, "Ogham"),
        (5792, 5887, "Runic"),
        (5888, 5919, "Tagalog"),
        (5920, 5951, "Hanunoo"),
        (5952, 5983, "Buhid"),
        (5984, 6015, "Tagbanwa"),
        (6016, 6143, "Khmer"),
        (6144, 6319, "Mongolian"),
        (6400, 6479, "Limbu"),
        (6480, 6527, "Tai Le"),
        (6528, 6623, "New Tai Lue"),
        (6624, 6655, "Khmer Symbols"),
        (6656, 6687, "Buginese"),
        (6688, 6831, "Tai Tham"),
        (6912, 7039, "Balinese"),
        (7040, 7103, "Sundanese"),
        (7104, 7167, "Batak"),
        (7168, 7247, "Lepcha"),
        (7248, 7295, "Ol Chiki"),
        (7424, 7551, "Phonetic Extensions"),
        (7552, 7615, "Phonetic Extensions Supplement"),
        (7680, 7935, "Latin Extended Additional"),
        (7936, 8191, "Greek Extended"),
        (8192, 8303, "General Punctuation"),
        (8304, 8351, "Superscripts and Subscripts"),
        (8352, 8399, "Currency Symbols"),
        (8400, 8447, "Combining Diacritical Marks for Symbols"),
        (8448, 8527, "Letterlike Symbols"),
        (8528, 8591, "Number Forms"),
        (8592, 8703, "Arrows"),
        (8704, 8959, "Mathematical Operators"),
        (8960, 9215, "Miscellaneous Technical"),
        (9216, 9279, "Control Pictures"),
        (9280, 9311, "Optical Character Recognition"),
        (9312, 9471, "Enclosed Alphanumerics"),
        (9472, 9599, "Box Drawing"),
        (9600, 9631, "Block Elements"),
        (9632, 9727, "Geometric Shapes"),
        (9728, 9983, "Miscellaneous Symbols"),
        (9984, 10175, "Dingbats"),
        (10176, 10223, "Miscellaneous Mathematical Symbols-A"),
        (10224, 10239, "Supplemental Arrows-A"),
        (10240, 10495, "Braille Patterns"),
        (10496, 10623, "Supplemental Arrows-B"),
        (10624, 10751, "Miscellaneous Mathematical Symbols-B"),
        (10752, 11007, "Supplemental Mathematical Operators"),
        (11008, 11263, "Miscellaneous Symbols and Arrows"),
        (11264, 11359, "Glagolitic"),
        (11360, 11391, "Latin Extended-C"),
        (11392, 11519, "Coptic"),
        (11520, 11567, "Georgian Supplement"),
        (11568, 11647, "Tifinagh"),
        (11648, 11743, "Ethiopic Extended"),
        (11744, 11775, "Cyrillic Extended-A"),
        (11776, 11903, "Supplemental Punctuation"),
        (11904, 12031, "CJK Radicals Supplement"),
        (12032, 12255, "Kangxi Radicals"),
        (12288, 12351, "CJK Symbols and Punctuation"),
        (12352, 12447, "Hiragana"),
        (12448, 12543, "Katakana"),
        (12544, 12591, "Bopomofo"),
        (12592, 12687, "Hangul Compatibility Jamo"),
        (12688, 12703, "Kanbun"),
        (12704, 12735, "Bopomofo Extended"),
        (12736, 12783, "CJK Strokes"),
        (12784, 12799, "Katakana Phonetic Extensions"),
        (12800, 13055, "Enclosed CJK Letters and Months"),
        (13056, 13311, "CJK Compatibility"),
        (13312, 19903, "CJK Unified Ideographs Extension A"),
        (19904, 19967, "Yijing Hexagram Symbols"),
        (19968, 40959, "CJK Unified Ideographs"),
        (40960, 42127, "Yi Syllables"),
        (42128, 42191, "Yi Radicals"),
        (42192, 42239, "Lisu"),
        (42240, 42559, "Vai"),
        (42560, 42655, "Cyrillic Extended-B"),
        (42656, 42751, "Bamum"),
        (42752, 42783, "Modifier Tone Letters"),
        (42784, 43007, "Latin Extended-D"),
        (43008, 43055, "Syloti Nagri"),
        (43056, 43071, "Common Indic Number Forms"),
        (43072, 43135, "Phags-pa"),
        (43136, 43231, "Saurashtra"),
        (43232, 43263, "Devanagari Extended"),
        (43264, 43311, "Kayah Li"),
        (43312, 43359, "Rejang"),
        (43360, 43391, "Hangul Jamo Extended-A"),
        (43392, 43487, "Javanese"),
        (43488, 43519, "Myanmar Extended-B"),
        (43520, 43615, "Cham"),
        (43616, 43647, "Myanmar Extended-A"),
        (43648, 43743, "Tai Viet"),
        (43744, 43775, "Meetei Mayek Extensions"),
        (43776, 43823, "Ethiopic Extended-A"),
        (43824, 43887, "Latin Extended-E"),
        (43888, 43967, "Cherokee Supplement"),
        (43968, 44031, "Meetei Mayek"),
        (44032, 55215, "Hangul Syllables"),
        (55216, 55295, "Hangul Jamo Extended-B"),
        (55296, 57343, "High Surrogates"),
        (57344, 63743, "Private Use Area"),
        (63744, 64255, "CJK Compatibility Ideographs"),
        (64256, 64335, "Alphabetic Presentation Forms"),
        (64336, 65023, "Arabic Presentation Forms-A"),
        (65024, 65039, "Variation Selectors"),
        (65040, 65055, "Vertical Forms"),
        (65056, 65071, "Combining Half Marks"),
        (65072, 65103, "CJK Compatibility Forms"),
        (65104, 65135, "Small Form Variants"),
        (65136, 65279, "Arabic Presentation Forms-B"),
        (65280, 65519, "Halfwidth and Fullwidth Forms"),
        (65520, 65535, "Specials")
    ]


def get_special_ranges() -> List[Tuple[int, int, str]]:
    """Get special token ID ranges that are likely to contain glitch tokens"""
    return [
        (0, 1000, "Early vocabulary (often special tokens)"),
        (1000, 5000, "Common tokens and punctuation"),
        (5000, 10000, "Extended vocabulary"),
        (10000, 20000, "Specialized tokens"),
        (20000, 50000, "Extended language tokens"),
        (50000, 100000, "Rare and specialized tokens"),
        (100000, 120000, "Very rare tokens"),
        (120000, 128000, "Extremely rare tokens"),
        (128000, 128256, "Reserved special tokens")
    ]


def find_tokens_in_range(
    tokenizer: AutoTokenizer,
    start_id: int,
    end_id: int,
    sample_rate: float = 1.0
) -> List[int]:
    """Find valid token IDs in a given range"""
    valid_tokens = []

    for token_id in range(start_id, min(end_id, tokenizer.vocab_size)):
        try:
            # Try to decode the token
            token_text = tokenizer.decode([token_id])

            # Skip empty tokens
            if not token_text:
                continue

            # Skip obvious special tokens (but include some for testing)
            if token_text.startswith('<|') and token_text.endswith('|>'):
                if random.random() < 0.1:  # Include 10% of special tokens
                    valid_tokens.append(token_id)
                continue

            # Apply sampling rate
            if random.random() < sample_rate:
                valid_tokens.append(token_id)

        except Exception:
            # Skip tokens that can't be decoded
            continue

    return valid_tokens


def range_based_mining(
    model_path: str,
    range_start: int = None,
    range_end: int = None,
    unicode_ranges: bool = False,
    special_ranges: bool = False,
    sample_rate: float = 0.1,
    max_tokens_per_range: int = 100,
    device: str = "auto",
    output_file: str = "range_mining_results.json"
) -> Dict[str, Any]:
    """
    Perform range-based mining for glitch tokens

    Args:
        model_path: Path to the model
        range_start: Starting token ID (if doing single range)
        range_end: Ending token ID (if doing single range)
        unicode_ranges: Whether to mine Unicode ranges
        special_ranges: Whether to mine special token ranges
        sample_rate: Fraction of tokens to sample from each range
        max_tokens_per_range: Maximum tokens to test per range
        device: Device to use
        output_file: Output file for results

    Returns:
        Dictionary with mining results
    """
    print(f"Loading model: {model_path}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    # Determine ranges to test
    ranges_to_test = []

    if range_start is not None and range_end is not None:
        ranges_to_test = [(range_start, range_end, f"Custom range {range_start}-{range_end}")]
    elif unicode_ranges:
        ranges_to_test = get_unicode_ranges()
    elif special_ranges:
        ranges_to_test = get_special_ranges()
    else:
        # Default: test vocabulary in chunks
        vocab_size = tokenizer.vocab_size
        chunk_size = 10000
        ranges_to_test = [
            (i, min(i + chunk_size, vocab_size), f"Chunk {i//chunk_size + 1}")
            for i in range(0, vocab_size, chunk_size)
        ]

    print(f"Testing {len(ranges_to_test)} ranges with sample rate {sample_rate}")

    # Results storage
    results = {
        "model_path": model_path,
        "timestamp": time.time(),
        "sample_rate": sample_rate,
        "max_tokens_per_range": max_tokens_per_range,
        "ranges_tested": [],
        "total_tokens_tested": 0,
        "total_glitch_tokens": 0,
        "glitch_tokens": []
    }

    # Create detailed log file
    log_file = f"range_mining_log_{int(time.time())}.jsonl"
    with open(log_file, 'w') as f:
        f.write("# Range-based mining log\n")
        f.write(json.dumps({"event": "start", "model": model_path, "ranges": len(ranges_to_test)}) + "\n")

    # Test each range
    for range_idx, (start, end, description) in enumerate(ranges_to_test):
        print(f"\n[{range_idx + 1}/{len(ranges_to_test)}] Testing range: {description} ({start}-{end})")

        # Find tokens in this range
        range_tokens = find_tokens_in_range(tokenizer, start, end, sample_rate)

        # Limit tokens per range
        if len(range_tokens) > max_tokens_per_range:
            range_tokens = random.sample(range_tokens, max_tokens_per_range)

        print(f"  Found {len(range_tokens)} tokens to test in this range")

        if not range_tokens:
            continue

        # Test tokens in this range
        range_glitch_tokens = []
        range_results = {
            "range_start": start,
            "range_end": end,
            "description": description,
            "tokens_tested": len(range_tokens),
            "glitch_tokens_found": 0,
            "glitch_tokens": []
        }

        for token_id in range_tokens:
            try:
                token_text = tokenizer.decode([token_id])

                # Log token being tested
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "testing_token",
                        "range": description,
                        "token_id": token_id,
                        "token_text": token_text
                    }) + "\n")

                # Test if it's a glitch token
                is_glitch = strictly_glitch_verify(
                    model, tokenizer, token_id, chat_template, log_file
                )

                if is_glitch:
                    glitch_info = {
                        "token_id": token_id,
                        "token_text": token_text,
                        "range": description,
                        "range_start": start,
                        "range_end": end
                    }
                    range_glitch_tokens.append(glitch_info)
                    results["glitch_tokens"].append(glitch_info)
                    print(f"    ✓ Found glitch token: '{token_text}' (ID: {token_id})")

                # Log result
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "token_result",
                        "token_id": token_id,
                        "token_text": token_text,
                        "is_glitch": is_glitch,
                        "range": description
                    }) + "\n")

            except Exception as e:
                print(f"    Error testing token {token_id}: {e}")
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "token_error",
                        "token_id": token_id,
                        "error": str(e),
                        "range": description
                    }) + "\n")

        # Update range results
        range_results["glitch_tokens_found"] = len(range_glitch_tokens)
        range_results["glitch_tokens"] = range_glitch_tokens
        results["ranges_tested"].append(range_results)
        results["total_tokens_tested"] += len(range_tokens)

        print(f"  Range summary: {len(range_glitch_tokens)}/{len(range_tokens)} tokens are glitches "
              f"({len(range_glitch_tokens)/len(range_tokens)*100:.1f}%)")

    # Final summary
    results["total_glitch_tokens"] = len(results["glitch_tokens"])
    glitch_rate = results["total_glitch_tokens"] / results["total_tokens_tested"] if results["total_tokens_tested"] > 0 else 0

    print(f"\n=== FINAL RESULTS ===")
    print(f"Total tokens tested: {results['total_tokens_tested']}")
    print(f"Total glitch tokens found: {results['total_glitch_tokens']}")
    print(f"Overall glitch rate: {glitch_rate:.1%}")
    print(f"Detailed log saved to: {log_file}")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Range-based mining for glitch tokens")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--range-start", type=int, help="Starting token ID for single range")
    parser.add_argument("--range-end", type=int, help="Ending token ID for single range")
    parser.add_argument("--unicode-ranges", action="store_true", help="Mine Unicode ranges")
    parser.add_argument("--special-ranges", action="store_true", help="Mine special token ranges")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Fraction of tokens to sample (0.0-1.0)")
    parser.add_argument("--max-tokens-per-range", type=int, default=100, help="Maximum tokens to test per range")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output", default="range_mining_results.json", help="Output file")

    args = parser.parse_args()

    # Validate arguments
    if args.range_start is not None and args.range_end is None:
        parser.error("--range-end is required when --range-start is specified")
    if args.range_end is not None and args.range_start is None:
        parser.error("--range-start is required when --range-end is specified")

    if args.sample_rate <= 0 or args.sample_rate > 1:
        parser.error("--sample-rate must be between 0 and 1")

    range_based_mining(
        model_path=args.model_path,
        range_start=args.range_start,
        range_end=args.range_end,
        unicode_ranges=args.unicode_ranges,
        special_ranges=args.special_ranges,
        sample_rate=args.sample_rate,
        max_tokens_per_range=args.max_tokens_per_range,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
