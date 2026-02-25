#!/usr/bin/env python3
"""
Test script for ASCII filtering functionality in genetic algorithm

This script tests the ASCII-only token filtering feature to ensure it correctly
identifies and filters tokens based on their decoded text content.

Usage:
    python test_ascii_filtering.py [model_name]

Author: Claude
Date: 2024
"""

import sys
import os
import tempfile
import json
from unittest.mock import Mock

# Add the glitcher package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from glitcher.genetic.reducer import GeneticProbabilityReducer


def create_mock_tokenizer():
    """Create a mock tokenizer for testing"""
    mock_tokenizer = Mock()

    # Mock token decoding results - mix of ASCII and non-ASCII
    decode_map = {
        12345: "hello",           # Pure ASCII
        67890: "café",            # Unicode (é)
        11111: "world",           # Pure ASCII
        22222: "测试",            # Chinese characters
        33333: "test\x00end",     # Control character
        44444: "<|endoftext|>",   # Special token with ASCII
        55555: "emoji🚀text",     # Emoji
        66666: "normal",          # Pure ASCII
        77777: "Ñoël",           # Unicode (Ñ, ë)
        88888: "tab\there",       # ASCII with tab
        99999: "\n\r\t",          # ASCII control chars only
        10101: "München",         # Unicode (ü)
        20202: "",                # Empty string
        30303: " ",               # Space only
    }

    def mock_decode(token_ids, skip_special_tokens=False):
        if isinstance(token_ids, list) and len(token_ids) == 1:
            token_id = token_ids[0]
            return decode_map.get(token_id, f"unknown_{token_id}")
        return "batch_decode_not_implemented"

    mock_tokenizer.decode = mock_decode
    return mock_tokenizer


def test_is_ascii_only():
    """Test the _is_ascii_only method"""
    print("Testing _is_ascii_only method...")

    # Create a reducer instance
    reducer = GeneticProbabilityReducer("test-model", "test text")

    # Test cases
    test_cases = [
        ("hello", True),           # Pure ASCII
        ("world123", True),        # ASCII alphanumeric
        ("Test!@#$%", True),       # ASCII with symbols
        ("café", False),           # Unicode
        ("测试", False),           # Chinese
        ("emoji🚀", False),        # Emoji
        ("Ñoël", False),          # Unicode letters
        ("München", False),        # Unicode umlaut
        ("tab\there", True),       # ASCII with tab
        ("\n\r\t", True),          # ASCII control chars
        ("test\x00end", True),     # ASCII with null byte
        ("", True),                # Empty string
        (" ", True),               # Space only
        ("normal text", True),     # ASCII with space
        ("mixed café test", False), # Mixed ASCII/Unicode
    ]

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result = reducer._is_ascii_only(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' -> {result} (expected: {expected})")

        if result == expected:
            passed += 1
        else:
            failed += 1

    print(f"_is_ascii_only tests: {passed} passed, {failed} failed\n")
    return failed == 0


def test_filter_ascii_tokens():
    """Test the _filter_ascii_tokens method"""
    print("Testing _filter_ascii_tokens method...")

    # Create a reducer instance with mock tokenizer
    reducer = GeneticProbabilityReducer("test-model", "test text")
    reducer.tokenizer = create_mock_tokenizer()

    # Test with mixed token IDs
    test_token_ids = [
        12345,  # "hello" - ASCII
        67890,  # "café" - Unicode
        11111,  # "world" - ASCII
        22222,  # "测试" - Chinese
        33333,  # "test\x00end" - ASCII with null
        44444,  # "<|endoftext|>" - ASCII special token
        55555,  # "emoji🚀text" - Emoji
        66666,  # "normal" - ASCII
        77777,  # "Ñoël" - Unicode
        88888,  # "tab\there" - ASCII with tab
        99999,  # "\n\r\t" - ASCII control only
        10101,  # "München" - Unicode
        20202,  # "" - Empty string
        30303,  # " " - Space only
    ]

    # Expected ASCII-only tokens
    expected_ascii = [12345, 11111, 33333, 44444, 66666, 88888, 99999, 20202, 30303]

    # Filter tokens
    filtered_tokens = reducer._filter_ascii_tokens(test_token_ids)

    print(f"  Input tokens: {len(test_token_ids)}")
    print(f"  Filtered tokens: {len(filtered_tokens)}")
    print(f"  Expected ASCII tokens: {len(expected_ascii)}")

    # Check if filtering worked correctly
    success = set(filtered_tokens) == set(expected_ascii)
    status = "✓" if success else "✗"
    print(f"  {status} Filtering result matches expected")

    if not success:
        print(f"    Expected: {sorted(expected_ascii)}")
        print(f"    Got:      {sorted(filtered_tokens)}")
        print(f"    Missing:  {set(expected_ascii) - set(filtered_tokens)}")
        print(f"    Extra:    {set(filtered_tokens) - set(expected_ascii)}")

    # Show details of filtered tokens
    print("\n  Filtered token details:")
    for token_id in filtered_tokens:
        decoded = reducer.tokenizer.decode([token_id])
        print(f"    {token_id}: '{decoded}'")

    print()
    return success


def test_load_glitch_tokens_with_ascii_filter():
    """Test load_glitch_tokens with ASCII filtering enabled"""
    print("Testing load_glitch_tokens with ASCII filtering...")

    # Create temporary token file
    test_tokens = [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_tokens, f)
        temp_file = f.name

    try:
        # Create reducer with mock tokenizer
        reducer = GeneticProbabilityReducer("test-model", "test text")
        reducer.tokenizer = create_mock_tokenizer()

        # Load tokens without ASCII filtering
        reducer.load_glitch_tokens(temp_file, ascii_only=False)
        all_tokens_count = len(reducer.glitch_tokens)
        print(f"  Without ASCII filtering: {all_tokens_count} tokens")

        # Load tokens with ASCII filtering
        reducer.load_glitch_tokens(temp_file, ascii_only=True)
        ascii_tokens_count = len(reducer.glitch_tokens)
        print(f"  With ASCII filtering: {ascii_tokens_count} tokens")

        # Should have fewer tokens after filtering
        success = ascii_tokens_count < all_tokens_count
        status = "✓" if success else "✗"
        print(f"  {status} ASCII filtering reduced token count")

        # Check specific tokens that should remain
        expected_remaining = [12345, 11111, 33333, 44444, 66666]  # ASCII tokens from test set
        actual_remaining = set(reducer.glitch_tokens)
        expected_set = set(expected_remaining)

        tokens_match = actual_remaining == expected_set
        status2 = "✓" if tokens_match else "✗"
        print(f"  {status2} Correct tokens remain after filtering")

        if not tokens_match:
            print(f"    Expected: {sorted(expected_set)}")
            print(f"    Got:      {sorted(actual_remaining)}")

        print()
        return success and tokens_match

    finally:
        # Clean up temp file
        os.unlink(temp_file)


def test_edge_cases():
    """Test edge cases for ASCII filtering"""
    print("Testing edge cases...")

    reducer = GeneticProbabilityReducer("test-model", "test text")

    # Test edge cases for _is_ascii_only
    edge_cases = [
        ("", True, "empty string"),
        (" " * 100, True, "long ASCII spaces"),
        ("a" * 1000, True, "long ASCII string"),
        ("\x00\x01\x02", True, "control characters"),
        ("\x7f", True, "DEL character (127)"),
        ("\x80", False, "first non-ASCII (128)"),
        ("test\xff", False, "ASCII with high byte"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in edge_cases:
        result = reducer._is_ascii_only(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: {result} (expected: {expected})")

        if result == expected:
            passed += 1
        else:
            failed += 1

    print(f"Edge case tests: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all ASCII filtering tests"""
    print("🧪 ASCII Filtering Test Suite")
    print("=" * 50)

    tests = [
        ("ASCII Detection", test_is_ascii_only),
        ("Token Filtering", test_filter_ascii_tokens),
        ("Integration Test", test_load_glitch_tokens_with_ascii_filter),
        ("Edge Cases", test_edge_cases),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! ASCII filtering is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
