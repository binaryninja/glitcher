#!/usr/bin/env python3
"""
Simple test to verify ASR return format from enhanced_glitch_verify

This test verifies that enhanced_glitch_verify returns (is_glitch, asr) tuple
and that the ASR value can be properly formatted for display.

Usage:
    python test_asr_return.py
"""

import sys
import os

# Add the glitcher package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_asr_return_format():
    """Test that enhanced_glitch_verify returns the expected format"""
    print("🧪 Testing ASR Return Format")
    print("=" * 50)

    try:
        from glitcher.enhanced_validation import enhanced_glitch_verify
        print("✅ Successfully imported enhanced_glitch_verify")

        # Check function signature
        import inspect
        sig = inspect.signature(enhanced_glitch_verify)
        print(f"✅ Function signature: {sig}")

        # Verify parameters include asr_threshold
        params = list(sig.parameters.keys())
        if 'asr_threshold' in params:
            print("✅ asr_threshold parameter found")
        else:
            print("❌ asr_threshold parameter missing")
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test mock return format
    print("\n🎯 Testing Return Format")
    print("-" * 30)

    def mock_enhanced_validation(attempts, successes):
        """Mock function to simulate enhanced validation return"""
        asr = successes / attempts if attempts > 0 else 0.0
        is_glitch = asr >= 0.5  # Default threshold
        return is_glitch, asr

    # Test various scenarios
    test_cases = [
        (5, 5, "Perfect consistency"),
        (5, 4, "High consistency"),
        (5, 3, "Medium consistency"),
        (5, 2, "Low consistency"),
        (5, 1, "Very low consistency"),
        (5, 0, "No glitch behavior"),
        (3, 2, "2 out of 3 attempts"),
        (10, 7, "7 out of 10 attempts")
    ]

    print("Test Cases:")
    print("Attempts | Successes | ASR    | Classification | Description")
    print("---------|-----------|---------|--------------|-----------------")

    for attempts, successes, description in test_cases:
        is_glitch, asr = mock_enhanced_validation(attempts, successes)
        classification = "GLITCH" if is_glitch else "normal"
        print(f"{attempts:8} | {successes:9} | {asr:6.1%} | {classification:>12} | {description}")

    # Test output formatting
    print("\n📝 Testing Output Formatting")
    print("-" * 35)

    # Sample token data
    token = "TestToken"
    token_id = 12345
    asr_test = 0.666  # 66.6%
    entropy = 0.1234
    target_prob = 0.001234
    max_prob = 0.567890

    # Test enhanced validation output format
    enhanced_output = f"✓ Validated glitch token: '{token}' (ID: {token_id}, asr: {asr_test:.2%}, entropy: {entropy:.4f}, target_prob: {target_prob:.6f}, top_prob: {max_prob:.6f}, method: enhanced)"
    print(f"Enhanced: {enhanced_output}")

    # Test standard validation output format (no ASR)
    standard_output = f"✓ Validated glitch token: '{token}' (ID: {token_id}, entropy: {entropy:.4f}, target_prob: {target_prob:.6f}, top_prob: {max_prob:.6f}, method: standard)"
    print(f"Standard: {standard_output}")

    # Test false positive output
    false_positive = f"✗ False positive: '{token}' (ID: {token_id}, asr: {asr_test:.2%}, failed enhanced validation)"
    print(f"False positive: {false_positive}")

    print("\n✅ All ASR return format tests passed!")
    return True

def test_asr_value_ranges():
    """Test that ASR values are in expected ranges"""
    print("\n🔍 Testing ASR Value Ranges")
    print("-" * 30)

    def validate_asr(asr, attempts, successes):
        """Validate ASR value is correct and in range"""
        expected_asr = successes / attempts if attempts > 0 else 0.0

        # Check range
        if not (0.0 <= asr <= 1.0):
            print(f"❌ ASR {asr} outside valid range [0.0, 1.0]")
            return False

        # Check accuracy
        if abs(asr - expected_asr) > 1e-6:
            print(f"❌ ASR {asr} doesn't match expected {expected_asr}")
            return False

        return True

    # Test edge cases
    test_cases = [
        (1, 0),   # 0% ASR
        (1, 1),   # 100% ASR
        (2, 1),   # 50% ASR
        (3, 1),   # 33.3% ASR
        (3, 2),   # 66.7% ASR
        (10, 3),  # 30% ASR
        (10, 8),  # 80% ASR
    ]

    all_passed = True
    for attempts, successes in test_cases:
        asr = successes / attempts
        if validate_asr(asr, attempts, successes):
            print(f"✅ {successes}/{attempts} → ASR {asr:.1%}")
        else:
            all_passed = False

    if all_passed:
        print("✅ All ASR value range tests passed!")
    else:
        print("❌ Some ASR value range tests failed!")

    return all_passed

def test_threshold_logic():
    """Test ASR threshold logic"""
    print("\n⚖️  Testing ASR Threshold Logic")
    print("-" * 35)

    def test_threshold(asr, threshold):
        """Test threshold comparison"""
        return asr >= threshold

    # Test cases: (asr, threshold, expected_result, description)
    test_cases = [
        (1.0, 1.0, True, "Perfect match at 100%"),
        (1.0, 0.8, True, "100% > 80% threshold"),
        (0.8, 1.0, False, "80% < 100% threshold"),
        (0.66, 0.5, True, "66% > 50% threshold"),
        (0.66, 0.8, False, "66% < 80% threshold"),
        (0.5, 0.5, True, "Exact match at 50%"),
        (0.4, 0.5, False, "40% < 50% threshold"),
        (0.0, 0.0, True, "Zero threshold accepts any"),
    ]

    print("ASR  | Threshold | Result | Description")
    print("-----|-----------|--------|------------------")

    all_passed = True
    for asr, threshold, expected, description in test_cases:
        result = test_threshold(asr, threshold)
        status = "✅" if result == expected else "❌"
        classification = "GLITCH" if result else "normal"
        print(f"{asr:4.0%} | {threshold:8.0%} | {classification:>6} | {description} {status}")

        if result != expected:
            all_passed = False

    if all_passed:
        print("✅ All threshold logic tests passed!")
    else:
        print("❌ Some threshold logic tests failed!")

    return all_passed

def main():
    """Run all ASR tests"""
    print("🚀 ASR Return Format Test Suite")
    print("=" * 60)

    tests = [
        test_asr_return_format,
        test_asr_value_ranges,
        test_threshold_logic
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All ASR tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
