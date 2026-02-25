#!/usr/bin/env python3
"""
Comprehensive Validation Explanation and Demo Script

This script demonstrates how glitcher's validation systems work to detect
glitch tokens and avoid false positives. It shows the difference between
standard and enhanced validation methods.

Author: Glitcher Development Team
"""

import torch
import json
import sys
import os
from typing import List, Dict, Any, Tuple

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from glitcher.model import (
    initialize_model_and_tokenizer,
    get_template_for_model,
    strictly_glitch_verify,
    glitch_verify_message1,
    glitch_verify_message2,
    glitch_verify_message3
)
from glitcher.enhanced_validation import enhanced_glitch_verify

def explain_validation_concepts():
    """Explain the core concepts of validation in glitcher"""
    print("=" * 80)
    print("GLITCHER VALIDATION SYSTEM EXPLANATION")
    print("=" * 80)

    print("""
🎯 WHAT ARE GLITCH TOKENS?
Glitch tokens are tokens in a language model's vocabulary that cause unexpected
behavior when the model encounters them. They often:
- Produce incoherent or random output
- Break the model's normal reasoning patterns
- Generate repetitive or nonsensical text
- Cause the model to ignore instructions

🔍 WHY DO WE NEED VALIDATION?
Simple entropy-based mining can produce many false positives - tokens that
appear to be glitches but actually behave normally in real usage. Validation
systems help distinguish true glitch tokens from false positives.

📊 KEY METRICS:
• ASR (Attack Success Rate): Percentage of attempts where token exhibits glitch behavior
• Probability Threshold: Minimum probability for considering a token "normal"
• Content Filtering: Pattern-based filtering for known false positive types

🎪 TWO VALIDATION METHODS:

1️⃣  STANDARD VALIDATION (strictly_glitch_verify):
   • Uses immediate next-token prediction
   • Checks 3 different prompt formats
   • Very strict probability thresholds (0.00001 - 0.00005)
   • Binary result: glitch or not glitch
   • Fast but can miss some edge cases

2️⃣  ENHANCED VALIDATION (enhanced_glitch_verify):
   • Generates full text sequences (up to max_tokens)
   • Searches for target token in generated text
   • Uses ASR across multiple attempts
   • Handles non-deterministic model behavior
   • More thorough but slower
""")

def explain_asr_system():
    """Explain the Attack Success Rate (ASR) system"""
    print("\n" + "=" * 80)
    print("ATTACK SUCCESS RATE (ASR) SYSTEM")
    print("=" * 80)

    print("""
🎯 ASR DEFINITION:
ASR = (Number of attempts showing glitch behavior) / (Total attempts) × 100%

🔄 HOW IT WORKS:
1. Run validation multiple times (num_attempts parameter)
2. Count how many attempts show glitch behavior
3. Calculate ASR percentage
4. Compare against threshold to make final decision

📈 ASR THRESHOLDS:
• ASR = 1.0 (100%): Token ALWAYS exhibits glitch behavior (strictest)
• ASR = 0.8 (80%):  Token exhibits glitch behavior 80%+ of time (high confidence)
• ASR = 0.5 (50%):  Token exhibits glitch behavior 50%+ of time (balanced, default)
• ASR = 0.3 (30%):  Token exhibits glitch behavior 30%+ of time (lenient)
• ASR = 0.0 (0%):   Any glitch behavior detected (most permissive)

💡 EXAMPLE SCENARIOS:

Token tested 5 times, glitch behavior detected in 3 attempts:
• ASR = 3/5 = 0.6 (60%)
• With --asr-threshold 0.5: Token classified as GLITCH ✓
• With --asr-threshold 0.7: Token classified as NORMAL ✗
• With --asr-threshold 1.0: Token classified as NORMAL ✗

🎪 WHY USE ASR?
Language models can be non-deterministic, especially with:
• Temperature > 0
• Different random seeds
• Model quantization effects
• Hardware variations

ASR accounts for this variability by requiring consistent glitch behavior
across multiple attempts rather than relying on a single test.
""")

def explain_false_positive_detection():
    """Explain false positive detection mechanisms"""
    print("\n" + "=" * 80)
    print("FALSE POSITIVE DETECTION MECHANISMS")
    print("=" * 80)

    print("""
🚫 WHAT ARE FALSE POSITIVES?
Tokens that appear to be glitches in initial screening but actually behave
normally when properly tested. Common causes:
• Probability calculation artifacts
• Unusual but valid tokens (programming terms, foreign characters)
• Context-dependent behavior
• Tokenization edge cases

🛡️ DETECTION MECHANISMS:

1️⃣  CONTENT-BASED FILTERING:
   Automatically skip tokens with known false positive patterns:
   • Brackets: '[', ']', '(', ')'
   • Programming symbols: '_', '$', '.'
   • Common prefixes: 'arg', 'prop', 'char'

   Example false positives caught:
   • '[INST]' - instruction formatting token
   • 'args' - programming parameter
   • '$var' - variable reference

2️⃣  PROBABILITY THRESHOLDS:
   Very strict thresholds for immediate next-token probability:
   • Llama 3.2: 0.00001 (0.001%)
   • Other models: 0.00005 (0.005%)

   If token has higher probability than threshold → likely not a glitch

3️⃣  MULTIPLE TEST FORMATS:
   Three different prompt formats to test token behavior:
   • Format 1: Direct completion
   • Format 2: Conversational format
   • Format 3: Instruction format

   Token must fail ALL formats to be considered a glitch

4️⃣  SEQUENCE GENERATION:
   Enhanced validation generates full text sequences and searches for:
   • Target token appearing in generated sequence
   • Token text appearing in generated text
   • Coherent vs incoherent output patterns

🎯 VALIDATION LOGIC:

Standard Validation Decision Tree:
├─ Content filtering: Skip known false positive patterns
├─ Probability check: If prob > threshold → NOT glitch
├─ Top token check: If token is top prediction → NOT glitch
└─ Multiple formats: Must fail ALL formats → GLITCH

Enhanced Validation Decision Tree:
├─ Content filtering: Same as standard
├─ Generate sequences: Create text with target token as context
├─ Search sequences: Look for token in generated text
├─ Multiple attempts: Repeat process num_attempts times
├─ Calculate ASR: Count glitch attempts / total attempts
└─ Threshold check: ASR >= threshold → GLITCH
""")

def demonstrate_validation_step_by_step(model, tokenizer, token_id: int, chat_template):
    """Demonstrate validation process step by step"""
    print(f"\n" + "=" * 80)
    print(f"STEP-BY-STEP VALIDATION DEMO: Token ID {token_id}")
    print("=" * 80)

    # Decode token
    token = tokenizer.decode([token_id])
    print(f"🎯 Target Token: '{token}' (ID: {token_id})")

    print(f"\n📝 STEP 1: GENERATE TEST PROMPTS")
    prompt1 = glitch_verify_message1(chat_template, token)
    prompt2 = glitch_verify_message2(chat_template, token)
    prompt3 = glitch_verify_message3(chat_template, token)

    print(f"   Format 1 (50 chars): '{prompt1[:50]}...'")
    print(f"   Format 2 (50 chars): '{prompt2[:50]}...'")
    print(f"   Format 3 (50 chars): '{prompt3[:50]}...'")

    print(f"\n🔍 STEP 2: CONTENT-BASED FILTERING")
    # Check content filtering
    should_skip = (
        ('[' in token or ']' in token) or
        ('(' in token or ')' in token) or
        ('_' in token) or
        ('arg' in token.lower()) or
        ('prop' in token.lower()) or
        ('char' in token.lower()) or
        (token.startswith('.')) or
        (token.startswith('$'))
    )

    if should_skip:
        print(f"   ❌ FILTERED: Token contains false positive patterns")
        print(f"   🏁 RESULT: NOT A GLITCH (filtered)")
        return False, 0.0
    else:
        print(f"   ✅ PASSED: No false positive patterns detected")

    print(f"\n📊 STEP 3: STANDARD VALIDATION")
    is_glitch_standard = strictly_glitch_verify(model, tokenizer, token_id, chat_template)
    print(f"   Standard result: {'GLITCH' if is_glitch_standard else 'NOT GLITCH'}")

    print(f"\n🚀 STEP 4: ENHANCED VALIDATION")
    print(f"   Running 3 attempts with ASR threshold 0.5...")
    is_glitch_enhanced, asr = enhanced_glitch_verify(
        model, tokenizer, token_id, chat_template,
        max_tokens=20, num_attempts=3, asr_threshold=0.5, quiet=True
    )

    print(f"   Enhanced result: {'GLITCH' if is_glitch_enhanced else 'NOT GLITCH'}")
    print(f"   ASR: {asr:.2%} ({'≥' if asr >= 0.5 else '<'} 50% threshold)")

    print(f"\n🏁 FINAL COMPARISON:")
    print(f"   Standard:  {'✓ GLITCH' if is_glitch_standard else '✗ NOT GLITCH'}")
    print(f"   Enhanced:  {'✓ GLITCH' if is_glitch_enhanced else '✗ NOT GLITCH'} (ASR: {asr:.1%})")

    if is_glitch_standard != is_glitch_enhanced:
        print(f"   ⚠️  DISAGREEMENT: Methods produced different results!")
        print(f"   This shows why enhanced validation with ASR is more reliable.")
    else:
        print(f"   ✅ AGREEMENT: Both methods agree on classification.")

    return is_glitch_enhanced, asr

def run_comprehensive_demo(model_path: str, test_tokens: List[int]):
    """Run comprehensive validation demo"""
    print("Loading model for comprehensive validation demo...")

    # Load model with int4 quantization for memory efficiency
    model, tokenizer = initialize_model_and_tokenizer(model_path, quant_type="int4")
    chat_template = get_template_for_model(model_path, tokenizer)

    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Template: {chat_template.template_name if hasattr(chat_template, 'template_name') else 'Unknown'}")

    # Run conceptual explanations
    explain_validation_concepts()
    explain_asr_system()
    explain_false_positive_detection()

    # Demo validation on test tokens
    print(f"\n" + "=" * 80)
    print("LIVE VALIDATION DEMONSTRATIONS")
    print("=" * 80)

    results = []
    for i, token_id in enumerate(test_tokens):
        try:
            is_glitch, asr = demonstrate_validation_step_by_step(
                model, tokenizer, token_id, chat_template
            )
            results.append({
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "is_glitch": is_glitch,
                "asr": asr
            })

            if i < len(test_tokens) - 1:
                input("\nPress Enter to continue to next token...")

        except Exception as e:
            print(f"❌ Error testing token {token_id}: {e}")
            continue

    # Summary
    print(f"\n" + "=" * 80)
    print("VALIDATION DEMO SUMMARY")
    print("=" * 80)

    glitch_count = sum(1 for r in results if r["is_glitch"])
    total_count = len(results)

    print(f"📊 Results: {glitch_count}/{total_count} tokens classified as glitches")
    print(f"\nDetailed Results:")
    for r in results:
        status = "GLITCH" if r["is_glitch"] else "NORMAL"
        print(f"   Token {r['token_id']} ('{r['token']}'): {status} (ASR: {r['asr']:.1%})")

    print(f"\n💡 KEY TAKEAWAYS:")
    print(f"• Enhanced validation uses ASR to handle non-deterministic behavior")
    print(f"• False positive filtering prevents common misclassifications")
    print(f"• Multiple test formats ensure robust validation")
    print(f"• ASR thresholds allow tuning sensitivity vs. specificity")

    # Save results
    results_file = "validation_demo_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model_path": model_path,
            "test_tokens": test_tokens,
            "results": results,
            "summary": {
                "total_tested": total_count,
                "glitches_found": glitch_count,
                "glitch_rate": glitch_count / total_count if total_count > 0 else 0
            }
        }, f, indent=2)

    print(f"\n📁 Results saved to: {results_file}")

def demonstrate_asr_thresholds(model_path: str, token_id: int):
    """Demonstrate how different ASR thresholds affect classification"""
    print(f"\n" + "=" * 80)
    print(f"ASR THRESHOLD DEMONSTRATION")
    print("=" * 80)

    model, tokenizer = initialize_model_and_tokenizer(model_path, quant_type="int4")
    chat_template = get_template_for_model(model_path, tokenizer)

    token = tokenizer.decode([token_id])
    print(f"🎯 Testing Token: '{token}' (ID: {token_id})")

    # Test with different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0]
    num_attempts = 5

    print(f"📊 Running {num_attempts} attempts with different ASR thresholds:")

    # Get base ASR by running enhanced validation once
    _, base_asr = enhanced_glitch_verify(
        model, tokenizer, token_id, chat_template,
        max_tokens=20, num_attempts=num_attempts, asr_threshold=0.5, quiet=True
    )

    print(f"\n🎲 Token achieved ASR of {base_asr:.1%} across {num_attempts} attempts")
    print(f"\n📈 Classification with different thresholds:")

    for threshold in thresholds:
        is_glitch = base_asr >= threshold
        status = "GLITCH" if is_glitch else "NORMAL"
        symbol = "✓" if is_glitch else "✗"
        print(f"   Threshold {threshold:.0%}: {symbol} {status}")

    print(f"\n💡 INTERPRETATION:")
    if base_asr == 1.0:
        print(f"• Token exhibits glitch behavior in ALL attempts (100% ASR)")
        print(f"• High-confidence glitch token")
    elif base_asr >= 0.8:
        print(f"• Token exhibits glitch behavior in most attempts ({base_asr:.0%} ASR)")
        print(f"• Likely glitch token")
    elif base_asr >= 0.5:
        print(f"• Token exhibits glitch behavior in some attempts ({base_asr:.0%} ASR)")
        print(f"• Possible glitch token (default threshold)")
    elif base_asr > 0.0:
        print(f"• Token exhibits glitch behavior occasionally ({base_asr:.0%} ASR)")
        print(f"• Likely normal token with some edge cases")
    else:
        print(f"• Token never exhibits glitch behavior (0% ASR)")
        print(f"• Normal token")

    print(f"\n🎛️ THRESHOLD RECOMMENDATIONS:")
    print(f"• Research/Academic: Use 0.8-1.0 for high confidence")
    print(f"• General Discovery: Use 0.5 (default) for balanced detection")
    print(f"• Broad Screening: Use 0.3-0.4 for comprehensive analysis")
    print(f"• Non-deterministic Models: Use lower thresholds with more attempts")

def main():
    """Main demonstration function"""
    # Configuration
    MIXTRAL_MODEL_PATH = os.environ.get("GLITCHER_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")
    TEST_TOKENS = [1000, 2000, 3000, 5000, 10000]  # Sample tokens for testing

    print("🎪 GLITCHER VALIDATION SYSTEM COMPREHENSIVE DEMO")
    print("=" * 80)
    print(f"Model: {MIXTRAL_MODEL_PATH}")
    print(f"Test Tokens: {TEST_TOKENS}")

    try:
        # Run main demo
        run_comprehensive_demo(MIXTRAL_MODEL_PATH, TEST_TOKENS)

        # Demo ASR thresholds on one token
        print(f"\n🎯 Running ASR threshold demo on token {TEST_TOKENS[0]}...")
        demonstrate_asr_thresholds(MIXTRAL_MODEL_PATH, TEST_TOKENS[0])

        print(f"\n🎉 Demo completed successfully!")
        print(f"Check the generated files for detailed results and logs.")

    except KeyboardInterrupt:
        print(f"\n❌ Demo interrupted by user")
        return 1

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
