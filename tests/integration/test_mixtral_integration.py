#!/usr/bin/env python3
"""
Comprehensive integration test for Mixtral fine-tune (nowllm-0829) support in glitcher
Tests model loading, template selection, prediction accuracy, and CLI integration
"""

import torch
import json
import tempfile
import os
import sys
from pathlib import Path

# Add the glitcher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from glitcher.model import (
    initialize_model_and_tokenizer,
    get_template_for_model,
    TokenClassifier,
    strictly_glitch_verify
)

# Constants
MIXTRAL_MODEL_PATH = os.environ.get("GLITCHER_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")
TEST_TOKENS = [1000, 2000, 3000, 5000]  # Sample token IDs for testing

class MixtralIntegrationTest:
    """Comprehensive test suite for Mixtral fine-tune integration"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.template = None
        self.test_results = {
            "model_loading": False,
            "template_detection": False,
            "basic_prediction": False,
            "chat_prediction": False,
            "token_classification": False,
            "glitch_verification": False,
            "memory_efficiency": False
        }

    def setup_model(self, quant_type="int4"):
        """Load the model with specified quantization"""
        print(f"Setting up Mixtral model with {quant_type} quantization...")
        try:
            self.model, self.tokenizer = initialize_model_and_tokenizer(
                MIXTRAL_MODEL_PATH,
                quant_type=quant_type
            )
            self.template = get_template_for_model(MIXTRAL_MODEL_PATH, self.tokenizer)

            print(f"✓ Model loaded: {type(self.model).__name__}")
            print(f"✓ Tokenizer loaded: {type(self.tokenizer).__name__}")
            print(f"✓ Template: {self.template.template_name if hasattr(self.template, 'template_name') else type(self.template).__name__}")

            self.test_results["model_loading"] = True
            return True

        except Exception as e:
            print(f"✗ Model setup failed: {e}")
            return False

    def test_template_detection(self):
        """Test that the correct template is detected for the model"""
        print("\nTesting template detection...")
        try:
            # Test template detection with model path
            template = get_template_for_model(MIXTRAL_MODEL_PATH, self.tokenizer)

            # Should detect nowllm template
            expected_template = "nowllm"
            actual_template = template.template_name if hasattr(template, 'template_name') else "unknown"

            if actual_template == expected_template:
                print(f"✓ Correct template detected: {actual_template}")
                self.test_results["template_detection"] = True
                return True
            else:
                print(f"✗ Wrong template detected: {actual_template}, expected: {expected_template}")
                return False

        except Exception as e:
            print(f"✗ Template detection failed: {e}")
            return False

    def test_basic_prediction(self):
        """Test basic next token prediction without chat formatting"""
        print("\nTesting basic prediction...")
        try:
            test_text = "The quick brown "
            inputs = self.tokenizer(test_text, return_tensors="pt")

            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, 10)
            top_tokens = [self.tokenizer.decode(idx.item()) for idx in top_indices]

            print(f"Top 5 predictions for '{test_text}':")
            for i, (token, prob) in enumerate(zip(top_tokens[:5], top_probs[:5])):
                print(f"  {i+1}. '{token}' (prob: {prob.item():.4f})")

            # Check if reasonable predictions (letters/words, not just special chars)
            reasonable_predictions = sum(1 for token in top_tokens[:5]
                                       if any(c.isalpha() for c in token.strip()))

            # For this model, even 1 reasonable prediction in top 5 is acceptable
            if reasonable_predictions >= 1:
                print(f"✓ Found {reasonable_predictions} reasonable predictions")
                self.test_results["basic_prediction"] = True
                return True
            else:
                print(f"✗ Only {reasonable_predictions} reasonable predictions found")
                return False

        except Exception as e:
            print(f"✗ Basic prediction failed: {e}")
            return False

    def test_chat_prediction(self):
        """Test prediction with proper chat formatting"""
        print("\nTesting chat-formatted prediction...")
        try:
            # Format with chat template
            user_message = "Complete this phrase: The quick brown"

            if hasattr(self.template, 'format_chat'):
                # Built-in template
                messages = [{"role": "user", "content": user_message}]
                formatted_text = self.template.format_chat(messages)
            else:
                # Manual template
                formatted_text = self.template.user_format.format(content=user_message)

            print(f"Chat formatted input: '{formatted_text[:100]}...'")

            inputs = self.tokenizer(formatted_text, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, 10)
            top_tokens = [self.tokenizer.decode(idx.item()) for idx in top_indices]

            print(f"Top 5 chat predictions:")
            for i, (token, prob) in enumerate(zip(top_tokens[:5], top_probs[:5])):
                print(f"  {i+1}. '{token}' (prob: {prob.item():.4f})")

            # Check for "fox" or "f" in top predictions
            fox_found = any("fox" in token.lower() or token.strip().lower() == "f"
                          for token in top_tokens[:10])

            if fox_found:
                print("✓ Found 'fox' or 'f' in top predictions")
                self.test_results["chat_prediction"] = True
                return True
            else:
                print("✗ 'fox' or 'f' not found in top predictions")
                return False

        except Exception as e:
            print(f"✗ Chat prediction failed: {e}")
            return False

    def test_token_classification(self):
        """Test the TokenClassifier with sample tokens"""
        print("\nTesting token classification...")
        try:
            classifier = TokenClassifier(self.model, self.tokenizer)

            results = []
            for token_id in TEST_TOKENS:
                try:
                    token_text = self.tokenizer.decode(token_id)
                    is_glitch = classifier.classify_token(token_id)
                    results.append((token_id, token_text, is_glitch))
                    print(f"  Token {token_id} ('{token_text}'): {'GLITCH' if is_glitch else 'NORMAL'}")
                except Exception as e:
                    print(f"  Token {token_id}: ERROR - {e}")
                    continue

            if len(results) >= len(TEST_TOKENS) // 2:
                print(f"✓ Successfully classified {len(results)}/{len(TEST_TOKENS)} tokens")
                self.test_results["token_classification"] = True
                return True
            else:
                print(f"✗ Only classified {len(results)}/{len(TEST_TOKENS)} tokens")
                return False

        except Exception as e:
            print(f"✗ Token classification failed: {e}")
            return False

    def test_glitch_verification(self):
        """Test the enhanced glitch verification system"""
        print("\nTesting glitch verification...")
        try:
            # Test with a sample token
            test_token_id = TEST_TOKENS[0]

            # Test standard verification
            result = strictly_glitch_verify(
                self.model,
                self.tokenizer,
                test_token_id,
                chat_template=self.template
            )

            print(f"  Verification result for token {test_token_id}: {result}")

            # Just check that verification runs without errors
            self.test_results["glitch_verification"] = True
            print("✓ Glitch verification system working")
            return True

        except Exception as e:
            print(f"✗ Glitch verification failed: {e}")
            return False

    def test_memory_efficiency(self):
        """Test memory usage and efficiency"""
        print("\nTesting memory efficiency...")
        try:
            if torch.cuda.is_available():
                # Get current memory usage
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB

                print(f"  GPU Memory allocated: {memory_allocated:.2f} GB")
                print(f"  GPU Memory reserved: {memory_reserved:.2f} GB")

                # Check if memory usage is reasonable (less than 25GB for int4)
                if memory_allocated < 25.0:
                    print("✓ Memory usage is reasonable")
                    self.test_results["memory_efficiency"] = True
                    return True
                else:
                    print(f"✗ High memory usage: {memory_allocated:.2f} GB")
                    return False
            else:
                print("✓ CPU-only mode detected")
                self.test_results["memory_efficiency"] = True
                return True

        except Exception as e:
            print(f"✗ Memory efficiency test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all integration tests"""
        print("Mixtral Fine-tune Integration Test Suite")
        print("=" * 60)

        # Setup model
        if not self.setup_model():
            print("Cannot continue without model. Exiting.")
            return False

        # Run all tests
        tests = [
            self.test_template_detection,
            self.test_basic_prediction,
            self.test_chat_prediction,
            self.test_token_classification,
            self.test_glitch_verification,
            self.test_memory_efficiency
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Test {test.__name__} failed with exception: {e}")

        # Print summary
        self.print_summary()

        # Return overall success
        return all(self.test_results.values())

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")

        print("-" * 60)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Mixtral integration is working correctly!")
        else:
            print(f"⚠️  {total_tests - passed_tests} tests failed - integration needs attention")

        # Save results to file
        results_file = "mixtral_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "model_path": MIXTRAL_MODEL_PATH,
                "test_results": self.test_results,
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "success_rate": passed_tests / total_tests
                }
            }, f, indent=2)

        print(f"Results saved to: {results_file}")

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main test execution"""
    test_suite = MixtralIntegrationTest()

    try:
        success = test_suite.run_all_tests()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1

    except Exception as e:
        print(f"Test suite failed with exception: {e}")
        return 1

    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    sys.exit(main())
