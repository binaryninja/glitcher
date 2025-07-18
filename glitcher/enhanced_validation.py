"""Enhanced glitch token validation with sequence generation"""

import torch
import torch.nn.functional as F
import json
import warnings
import os
from .model import get_template_for_model, glitch_verify_message1, glitch_verify_message2, glitch_verify_message3


def enhanced_glitch_verify(model, tokenizer, token_id, chat_template=None, log_file=None, max_tokens=100, quiet=True, num_attempts=1, asr_threshold=0.5):
    """
    Enhanced verification that generates a sequence of tokens and searches for the target token.
    Uses Attack Success Rate (ASR) to determine if a token is a glitch.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_id: The token ID to verify
        chat_template: Chat template for formatting (optional)
        log_file: File to log detailed results (optional)
        max_tokens: Maximum number of tokens to generate and search through
        quiet: Whether to suppress transformer warnings (default: True)
        num_attempts: Number of times to run the validation (default: 1)
        asr_threshold: ASR threshold for considering token a glitch (default: 0.5)

    Returns:
        tuple: (is_glitch: bool, asr: float) - validation result and attack success rate
    """
    # Get model device
    device = model.device

    # Decode token_id to get text
    token = tokenizer.decode([token_id])

    # Get chat template for model if not provided
    if chat_template is None:
        chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Use proper chat templates for validation
    formatted_input1 = glitch_verify_message1(chat_template, token)
    formatted_input2 = glitch_verify_message2(chat_template, token)
    formatted_input3 = glitch_verify_message3(chat_template, token)

    # Convert to input IDs
    input_ids1 = tokenizer.encode(formatted_input1, return_tensors='pt').to(device)
    input_ids2 = tokenizer.encode(formatted_input2, return_tensors='pt').to(device)
    input_ids3 = tokenizer.encode(formatted_input3, return_tensors='pt').to(device)

    inputs = [input_ids1, input_ids2, input_ids3]
    formatted_inputs = [formatted_input1, formatted_input2, formatted_input3]

    # Check if the token is a glitch token
    is_llama32 = "llama3.2" in model.config._name_or_path.lower() or "llama32" in model.config._name_or_path.lower()

    # Run multiple attempts to account for non-deterministic nature of LLMs
    all_attempts_results = []
    attempt_glitch_counts = []

    for attempt in range(num_attempts):
        tests_failed = 0
        tests_count = len(inputs)
        test_results = []

        for i, (input_ids, formatted_input) in enumerate(zip(inputs, formatted_inputs)):
            with torch.no_grad():
                # Create attention mask
                attention_mask = torch.ones_like(input_ids)

                # Set up proper stop tokens
                stop_token_ids = [tokenizer.eos_token_id]
                if hasattr(tokenizer, 'get_added_vocab'):
                    # Add <|eot_id|> if available
                    eot_token_id = tokenizer.get_added_vocab().get('<|eot_id|>')
                    if eot_token_id:
                        stop_token_ids.append(eot_token_id)

                # Suppress warnings if requested
                if quiet:
                    # Temporarily suppress transformers warnings
                    original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "info")
                    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

                        # Generate a sequence of tokens - use minimal parameters to avoid warnings
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_tokens,
                            do_sample=False,  # Use greedy decoding for consistency
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=stop_token_ids,
                            use_cache=True
                        )

                    # Restore original verbosity
                    os.environ["TRANSFORMERS_VERBOSITY"] = original_verbosity
                else:
                    # Generate a sequence of tokens
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        do_sample=False,  # Use greedy decoding for consistency
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=stop_token_ids,
                        use_cache=True
                    )

                # Extract only the newly generated tokens (excluding the input)
                new_tokens = generated_ids[0][input_ids.shape[1]:]

                # Convert to text for analysis
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Check if target token appears in the generated sequence
                token_found_in_sequence = token_id in new_tokens.tolist()

                # Also check if the token text appears in the generated text
                # This handles cases where the token might be generated as part of a larger token
                token_text_found = token.strip() in generated_text

                # Get the first few tokens for analysis
                first_5_tokens = new_tokens[:5].tolist()
                first_5_token_texts = [tokenizer.decode([tid]) for tid in first_5_tokens]

                # Check immediate next token probability (for comparison with old method)
                outputs = model(input_ids=input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                target_prob = next_token_probs[0, token_id].item()

                # Get top 5 predicted tokens for the immediate next position
                top5_values, top5_indices = torch.topk(next_token_probs[0], 5)
                top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices]
                top5_probs = top5_values.tolist()

                # Determine if this test indicates a glitch token
                # The token is NOT a glitch if:
                # 1. It appears in the generated sequence, OR
                # 2. The token text appears in the generated text
                is_test_glitch = not (token_found_in_sequence or token_text_found)

                # Store detailed test results
                test_result = {
                    "test_number": i + 1,
                    "formatted_input": formatted_input,
                    "generated_text": generated_text,
                    "generated_tokens": new_tokens.tolist(),
                    "target_token_id": token_id,
                    "target_token": token,
                    "token_found_in_sequence": token_found_in_sequence,
                    "token_text_found": token_text_found,
                    "first_5_tokens": first_5_token_texts,
                    "first_5_token_ids": first_5_tokens,
                    "immediate_next_token_probability": target_prob,
                    "top5_immediate_predictions": top5_tokens,
                    "top5_immediate_probabilities": top5_probs,
                    "is_test_glitch": is_test_glitch,
                    "max_tokens_generated": len(new_tokens),
                    "reasoning": "Token found in generated sequence" if token_found_in_sequence else
                               "Token text found in generated text" if token_text_found else
                               "Token not found in generated sequence or text"
                }
                test_results.append(test_result)

                if is_test_glitch:
                    tests_failed += 1
                elif not is_llama32:
                    # For non-Llama 3.2 models, if any test passes, it's not a glitch
                    break

        # Determine result for this attempt
        if is_llama32:
            # For Llama 3.2, token is a glitch only if ALL tests failed
            attempt_is_glitch = tests_failed == tests_count
        else:
            # For other models, if any test passes, it's not a glitch
            attempt_is_glitch = tests_failed == tests_count

        # Store results for this attempt
        attempt_result = {
            "attempt": attempt + 1,
            "tests_failed": tests_failed,
            "tests_count": tests_count,
            "is_glitch": attempt_is_glitch,
            "test_results": test_results
        }
        all_attempts_results.append(attempt_result)
        attempt_glitch_counts.append(1 if attempt_is_glitch else 0)

    # Determine final result based on Attack Success Rate (ASR)
    glitch_attempts = sum(attempt_glitch_counts)
    asr = glitch_attempts / num_attempts if num_attempts > 0 else 0.0
    is_glitch = asr >= asr_threshold

    # Log verification details if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            verification_details = {
                "event": "enhanced_token_verification",
                "token": token,
                "token_id": token_id,
                "is_llama32": is_llama32,
                "num_attempts": num_attempts,
                "glitch_attempts": glitch_attempts,
                "asr": asr,
                "asr_threshold": asr_threshold,
                "is_glitch": is_glitch,
                "max_tokens": max_tokens,
                "all_attempts_results": all_attempts_results,
                "final_decision_reason": f"ASR {asr:.2%} >= threshold {asr_threshold:.2%}" if is_glitch else
                                        f"ASR {asr:.2%} < threshold {asr_threshold:.2%}"
            }
            f.write(json.dumps(verification_details) + "\n")

    return is_glitch, asr


def batch_enhanced_verify(model, tokenizer, token_ids, chat_template=None, log_file=None, max_tokens=100, quiet=True, num_attempts=1, asr_threshold=0.5):
    """
    Batch version of enhanced verification for multiple tokens.
    Uses Attack Success Rate (ASR) to determine if tokens are glitches.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_ids: List of token IDs to verify
        chat_template: Chat template for formatting (optional)
        log_file: File to log detailed results (optional)
        max_tokens: Maximum number of tokens to generate and search through
        quiet: Whether to suppress transformer warnings (default: True)
        num_attempts: Number of times to run the validation per token (default: 1)
        asr_threshold: ASR threshold for considering token a glitch (default: 0.5)

    Returns:
        List[bool]: List of boolean results for each token
    """
    results = []

    for i, token_id in enumerate(token_ids):
        try:
            token = tokenizer.decode([token_id])
            is_glitch, asr = enhanced_glitch_verify(
                model, tokenizer, token_id, chat_template, log_file, max_tokens, quiet, num_attempts, asr_threshold
            )

            result = {
                "token_id": token_id,
                "token": token,
                "is_glitch": is_glitch
            }
            results.append(result)

            print(f"[{i+1}/{len(token_ids)}] Token: '{token}', ID: {token_id}, Is glitch: {is_glitch}")

        except Exception as e:
            print(f"Error testing token {token_id}: {e}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        "event": "token_test_error",
                        "token_id": token_id,
                        "error": str(e)
                    }) + "\n")

    return results


def compare_validation_methods(model, tokenizer, token_id, chat_template=None, max_tokens=100, quiet=True, num_attempts=1, asr_threshold=0.5):
    """
    Compare the original validation method with the enhanced method.
    Enhanced method uses Attack Success Rate (ASR) for determination.

    Args:
        model: The language model
        tokenizer: The tokenizer
        token_id: The token ID to verify
        chat_template: Chat template for formatting (optional)
        max_tokens: Maximum number of tokens to generate and search through
        quiet: Whether to suppress transformer warnings (default: True)
        num_attempts: Number of times to run the enhanced validation (default: 1)
        asr_threshold: ASR threshold for enhanced validation (default: 0.5)

    Returns:
        dict: Comparison results including both methods' outputs and ASR data
    """
    from .model import strictly_glitch_verify

    # Run original validation
    original_result = strictly_glitch_verify(model, tokenizer, token_id, chat_template)

    # Run enhanced validation
    enhanced_result, enhanced_asr = enhanced_glitch_verify(model, tokenizer, token_id, chat_template, max_tokens=max_tokens, quiet=quiet, num_attempts=num_attempts, asr_threshold=asr_threshold)

    token = tokenizer.decode([token_id])

    return {
        "token_id": token_id,
        "token": token,
        "original_method": original_result,
        "enhanced_method": enhanced_result,
        "enhanced_asr": enhanced_asr,
        "methods_agree": original_result == enhanced_result,
        "difference": "Enhanced method is more lenient" if original_result and not enhanced_result else
                     "Enhanced method is more strict" if not original_result and enhanced_result else
                     "Methods agree"
    }
