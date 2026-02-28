#!/usr/bin/env python3
"""
ASR Stochasticity Audit Harness

Tests whether the enhanced_validation.py ASR pipeline produces genuinely
independent, stochastic outputs across attempts -- or whether greedy
decoding (do_sample=False) makes every attempt identical.

Usage:
    python experiments/asr_audit.py MODEL [--token-ids 8673,10939,...] \
        [--num-attempts 10] [--patched] [--output data/asr_audit_raw.json]
"""

import argparse
import json
import os
import sys
import time

import torch

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glitcher.model import initialize_model_and_tokenizer, get_template_for_model
from glitcher.model import glitch_verify_message1, glitch_verify_message2, glitch_verify_message3


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_once(model, tokenizer, input_ids, max_tokens, patched, attempt_idx):
    """
    Run a single generation call.

    If *patched* is True, use do_sample=True with temperature=0.7 and a
    unique seed per attempt so that outputs genuinely vary.
    """
    device = model.device
    attention_mask = torch.ones_like(input_ids)

    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "get_added_vocab"):
        eot = tokenizer.get_added_vocab().get("<|eot_id|>")
        if eot:
            stop_token_ids.append(eot)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_token_ids,
        use_cache=True,
    )

    if patched:
        # Stochastic generation with a fresh seed per attempt
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.95
        torch.manual_seed(int(time.time() * 1000) + attempt_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(time.time() * 1000) + attempt_idx)
    else:
        # Original (buggy) greedy decoding
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)

    new_tokens = generated_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "token_ids": new_tokens.tolist(),
        "text": generated_text,
        "generation_params": {
            "do_sample": gen_kwargs.get("do_sample"),
            "temperature": gen_kwargs.get("temperature"),
            "top_p": gen_kwargs.get("top_p"),
        },
    }


def _is_glitch_indicator(token_id, token_text, new_token_ids, generated_text):
    """
    Replicate the anomaly indicator from enhanced_validation.py.
    Returns True if the token is detected as a glitch (not found).
    """
    stripped = token_text.strip()
    text_found = (token_text in generated_text) or (stripped and stripped in generated_text)
    id_found = token_id in new_token_ids
    token_found_in_sequence = id_found or text_found
    token_text_found = text_found
    return not (token_found_in_sequence or token_text_found)


# ---------------------------------------------------------------------------
# Main audit loop
# ---------------------------------------------------------------------------

def audit_token(model, tokenizer, token_id, num_attempts, max_tokens, patched):
    """Run the audit for a single token."""
    device = model.device
    token_text = tokenizer.decode([token_id])
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    prompts = [
        glitch_verify_message1(chat_template, token_text),
        glitch_verify_message2(chat_template, token_text),
        glitch_verify_message3(chat_template, token_text),
    ]
    input_ids_list = [
        tokenizer.encode(p, return_tensors="pt").to(device) for p in prompts
    ]

    attempts = []
    for attempt_idx in range(num_attempts):
        test_results = []
        tests_failed = 0
        for test_idx, input_ids in enumerate(input_ids_list):
            out = _generate_once(model, tokenizer, input_ids, max_tokens, patched, attempt_idx)
            is_glitch = _is_glitch_indicator(
                token_id, token_text, out["token_ids"], out["text"]
            )
            if is_glitch:
                tests_failed += 1
            test_results.append({
                "test_idx": test_idx,
                "text": out["text"][:300],
                "token_ids_head": out["token_ids"][:30],
                "is_glitch": is_glitch,
                "generation_params": out["generation_params"],
            })
        attempt_is_glitch = tests_failed == len(input_ids_list)
        attempts.append({
            "attempt_idx": attempt_idx,
            "tests_failed": tests_failed,
            "attempt_is_glitch": attempt_is_glitch,
            "test_results": test_results,
        })

    # --- Compute diagnostics ---
    # Unique outputs per test prompt across attempts
    unique_per_test = {}
    for test_idx in range(len(input_ids_list)):
        texts = [a["test_results"][test_idx]["text"] for a in attempts]
        unique_per_test[test_idx] = len(set(texts))

    # Per-token ASR
    glitch_count = sum(1 for a in attempts if a["attempt_is_glitch"])
    asr = glitch_count / num_attempts if num_attempts else 0.0

    # Are all outputs identical?
    all_identical_per_test = {k: (v == 1) for k, v in unique_per_test.items()}
    all_tests_identical = all(all_identical_per_test.values())

    # Indicator consistency: does indicator give same result on all attempts?
    indicators_per_test = {}
    for test_idx in range(len(input_ids_list)):
        vals = [a["test_results"][test_idx]["is_glitch"] for a in attempts]
        indicators_per_test[test_idx] = {
            "all_same": len(set(vals)) == 1,
            "values": vals,
        }

    return {
        "token_id": token_id,
        "token_text": token_text,
        "patched": patched,
        "num_attempts": num_attempts,
        "asr": asr,
        "glitch_count": glitch_count,
        "unique_outputs_per_test": unique_per_test,
        "all_outputs_identical": all_tests_identical,
        "indicator_consistency_per_test": indicators_per_test,
        "attempts": attempts,
    }


def main():
    parser = argparse.ArgumentParser(description="ASR Stochasticity Audit")
    parser.add_argument("model", help="Model path or name")
    parser.add_argument(
        "--token-ids",
        type=str,
        default=None,
        help="Comma-separated token IDs to test. If not provided, uses glitch_tokens.json",
    )
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--patched", action="store_true", help="Use stochastic generation")
    parser.add_argument("--output", type=str, default="data/asr_audit_raw.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quant-type", type=str, default="bfloat16")
    args = parser.parse_args()

    # Resolve token IDs
    if args.token_ids:
        token_ids = [int(x.strip()) for x in args.token_ids.split(",")]
    else:
        # Try loading from glitch_tokens.json
        gt_path = os.path.join(os.path.dirname(__file__), "..", "glitch_tokens.json")
        if os.path.exists(gt_path):
            with open(gt_path) as f:
                data = json.load(f)
            token_ids = list(set(data.get("glitch_token_ids", [])))
        else:
            print("No --token-ids given and glitch_tokens.json not found.", file=sys.stderr)
            sys.exit(1)

    print(f"Auditing {len(token_ids)} tokens, {args.num_attempts} attempts each "
          f"({'PATCHED' if args.patched else 'ORIGINAL'})")
    print(f"Model: {args.model}")

    model, tokenizer = initialize_model_and_tokenizer(args.model, args.device, args.quant_type)
    model.eval()

    results = []
    for idx, tid in enumerate(token_ids):
        print(f"\n[{idx+1}/{len(token_ids)}] Token ID {tid} "
              f"('{tokenizer.decode([tid])}')")
        result = audit_token(model, tokenizer, tid, args.num_attempts, args.max_tokens, args.patched)
        results.append(result)
        print(f"  ASR={result['asr']:.0%}  "
              f"all_identical={result['all_outputs_identical']}  "
              f"unique_per_test={result['unique_outputs_per_test']}")

    # --- Summary ---
    n_all_identical = sum(1 for r in results if r["all_outputs_identical"])
    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_all_identical}/{len(results)} tokens had identical outputs "
          f"across all {args.num_attempts} attempts ({n_all_identical/len(results)*100:.0f}%)")
    asr_values = [r["asr"] for r in results]
    asr_0 = sum(1 for a in asr_values if a == 0.0)
    asr_100 = sum(1 for a in asr_values if a == 1.0)
    asr_intermediate = sum(1 for a in asr_values if 0.0 < a < 1.0)
    print(f"ASR distribution: 0%={asr_0}  100%={asr_100}  intermediate={asr_intermediate}")
    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "model": args.model,
        "patched": args.patched,
        "num_attempts": args.num_attempts,
        "timestamp": time.time(),
        "summary": {
            "total_tokens": len(results),
            "all_identical_count": n_all_identical,
            "all_identical_fraction": n_all_identical / len(results) if results else 0,
            "asr_0_count": asr_0,
            "asr_100_count": asr_100,
            "asr_intermediate_count": asr_intermediate,
        },
        "tokens": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
