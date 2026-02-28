#!/usr/bin/env python3
"""
Sampling-Based Vocabulary Scanner

Scans a stratified random sample of the full model vocabulary for glitch tokens,
using multi-attempt ASR measurement. Compares findings against the entropy-guided
mining pipeline to quantify coverage gaps.

Usage:
    python experiments/vocab_scanner.py MODEL [--sample-size 2000] \
        [--num-attempts 5] [--patched] [--checkpoint data/vocab_scan_results.json]
"""

import argparse
import json
import math
import os
import random
import sys
import time
import unicodedata

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from glitcher.model import (
    initialize_model_and_tokenizer,
    get_template_for_model,
    glitch_verify_message1,
    glitch_verify_message2,
    glitch_verify_message3,
)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _unicode_category(text):
    """Classify a token string by its dominant Unicode general category."""
    if not text.strip():
        return "whitespace"
    cats = [unicodedata.category(c) for c in text if not c.isspace()]
    if not cats:
        return "whitespace"
    from collections import Counter
    dominant = Counter(cats).most_common(1)[0][0]
    return dominant


def _get_l2_norms(model, tokenizer):
    """Compute L2 norms for all embeddings."""
    embedding_matrix = model.get_input_embeddings().weight.data
    norms = torch.norm(embedding_matrix.float(), dim=1).cpu().numpy()
    return norms


def stratified_sample(tokenizer, l2_norms, sample_size, seed=42):
    """
    Build a stratified sample of the vocabulary.

    Strata:
      - 5 L2 norm quintiles (each gets sample_size // 5 tokens)
      - Oversampled: whitespace, CJK, Cyrillic, code fragments
    """
    rng = random.Random(seed)
    vocab_size = len(l2_norms)
    all_ids = list(range(vocab_size))

    # Compute quintile boundaries
    sorted_norms = np.sort(l2_norms)
    quintile_bounds = [np.percentile(sorted_norms, q) for q in [0, 20, 40, 60, 80, 100]]
    quintile_bounds[-1] += 0.001  # slightly above max to include all tokens

    # Assign each token to a quintile
    quintile_bins = {i: [] for i in range(5)}
    for tid in all_ids:
        norm = l2_norms[tid]
        for q in range(5):
            if quintile_bounds[q] <= norm < quintile_bounds[q + 1]:
                quintile_bins[q].append(tid)
                break

    # Sample from each quintile
    per_quintile = max(1, sample_size // 5)
    sampled = set()
    for q in range(5):
        pool = quintile_bins[q]
        rng.shuffle(pool)
        for tid in pool[:per_quintile]:
            sampled.add(tid)

    # Oversample special categories
    special_categories = {"whitespace": [], "cjk": [], "cyrillic": [], "code": []}
    for tid in all_ids:
        text = tokenizer.decode([tid])
        if not text.strip():
            special_categories["whitespace"].append(tid)
        elif any("\u4e00" <= c <= "\u9fff" for c in text):
            special_categories["cjk"].append(tid)
        elif any("\u0400" <= c <= "\u04ff" for c in text):
            special_categories["cyrillic"].append(tid)
        elif any(c in text for c in "{}[]();<>\\"):
            special_categories["code"].append(tid)

    extra = max(1, sample_size // 20)
    for cat, ids in special_categories.items():
        rng.shuffle(ids)
        for tid in ids[:extra]:
            sampled.add(tid)

    # Trim to exact sample size if needed
    sampled = list(sampled)
    if len(sampled) > sample_size:
        rng.shuffle(sampled)
        sampled = sampled[:sample_size]

    return sampled


# ---------------------------------------------------------------------------
# ASR measurement
# ---------------------------------------------------------------------------

def measure_asr(model, tokenizer, token_id, num_attempts, max_tokens, patched):
    """
    Measure ASR for a single token with multi-attempt generation.

    Returns dict with asr, raw results, and whether all outputs were identical.
    """
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

    stop_token_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "get_added_vocab"):
        eot = tokenizer.get_added_vocab().get("<|eot_id|>")
        if eot:
            stop_token_ids.append(eot)

    glitch_attempts = 0
    outputs_per_test = {i: [] for i in range(len(prompts))}

    for attempt_idx in range(num_attempts):
        tests_failed = 0
        for test_idx, input_ids in enumerate(input_ids_list):
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                use_cache=True,
            )
            if patched:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = 0.7
                gen_kwargs["top_p"] = 0.95
                torch.manual_seed(int(time.time() * 1000) + attempt_idx * 100 + test_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(time.time() * 1000) + attempt_idx * 100 + test_idx)
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                generated = model.generate(**gen_kwargs)
            new_tokens = generated[0][input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            outputs_per_test[test_idx].append(text)

            # Anomaly indicator
            stripped = token_text.strip()
            text_found = (token_text in text) or (stripped and stripped in text)
            id_found = token_id in new_tokens.tolist()
            is_glitch = not (id_found or text_found)
            if is_glitch:
                tests_failed += 1

        if tests_failed == len(input_ids_list):
            glitch_attempts += 1

    asr = glitch_attempts / num_attempts if num_attempts else 0.0
    unique_per_test = {k: len(set(v)) for k, v in outputs_per_test.items()}
    all_identical = all(u == 1 for u in unique_per_test.values())

    return {
        "asr": asr,
        "glitch_attempts": glitch_attempts,
        "num_attempts": num_attempts,
        "unique_per_test": unique_per_test,
        "all_identical": all_identical,
    }


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vocabulary Glitch Token Scanner")
    parser.add_argument("model", help="Model path or name")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--num-attempts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--patched", action="store_true")
    parser.add_argument("--asr-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="data/vocab_scan_results.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quant-type", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = initialize_model_and_tokenizer(args.model, args.device, args.quant_type)
    model.eval()

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Compute L2 norms
    print("Computing embedding L2 norms...")
    l2_norms = _get_l2_norms(model, tokenizer)
    norm_mean = float(np.mean(l2_norms))
    norm_std = float(np.std(l2_norms))
    print(f"L2 norm stats: mean={norm_mean:.4f}, std={norm_std:.4f}")

    # Build stratified sample
    print(f"Building stratified sample of {args.sample_size} tokens...")
    sample_ids = stratified_sample(tokenizer, l2_norms, args.sample_size, args.seed)
    print(f"Sampled {len(sample_ids)} tokens")

    # Load checkpoint if exists
    completed = {}
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            checkpoint_data = json.load(f)
        for entry in checkpoint_data.get("results", []):
            completed[entry["token_id"]] = entry
        print(f"Loaded {len(completed)} completed results from checkpoint")

    # Scan
    results = list(completed.values())
    remaining = [tid for tid in sample_ids if tid not in completed]
    print(f"Scanning {len(remaining)} remaining tokens...")

    for idx, tid in enumerate(remaining):
        token_text = tokenizer.decode([tid])
        ucat = _unicode_category(token_text)
        l2 = float(l2_norms[tid])

        try:
            asr_result = measure_asr(
                model, tokenizer, tid, args.num_attempts, args.max_tokens, args.patched
            )
        except Exception as e:
            print(f"  Error on token {tid}: {e}")
            asr_result = {"asr": 0.0, "glitch_attempts": 0, "num_attempts": args.num_attempts,
                          "unique_per_test": {}, "all_identical": True, "error": str(e)}

        entry = {
            "token_id": tid,
            "token_text": token_text,
            "l2_norm": l2,
            "l2_norm_zscore": (l2 - norm_mean) / norm_std if norm_std > 0 else 0.0,
            "unicode_category": ucat,
            "token_length": len(token_text),
            "is_glitch": asr_result["asr"] >= args.asr_threshold,
            **asr_result,
        }
        results.append(entry)

        if (idx + 1) % 10 == 0 or idx == len(remaining) - 1:
            glitch_count = sum(1 for r in results if r.get("is_glitch"))
            print(f"  [{idx+1}/{len(remaining)}] Token {tid} "
                  f"'{token_text[:20]}' ASR={asr_result['asr']:.0%} "
                  f"L2={l2:.3f} | Total glitches so far: {glitch_count}")

            # Checkpoint
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            _save_checkpoint(args, results, l2_norms, norm_mean, norm_std)

    # Final save
    _save_checkpoint(args, results, l2_norms, norm_mean, norm_std)

    # Summary
    glitch_tokens = [r for r in results if r.get("is_glitch")]
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"Tokens scanned: {len(results)}")
    print(f"Glitch tokens found: {len(glitch_tokens)} ({len(glitch_tokens)/len(results)*100:.1f}%)")
    if glitch_tokens:
        norms = [r["l2_norm"] for r in glitch_tokens]
        print(f"Glitch L2 norms: min={min(norms):.4f} max={max(norms):.4f} "
              f"mean={sum(norms)/len(norms):.4f}")
        within_1sigma = sum(1 for r in glitch_tokens if abs(r["l2_norm_zscore"]) <= 1.0)
        print(f"Glitch tokens with L2 norm within 1 sigma of mean: "
              f"{within_1sigma}/{len(glitch_tokens)} ({within_1sigma/len(glitch_tokens)*100:.0f}%)")
    print(f"{'='*60}")


def _save_checkpoint(args, results, l2_norms, norm_mean, norm_std):
    checkpoint_data = {
        "model": args.model,
        "patched": args.patched,
        "num_attempts": args.num_attempts,
        "asr_threshold": args.asr_threshold,
        "sample_size": args.sample_size,
        "l2_norm_mean": norm_mean,
        "l2_norm_std": norm_std,
        "timestamp": time.time(),
        "results": results,
    }
    with open(args.checkpoint, "w") as f:
        json.dump(checkpoint_data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
