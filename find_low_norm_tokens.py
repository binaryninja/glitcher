#!/usr/bin/env python3
"""
Find tokens with the lowest L2 norms in a language model.
These tokens are highly likely to be glitch tokens.

Usage:
    python find_low_norm_tokens.py meta-llama/Llama-3.2-1B-Instruct --top-k 100
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import time


def find_low_norm_tokens(
    model_path: str,
    top_k: int = 100,
    device: str = "auto",
    output_file: str = "low_norm_tokens.json"
) -> List[Tuple[int, float, str]]:
    """
    Find tokens with the lowest L2 norms in the model's embedding matrix.

    Args:
        model_path: Path to the model
        top_k: Number of lowest norm tokens to return
        device: Device to use for computation
        output_file: File to save results

    Returns:
        List of (token_id, l2_norm, token_text) tuples
    """
    print(f"Loading model: {model_path}")

    # Load tokenizer and model
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

    # Get embedding matrix
    embeddings = model.get_input_embeddings().weight
    vocab_size = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]

    print(f"Analyzing {vocab_size} tokens with {embedding_dim}-dimensional embeddings...")

    # Calculate L2 norms for all tokens
    norms = []
    batch_size = 1000  # Process in batches to avoid memory issues

    for i in range(0, vocab_size, batch_size):
        end_idx = min(i + batch_size, vocab_size)
        batch_embeddings = embeddings[i:end_idx].float()
        batch_norms = torch.norm(batch_embeddings, dim=1).detach().cpu().numpy()

        for j, norm in enumerate(batch_norms):
            token_id = i + j
            try:
                token_text = tokenizer.decode([token_id])
                norms.append((token_id, float(norm), token_text))
            except Exception as e:
                # Skip tokens that can't be decoded
                norms.append((token_id, float(norm), f"<DECODE_ERROR_{token_id}>"))

        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(end_idx, vocab_size)}/{vocab_size} tokens...")

    # Sort by L2 norm (ascending)
    norms.sort(key=lambda x: x[1])

    # Get top-k lowest norm tokens
    lowest_norm_tokens = norms[:top_k]

    print(f"\nTop {top_k} tokens with lowest L2 norms:")
    print("-" * 70)
    for i, (token_id, norm, token_text) in enumerate(lowest_norm_tokens):
        # Clean up token text for display
        display_text = repr(token_text) if len(token_text) > 20 or '\n' in token_text or '\t' in token_text else token_text
        print(f"{i+1:3d}. ID {token_id:6d}: {display_text:30} L2 norm: {norm:.6f}")

    # Save results
    results = {
        "model_path": model_path,
        "timestamp": time.time(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "top_k": top_k,
        "lowest_norm_tokens": [
            {
                "rank": i + 1,
                "token_id": token_id,
                "l2_norm": norm,
                "token_text": token_text
            }
            for i, (token_id, norm, token_text) in enumerate(lowest_norm_tokens)
        ],
        "statistics": {
            "min_norm": float(norms[0][1]),
            "max_norm": float(norms[-1][1]),
            "median_norm": float(norms[len(norms)//2][1]),
            "mean_norm": float(sum(norm for _, norm, _ in norms) / len(norms))
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Statistics:")
    print(f"  Minimum L2 norm: {results['statistics']['min_norm']:.6f}")
    print(f"  Maximum L2 norm: {results['statistics']['max_norm']:.6f}")
    print(f"  Median L2 norm:  {results['statistics']['median_norm']:.6f}")
    print(f"  Mean L2 norm:    {results['statistics']['mean_norm']:.6f}")

    return lowest_norm_tokens


def main():
    parser = argparse.ArgumentParser(description="Find tokens with lowest L2 norms")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--top-k", type=int, default=100, help="Number of lowest norm tokens to find")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--output", default="low_norm_tokens.json", help="Output file")

    args = parser.parse_args()

    find_low_norm_tokens(
        model_path=args.model_path,
        top_k=args.top_k,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
