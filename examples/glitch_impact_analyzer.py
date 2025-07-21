#!/usr/bin/env python3
"""
Glitch Token Impact Analyzer

This tool analyzes how different glitch tokens impact the probability distribution
when inserted before "The quick brown". It helps identify which glitch tokens
have the strongest effect on model predictions and attention patterns.

Usage:
    python glitch_impact_analyzer.py

The tool will:
1. Load glitch tokens from email_llams321.json
2. Test each token's impact on "The quick brown" predictions
3. Rank tokens by their impact strength
4. Show detailed analysis for the most impactful tokens
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple, Optional


def load_model_with_attention():
    """Load a language model configured to output attention patterns"""
    try:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"  # Required for attention outputs
        )
        print(f"✓ Model loaded successfully on {model.device}")
        return model, tokenizer, model_name

    except Exception as e:
        print(f"Failed to load Llama: {e}")
        print("Falling back to GPT-2...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
        print("✓ GPT-2 loaded successfully")
        return model, tokenizer, model_name


def load_glitch_tokens(json_file: str = "../email_llams321.json", limit: int = 50) -> List[Dict]:
    """Load glitch tokens from the classification JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        glitch_tokens = []
        for classification in data.get('classifications', [])[:limit]:
            token_info = {
                'token_id': classification['token_id'],
                'token': classification['token'],
                'token_display': repr(classification['token'])
            }
            glitch_tokens.append(token_info)

        print(f"✓ Loaded {len(glitch_tokens)} glitch tokens from {json_file}")
        return glitch_tokens

    except Exception as e:
        print(f"❌ Error loading glitch tokens: {e}")
        # Fallback to some known glitch tokens
        return [
            {'token_id': 89472, 'token': 'useRalative', 'token_display': "'useRalative'"},
            {'token_id': 110024, 'token': ' CLIIIK', 'token_display': "' CLIIIK'"},
            {'token_id': 116552, 'token': 'ujícím', 'token_display': "'ujícím'"}
        ]


def calculate_entropy(probs: torch.Tensor) -> float:
    """Calculate entropy of probability distribution"""
    # Ensure probabilities are positive and normalized
    probs = torch.clamp(probs, min=1e-15)
    probs = probs / torch.sum(probs)
    # Use log2 for entropy in bits
    entropy = -torch.sum(probs * torch.log2(probs))
    return entropy.item() if torch.isfinite(entropy) else 0.0


def calculate_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Calculate KL divergence between two probability distributions"""
    # Ensure probabilities are positive and normalized
    eps = 1e-15
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = p / torch.sum(p)
    q = q / torch.sum(q)

    # Calculate KL divergence with numerical stability
    kl = torch.sum(p * (torch.log(p) - torch.log(q)))
    return kl.item() if torch.isfinite(kl) else float('inf')


def calculate_manhattan_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Calculate Manhattan distance between two probability distributions"""
    # Normalize distributions
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    return torch.sum(torch.abs(p - q)).item()


def calculate_hellinger_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Calculate Hellinger distance between two probability distributions"""
    # Normalize distributions
    p = p / torch.sum(p)
    q = q / torch.sum(q)
    # Hellinger distance
    hellinger = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2)) / np.sqrt(2)
    return hellinger.item()


def analyze_baseline_prediction(model, tokenizer, base_text: str = "The quick brown") -> Dict:
    """Analyze baseline prediction without any glitch token"""

    input_ids = tokenizer.encode(base_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    # Get top predictions
    top_probs, top_indices = torch.topk(probs[0], 10)

    baseline_data = {
        'text': base_text,
        'tokens': [tokenizer.decode([token_id]) for token_id in input_ids[0]],
        'probs': probs[0],
        'entropy': calculate_entropy(probs[0]),
        'top_token_id': top_indices[0].item(),
        'top_token': tokenizer.decode([top_indices[0].item()]),
        'top_prob': top_probs[0].item(),
        'top_10_probs': top_probs.tolist(),
        'top_10_tokens': [tokenizer.decode([idx.item()]) for idx in top_indices],
        'outputs': outputs
    }

    return baseline_data


def analyze_glitch_impact(model, tokenizer, glitch_token: str, base_text: str = "The quick brown") -> Dict:
    """Analyze how a glitch token impacts the prediction"""

    # Create text with glitch token inserted
    modified_text = f"{glitch_token} {base_text}"

    input_ids = tokenizer.encode(modified_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

    # Get top predictions
    top_probs, top_indices = torch.topk(probs[0], 10)

    impact_data = {
        'modified_text': modified_text,
        'tokens': [tokenizer.decode([token_id]) for token_id in input_ids[0]],
        'probs': probs[0],
        'entropy': calculate_entropy(probs[0]),
        'top_token_id': top_indices[0].item(),
        'top_token': tokenizer.decode([top_indices[0].item()]),
        'top_prob': top_probs[0].item(),
        'top_10_probs': top_probs.tolist(),
        'top_10_tokens': [tokenizer.decode([idx.item()]) for idx in top_indices],
        'outputs': outputs
    }

    return impact_data


def calculate_impact_metrics(baseline: Dict, glitch_result: Dict) -> Dict:
    """Calculate various impact metrics comparing baseline to glitch result"""

    # Probability distribution changes
    kl_div = calculate_kl_divergence(glitch_result['probs'], baseline['probs'])
    entropy_change = glitch_result['entropy'] - baseline['entropy']

    # Top token changes
    top_token_changed = baseline['top_token_id'] != glitch_result['top_token_id']
    prob_change = glitch_result['top_prob'] - baseline['top_prob']

    # Calculate multiple distance metrics
    p = baseline['probs']
    q = glitch_result['probs']

    # JS divergence (symmetric measure)
    eps = 1e-15
    p_norm = torch.clamp(p, min=eps)
    q_norm = torch.clamp(q, min=eps)
    p_norm = p_norm / torch.sum(p_norm)
    q_norm = q_norm / torch.sum(q_norm)
    m = 0.5 * (p_norm + q_norm)

    kl_pm = calculate_kl_divergence(p_norm, m)
    kl_qm = calculate_kl_divergence(q_norm, m)
    js_div = 0.5 * (kl_pm + kl_qm)

    # Additional distance metrics
    manhattan_dist = calculate_manhattan_distance(p, q)
    hellinger_dist = calculate_hellinger_distance(p, q)

    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'manhattan_distance': manhattan_dist,
        'hellinger_distance': hellinger_dist,
        'entropy_change': entropy_change,
        'top_token_changed': top_token_changed,
        'top_prob_change': prob_change,
        'baseline_top_token': baseline['top_token'],
        'glitch_top_token': glitch_result['top_token'],
        'baseline_top_prob': baseline['top_prob'],
        'glitch_top_prob': glitch_result['top_prob']
    }


def rank_glitch_tokens_by_impact(impact_results: List[Dict]) -> List[Dict]:
    """Rank glitch tokens by their impact strength"""

    # Sort by Manhattan distance (most robust for this case)
    # Use Manhattan distance as primary, Hellinger as secondary
    def impact_score(x):
        metrics = x['metrics']
        # Primary: Manhattan distance, Secondary: Hellinger distance
        return (metrics['manhattan_distance'], metrics['hellinger_distance'])

    sorted_results = sorted(impact_results, key=impact_score, reverse=True)
    return sorted_results


def analyze_attention_impact(baseline_outputs, glitch_outputs, baseline_tokens, glitch_tokens):
    """Analyze how attention patterns change due to glitch token"""

    if baseline_outputs.attentions is None or glitch_outputs.attentions is None:
        return None

    # Get attention patterns from last layer
    baseline_attention = baseline_outputs.attentions[-1][0].mean(dim=0)
    glitch_attention = glitch_outputs.attentions[-1][0].mean(dim=0)

    # Attention from last position
    baseline_last_attention = baseline_attention[-1, :]
    glitch_last_attention = glitch_attention[-1, :]

    # Calculate attention shifts for overlapping tokens
    # Map baseline tokens to glitch tokens (accounting for inserted glitch token)
    attention_shifts = {}
    baseline_token_map = {}

    # Create mapping of baseline tokens to their positions in glitch sequence
    for i, token in enumerate(baseline_tokens):
        if token == "'<|begin_of_text|>'":
            baseline_token_map[i] = 0  # Begin token stays at position 0
        else:
            # Other tokens shift by 1 position due to inserted glitch token
            baseline_token_map[i] = i + 1

    # Calculate attention changes for mapped tokens
    for baseline_idx, glitch_idx in baseline_token_map.items():
        if glitch_idx < len(glitch_last_attention):
            baseline_att = baseline_last_attention[baseline_idx].item()
            glitch_att = glitch_last_attention[glitch_idx].item()
            attention_shifts[baseline_tokens[baseline_idx]] = {
                'baseline': baseline_att,
                'with_glitch': glitch_att,
                'change': glitch_att - baseline_att
            }

    attention_analysis = {
        'baseline_tokens': baseline_tokens,
        'glitch_tokens': glitch_tokens,
        'baseline_attention': baseline_last_attention.tolist(),
        'glitch_attention': glitch_last_attention.tolist(),
        'attention_shifts': attention_shifts,
        'glitch_token_attention': glitch_last_attention[1].item() if len(glitch_last_attention) > 1 else 0.0
    }

    return attention_analysis


def display_impact_summary(impact_results: List[Dict], top_k: int = 10):
    """Display a summary of the most impactful glitch tokens"""

    print(f"\n🏆 TOP {top_k} MOST IMPACTFUL GLITCH TOKENS")
    print("=" * 80)
    print(f"{'Rank':<4} {'Token':<20} {'Manhattan':<10} {'Hellinger':<10} {'Entropy Δ':<12} {'Token Δ':<8} {'Prob Δ':<12}")
    print("-" * 90)

    for i, result in enumerate(impact_results[:top_k], 1):
        token_display = result['glitch_token']['token_display']
        metrics = result['metrics']

        # Format changes
        token_change = "✓" if metrics['top_token_changed'] else "✗"
        manhattan = f"{metrics['manhattan_distance']:.6f}"
        hellinger = f"{metrics['hellinger_distance']:.6f}"
        entropy_change = f"{metrics['entropy_change']:.6f}"
        prob_change = f"{metrics['top_prob_change']:+.6f}"

        print(f"{i:<4} {token_display:<20} {manhattan:<10} {hellinger:<10} {entropy_change:<12} {token_change:<8} {prob_change:<12}")

    print("-" * 90)


def display_detailed_analysis(result: Dict, rank: int):
    """Display detailed analysis for a specific glitch token"""

    print(f"\n📊 DETAILED ANALYSIS #{rank}")
    print("=" * 70)

    token_info = result['glitch_token']
    metrics = result['metrics']
    baseline = result['baseline']
    glitch = result['glitch_result']

    print(f"Glitch Token: {token_info['token_display']} (ID: {token_info['token_id']})")
    print(f"Modified Text: '{glitch['modified_text']}'")
    print()

    print("IMPACT METRICS:")
    print(f"  • Manhattan Distance: {metrics['manhattan_distance']:.6f}")
    print(f"  • Hellinger Distance: {metrics['hellinger_distance']:.6f}")
    print(f"  • JS Divergence: {metrics['js_divergence']:.6f}")
    print(f"  • KL Divergence: {metrics['kl_divergence']:.6f}")
    print(f"  • Entropy Change: {metrics['entropy_change']:+.6f}")
    print(f"  • Top Token Changed: {'Yes' if metrics['top_token_changed'] else 'No'}")
    print()

    print("PREDICTION CHANGES:")
    print(f"  Baseline: '{baseline['top_token']}' ({baseline['top_prob']:.6f})")
    print(f"  With Glitch: '{glitch['top_token']}' ({glitch['top_prob']:.6f})")
    print(f"  Probability Change: {metrics['top_prob_change']:+.6f}")
    print()

    print("TOP 5 PREDICTIONS COMPARISON:")
    print(f"{'Rank':<4} {'Baseline':<20} {'Prob':<10} {'With Glitch':<20} {'Prob':<10}")
    print("-" * 70)

    for i in range(min(5, len(baseline['top_10_tokens']))):
        baseline_token = baseline['top_10_tokens'][i]
        baseline_prob = baseline['top_10_probs'][i]
        glitch_token = glitch['top_10_tokens'][i]
        glitch_prob = glitch['top_10_probs'][i]

        print(f"{i+1:<4} {repr(baseline_token):<20} {baseline_prob:<10.6f} {repr(glitch_token):<20} {glitch_prob:<10.6f}")

    # Show attention analysis if available
    if 'attention_analysis' in result and result['attention_analysis']:
        attention = result['attention_analysis']
        print(f"\nATTENTION PATTERN CHANGES:")
        print(f"Baseline tokens: {[repr(t) for t in attention['baseline_tokens']]}")
        print(f"Glitch tokens: {[repr(t) for t in attention['glitch_tokens']]}")

        if 'attention_shifts' in attention:
            print(f"\nAttention shifts for original tokens:")
            print(f"{'Token':<20} {'Baseline':<12} {'With Glitch':<12} {'Change':<12} {'Visual'}")
            print("-" * 75)

            for token, shift_data in attention['attention_shifts'].items():
                baseline_att = shift_data['baseline']
                glitch_att = shift_data['with_glitch']
                change = shift_data['change']

                # Create visual representation
                baseline_bar = "█" * int(baseline_att * 20)
                glitch_bar = "█" * int(glitch_att * 20)
                change_indicator = "↑" if change > 0 else "↓" if change < 0 else "="

                print(f"{repr(token):<20} {baseline_att:<12.6f} {glitch_att:<12.6f} {change:+.6f} {change_indicator}")

            glitch_token_att = attention.get('glitch_token_attention', 0.0)
            print(f"\nGlitch token attention: {glitch_token_att:.6f}")
            glitch_bar = "█" * int(glitch_token_att * 20)
            print(f"Visual: {glitch_bar}")


def main():
    """Main analysis function"""

    print("🔍 GLITCH TOKEN IMPACT ANALYZER")
    print("=" * 80)
    print("Analyzing how different glitch tokens impact predictions after 'The quick brown'")
    print()

    # Load model and glitch tokens
    model, tokenizer, model_name = load_model_with_attention()
    glitch_tokens = load_glitch_tokens(limit=20)  # Limit for faster testing

    if not glitch_tokens:
        print("❌ No glitch tokens loaded. Exiting.")
        return

    print(f"\nModel: {model_name}")
    print(f"Testing {len(glitch_tokens)} glitch tokens")
    print()

    # Analyze baseline
    print("📊 Analyzing baseline prediction...")
    baseline = analyze_baseline_prediction(model, tokenizer)
    print(f"Baseline: '{baseline['text']}' → '{baseline['top_token']}' ({baseline['top_prob']:.6f})")

    # Analyze each glitch token
    print(f"\n🧪 Testing glitch token impacts...")
    impact_results = []

    for i, glitch_token in enumerate(glitch_tokens):
        print(f"Testing {i+1}/{len(glitch_tokens)}: {glitch_token['token_display']}")

        try:
            # Analyze glitch impact
            glitch_result = analyze_glitch_impact(model, tokenizer, glitch_token['token'])

            # Calculate impact metrics
            metrics = calculate_impact_metrics(baseline, glitch_result)

            # Analyze attention changes
            attention_analysis = analyze_attention_impact(
                baseline['outputs'], glitch_result['outputs'],
                baseline['tokens'], glitch_result['tokens']
            )

            # Store results
            result = {
                'glitch_token': glitch_token,
                'baseline': baseline,
                'glitch_result': glitch_result,
                'metrics': metrics,
                'attention_analysis': attention_analysis
            }
            impact_results.append(result)

        except Exception as e:
            print(f"  ❌ Error testing {glitch_token['token_display']}: {e}")

    # Rank by impact
    ranked_results = rank_glitch_tokens_by_impact(impact_results)

    # Display summary
    display_impact_summary(ranked_results)

    # Show detailed analysis for top 3
    print(f"\n" + "=" * 80)
    print("DETAILED ANALYSIS OF TOP 3 MOST IMPACTFUL TOKENS")
    print("=" * 80)

    for i, result in enumerate(ranked_results[:3], 1):
        display_detailed_analysis(result, i)

    print(f"\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print(f"• Tested {len(impact_results)} glitch tokens")
    print(f"• Most impactful: {ranked_results[0]['glitch_token']['token_display']}")
    print(f"• Strongest impact: {ranked_results[0]['metrics']['manhattan_distance']:.6f} Manhattan distance")
    print("• Higher distance metrics = stronger impact on probability distribution")
    print("• Manhattan distance measures total variation between distributions")


if __name__ == "__main__":
    main()
