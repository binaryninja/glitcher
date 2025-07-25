#!/usr/bin/env python3
"""
Full Glitch Token Impact Analyzer

A comprehensive tool for analyzing how ALL glitch tokens affect language model
predictions and attention patterns. Processes the complete dataset of 464 glitch
tokens with progress tracking, batch processing, and detailed output options.

Usage:
    python glitch_impact_analyzer_full.py --help
    python glitch_impact_analyzer_full.py --limit 50 --base-text "The quick brown"
    python glitch_impact_analyzer_full.py --all --save-results results.json
    python glitch_impact_analyzer_full.py --batch-size 10 --detailed-analysis 5
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys
import os


class GlitchImpactAnalyzer:
    """Main class for analyzing glitch token impacts"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.results = []

    def load_model(self):
        """Load the language model with attention support"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="eager"  # Required for attention outputs
            )
            print(f"✓ Model loaded successfully on {self.model.device}")
            return True

        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            print("Falling back to GPT-2...")
            try:
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    attn_implementation="eager"
                )
                print("✓ GPT-2 loaded successfully")
                return True
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                return False

    def load_glitch_tokens(self, json_file: str, limit: Optional[int] = None,
                          start_idx: int = 0) -> List[Dict]:
        """Load glitch tokens from the classification JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_tokens = len(data.get('classifications', []))
            print(f"Found {total_tokens} glitch tokens in {json_file}")

            # Apply start index and limit
            classifications = data.get('classifications', [])[start_idx:]
            if limit:
                classifications = classifications[:limit]

            glitch_tokens = []
            for classification in classifications:
                token_info = {
                    'token_id': classification['token_id'],
                    'token': classification['token'],
                    'token_display': repr(classification['token'])
                }
                glitch_tokens.append(token_info)

            print(f"✓ Loaded {len(glitch_tokens)} glitch tokens (starting from index {start_idx})")
            return glitch_tokens

        except Exception as e:
            print(f"❌ Error loading glitch tokens: {e}")
            return []

    def calculate_entropy(self, probs: torch.Tensor) -> float:
        """Calculate entropy of probability distribution"""
        probs = torch.clamp(probs, min=1e-15)
        probs = probs / torch.sum(probs)
        entropy = -torch.sum(probs * torch.log2(probs))
        return entropy.item() if torch.isfinite(entropy) else 0.0

    def calculate_distance_metrics(self, p: torch.Tensor, q: torch.Tensor) -> Dict[str, float]:
        """Calculate multiple distance metrics between probability distributions"""
        eps = 1e-15

        # Normalize distributions
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        p = p / torch.sum(p)
        q = q / torch.sum(q)

        # Manhattan distance
        manhattan = torch.sum(torch.abs(p - q)).item()

        # Hellinger distance
        hellinger = (torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2)) / np.sqrt(2)).item()

        # KL divergence (with numerical stability)
        kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)))
        kl_div = kl_pq.item() if torch.isfinite(kl_pq) else float('inf')

        # JS divergence
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
        js_div = 0.5 * (kl_pm + kl_qm)
        js_div = js_div.item() if torch.isfinite(js_div) else float('inf')

        return {
            'manhattan_distance': manhattan,
            'hellinger_distance': hellinger,
            'kl_divergence': kl_div,
            'js_divergence': js_div
        }

    def analyze_baseline_prediction(self, base_text: str) -> Dict:
        """Analyze baseline prediction without any glitch token"""
        input_ids = self.tokenizer.encode(base_text, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], 10)

        return {
            'text': base_text,
            'tokens': [self.tokenizer.decode([token_id]) for token_id in input_ids[0]],
            'probs': probs[0],
            'entropy': self.calculate_entropy(probs[0]),
            'top_token_id': top_indices[0].item(),
            'top_token': self.tokenizer.decode([top_indices[0].item()]),
            'top_prob': top_probs[0].item(),
            'top_10_probs': top_probs.tolist(),
            'top_10_tokens': [self.tokenizer.decode([idx.item()]) for idx in top_indices],
            'outputs': outputs
        }

    def analyze_glitch_impact(self, glitch_token: str, base_text: str) -> Dict:
        """Analyze how a glitch token impacts the prediction"""

        #gs = ["'ujících'", "'ávající'"]
        #glitch_string = ' '.join(gs)
        #modified_text = f"{glitch_string} {glitch_token} {base_text} "
        modified_text = f"{glitch_token} {base_text} "
        print(f"Modified Text: {modified_text}")
        input_ids = self.tokenizer.encode(modified_text, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], 10)

        return {
            'modified_text': modified_text,
            'tokens': [self.tokenizer.decode([token_id]) for token_id in input_ids[0]],
            'probs': probs[0],
            'entropy': self.calculate_entropy(probs[0]),
            'top_token_id': top_indices[0].item(),
            'top_token': self.tokenizer.decode([top_indices[0].item()]),
            'top_prob': top_probs[0].item(),
            'top_10_probs': top_probs.tolist(),
            'top_10_tokens': [self.tokenizer.decode([idx.item()]) for idx in top_indices],
            'outputs': outputs
        }

    def analyze_attention_changes(self, baseline_outputs, glitch_outputs,
                                baseline_tokens, glitch_tokens) -> Optional[Dict]:
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
        attention_shifts = {}
        baseline_token_map = {}

        # Create mapping of baseline tokens to their positions in glitch sequence
        for i, token in enumerate(baseline_tokens):
            if token == "'<|begin_of_text|>'" or token == "<|begin_of_text|>":
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

        return {
            'baseline_tokens': baseline_tokens,
            'glitch_tokens': glitch_tokens,
            'attention_shifts': attention_shifts,
            'glitch_token_attention': glitch_last_attention[1].item() if len(glitch_last_attention) > 1 else 0.0
        }

    def process_glitch_token(self, glitch_token: Dict, baseline: Dict, base_text: str) -> Dict:
        """Process a single glitch token and return impact analysis"""
        try:
            # Analyze glitch impact
            glitch_result = self.analyze_glitch_impact(glitch_token['token'], base_text)

            # Calculate distance metrics
            distance_metrics = self.calculate_distance_metrics(baseline['probs'], glitch_result['probs'])

            # Calculate additional metrics
            entropy_change = glitch_result['entropy'] - baseline['entropy']
            top_token_changed = baseline['top_token_id'] != glitch_result['top_token_id']
            prob_change = glitch_result['top_prob'] - baseline['top_prob']

            # Calculate negative probability impact - how much the baseline token's probability decreased
            baseline_token_id = baseline['top_token_id']
            baseline_token_prob_with_glitch = glitch_result['probs'][baseline_token_id].item()
            negative_prob_impact = baseline['top_prob'] - baseline_token_prob_with_glitch

            metrics = {
                **distance_metrics,
                'entropy_change': entropy_change,
                'top_token_changed': top_token_changed,
                'top_prob_change': prob_change,
                'negative_prob_impact': negative_prob_impact,
                'baseline_top_token': baseline['top_token'],
                'glitch_top_token': glitch_result['top_token'],
                'baseline_top_prob': baseline['top_prob'],
                'glitch_top_prob': glitch_result['top_prob'],
                'baseline_token_prob_with_glitch': baseline_token_prob_with_glitch
            }

            # Analyze attention changes
            attention_analysis = self.analyze_attention_changes(
                baseline['outputs'], glitch_result['outputs'],
                baseline['tokens'], glitch_result['tokens']
            )

            return {
                'glitch_token': glitch_token,
                'baseline': baseline,
                'glitch_result': glitch_result,
                'metrics': metrics,
                'attention_analysis': attention_analysis,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'glitch_token': glitch_token,
                'success': False,
                'error': str(e),
                'metrics': None
            }

    def run_batch_analysis(self, glitch_tokens: List[Dict], base_text: str,
                          batch_size: int = 10) -> List[Dict]:
        """Run analysis on batches of glitch tokens with progress tracking"""
        print(f"\n🧪 Processing {len(glitch_tokens)} glitch tokens...")
        print(f"Base text: '{base_text}'")
        print(f"Batch size: {batch_size}")

        # Analyze baseline once
        print("📊 Analyzing baseline prediction...")
        baseline = self.analyze_baseline_prediction(base_text)
        print(f"Baseline: '{baseline['text']}' → '{baseline['top_token']}' ({baseline['top_prob']:.6f})")

        # Process tokens in batches
        results = []
        start_time = time.time()

        with tqdm(total=len(glitch_tokens), desc="Processing tokens") as pbar:
            for i in range(0, len(glitch_tokens), batch_size):
                batch = glitch_tokens[i:i + batch_size]
                batch_results = []

                for glitch_token in batch:
                    result = self.process_glitch_token(glitch_token, baseline, base_text)
                    batch_results.append(result)

                    # Update progress
                    pbar.update(1)

                    # Show current token info
                    if result['success']:
                        impact = result['metrics']['manhattan_distance']
                        token_display = result['glitch_token']['token_display']
                        pbar.set_postfix({
                            'current': token_display[:15],
                            'impact': f"{impact:.4f}",
                            'errors': sum(1 for r in results + batch_results if not r['success'])
                        })

                results.extend(batch_results)

                # Memory cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"\n✅ Batch processing complete!")
        print(f"• Total time: {elapsed:.1f} seconds ({elapsed/len(glitch_tokens):.2f}s per token)")
        print(f"• Successful: {successful}/{len(glitch_tokens)}")
        print(f"• Failed: {failed}")

        return results

    def rank_results(self, results: List[Dict], sort_by: str = 'negative_prob_impact') -> List[Dict]:
        """Rank results by impact metric"""
        successful_results = [r for r in results if r['success']]

        if sort_by == 'combined':
            # Combined ranking using multiple metrics
            def combined_score(r):
                m = r['metrics']
                return (m['manhattan_distance'] * 0.4 +
                       m['hellinger_distance'] * 0.3 +
                       abs(m['entropy_change']) * 0.2 +
                       abs(m['top_prob_change']) * 0.1)
            return sorted(successful_results, key=combined_score, reverse=True)
        elif sort_by == 'negative_prob_impact':
            # For negative probability impact, we want the highest positive values first
            # (actual reductions), then the least negative values (smallest increases)
            return sorted(successful_results, key=lambda x: x['metrics'][sort_by], reverse=True)
        else:
            return sorted(successful_results, key=lambda x: x['metrics'][sort_by], reverse=True)

    def display_summary(self, ranked_results: List[Dict], top_k: int = 20):
        """Display summary of most impactful tokens"""
        # Check if we have any actual reductions (positive negative_prob_impact)
        reductions = [r for r in ranked_results if r['metrics']['negative_prob_impact'] > 0]

        if reductions:
            print(f"\n🏆 TOP {min(top_k, len(ranked_results))} TOKENS THAT REDUCE BASELINE PROBABILITY")
        else:
            print(f"\n📊 TOP {min(top_k, len(ranked_results))} TOKENS BY PROBABILITY IMPACT")
            print("(All tokens increase baseline probability - showing least disruptive first)")

        print("=" * 115)
        print(f"{'Rank':<4} {'Token':<25} {'Prob Impact':<12} {'Effect':<12} {'Baseline→New':<15} {'Manhattan':<10} {'Token Δ':<8}")
        print("-" * 115)

        for i, result in enumerate(ranked_results[:top_k], 1):
            token_display = result['glitch_token']['token_display'][:22]
            metrics = result['metrics']

            token_change = "✓" if metrics['top_token_changed'] else "✗"
            neg_impact = metrics['negative_prob_impact']

            # Format impact and effect
            if neg_impact > 0:
                impact_str = f"+{neg_impact:.6f}"
                effect_str = "REDUCES"
            else:
                impact_str = f"{neg_impact:.6f}"
                effect_str = "INCREASES"

            baseline_prob = f"{metrics['baseline_top_prob']:.3f}"
            glitch_prob = f"{metrics['baseline_token_prob_with_glitch']:.3f}"
            prob_transition = f"{baseline_prob}→{glitch_prob}"
            manhattan = f"{metrics['manhattan_distance']:.6f}"

            print(f"{i:<4} {token_display:<25} {impact_str:<12} {effect_str:<12} {prob_transition:<15} {manhattan:<10} {token_change:<8}")

        print("-" * 115)

    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        try:
            # Prepare data for JSON serialization
            serializable_results = []
            for result in results:
                if result['success']:
                    # Remove torch tensors and model outputs
                    clean_result = {
                        'glitch_token': result['glitch_token'],
                        'metrics': result['metrics'],
                        'baseline_summary': {
                            'text': result['baseline']['text'],
                            'top_token': result['baseline']['top_token'],
                            'top_prob': result['baseline']['top_prob'],
                            'entropy': result['baseline']['entropy']
                        },
                        'glitch_summary': {
                            'modified_text': result['glitch_result']['modified_text'],
                            'top_token': result['glitch_result']['top_token'],
                            'top_prob': result['glitch_result']['top_prob'],
                            'entropy': result['glitch_result']['entropy']
                        }
                    }
                    if result['attention_analysis']:
                        clean_result['attention_summary'] = {
                            'glitch_token_attention': result['attention_analysis']['glitch_token_attention']
                        }
                    serializable_results.append(clean_result)
                else:
                    serializable_results.append({
                        'glitch_token': result['glitch_token'],
                        'success': False,
                        'error': result['error']
                    })

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': self.model_name,
                    'total_tokens_tested': len(results),
                    'successful_results': len([r for r in results if r['success']]),
                    'results': serializable_results
                }, f, indent=2, ensure_ascii=False)

            print(f"💾 Results saved to {filename}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Full Glitch Token Impact Analyzer')

    # Input options
    parser.add_argument('--json-file', default='../email_llams321.json',
                       help='Path to glitch tokens JSON file (default: ../email_llams321.json)')
    parser.add_argument('--base-text', default='The quick brown',
                       help='Base text to analyze (default: "The quick brown")')

    # Processing options
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tokens to process (default: all 464)')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Starting index in token list (default: 0)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing (default: 10)')
    parser.add_argument('--all', action='store_true',
                       help='Process all tokens (overrides --limit)')

    # Output options
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top results to display (default: 20)')
    parser.add_argument('--detailed-analysis', type=int, default=3,
                       help='Number of tokens for detailed analysis (default: 3)')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--sort-by', choices=['negative_prob_impact', 'manhattan_distance', 'hellinger_distance',
                                             'js_divergence', 'entropy_change', 'combined'],
                       default='negative_prob_impact',
                       help='Metric to sort results by (default: negative_prob_impact)')

    # Model options
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct',
                       help='Model to use (default: meta-llama/Llama-3.2-1B-Instruct)')

    args = parser.parse_args()

    # Override limit if --all is specified
    if args.all:
        args.limit = None

    print("🔍 FULL GLITCH TOKEN IMPACT ANALYZER")
    print("=" * 80)
    print("Comprehensive analysis of glitch token impacts on language model predictions")
    print()

    # Initialize analyzer
    analyzer = GlitchImpactAnalyzer(args.model)

    # Load model
    if not analyzer.load_model():
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)

    # Load glitch tokens
    glitch_tokens = analyzer.load_glitch_tokens(args.json_file, args.limit, args.start_index)
    if not glitch_tokens:
        print("❌ No glitch tokens loaded. Exiting.")
        sys.exit(1)

    # Show configuration
    print(f"Configuration:")
    print(f"• Model: {analyzer.model_name}")
    print(f"• Base text: '{args.base_text}'")
    print(f"• Tokens to process: {len(glitch_tokens)}")
    print(f"• Batch size: {args.batch_size}")
    print(f"• Sort by: {args.sort_by}")

    # Run analysis
    results = analyzer.run_batch_analysis(glitch_tokens, args.base_text, args.batch_size)

    # Rank results
    ranked_results = analyzer.rank_results(results, args.sort_by)

    # Display summary
    analyzer.display_summary(ranked_results, args.top_k)

    # Save results if requested
    if args.save_results:
        analyzer.save_results(results, args.save_results)

    # Show detailed analysis for top tokens
    if args.detailed_analysis > 0 and ranked_results:
        print(f"\n" + "=" * 80)
        print(f"DETAILED ANALYSIS OF TOP {min(args.detailed_analysis, len(ranked_results))} TOKENS")
        print("=" * 80)

        for i, result in enumerate(ranked_results[:args.detailed_analysis], 1):
            print(f"\n📊 DETAILED ANALYSIS #{i}")
            print("=" * 70)

            token_info = result['glitch_token']
            metrics = result['metrics']
            baseline = result['baseline']
            glitch = result['glitch_result']

            print(f"Token: {token_info['token_display']} (ID: {token_info['token_id']})")
            print(f"Modified Text: '{glitch['modified_text']}'")
            print()

            print("IMPACT METRICS:")
            neg_impact = metrics['negative_prob_impact']
            if neg_impact > 0:
                print(f"  • Probability Impact: +{neg_impact:.6f} (REDUCES baseline token probability)")
            else:
                print(f"  • Probability Impact: {neg_impact:.6f} (INCREASES baseline token probability)")
            print(f"  • Baseline Token Probability: {baseline['top_prob']:.6f} → {metrics['baseline_token_prob_with_glitch']:.6f}")
            print(f"  • Manhattan Distance: {metrics['manhattan_distance']:.6f}")
            print(f"  • Hellinger Distance: {metrics['hellinger_distance']:.6f}")
            print(f"  • Entropy Change: {metrics['entropy_change']:+.6f}")
            print(f"  • Top Token Changed: {'Yes' if metrics['top_token_changed'] else 'No'}")
            print()

            print("PREDICTION CHANGES:")
            print(f"  Baseline: '{baseline['top_token']}' ({baseline['top_prob']:.6f})")
            print(f"  With Glitch: '{glitch['top_token']}' ({glitch['top_prob']:.6f})")
            print(f"  New Top Token Probability Change: {metrics['top_prob_change']:+.6f}")

    print(f"\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    successful = len(ranked_results)
    total = len(results)
    print(f"• Successfully analyzed {successful}/{total} glitch tokens")
    if ranked_results:
        top_token = ranked_results[0]['glitch_token']['token_display']
        if args.sort_by == 'negative_prob_impact':
            impact_val = ranked_results[0]['metrics'][args.sort_by]
            baseline_prob = ranked_results[0]['metrics']['baseline_top_prob']
            new_prob = ranked_results[0]['metrics']['baseline_token_prob_with_glitch']
            baseline_token = ranked_results[0]['metrics']['baseline_top_token']

            if impact_val > 0:
                print(f"• Most disruptive: {top_token}")
                print(f"• Highest reduction: +{impact_val:.6f} (reduced '{baseline_token}' from {baseline_prob:.3f} to {new_prob:.3f})")
            else:
                print(f"• Least disruptive: {top_token}")
                print(f"• Smallest increase: {impact_val:.6f} (increased '{baseline_token}' from {baseline_prob:.3f} to {new_prob:.3f})")
        else:
            print(f"• Most impactful: {top_token}")
            print(f"• Highest impact: {ranked_results[0]['metrics'][args.sort_by]:.6f} ({args.sort_by})")
    print("• Use --help to see all available options")


if __name__ == "__main__":
    main()
