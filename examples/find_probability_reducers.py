#!/usr/bin/env python3
"""
Find Probability Reducer Tokens

This script specifically searches for glitch tokens that REDUCE the probability
of the baseline prediction. It measures the reduction in probability of the
original predicted token when a glitch token is inserted.

Usage:
    python find_probability_reducers.py --base-text "Hello world" --limit 100
    python find_probability_reducers.py --all --save-reducers reducers.json
"""

import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sys


class ProbabilityReducerFinder:
    """Find glitch tokens that reduce baseline prediction probabilities"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the language model"""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"✓ Model loaded on {self.model.device}")
            return True
        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            print("Trying GPT-2...")
            try:
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                print("✓ GPT-2 loaded")
                return True
            except Exception as e2:
                print(f"Failed to load fallback: {e2}")
                return False

    def load_glitch_tokens(self, json_file: str, limit: Optional[int] = None) -> List[Dict]:
        """Load glitch tokens from JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            classifications = data.get('classifications', [])
            if limit:
                classifications = classifications[:limit]

            tokens = []
            for item in classifications:
                tokens.append({
                    'token_id': item['token_id'],
                    'token': item['token'],
                    'display': repr(item['token'])
                })

            print(f"✓ Loaded {len(tokens)} glitch tokens")
            return tokens

        except Exception as e:
            print(f"❌ Error loading tokens: {e}")
            return []

    def get_baseline_prediction(self, text: str) -> Dict:
        """Get baseline prediction for text"""
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Get top prediction
            top_prob, top_idx = torch.topk(probs[0], 1)
            top_token = self.tokenizer.decode([top_idx[0].item()])

            return {
                'text': text,
                'top_token_id': top_idx[0].item(),
                'top_token': top_token,
                'top_prob': top_prob[0].item(),
                'full_probs': probs[0]
            }

    def test_glitch_token(self, glitch_token: str, baseline: Dict) -> Dict:
        """Test how a glitch token affects the baseline prediction"""
        # Insert glitch token before baseline text
        modified_text = f"{glitch_token} {baseline['text']}"

        input_ids = self.tokenizer.encode(modified_text, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Get probability of baseline token in modified context
            baseline_token_prob = probs[0, baseline['top_token_id']].item()

            # Calculate probability reduction
            prob_reduction = baseline['top_prob'] - baseline_token_prob

            # Get new top prediction
            new_top_prob, new_top_idx = torch.topk(probs[0], 1)
            new_top_token = self.tokenizer.decode([new_top_idx[0].item()])

            return {
                'modified_text': modified_text,
                'baseline_token_prob_after': baseline_token_prob,
                'probability_reduction': prob_reduction,
                'new_top_token_id': new_top_idx[0].item(),
                'new_top_token': new_top_token,
                'new_top_prob': new_top_prob[0].item(),
                'top_token_changed': baseline['top_token_id'] != new_top_idx[0].item()
            }

    def find_reducers(self, base_text: str, glitch_tokens: List[Dict]) -> List[Dict]:
        """Find tokens that reduce baseline prediction probability"""
        print(f"\n🔍 Searching for probability reducers...")
        print(f"Base text: '{base_text}'")

        # Get baseline
        baseline = self.get_baseline_prediction(base_text)
        print(f"Baseline: '{baseline['text']}' → '{baseline['top_token']}' ({baseline['top_prob']:.6f})")

        results = []
        reducers = []

        with tqdm(total=len(glitch_tokens), desc="Testing tokens") as pbar:
            for glitch_token in glitch_tokens:
                try:
                    result = self.test_glitch_token(glitch_token['token'], baseline)

                    # Combine data
                    full_result = {
                        'glitch_token': glitch_token,
                        'baseline': baseline,
                        'result': result,
                        'success': True
                    }

                    results.append(full_result)

                    # Track actual reducers
                    if result['probability_reduction'] > 0:
                        reducers.append(full_result)
                        pbar.set_postfix({
                            'reducers': len(reducers),
                            'current_reduction': f"{result['probability_reduction']:.6f}",
                            'token_changed': result['top_token_changed']
                        })

                except Exception as e:
                    results.append({
                        'glitch_token': glitch_token,
                        'success': False,
                        'error': str(e)
                    })

                pbar.update(1)

        # Sort by probability reduction (highest first)
        results.sort(key=lambda x: x['result']['probability_reduction'] if x['success'] else -999, reverse=True)

        print(f"\n✅ Analysis complete!")
        print(f"• Total tokens tested: {len(glitch_tokens)}")
        print(f"• Actual reducers found: {len(reducers)}")
        print(f"• Tokens that change prediction: {sum(1 for r in results if r['success'] and r['result']['top_token_changed'])}")

        return results, reducers

    def display_results(self, results: List[Dict], top_k: int = 20):
        """Display results focusing on probability reduction"""
        successful = [r for r in results if r['success']]
        reducers = [r for r in successful if r['result']['probability_reduction'] > 0]

        if reducers:
            print(f"\n🎯 TOP {min(top_k, len(reducers))} PROBABILITY REDUCERS")
            print("=" * 100)
            print(f"{'Rank':<4} {'Token':<25} {'Reduction':<12} {'Changed':<8} {'Baseline→New':<20} {'New Token':<15}")
            print("-" * 100)

            for i, result in enumerate(reducers[:top_k], 1):
                token_display = result['glitch_token']['display'][:22]
                reduction = result['result']['probability_reduction']
                changed = "✓" if result['result']['top_token_changed'] else "✗"

                baseline_prob = result['baseline']['top_prob']
                new_prob = result['result']['baseline_token_prob_after']
                prob_change = f"{baseline_prob:.3f}→{new_prob:.3f}"

                new_token = result['result']['new_top_token'][:12]

                print(f"{i:<4} {token_display:<25} {reduction:<12.6f} {changed:<8} {prob_change:<20} {new_token:<15}")

        else:
            print(f"\n❌ NO PROBABILITY REDUCERS FOUND")
            print("All tokens increase or maintain baseline probability")
            print("\nTop tokens by smallest increase:")
            print("=" * 100)
            print(f"{'Rank':<4} {'Token':<25} {'Increase':<12} {'Baseline→New':<20} {'Effect'}")
            print("-" * 100)

            for i, result in enumerate(successful[:top_k], 1):
                token_display = result['glitch_token']['display'][:22]
                increase = -result['result']['probability_reduction']  # Make positive

                baseline_prob = result['baseline']['top_prob']
                new_prob = result['result']['baseline_token_prob_after']
                prob_change = f"{baseline_prob:.3f}→{new_prob:.3f}"

                effect = "Strengthens prediction"

                print(f"{i:<4} {token_display:<25} {increase:<12.6f} {prob_change:<20} {effect}")

    def save_reducers(self, reducers: List[Dict], filename: str):
        """Save reducer tokens to JSON file"""
        if not reducers:
            print("No reducers to save")
            return

        try:
            data = {
                'model_name': self.model_name,
                'reducer_count': len(reducers),
                'reducers': []
            }

            for r in reducers:
                data['reducers'].append({
                    'token_id': r['glitch_token']['token_id'],
                    'token': r['glitch_token']['token'],
                    'probability_reduction': r['result']['probability_reduction'],
                    'baseline_token': r['baseline']['top_token'],
                    'baseline_prob': r['baseline']['top_prob'],
                    'reduced_prob': r['result']['baseline_token_prob_after'],
                    'new_top_token': r['result']['new_top_token'],
                    'top_token_changed': r['result']['top_token_changed']
                })

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {len(reducers)} reducers to {filename}")

        except Exception as e:
            print(f"❌ Error saving: {e}")


def main():
    parser = argparse.ArgumentParser(description='Find Probability Reducer Tokens')

    parser.add_argument('--base-text', default='The quick brown',
                       help='Base text to test (default: "The quick brown")')
    parser.add_argument('--json-file', default='../email_llams321.json',
                       help='Path to glitch tokens JSON file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tokens to test')
    parser.add_argument('--all', action='store_true',
                       help='Test all tokens (overrides limit)')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top results to show')
    parser.add_argument('--save-reducers', type=str, default=None,
                       help='Save reducer tokens to JSON file')
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B-Instruct',
                       help='Model to use')

    args = parser.parse_args()

    if args.all:
        args.limit = None

    print("🔍 PROBABILITY REDUCER FINDER")
    print("=" * 60)
    print("Finding glitch tokens that REDUCE baseline prediction probabilities")
    print()

    # Initialize finder
    finder = ProbabilityReducerFinder(args.model)

    if not finder.load_model():
        print("❌ Failed to load model")
        sys.exit(1)

    # Load tokens
    tokens = finder.load_glitch_tokens(args.json_file, args.limit)
    if not tokens:
        print("❌ No tokens loaded")
        sys.exit(1)

    # Find reducers
    results, reducers = finder.find_reducers(args.base_text, tokens)

    # Display results
    finder.display_results(results, args.top_k)

    # Save reducers if requested
    if args.save_reducers and reducers:
        finder.save_reducers(reducers, args.save_reducers)

    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    if reducers:
        print(f"• Found {len(reducers)} tokens that reduce baseline probability")
        print(f"• Best reducer: {reducers[0]['glitch_token']['display']}")
        print(f"• Maximum reduction: {reducers[0]['result']['probability_reduction']:.6f}")
    else:
        print("• No probability reducers found")
        print("• All glitch tokens strengthen the baseline prediction")
        print("• Try different base texts with weaker predictions")


if __name__ == "__main__":
    main()
