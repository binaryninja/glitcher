#!/usr/bin/env python3
"""
Glitcher CLI - A command-line tool for mining and testing glitch tokens in language models.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from glitcher.model import (
    mine_glitch_tokens,
    strictly_glitch_verify,
    get_template_for_model,
    chat_token,
    initialize_model_and_tokenizer
)


class GlitcherCLI:
    """CLI for Glitcher to find and test glitch tokens with save/resume functionality."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args = None
        self.progress = {
            "model_path": "",
            "num_iterations": 0,
            "iterations_completed": 0,
            "total_tokens_checked": 0,
            "glitch_tokens": [],
            "glitch_token_ids": [],
            "start_time": 0,
            "last_saved": 0,
        }
        self.running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def setup_parser(self):
        """Setup command line argument parser."""
        parser = argparse.ArgumentParser(
            description="GlitchMiner CLI - Find and test glitch tokens in language models"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command to run")
        
        # Mine command
        mine_parser = subparsers.add_parser("mine", help="Mine for glitch tokens")
        mine_parser.add_argument("model_path", help="Path or name of the model to test")
        mine_parser.add_argument(
            "--num-iterations", type=int, default=50, 
            help="Number of iterations to run (default: 50)"
        )
        mine_parser.add_argument(
            "--batch-size", type=int, default=8, 
            help="Batch size for token testing (default: 8)"
        )
        mine_parser.add_argument(
            "--k", type=int, default=32, 
            help="Number of nearest tokens to consider (default: 32)"
        )
        mine_parser.add_argument(
            "--output", type=str, default="glitch_tokens.json", 
            help="Output file for results (default: glitch_tokens.json)"
        )
        mine_parser.add_argument(
            "--save-interval", type=int, default=5, 
            help="Save progress every N iterations (default: 5)"
        )
        mine_parser.add_argument(
            "--resume", action="store_true", 
            help="Resume from previous progress file"
        )
        mine_parser.add_argument(
            "--progress-file", type=str, default="glitch_progress.json", 
            help="File to save/load progress (default: glitch_progress.json)"
        )
        mine_parser.add_argument(
            "--device", type=str, default="cuda", 
            help="Device to use (default: cuda)"
        )
        mine_parser.add_argument(
            "--quant-type", type=str, default="bfloat16", 
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        
        # Test command
        test_parser = subparsers.add_parser("test", help="Test specific tokens")
        test_parser.add_argument("model_path", help="Path or name of the model to test")
        test_parser.add_argument(
            "--token-ids", type=str, 
            help="Comma-separated list of token IDs to test"
        )
        test_parser.add_argument(
            "--token-file", type=str, 
            help="JSON file containing token IDs to test"
        )
        test_parser.add_argument(
            "--output", type=str, default="test_results.json", 
            help="Output file for results (default: test_results.json)"
        )
        test_parser.add_argument(
            "--device", type=str, default="cuda", 
            help="Device to use (default: cuda)"
        )
        test_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        
        # Chat command
        chat_parser = subparsers.add_parser("chat", help="Chat with a token to see its behavior")
        chat_parser.add_argument("model_path", help="Path or name of the model to test")
        chat_parser.add_argument("token_id", type=int, help="Token ID to test")
        chat_parser.add_argument(
            "--max-size", type=int, default=10, 
            help="Maximum number of tokens to generate (default: 10)"
        )
        chat_parser.add_argument(
            "--device", type=str, default="cuda", 
            help="Device to use (default: cuda)"
        )
        chat_parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )
        
        return parser

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.args.model_path}")
        self.model, self.tokenizer = initialize_model_and_tokenizer(
            model_path=self.args.model_path,
            device=self.args.device,
            quant_type=self.args.quant_type
        )
        print(f"Model loaded on {self.args.device}")

    def load_progress(self):
        """Load progress from file if it exists."""
        progress_path = Path(self.args.progress_file)
        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    self.progress = json.load(f)
                print(f"Loaded progress: {self.progress['iterations_completed']} iterations completed")
                print(f"Found {len(self.progress['glitch_token_ids'])} glitch tokens so far")
                return True
            except Exception as e:
                print(f"Error loading progress: {e}")
        return False

    def save_progress(self, final=False):
        """Save current progress to file."""
        try:
            # Update the progress data
            self.progress["last_saved"] = time.time()
            
            with open(self.args.progress_file, "w") as f:
                json.dump(self.progress, f, indent=2)
            
            if final:
                print("Final progress saved.")
            else:
                print(f"Progress saved after {self.progress['iterations_completed']} iterations.")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def save_results(self, filename, data):
        """Save results to a file."""
        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def handle_interrupt(self, sig, frame):
        """Handle interrupt signal (Ctrl+C) to save progress and exit gracefully."""
        print("\nInterrupted. Saving progress and exiting...")
        self.running = False

    def run_mining(self):
        """Run the mining process."""
        # Initialize progress
        if self.args.resume and self.load_progress():
            # Verify the model path is the same
            if self.progress["model_path"] != self.args.model_path:
                print("Warning: Model path in progress file doesn't match the specified model.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != "y":
                    return
            
            remaining_iterations = self.args.num_iterations - self.progress["iterations_completed"]
        else:
            # Initialize new progress
            self.progress = {
                "model_path": self.args.model_path,
                "num_iterations": self.args.num_iterations,
                "iterations_completed": 0,
                "total_tokens_checked": 0,
                "glitch_tokens": [],
                "glitch_token_ids": [],
                "start_time": time.time(),
                "last_saved": time.time(),
            }
            remaining_iterations = self.args.num_iterations
        
        if remaining_iterations <= 0:
            print("All iterations completed. Use a higher --num-iterations value to continue.")
            return
        
        # Set up mining parameters
        batch_size = self.args.batch_size
        k = self.args.k
        
        # Run the mining process
        print(f"Starting mining process with {remaining_iterations} iterations...")
        print(f"Using batch_size={batch_size}, k={k}")
        
        self.progress["start_time"] = time.time()
        
        def checkpoint_callback(data):
            """Callback function to save progress during mining"""
            self.progress["iterations_completed"] = data["iteration"] + 1
            self.progress["total_tokens_checked"] = data["total_tokens_checked"]
            self.progress["glitch_tokens"] = data["glitch_tokens"]
            self.progress["glitch_token_ids"] = data["glitch_token_ids"]
            
            # Save progress at the specified interval
            if (data["iteration"] + 1) % self.args.save_interval == 0:
                self.save_progress()
            
            # Check if we should continue running
            return self.running
        
        try:
            # Run glitch token mining with the specified number of iterations
            glitch_tokens, glitch_token_ids = mine_glitch_tokens(
                model=self.model,
                tokenizer=self.tokenizer,
                num_iterations=remaining_iterations,
                batch_size=batch_size,
                k=k,
                verbose=True,
                language="ENG",
                checkpoint_callback=checkpoint_callback
            )
            
            # Update progress
            self.progress["iterations_completed"] = self.args.num_iterations
            self.progress["glitch_tokens"].extend(glitch_tokens)
            self.progress["glitch_token_ids"].extend(glitch_token_ids)
            
            # Save final results
            self.save_progress(final=True)
            
            # Save results
            results = {
                "model_path": self.args.model_path,
                "glitch_tokens": self.progress["glitch_tokens"],
                "glitch_token_ids": self.progress["glitch_token_ids"],
                "total_iterations": self.args.num_iterations,
                "runtime_seconds": time.time() - self.progress["start_time"],
            }
            
            self.save_results(self.args.output, results)
            
        except Exception as e:
            print(f"Error during mining: {e}")
            # Save progress on error
            self.save_progress(final=True)
        
    def run_token_test(self):
        """Test specific tokens for glitch properties."""
        # Get token IDs to test
        token_ids = []
        
        if self.args.token_ids:
            try:
                token_ids = [int(tid.strip()) for tid in self.args.token_ids.split(",")]
            except ValueError:
                print("Error: Token IDs must be comma-separated integers")
                return
        
        elif self.args.token_file:
            try:
                with open(self.args.token_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        token_ids = data
                    elif "glitch_token_ids" in data:
                        token_ids = data["glitch_token_ids"]
                    else:
                        print("Error: Could not find token IDs in the file")
                        return
            except Exception as e:
                print(f"Error loading token file: {e}")
                return
        
        if not token_ids:
            print("Error: No token IDs provided")
            return
        
        print(f"Testing {len(token_ids)} tokens...")
        
        # Get chat template
        chat_template = get_template_for_model(self.model.config._name_or_path)
        
        # Test each token
        results = []
        for i, token_id in enumerate(token_ids):
            if not self.running:
                break
                
            try:
                token = self.tokenizer.decode([token_id])
                is_glitch = strictly_glitch_verify(self.model, self.tokenizer, token_id, chat_template)
                
                result = {
                    "token_id": token_id,
                    "token": token,
                    "is_glitch": is_glitch
                }
                
                results.append(result)
                print(f"[{i+1}/{len(token_ids)}] Token: '{token}', ID: {token_id}, Is glitch: {is_glitch}")
                
            except Exception as e:
                print(f"Error testing token {token_id}: {e}")
        
        # Save results
        test_results = {
            "model_path": self.args.model_path,
            "test_results": results,
            "timestamp": time.time()
        }
        
        self.save_results(self.args.output, test_results)

    def run_chat_test(self):
        """Test how a model responds to a specific token in a chat context."""
        token_id = self.args.token_id
        
        try:
            # Run chat test
            print(f"Testing token ID {token_id} in chat context...")
            result = chat_token(
                self.args.model_path, 
                token_id, 
                max_size=self.args.max_size,
                device=self.args.device,
                quant_type=self.args.quant_type
            )
            
            # Print results
            print("\nChat Test Results:")
            print(f"Token ID: {result['检测token id']}")
            print(f"Token: '{result['检测token']}'")
            print(f"Probability of correct repetition: {result['模型输出的第一个token为检测token的概率']:.6f}")
            print(f"Most likely token: '{result['概率最大的token']}' (ID: {result['概率最大的token下标']})")
            print(f"Most likely token probability: {result['概率最大的token概率']:.6f}")
            print("\nModel output:")
            print(result['大模型对话的结果'])
            
            # Save result
            chat_result = {
                "model_path": self.args.model_path,
                "token_id": token_id,
                "token": result['检测token'],
                "correct_prob": result['模型输出的第一个token为检测token的概率'],
                "top_tokens": [int(i) for i in result['模型输出的第一个token位置上概率最大的5个下标']],
                "top_token": result['概率最大的token'],
                "top_token_id": result['概率最大的token下标'],
                "top_token_prob": result['概率最大的token概率'],
                "model_output": result['大模型对话的结果'],
                "timestamp": time.time()
            }
            
            output_file = f"chat_token_{token_id}.json"
            self.save_results(output_file, chat_result)
            
        except Exception as e:
            print(f"Error testing token in chat: {e}")

    def run(self):
        """Run the CLI tool."""
        parser = self.setup_parser()
        self.args = parser.parse_args()
        
        if not self.args.command:
            parser.print_help()
            return
        
        # Load model for all commands
        self.load_model()
        
        if self.args.command == "mine":
            self.run_mining()
        elif self.args.command == "test":
            self.run_token_test()
        elif self.args.command == "chat":
            self.run_chat_test()
        else:
            parser.print_help()


def main():
    """Main entry point for the CLI."""
    cli = GlitcherCLI()
    cli.run()


if __name__ == "__main__":
    main()