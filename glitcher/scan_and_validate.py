#!/usr/bin/env python3
"""
Glitch token scanner and validator - Automatically mines, validates and logs glitch tokens
"""

import argparse
import json
import os
import time
from pathlib import Path
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from glitcher.model import (
    mine_glitch_tokens,
    strictly_glitch_verify,
    get_template_for_model,
    chat_token,
    initialize_model_and_tokenizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("glitch_scan.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GlitchScanner")

class GlitchScanner:
    """Continuously scans for and validates glitch tokens"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.args = None
        self.validated_tokens = {
            "model_path": "",
            "glitch_tokens": [],
            "glitch_token_ids": [],
            "validation_results": [],
            "start_time": 0,
            "scan_iterations": 0,
            "total_tokens_checked": 0
        }

    def setup_parser(self):
        """Setup command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Glitch Token Scanner - Mine and validate glitch tokens automatically"
        )

        parser.add_argument("model_path", help="Path or name of the model to scan")
        parser.add_argument(
            "--num-iterations", type=int, default=50,
            help="Number of iterations to run (default: 50)"
        )
        parser.add_argument(
            "--batch-size", type=int, default=8,
            help="Batch size for token testing (default: 8)"
        )
        parser.add_argument(
            "--k", type=int, default=32,
            help="Number of nearest tokens to consider (default: 32)"
        )
        parser.add_argument(
            "--output", type=str, default="validated_glitch_tokens.json",
            help="Output file for results (default: validated_glitch_tokens.json)"
        )
        parser.add_argument(
            "--chat-sample-size", type=int, default=3,
            help="Number of validated tokens to test with chat (default: 3)"
        )
        parser.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use (default: cuda)"
        )
        parser.add_argument(
            "--quant-type", type=str, default="bfloat16",
            choices=["bfloat16", "float16", "int8", "int4"],
            help="Quantization type (default: bfloat16)"
        )

        return parser

    def load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading model: {self.args.model_path}")
        self.model, self.tokenizer = initialize_model_and_tokenizer(
            model_path=self.args.model_path,
            device=self.args.device,
            quant_type=self.args.quant_type
        )
        logger.info(f"Model loaded on {self.args.device}")

    def save_results(self):
        """Save the validated tokens and results"""
        # Update runtime info
        self.validated_tokens["runtime_seconds"] = time.time() - self.validated_tokens["start_time"]

        # Save to file
        try:
            with open(self.args.output, "w") as f:
                json.dump(self.validated_tokens, f, indent=2)
            logger.info(f"Results saved to {self.args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def validate_token(self, token_id, token):
        """Validate if a token is truly a glitch token"""
        # Get chat template for model
        chat_template = get_template_for_model(self.model.config._name_or_path, self.tokenizer)

        # Perform strict validation
        try:
            is_glitch = strictly_glitch_verify(self.model, self.tokenizer, token_id, chat_template)
            result = {
                "token_id": token_id,
                "token": token,
                "is_glitch": is_glitch,
                "timestamp": time.time()
            }
            return result
        except Exception as e:
            logger.error(f"Error validating token {token_id}: {e}")
            return None

    def test_token_in_chat(self, token_id):
        """Test a validated glitch token in chat context"""
        try:
            logger.info(f"Testing token ID {token_id} in chat context")

            # Decode token to get text
            token = self.tokenizer.decode([token_id])

            # Get chat template for model
            chat_template = get_template_for_model(self.model.config._name_or_path, self.tokenizer)

            # Get template formats
            system_format = chat_template.system_format
            user_format = chat_template.user_format
            assistant_format = chat_template.assistant_format
            stop_word = chat_template.stop_word
            system_message = ''

            with torch.no_grad():
                # Use a clearer prompt without special characters surrounding the token
                user_prompt = f'''Please respond with ONLY the exact characters inside quotes: "{token}"'''
                assistant_prefill = ' '

                # Construct input
                if system_format is not None:
                    formatted_system = system_format.format(content=system_message)
                else:
                    formatted_system = ""

                formatted_user = user_format.format(content=user_prompt)

                if assistant_prefill:
                    formatted_input = formatted_system + formatted_user + assistant_prefill
                else:
                    formatted_input = formatted_system + formatted_user

                # Tokenize input
                inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)

                # Generate model response
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                        output_scores=True, return_dict_in_generate=True)

                # Get probability distribution for first generated token
                first_token_probs = torch.softmax(outputs.scores[0][0], dim=-1)

                # Get probability of the target token
                target_prob = first_token_probs[token_id].item()

                # Get top 5 token indices
                top_5_indices = torch.topk(first_token_probs, k=5).indices

                # Get most probable token and its probability
                max_prob_index = torch.argmax(first_token_probs)
                max_prob_token = self.tokenizer.decode([max_prob_index])
                max_prob = first_token_probs[max_prob_index].item()

                # Decode generated text
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)

                # Prepare result
                result = {
                    "token_id": token_id,
                    "token": token,
                    "target_token_prob": target_prob,
                    "top_5_indices": top_5_indices.tolist(),
                    "top_token": max_prob_token,
                    "top_token_prob": max_prob,
                    "top_token_id": max_prob_index.item(),
                    "generated_text": generated_text
                }

            # Log results
            logger.info(f"Chat test for token '{result['token']}' (ID: {token_id}):")
            logger.info(f"Probability of correct repetition: {result['target_token_prob']:.6f}")
            logger.info(f"Most likely token: '{result['top_token']}' (ID: {result['top_token_id']})")
            logger.info(f"Model output: {result['generated_text']}")

            # Save chat result to separate file
            chat_result = {
                "model_path": self.args.model_path,
                "token_id": token_id,
                "token": result['token'],
                "correct_prob": result['target_token_prob'],
                "top_tokens": result['top_5_indices'],
                "top_token": result['top_token'],
                "top_token_id": result['top_token_id'],
                "top_token_prob": result['top_token_prob'],
                "model_output": result['generated_text'],
                "timestamp": time.time()
            }

            output_file = f"chat_token_{token_id}.json"
            with open(output_file, "w") as f:
                json.dump(chat_result, f, indent=2)

            return chat_result

        except Exception as e:
            logger.error(f"Error testing token in chat: {e}")
            return None

    def mining_checkpoint_callback(self, data):
        """Process new potential glitch tokens as mining progresses"""
        iteration = data["iteration"]
        total_tokens_checked = data["total_tokens_checked"]
        glitch_tokens = data["glitch_tokens"]
        glitch_token_ids = data["glitch_token_ids"]

        # Update our tracking stats
        self.validated_tokens["scan_iterations"] = iteration + 1
        self.validated_tokens["total_tokens_checked"] = total_tokens_checked

        # Get new tokens found in this iteration
        current_tokens = set(self.validated_tokens["glitch_token_ids"])
        new_token_indices = [i for i, tid in enumerate(glitch_token_ids) if tid not in current_tokens]

        # Validate each new token
        for i in new_token_indices:
            token_id = glitch_token_ids[i]
            token = glitch_tokens[i]

            logger.info(f"New potential glitch token found: '{token}' (ID: {token_id})")

            # Validate the token
            validation_result = self.validate_token(token_id, token)

            if validation_result and validation_result["is_glitch"]:
                # Token is confirmed as a glitch
                logger.info(f"CONFIRMED GLITCH TOKEN: '{token}' (ID: {token_id})")

                # Add to our validated tokens list
                self.validated_tokens["glitch_tokens"].append(token)
                self.validated_tokens["glitch_token_ids"].append(token_id)
                self.validated_tokens["validation_results"].append(validation_result)

                # Save updated results
                self.save_results()
            else:
                logger.info(f"Token '{token}' (ID: {token_id}) is not actually a glitch token")

        # Continue mining
        return True

    def run_scan(self):
        """Run the scanning and validation process"""
        # Initialize
        self.validated_tokens = {
            "model_path": self.args.model_path,
            "glitch_tokens": [],
            "glitch_token_ids": [],
            "validation_results": [],
            "chat_results": [],
            "start_time": time.time(),
            "scan_iterations": 0,
            "total_tokens_checked": 0
        }

        # Set up mining parameters
        batch_size = self.args.batch_size
        k = self.args.k

        # Start mining process
        logger.info(f"Starting mining process with {self.args.num_iterations} iterations")
        logger.info(f"Using batch_size={batch_size}, k={k}")

        try:
            # Run glitch token mining with validation during the process
            mine_glitch_tokens(
                model=self.model,
                tokenizer=self.tokenizer,
                num_iterations=self.args.num_iterations,
                batch_size=batch_size,
                k=k,
                verbose=True,
                language="ENG",
                checkpoint_callback=self.mining_checkpoint_callback
            )

            # Mining complete - log summary
            logger.info("Mining process complete")
            logger.info(f"Total tokens checked: {self.validated_tokens['total_tokens_checked']}")
            logger.info(f"Confirmed glitch tokens found: {len(self.validated_tokens['glitch_tokens'])}")

            # Test a sample of tokens in chat context
            if self.validated_tokens["glitch_token_ids"]:
                chat_sample_size = min(self.args.chat_sample_size, len(self.validated_tokens["glitch_token_ids"]))

                if chat_sample_size > 0:
                    logger.info(f"Testing {chat_sample_size} glitch tokens in chat context")

                    # Take a sample of tokens (first few found)
                    sample_token_ids = self.validated_tokens["glitch_token_ids"][:chat_sample_size]

                    # Test each token in chat
                    for token_id in sample_token_ids:
                        chat_result = self.test_token_in_chat(token_id)
                        if chat_result:
                            self.validated_tokens["chat_results"].append(chat_result)

            # Save final results
            self.save_results()

        except Exception as e:
            logger.error(f"Error during scanning process: {e}")
            # Save any results we have so far
            self.save_results()

    def run(self):
        """Run the scanner"""
        parser = self.setup_parser()
        self.args = parser.parse_args()

        # Print banner
        logger.info("=" * 80)
        logger.info(f"GLITCH TOKEN SCANNER - Starting scan of {self.args.model_path}")
        logger.info("=" * 80)

        # Load model
        self.load_model()

        # Run the scanning process
        self.run_scan()

        logger.info("=" * 80)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 80)

def main():
    """Main entry point for the scanner"""
    scanner = GlitchScanner()
    scanner.run()

if __name__ == "__main__":
    main()
