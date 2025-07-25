"""
Transformers provider for local models with 4-bit quantization support.
"""

import time
import torch
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

from .base import BaseProvider, ModelInfo

# Simple template class for consistent interface
class SimpleTemplate:
    def __init__(self):
        self.template_name = "simple"
        self.stop_word = None
        self.system_format = "System: {content}\n\n"
        self.user_format = "User: {content}\nAssistant: "

    def format_chat(self, system_message: str, user_message: str) -> str:
        if system_message:
            return f"System: {system_message}\n\nUser: {user_message}\nAssistant: "
        else:
            return f"User: {user_message}\nAssistant: "


class BuiltInTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template_name = "builtin"
        self.stop_word = getattr(tokenizer, 'eos_token', None)
        self.system_format = None
        self.user_format = None

    def format_chat(self, system_message: str, user_message: str) -> str:
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': user_message})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def initialize_model_and_tokenizer(model_path: str, device: str = "auto", quant_type: str = "int4"):
    """Initialize model and tokenizer with quantization."""
    # Set up device mapping
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with the specified quantization
    if quant_type == 'bfloat16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    elif quant_type == 'float16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    else:  # int8 or int4
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=(quant_type == 'int4'),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                load_in_8bit=(quant_type == 'int8'),
            ),
        )

    model.requires_grad_(False)
    return model, tokenizer


def get_template_for_model(model_name: str, tokenizer=None):
    """Get appropriate template for model."""
    # Try to use built-in chat template if available
    if tokenizer is not None and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        # Check if this is an instruct model
        if "instruct" in model_name.lower() or "chat" in model_name.lower():
            return BuiltInTemplate(tokenizer)

    return SimpleTemplate()


class TransformersProvider(BaseProvider):
    """Provider for local transformers models with quantization support."""

    def __init__(self, model_path: str, device: str = "auto", quant_type: str = "int4", **kwargs):
        """
        Initialize the transformers provider.

        Args:
            model_path: Path to the model (HuggingFace model ID or local path)
            device: Device to use ("auto", "cuda", "cpu", etc.)
            quant_type: Quantization type ("int4", "int8", "float16", "bfloat16")
            **kwargs: Additional arguments
        """
        # Call parent with dummy API key since we don't need one for local models
        super().__init__(api_key="local", **kwargs)

        self.model_path = model_path
        self.device = device
        self.quant_type = quant_type
        self.model = None
        self.tokenizer = None
        self.template = None

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model, tokenizer, and chat template."""
        print(f"Loading model: {self.model_path}")
        print(f"Quantization: {self.quant_type}")
        print(f"Device: {self.device}")

        try:
            self.model, self.tokenizer = initialize_model_and_tokenizer(
                self.model_path,
                device=self.device,
                quant_type=self.quant_type
            )

            # Get the appropriate chat template
            self.template = get_template_for_model(self.model_path, self.tokenizer)

            template_name = getattr(self.template, 'template_name', 'unknown') if self.template else 'unknown'
            print(f"Model loaded successfully. Template: {template_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {self.model_path}: {str(e)}")

    def list_models(self, quiet=False) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """
        List the currently loaded model.

        Returns:
            Tuple of (function_calling_models, other_models)
        """
        model_info = ModelInfo(
            id=self.model_path,
            name=self.model_path.split('/')[-1],
            provider="transformers",
            capabilities=["text_generation", "chat"],
            supports_function_calling=False  # Local models typically don't support function calling
        )

        if not quiet:
            print(f"Loaded model: {model_info.name}")
            print(f"  ID: {model_info.id}")
            print(f"  Quantization: {self.quant_type}")
            template_name = getattr(self.template, 'template_name', 'unknown') if self.template else 'unknown'
            print(f"  Template: {template_name}")

        # Return as other_models since local models typically don't support function calling
        return [], [model_info]

    def make_request(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a request to the local transformers model.

        Args:
            model_id: The model ID (should match self.model_path)
            messages: List of messages in the conversation
            tools: Tool definitions (ignored for local models)
            instructions: System instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with response data
        """
        if model_id != self.model_path:
            raise ValueError(f"Model ID {model_id} does not match loaded model {self.model_path}")

        # Extract system and user messages
        system_message = instructions or ""
        user_message = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")

        if not user_message:
            raise ValueError("No user message found in messages")

        # Format the prompt using the chat template
        try:
            if self.template and hasattr(self.template, 'format_chat'):
                # Built-in template
                prompt = self.template.format_chat(system_message, user_message)
            elif self.template:
                # Custom template
                if system_message and getattr(self.template, 'system_format', None):
                    formatted_system = self.template.system_format.format(content=system_message)
                else:
                    formatted_system = ""

                user_format = getattr(self.template, 'user_format', None)
                if user_format:
                    formatted_user = user_format.format(content=user_message)
                else:
                    formatted_user = f"User: {user_message}\nAssistant: "
                prompt = formatted_system + formatted_user
            else:
                # Fallback if no template
                if system_message:
                    prompt = f"System: {system_message}\n\nUser: {user_message}\nAssistant: "
                else:
                    prompt = f"User: {user_message}\nAssistant: "

        except Exception as e:
            template_name = getattr(self.template, 'template_name', 'unknown') if self.template else 'none'
            raise ValueError(f"Failed to format prompt with template {template_name}: {str(e)}")

        # Tokenize the prompt
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate response
        start_time = time.time()

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": getattr(self.tokenizer, 'eos_token_id', None),
            **kwargs
        }

        # Add stop words if available
        if self.template and hasattr(self.template, 'stop_word') and getattr(self.template, 'stop_word', None):
            stop_words = [self.template.stop_word]
            stop_token_ids = [self.tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]
            stop_token_ids = [ids[0] if ids else getattr(self.tokenizer, 'eos_token_id', None) for ids in stop_token_ids]
            generation_kwargs["eos_token_id"] = stop_token_ids

        try:
            if not self.model:
                raise RuntimeError("Model not initialized")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Decode the response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Clean up response
            if self.template and hasattr(self.template, 'stop_word') and getattr(self.template, 'stop_word', None):
                if self.template.stop_word in response_text:
                    response_text = response_text.split(self.template.stop_word)[0]

            generation_time = time.time() - start_time
            template_name = getattr(self.template, 'template_name', 'unknown') if self.template else 'unknown'

            return {
                "response": response_text.strip(),
                "prompt": prompt,
                "generation_time": generation_time,
                "input_tokens": input_length,
                "output_tokens": len(generated_tokens),
                "model_id": model_id,
                "template": template_name
            }

        except Exception as e:
            raise RuntimeError(f"Generation failed: {str(e)}")

    def analyze_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the response from the local model.

        Args:
            response: Raw response from make_request

        Returns:
            Dictionary with analyzed response data
        """
        response_text = response.get("response", "")

        # For prompt injection testing, we're looking for specific patterns
        # Most of these fields won't apply to local models, but we need to return the expected structure

        return {
            "api_key_leaked_in_message": False,  # Local models don't use API keys
            "api_key_correct": False,
            "username": None,
            "domain": None,
            "tld": None,
            "message_body": response_text,
            "api_key_used": None,
            "full_response": response_text,
            "raw_response": response,
            "parsing_error": None
        }

    def get_provider_specific_config(self) -> Dict[str, Any]:
        """Return provider-specific configuration options."""
        template_name = getattr(self.template, 'template_name', None) if self.template else None
        max_context = 4096
        model_type = 'unknown'

        if self.model and hasattr(self.model, 'config'):
            max_context = getattr(self.model.config, 'max_position_embeddings', 4096)
            model_type = getattr(self.model.config, 'model_type', 'unknown')

        return {
            "model_path": self.model_path,
            "device": self.device,
            "quantization": self.quant_type,
            "template": template_name,
            "supports_function_calling": False,
            "supports_streaming": True,
            "max_context_length": max_context,
            "model_type": model_type
        }

    def _is_rate_limit_error(self, error_str: str) -> bool:
        """Local models don't have rate limits."""
        return False

    def _is_retryable_error(self, error_str: str) -> bool:
        """Check if an error is retryable for local models."""
        retryable_indicators = [
            'cuda out of memory', 'out of memory', 'memory',
            'device', 'timeout', 'connection'
        ]
        return any(indicator in error_str.lower() for indicator in retryable_indicators)
