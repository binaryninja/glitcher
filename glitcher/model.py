#!/usr/bin/env python3
"""
Core functionality for glitch token mining and testing
"""

import time
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from typing import List, Tuple, Dict, Any, Optional, Union

# Harmony support (runtime import to avoid hard dependency here)
HARMONY_AVAILABLE = False
HarmonyEncodingName = None
load_harmony_encoding = None
Conversation = None
Message = None
Role = None
SystemContent = None
DeveloperContent = None

try:
    _harmony = __import__(
        "openai_harmony",
        fromlist=[
            "HarmonyEncodingName",
            "load_harmony_encoding",
            "Conversation",
            "Message",
            "Role",
            "SystemContent",
            "DeveloperContent",
        ],
    )
    HarmonyEncodingName = getattr(_harmony, "HarmonyEncodingName", None)
    load_harmony_encoding = getattr(_harmony, "load_harmony_encoding", None)
    Conversation = getattr(_harmony, "Conversation", None)
    Message = getattr(_harmony, "Message", None)
    Role = getattr(_harmony, "Role", None)
    SystemContent = getattr(_harmony, "SystemContent", None)
    DeveloperContent = getattr(_harmony, "DeveloperContent", None)
    HARMONY_AVAILABLE = all([
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    ])
except Exception:
    HARMONY_AVAILABLE = False


def _is_gpt_oss_name(name: str) -> bool:
    """Detect gpt-oss family by model name"""
    return "gpt-oss" in name.lower() or "gptoss" in name.lower()


def build_harmony_prefill(system_message: str, user_prompt: str, tokenizer, model):
    """
    Build Harmony prefill ids and stop token ids for gpt-oss with configurable reasoning level.

    Returns: (prefill_ids, stop_ids, prefill_text, encoding)
    """
    if not HARMONY_AVAILABLE:
        return [], [], "", None

    reasoning_level = os.environ.get("GLITCHER_REASONING_LEVEL", "medium")
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Put reasoning level into developer instructions ahead of the policy/system guidance
    developer_instructions = f"Reasoning: {reasoning_level}\n\n{system_message}"

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(developer_instructions),
        ),
        Message.from_role_and_content(Role.USER, user_prompt),
    ])

    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    stop_ids = encoding.stop_tokens_for_assistant_actions()
    prefill_text = tokenizer.decode(prefill_ids)

    return prefill_ids, stop_ids, prefill_text, encoding


def parse_harmony_final(new_token_ids: List[int], tokenizer, encoding) -> str:
    """
    Parse Harmony completion tokens and return only final channel text.
    Falls back to string splitting if structured parsing fails.
    """
    if HARMONY_AVAILABLE and encoding is not None:
        try:
            entries = encoding.parse_messages_from_completion_tokens(new_token_ids, Role.ASSISTANT)
            parsed = [m.to_dict() for m in entries]
            finals = [m.get("content", "") for m in parsed if str(m.get("channel", "")).lower() == "final"]
            if finals:
                return finals[-1]
        except Exception:
            pass

    # Fallback: heuristic split by Harmony markers
    full_decoded = tokenizer.decode(new_token_ids)
    tail = full_decoded.split("<|channel|>final<|message|>")[-1]
    return tail.split("<|end|>")[0].strip() if "<|end|>" in tail else tail.strip()


def entropy(probs):
    """Calculate entropy of probability distribution"""
    return -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)


class Template:
    """Chat template for different models"""
    def __init__(
        self,
        template_name: str,
        system_format: Optional[str],
        user_format: str,
        assistant_format: str,
        system: Optional[str] = None,
        stop_word: Optional[str] = None,
    ):
        self.template_name = template_name
        self.system_format = system_format
        self.user_format = user_format
        self.assistant_format = assistant_format
        self.system = system
        self.stop_word = stop_word


class BuiltInTemplate:
    """Wrapper for tokenizer's built-in chat template"""
    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.template_name = f"builtin_{model_name.split('/')[-1].lower()}"
        self.system_format = None  # Not used for built-in templates
        self.user_format = None    # Not used for built-in templates
        self.assistant_format = None  # Not used for built-in templates
        self.system = None
        name_lower = model_name.lower()
        self.stop_word = '<|eot_id|>' if 'llama' in name_lower else None

    def format_chat(self, system_message: str, user_message: str) -> str:
        """Format a chat using the built-in template"""
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


_TEMPLATES = {
    # Default template
    'default': Template(
        template_name='default',
        system_format='System: {content}\n\n',
        user_format='User: {content}\nAssistant: ',
        assistant_format='{content} {stop_token}',
        system=None,
        stop_word=None
    ),

    # Llama 3 template
    'llama3': Template(
        template_name='llama3',
        system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
        user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
        assistant_format='{content}<|eot_id|>',
        system=None,
        stop_word='<|eot_id|>'
    ),

    # Llama 3.2 template
    'llama32': Template(
        template_name='llama32',
        system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
        user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
        assistant_format='{content}<|eot_id|>',
        system=None,
        stop_word='<|eot_id|>'
    ),

    # Llama 3.3 template
    'llama33': Template(
        template_name='llama32',
        system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
        user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
        assistant_format='{content}<|eot_id|>',
        system=None,
        stop_word='<|eot_id|>'
    ),
    # Llama 2 template
    'llama2': Template(
        template_name='llama2',
        system_format='<<SYS>>\n{content}\n<</SYS>>\n\n',
        user_format='[INST]{content}[/INST]',
        assistant_format='{content} </s>',
        system="You are a helpful, respectful and honest assistant.",
        stop_word='</s>'
    ),

    # Mistral template
    'mistral': Template(
        template_name='mistral',
        system_format='<s>',
        user_format='[INST]{content}[/INST]',
        assistant_format='{content}</s>',
        system='',
        stop_word='</s>'
    ),

    # nowllm template (identical to Mistral)
    'nowllm': Template(
        template_name='nowllm',
        system_format='<s>',
        user_format='[INST]{content}[/INST]',
        assistant_format='{content}</s>',
        system='',
        stop_word='</s>'
    ),

    # Gemma template
    'gemma': Template(
        template_name='gemma',
        system_format='<bos>',
        user_format='<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
        assistant_format='{content}<eos>\n',
        system='',
        stop_word='<eos>'
    ),
}


def get_template_for_model(model_name: str, tokenizer=None) -> Union[Template, BuiltInTemplate]:
    """Get the appropriate chat template for a model"""
    # If tokenizer is provided and has a built-in chat template, use that for instruct-like models
    if tokenizer is not None and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        lower_name = model_name.lower()
        # Include gpt-oss family in built-in template usage
        if ("instruct" in lower_name or "chat" in lower_name or "gpt-oss" in lower_name or "gptoss" in lower_name):
            return BuiltInTemplate(tokenizer, model_name)

    # Only take the last part of the path and normalize
    original_name = model_name
    model_name = model_name.split('/')[-1].lower().replace('-', '').replace('_', '')

    # Special case for Llama 3.2 models
    if "llama3.2" in model_name or "llama32" in model_name:
        return _TEMPLATES['llama32']

    # Special case for Llama 3 models that aren't 3.2
    if "llama3" in model_name and "3.2" not in original_name and "32" not in model_name:
        return _TEMPLATES['llama3']

    # Special case for nowllm-0829 model (Mixtral fine-tune)
    if "nowllm0829" in model_name or "/nowllm-0829" in original_name:
        return _TEMPLATES['nowllm']

    # Try to find matching template
    matching_templates = []
    for template_name, template in _TEMPLATES.items():
        if template_name.lower() in model_name:
            matching_templates.append((template_name, template))

    if not matching_templates:
        return _TEMPLATES['default']

    # Choose the best match (longest template name)
    best_match = max(matching_templates, key=lambda x: len(x[0]))
    return best_match[1]


class TokenClassifier:
    """Classifies tokens for filtering"""

    NORMAL = 0
    SPECIAL = 1
    UNDECODEABLE = 2
    UNREACHABLE = 3

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def classify_token(self, token_id: int) -> int:
        """Classify token as normal, special, undecodeable, or unreachable"""
        try:
            token = self.tokenizer.decode([token_id])
            if not token:
                return self.UNDECODEABLE

            # Check if it contains special characters
            if token.startswith('<') and token.endswith('>'):
                return self.SPECIAL

            # Basic check to see if token is potentially harmful
            if '<script>' in token.lower() or 'javascript:' in token.lower():
                return self.SPECIAL

            # Consider all other tokens normal
            return self.NORMAL

        except Exception:
            return self.UNDECODEABLE

    def filter_tokens(self) -> List[int]:
        """Return a list of token IDs that should be filtered out"""
        filtered_tokens = []
        vocab_size = self.model.get_input_embeddings().weight.shape[0]

        for token_id in range(vocab_size):
            token_class = self.classify_token(token_id)
            if token_class in [self.SPECIAL, self.UNDECODEABLE, self.UNREACHABLE]:
                filtered_tokens.append(token_id)

        return filtered_tokens


def initialize_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    quant_type: str = "bfloat16"
) -> Tuple[Any, Any]:
    """Initialize a model and tokenizer with the specified configuration"""
    # Set up device mapping
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Special handling for nowllm-0829 Mixtral fine-tune
    if "nowllm-0829" in model_path:
        print(f"Loading nowllm-0829 Mixtral fine-tune with {quant_type} precision")

    # Load model with the specified quantization
    if quant_type == 'auto':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype="auto",
        )
    elif quant_type == 'bfloat16':
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


def glitch_verify_message1(chat_template, token):
    """Format a verification message that asks to repeat a token"""
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' '
    # Match the system message from token_repetition.py
    system_message = 'You are a helpful assistant. When asked to repeat something, repeat it exactly, character for character.'
    # Match the user prompt from token_repetition.py
    user_prompt = f"Please respond with exactly this: '{token}'"

    # Use built-in template if available, otherwise use custom template
    if isinstance(chat_template, BuiltInTemplate):
        formatted_input = chat_template.format_chat(system_message, user_prompt)
    else:
        # Construct input using custom template
        formatted_system = system_format.format(content=system_message) if system_format else ""
        formatted_user = user_format.format(content=user_prompt)

        # For Llama 3.2, we need to ensure we have the right prefill token for best results
        if chat_template.template_name in ['llama3', 'llama32']:
            assistant_prefill = ""

        formatted_input = formatted_system + formatted_user + assistant_prefill

    return formatted_input


def glitch_verify_message2(chat_template, token):
    """Format an alternative verification message asking for token meaning"""
    if isinstance(chat_template, BuiltInTemplate):
        # For built-in templates, we can't easily do multi-turn conversations, so use a simple format
        system_message = 'This helpful assistant explains what a phrase means, replying with identical responses to identical questions for reliability.'
        user_prompt = f"What does '{token}' mean?"
        return chat_template.format_chat(system_message, user_prompt)
    else:
        system_format = chat_template.system_format
        user_format = chat_template.user_format
        assistant_format = chat_template.assistant_format
        assistant_prefill = "'"
        system_message = 'This helpful assistant explains what a phrase means, replying with identical responses to identical questions for reliability.'

        user_prompt1 = "What does 'edslfkj' mean?"
        assistant_answer1 = "'edslfkj' has no specific meaning I am aware of."
        user_prompt2 = f"What does '{token}' mean?"

        formatted_system = system_format.format(content=system_message) if system_format else ""
        formatted_user1 = user_format.format(content=user_prompt1)
        formatted_assitant1 = assistant_format.format(content=assistant_answer1)
        formatted_user2 = user_format.format(content=user_prompt2)

        # For Llama 3.2, we need to ensure we have the right prefill token for best results
        if chat_template.template_name in ['llama3', 'llama32']:
            assistant_prefill = ""

        return formatted_system + formatted_user1 + formatted_assitant1 + formatted_user2 + assistant_prefill


def glitch_verify_message3(chat_template, token):
    """Format a third verification message with different phrasing"""
    if isinstance(chat_template, BuiltInTemplate):
        system_message = 'You are a helpful assistant that follows instructions carefully.'
        user_prompt = f"Question: Please can you repeat the string '{token}'?"
        return chat_template.format_chat(system_message, user_prompt)
    else:
        system_format = chat_template.system_format
        user_format = chat_template.user_format
        assistant_prefill = " Of course! Here is the repeated string:'"
        system_message = 'You are a helpful assistant that follows instructions carefully.'
        user_prompt = f"Question: Please can you repeat the string '{token}'?"

        formatted_system = system_format.format(content=system_message) if system_format else ""
        formatted_user = user_format.format(content=user_prompt)

        # For Llama 3.2, adjust the prefill tokens for better reliability
        if chat_template.template_name in ['llama3', 'llama32']:
            assistant_prefill = ""

        return formatted_system + formatted_user + assistant_prefill


def strictly_glitch_verify(model, tokenizer, token_id, chat_template=None, log_file=None):
    """Verify if a token is a glitch token using multiple prompt formats"""
    import json

    # Get model device
    device = model.device

    # Decode token_id to get text
    token = tokenizer.decode([token_id])

    # Get chat template for model if not provided
    if chat_template is None:
        chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Prepare test prompts
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
    is_glitch = True

    # For Llama 3.2, require a stronger verification by counting how many tests fail
    is_llama32 = "llama3.2" in model.config._name_or_path.lower() or "llama32" in model.config._name_or_path.lower()
    tests_failed = 0
    tests_count = len(inputs)
    test_results = []

    for i, (input_ids, formatted_input) in enumerate(zip(inputs, formatted_inputs)):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # Get logits for the next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Check probability of the target token
            target_prob = next_token_probs[0, token_id].item()

            # Get top 5 predicted tokens and their probabilities
            top5_values, top5_indices = torch.topk(next_token_probs[0], 5)
            top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices]
            top5_probs = top5_values.tolist()

            # Get most probable token
            predicted_token_id = next_token_probs.argmax(dim=-1).item()
            predicted_token = tokenizer.decode([predicted_token_id])

            # Set threshold based on model type - use extremely low threshold
            # Use a much more strict threshold - true glitches have probabilities near zero
            probability_threshold = 0.00005

            # Add content-based filtering to avoid certain patterns we know generate false positives
            token_text = token
            # Skip tokens with brackets, common programming symbols, or specific prefixes that tend to be false positives
            should_skip = (
                # ('[' in token_text or ']' in token_text) or
                # ('(' in token_text or ')' in token_text) or
                # ('_' in token_text) or
                # ('arg' in token_text.lower()) or
                # ('prop' in token_text.lower()) or
                # ('char' in token_text.lower()) or
                # (token_text.startswith('.')) or
                (token_text.startswith('XXXXXXXXXXX'))
            )

            # Three criteria for a token not being a glitch:
            # 1. It has a high probability of being predicted (>threshold)
            # 2. OR it's actually the top predicted token
            # 3. OR it matches our skip patterns for false positives
            is_test_glitch = (target_prob < probability_threshold) and (predicted_token_id != token_id) and not should_skip

            # Store test result for logging
            test_result = {
                "test_number": i+1,
                "formatted_input": formatted_input,
                "target_probability": target_prob,
                "predicted_token_id": predicted_token_id,
                "predicted_token": predicted_token,
                "probability_threshold": probability_threshold,
                "top5_tokens": top5_tokens,
                "top5_probabilities": top5_probs,
                "should_skip": should_skip,
                "is_test_glitch": is_test_glitch,
                "reason": "pattern filtered" if should_skip else
                         "probability below threshold and not top token" if is_test_glitch else
                         "probability above threshold" if target_prob >= probability_threshold else
                         "is top predicted token"
            }
            test_results.append(test_result)

            if is_test_glitch:
                tests_failed += 1
            elif not is_llama32:
                # For non-Llama 3.2 models, use the original approach - if any test passes, it's not a glitch
                is_glitch = False
                break

    # For Llama 3.2, token is a glitch only if ALL tests failed
    if is_llama32:
        is_glitch = tests_failed == tests_count

    # Log verification details if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            verification_details = {
                "event": "token_strict_verification",
                "token": token,
                "token_id": token_id,
                "is_llama32": is_llama32,
                "tests_failed": tests_failed,
                "tests_count": tests_count,
                "is_glitch": is_glitch,
                "test_results": test_results,
                "final_decision_reason": "All tests failed" if is_glitch and is_llama32 else
                                        "At least one test passed" if not is_glitch and is_llama32 else
                                        "All tests failed" if is_glitch and not is_llama32 else
                                        "At least one test passed" if not is_glitch and not is_llama32 else
                                        "Unknown"
            }
            f.write(json.dumps(verification_details) + "\n")

    return is_glitch


def chat_token(model_path, token_id, max_size=10, device="auto", quant_type="bfloat16", model=None, tokenizer=None):
    """Test how a model responds to a token in a chat context using proper chat template"""
    # Initialize model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = initialize_model_and_tokenizer(model_path, device, quant_type)
        model.eval()

    with torch.no_grad():
        # Decode token
        token = tokenizer.decode([token_id])

        # Get chat template for proper formatting
        chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

        # Format message using proper chat template
        system_message = 'You are a helpful assistant. When asked to repeat something, repeat it exactly, character for character.'
        user_prompt = f"Repeat this exactly once: {token}"

        # Use built-in template if available, otherwise use custom template
        if isinstance(chat_template, BuiltInTemplate):
            formatted_prompt = chat_template.format_chat(system_message, user_prompt)
        else:
            # Build the properly formatted chat prompt using custom template
            formatted_system = chat_template.system_format.format(content=system_message) if chat_template.system_format else ""
            formatted_user = chat_template.user_format.format(content=user_prompt)

            # For Llama 3/3.2, don't add any assistant prefill to get clean probability measurements
            assistant_prefill = ""
            if chat_template.template_name not in ['llama3', 'llama32']:
                assistant_prefill = " "

            formatted_prompt = formatted_system + formatted_user + assistant_prefill

        # Build inputs and stop tokens (Harmony for gpt-oss, otherwise tokenizer path)
        if _is_gpt_oss_name(model.config._name_or_path) and HARMONY_AVAILABLE:
            prefill_ids, stop_token_ids, formatted_prompt, _enc = build_harmony_prefill(
                system_message, user_prompt, tokenizer, model
            )
            inputs = {"input_ids": torch.tensor([prefill_ids], device=model.device)}
        else:
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # Set up proper stop tokens based on chat template
            stop_token_ids = [tokenizer.eos_token_id]
            if hasattr(tokenizer, 'get_added_vocab'):
                # Add model-specific stop tokens
                if chat_template.template_name in ['llama3', 'llama32']:
                    eot_token_id = tokenizer.get_added_vocab().get('<|eot_id|>')
                    if eot_token_id:
                        stop_token_ids.append(eot_token_id)
                elif chat_template.stop_word:
                    # Try to get the stop word token ID
                    try:
                        stop_word_ids = tokenizer.encode(chat_template.stop_word, add_special_tokens=False)
                        if stop_word_ids:
                            stop_token_ids.extend(stop_word_ids)
                    except:
                        pass


        generation_kwargs = {
            "max_new_tokens": max_size,
            "do_sample": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": stop_token_ids,
            "use_cache": True
        }

        outputs = model.generate(**inputs, **generation_kwargs)

        # Get probability distribution for first generated token
        first_token_probs = torch.softmax(outputs.scores[0][0], dim=-1)

        # Get probability of the target token
        target_prob = first_token_probs[token_id].item()

        # Get top 5 token indices
        top_5_indices = torch.topk(first_token_probs, k=5).indices

        # Get most probable token and its probability
        max_prob_index = torch.argmax(first_token_probs)
        max_prob_token = tokenizer.decode([max_prob_index])
        max_prob = first_token_probs[max_prob_index].item()

        # Decode only the newly generated tokens (excluding the input prompt)
        input_length = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[0][input_length:]
        if _is_gpt_oss_name(model.config._name_or_path) and HARMONY_AVAILABLE:
            generated_text = parse_harmony_final(generated_tokens.tolist(), tokenizer, _enc if '_enc' in locals() else None)
        else:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Prepare result
        result = {
            "token_id": token_id,
            "token": token,
            "target_token_prob": target_prob,
            "top_5_indices": top_5_indices.tolist(),
            "top_token": max_prob_token,
            "top_token_prob": max_prob,
            "top_token_id": max_prob_index.item(),
            "generated_text": generated_text,
            "formatted_prompt": formatted_prompt,
            "chat_template_used": chat_template.template_name
        }

        return result


def mine_glitch_tokens(
    model,
    tokenizer,
    num_iterations: int = 50,
    batch_size: int = 8,
    k: int = 32,
    verbose: bool = True,
    language: str = "ENG",
    checkpoint_callback = None,
    log_file: str = "glitch_mining_log.jsonl",
    enhanced_validation: bool = True,
    max_tokens: int = 50,
    num_attempts: int = 1,
    asr_threshold: float = 0.5
) -> Tuple[List[str], List[int]]:
    """
    Mine for glitch tokens in a language model

    Args:
        model: The language model
        tokenizer: The tokenizer
        num_iterations: Number of mining iterations to run
        batch_size: Batch size for testing tokens
        k: Number of nearest tokens to consider
        verbose: Whether to print progress
        language: Output language (ENG or CN)
        checkpoint_callback: Optional callback function that receives progress info
        log_file: Path to detailed logging file
        enhanced_validation: Whether to use enhanced validation method (default: True)
        max_tokens: Maximum tokens to generate in enhanced validation (default: 50)
        num_attempts: Number of attempts for enhanced validation (default: 1)
        asr_threshold: ASR threshold for considering token a glitch (default: 0.5)

    Returns:
        Tuple of (glitch_tokens, glitch_token_ids)
    """
    import json
    from tqdm import tqdm

    device = model.device

    # Start logging
    with open(log_file, 'w') as f:
        f.write("# Glitch Mining Log - Detailed token evaluation\n")
        model_info = {
            "model_name": model.config._name_or_path,
            "device": str(device),
            "is_llama32": "llama3.2" in model.config._name_or_path.lower() or "llama32" in model.config._name_or_path.lower(),
            "mining_params": {
                "num_iterations": num_iterations,
                "batch_size": batch_size,
                "k": k,
                "enhanced_validation": enhanced_validation,
                "max_tokens": max_tokens if enhanced_validation else None,
                "num_attempts": num_attempts if enhanced_validation else None,
                "asr_threshold": asr_threshold if enhanced_validation else None
            }
        }
        f.write(json.dumps({"event": "start", "info": model_info}) + "\n")

    # Get all token embeddings
    all_token_embeddings = model.get_input_embeddings().weight.detach().to(device)

    # Filter tokens
    classifier = TokenClassifier(model, tokenizer)
    skip_tokens = classifier.filter_tokens()

    # Create mask for tokens to include
    mask = torch.ones(all_token_embeddings.shape[0], dtype=torch.bool, device=device)
    mask[skip_tokens] = False

    # Get template
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    # Template formats
    if isinstance(chat_template, BuiltInTemplate):
        system_format = None
        user_format = None
        assistant_prefill = None
    else:
        system_format = chat_template.system_format
        user_format = chat_template.user_format
        assistant_prefill = ' '

    # Match the system message from token_repetition.py
    system_message = 'You are a helpful assistant. When asked to repeat something, repeat it exactly, character for character.'

    # Import enhanced validation
    from .enhanced_validation import enhanced_glitch_verify

    # Log template info
    with open(log_file, 'a') as f:
        template_info = {
            "template_name": chat_template.template_name,
            "system_format": system_format,
            "user_format": user_format,
            "assistant_prefill": assistant_prefill,
            "uses_builtin_template": isinstance(chat_template, BuiltInTemplate)
        }
        f.write(json.dumps({"event": "template_info", "info": template_info}) + "\n")

    # Find token with smallest L2 norm as starting point
    norms = []
    all_token_embeddings_cpu = all_token_embeddings.cpu().clone()
    valid_tokens = torch.where(mask)[0]

    for token_id in valid_tokens:
        l2_norm = torch.norm(all_token_embeddings_cpu[token_id], p=2)
        norms.append((token_id.item(), l2_norm.item()))

    norms.sort(key=lambda x: x[1])
    start_id = norms[0][0]
    current_token_id = start_id

    # Initialize counters
    total_tokens_checked = 0
    glitch_tokens_count = 0
    glitch_tokens = []
    glitch_token_ids = []

    # Set up progress bar
    pbar = tqdm(total=num_iterations, desc="Mining glitch tokens")

    for iteration in range(num_iterations):
        # Step 1: Calculate entropy and gradient for current token
        token = tokenizer.decode([current_token_id])
        # Match the user prompt from token_repetition.py
        user_prompt = f"Please respond with exactly this: '{token}'"

        # Format input using appropriate template
        if _is_gpt_oss_name(model.config._name_or_path) and HARMONY_AVAILABLE:
            # Build Harmony-based prefill for gpt-oss with reasoning control
            _prefill_ids, _stop_ids, _prefill_text, _enc = build_harmony_prefill(system_message, user_prompt, tokenizer, model)
            formatted_input = _prefill_text
        elif isinstance(chat_template, BuiltInTemplate):
            formatted_input = chat_template.format_chat(system_message, user_prompt)
        else:
            formatted_user = (user_format or "User: {content}\nAssistant: ").format(content=user_prompt)
            formatted_input = (system_format.format(content=system_message) if system_format else "") + formatted_user + (assistant_prefill or "")

        # Log the current token and formatted input
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                "event": "iteration_start",
                "iteration": iteration,
                "current_token": token,
                "current_token_id": current_token_id,
                "formatted_input": formatted_input
            }) + "\n")

        # Find position of the token in the input
        input_ids = tokenizer.encode(formatted_input)
        # Try to find position after the quote - use single quote now instead of double
        quote_indices = [i for i, id in enumerate(input_ids) if tokenizer.decode([id]) == "'"]
        if len(quote_indices) >= 2:
            current_token_position = quote_indices[-2] + 1
        else:
            # Fallback to a fixed position if quotes can't be found
            current_token_position = len(input_ids) - 3

        input_ids = torch.tensor([input_ids], device=device)
        inputs_embeds = model.get_input_embeddings()(input_ids).clone().detach()
        inputs_embeds.requires_grad_(True)

        outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        entropy_value = entropy(next_token_probs)

        grads = torch.autograd.grad(
            outputs=entropy_value,
            inputs=inputs_embeds,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]

        # Step 2: Find nearest tokens and calculate approximated entropy
        current_token_embedding = model.get_input_embeddings().weight[current_token_id].detach()
        grad = grads[0, current_token_position, :].detach()

        valid_mask = mask.clone()
        valid_mask[current_token_id] = False
        valid_token_ids = torch.where(valid_mask)[0]
        valid_embeddings = model.get_input_embeddings().weight[valid_token_ids]

        # Normalize embeddings
        normalized_current = F.normalize(current_token_embedding.unsqueeze(0), p=2, dim=1)
        normalized_valid = F.normalize(valid_embeddings, p=2, dim=1)

        # Calculate distances
        normalized_l2_distances = torch.norm(normalized_valid - normalized_current, p=2, dim=1)
        nearest_indices = torch.topk(normalized_l2_distances, k=min(k, len(valid_token_ids)), largest=False).indices

        candidate_token_ids = valid_token_ids[nearest_indices]

        delta_embeddings = valid_embeddings[nearest_indices] - current_token_embedding
        entropy_approximations = entropy_value.item() + (delta_embeddings @ grad).detach()

        # Step 3: Select top batch_size tokens with highest entropy approximation
        top_batch_indices = torch.topk(entropy_approximations, k=min(batch_size, len(candidate_token_ids))).indices
        batch_token_ids = candidate_token_ids[top_batch_indices]

        # Step 4: Evaluate if tokens are glitch tokens
        model.eval()
        with torch.no_grad():
            batch_entropies = []
            batch_iteration_pbar = tqdm(batch_token_ids, desc=f"Batch {iteration+1}", leave=False) if verbose else batch_token_ids

            for token_id in batch_iteration_pbar:
                token_id_value = token_id.item()
                token = tokenizer.decode([token_id_value])
                # Match the user prompt from token_repetition.py
                user_prompt = f"Please respond with exactly this: '{token}'"

                # Format input using appropriate template
                if _is_gpt_oss_name(model.config._name_or_path) and HARMONY_AVAILABLE:
                    _prefill_ids, _stop_ids, _prefill_text, _enc = build_harmony_prefill(system_message, user_prompt, tokenizer, model)
                    formatted_input = _prefill_text
                elif isinstance(chat_template, BuiltInTemplate):
                    formatted_input = chat_template.format_chat(system_message, user_prompt)
                else:
                    formatted_input = (system_format.format(content=system_message) if system_format else "") + (user_format or "User: {content}\nAssistant: ").format(content=user_prompt) + (assistant_prefill or "")
                input_ids = tokenizer(formatted_input, return_tensors="pt").to(device)

                outputs = model(input_ids=input_ids.input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                entropy_value = entropy(next_token_probs)
                batch_entropies.append(entropy_value.item())

                # Get top 5 most likely tokens and their probabilities
                top5_values, top5_indices = torch.topk(next_token_probs[0], 5)
                top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices]
                top5_probs = top5_values.tolist()

                max_prob_token_id = next_token_probs.argmax().item()
                max_prob_token = tokenizer.decode([max_prob_token_id])
                max_prob = next_token_probs[0, max_prob_token_id].item()

                # Check probability of the target token
                target_prob = next_token_probs[0, token_id_value].item()

                # Set threshold based on model type - use extremely low threshold
                is_llama32 = "llama3.2" in model.config._name_or_path.lower() or "llama32" in model.config._name_or_path.lower()
                # Use a much more strict threshold - true glitches have probabilities near zero
                probability_threshold = 0.00001 if is_llama32 else 0.00005

                # Add content-based filtering to avoid certain patterns we know generate false positives
                token_text = token
                # Skip tokens with brackets, common programming symbols, or specific prefixes that tend to be false positives
                should_skip = (
                    ('[' in token_text or ']' in token_text) or
                    ('(' in token_text or ')' in token_text) or
                    ('_' in token_text) or
                    ('arg' in token_text.lower()) or
                    ('prop' in token_text.lower()) or
                    ('char' in token_text.lower()) or
                    (token_text.startswith('.')) or
                    (token_text.startswith('$'))
                )

                # Two criteria for a token being a glitch:
                # 1. It has a low probability of being predicted (below threshold)
                # 2. AND it's not the top predicted token
                # 3. AND it doesn't match our skip patterns
                is_glitch = (target_prob < probability_threshold) and (max_prob_token_id != token_id_value) and not should_skip

                # Log token verification details
                with open(log_file, 'a') as f:
                    token_details = {
                        "event": "token_verification",
                        "iteration": iteration,
                        "token": token,
                        "token_id": token_id_value,
                        "formatted_input": formatted_input,
                        "entropy": entropy_value.item(),
                        "target_probability": target_prob,
                        "top_token": max_prob_token,
                        "top_token_id": max_prob_token_id,
                        "top_token_probability": max_prob,
                        "top5_tokens": top5_tokens,
                        "top5_probabilities": top5_probs,
                        "probability_threshold": probability_threshold,
                        "is_llama32": is_llama32,
                        "should_skip": should_skip,
                        "is_glitch": is_glitch,
                        "reason": "pattern filtered" if should_skip else
                                 "probability below threshold and not top token" if is_glitch else
                                 "probability above threshold" if target_prob >= probability_threshold else
                                 "is top predicted token"
                    }
                    f.write(json.dumps(token_details) + "\n")

                total_tokens_checked += 1

                if is_glitch:
                    # Validate the potential glitch token
                    if enhanced_validation:
                        # Use enhanced validation
                        validation_result, asr = enhanced_glitch_verify(
                            model, tokenizer, token_id_value, chat_template, log_file,
                            max_tokens, quiet=True, num_attempts=num_attempts, asr_threshold=asr_threshold
                        )

                        # Log validation result
                        with open(log_file, 'a') as f:
                            validation_details = {
                                "event": "mining_validation",
                                "iteration": iteration,
                                "token": token,
                                "token_id": token_id_value,
                                "validation_method": "enhanced",
                                "validation_result": validation_result,
                                "asr": asr,
                                "max_tokens": max_tokens,
                                "num_attempts": num_attempts,
                                "asr_threshold": asr_threshold
                            }
                            f.write(json.dumps(validation_details) + "\n")
                    else:
                        # Use standard validation
                        validation_result = strictly_glitch_verify(
                            model, tokenizer, token_id_value, chat_template, log_file
                        )

                        # Log validation result
                        with open(log_file, 'a') as f:
                            validation_details = {
                                "event": "mining_validation",
                                "iteration": iteration,
                                "token": token,
                                "token_id": token_id_value,
                                "validation_method": "standard",
                                "validation_result": validation_result
                            }
                            f.write(json.dumps(validation_details) + "\n")

                        # Set asr to None for standard validation
                        asr = None

                    # Only add to results if validation confirms it's a glitch
                    if validation_result:
                        glitch_tokens_count += 1
                        glitch_tokens.append(token)
                        glitch_token_ids.append(token_id_value)

                        if verbose:
                            validation_method = "enhanced" if enhanced_validation else "standard"
                            if enhanced_validation and asr is not None:
                                tqdm.write(f"✓ Validated glitch token: '{token}' (ID: {token_id_value}, asr: {asr:.2%}, entropy: {entropy_value.item():.4f}, target_prob: {target_prob:.6f}, top_prob: {max_prob:.6f}, method: {validation_method})")
                            else:
                                tqdm.write(f"✓ Validated glitch token: '{token}' (ID: {token_id_value}, entropy: {entropy_value.item():.4f}, target_prob: {target_prob:.6f}, top_prob: {max_prob:.6f}, method: {validation_method})")
                    else:
                        if verbose:
                            validation_method = "enhanced" if enhanced_validation else "standard"
                            if enhanced_validation and asr is not None:
                                tqdm.write(f"✗ False positive: '{token}' (ID: {token_id_value}, asr: {asr:.2%}, failed {validation_method} validation)")
                            else:
                                tqdm.write(f"✗ False positive: '{token}' (ID: {token_id_value}, failed {validation_method} validation)")

        # Step 5: Choose highest entropy token for next iteration
        max_entropy_index = torch.argmax(torch.tensor(batch_entropies))
        current_token_id = batch_token_ids[max_entropy_index].item()

        # Update mask
        mask[batch_token_ids] = False

        # Clean up memory
        del outputs, entropy_value, grads
        torch.cuda.empty_cache()

        # Call checkpoint callback if provided
        if checkpoint_callback is not None:
            checkpoint_data = {
                "iteration": iteration,
                "total_iterations": num_iterations,
                "total_tokens_checked": total_tokens_checked,
                "glitch_tokens_count": glitch_tokens_count,
                "glitch_tokens": glitch_tokens.copy(),
                "glitch_token_ids": glitch_token_ids.copy(),
                "current_token_id": current_token_id
            }
            checkpoint_callback(checkpoint_data)

        # Update progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    # Finalize
    glitch_frequency = glitch_tokens_count / total_tokens_checked if total_tokens_checked > 0 else 0

    if verbose:
        print(f"Summary: Found {glitch_tokens_count} glitch tokens out of {total_tokens_checked} checked ({glitch_frequency:.2%})")

    return glitch_tokens, glitch_token_ids
