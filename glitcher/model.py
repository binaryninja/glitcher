#!/usr/bin/env python3
"""
Core functionality for glitch token mining and testing
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple, Dict, Any, Optional, Union


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


def get_template_for_model(model_name: str) -> Template:
    """Get the appropriate chat template for a model"""
    # Only take the last part of the path and normalize
    original_name = model_name
    model_name = model_name.split('/')[-1].lower().replace('-', '').replace('_', '')
    
    # Special case for Llama 3.2 models
    if "llama3.2" in model_name or "llama32" in model_name:
        return _TEMPLATES['llama32']
        
    # Special case for Llama 3 models that aren't 3.2
    if "llama3" in model_name and "3.2" not in original_name and "32" not in model_name:
        return _TEMPLATES['llama3']
    
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


def glitch_verify_message1(chat_template, token):
    """Format a verification message that asks to repeat a token"""
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' '
    # Match the system message from token_repetition.py
    system_message = 'You are a helpful assistant. When asked to repeat something, repeat it exactly, character for character.'
    # Match the user prompt from token_repetition.py
    user_prompt = f"Please respond with exactly this: '{token}'"
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    
    # For Llama 3.2, we need to ensure we have the right prefill token for best results
    if chat_template.template_name in ['llama3', 'llama32']:
        assistant_prefill = ''
    
    return formatted_system + formatted_user + assistant_prefill


def glitch_verify_message2(chat_template, token):
    """Format an alternative verification message asking for token meaning"""
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
        chat_template = get_template_for_model(model.config._name_or_path)
    
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
                (token_text.startswith('$'))$'))
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


def chat_token(model_path, token_id, max_size=10, device="auto", quant_type="bfloat16"):
    """Test how a model responds to a token in a chat context"""
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_path, device, quant_type)
    model.eval()
    
    # Get chat template
    chat_template = get_template_for_model(model.config._name_or_path)
    
    # Get template formats
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_format = chat_template.assistant_format
    stop_word = chat_template.stop_word
    system_message = 'You are a helpful assistant that repeats strings exactly as requested.'
    
    with torch.no_grad():
        # Decode token
        token = tokenizer.decode([token_id])
        user_prompt = f'''Please repeat the string: "{token}"'''
        assistant_prefill = ' Sure, the string is: "'
        
        # For Llama 3.2, adjust the prefill for better results
        if chat_template.template_name in ['llama3', 'llama32']:
            assistant_prefill = ""
        
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
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
        
        # Generate model response - use more conservative parameters for Llama 3.2
        generation_kwargs = {
            "max_new_tokens": max_size,
            "do_sample": False,
            "output_scores": True,
            "return_dict_in_generate": True
        }
        
        # Add specific parameters if needed for this model
        if "llama" in model.config._name_or_path.lower():
            generation_kwargs.update({
                "repetition_penalty": 1.0,  # No repetition penalty
                "temperature": 0.0,  # Use greedy decoding
            })
            
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
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
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
    log_file: str = "glitch_mining_log.jsonl"
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
                "k": k
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
    chat_template = get_template_for_model(model.config._name_or_path)
    
    # Template formats
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = ' '
    # Match the system message from token_repetition.py
    system_message = 'You are a helpful assistant. When asked to repeat something, repeat it exactly, character for character.'
    
    # Log template info
    with open(log_file, 'a') as f:
        template_info = {
            "template_name": chat_template.template_name,
            "system_format": system_format,
            "user_format": user_format,
            "assistant_prefill": assistant_prefill
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
        formatted_user = user_format.format(content=user_prompt)
        formatted_input = (system_format.format(content=system_message) if system_format else "") + formatted_user + assistant_prefill
        
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
                formatted_input = (system_format.format(content=system_message) if system_format else "") + user_format.format(content=user_prompt) + assistant_prefill
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
                    glitch_tokens_count += 1
                    glitch_tokens.append(token)
                    glitch_token_ids.append(token_id_value)
                    
                    if verbose:
                        tqdm.write(f"✓ Found glitch token: '{token}' (ID: {token_id_value}, entropy: {entropy_value.item():.4f}, target_prob: {target_prob:.6f}, top_prob: {max_prob:.6f})")
        
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