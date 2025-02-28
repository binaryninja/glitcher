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
    model_name = model_name.split('/')[-1].lower().replace('-', '').replace('_', '')
    
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
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''
    user_prompt = f'Please repeat the string: "«{token}»"'
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    
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
    
    return formatted_system + formatted_user1 + formatted_assitant1 + formatted_user2 + assistant_prefill


def glitch_verify_message3(chat_template, token):
    """Format a third verification message with different phrasing"""
    system_format = chat_template.system_format
    user_format = chat_template.user_format
    assistant_prefill = " Of course! Here is the repeated string:'"
    system_message = ''
    user_prompt = f"Question: Please can you repeat the string '{token}'?"
    
    formatted_system = system_format.format(content=system_message) if system_format else ""
    formatted_user = user_format.format(content=user_prompt)
    
    return formatted_system + formatted_user + assistant_prefill


def strictly_glitch_verify(model, tokenizer, token_id, chat_template=None):
    """Verify if a token is a glitch token using multiple prompt formats"""
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
    
    # Check if the token is a glitch token
    is_glitch = True
    for input_ids in inputs:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # Get logits for the next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            # Get most probable token
            predicted_token_id = next_token_probs.argmax(dim=-1).item()
            
            # Check if predicted token matches input token
            is_glitch = predicted_token_id != token_id
        
        # If any test passes, token is not a glitch
        if not is_glitch:
            break
    
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
    system_message = ''
    
    with torch.no_grad():
        # Decode token
        token = tokenizer.decode([token_id])
        user_prompt = f'''Please repeat the string: "«{token}»"'''
        assistant_prefill = ' Sure, the string is: "«'
        
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
        
        # Generate model response
        outputs = model.generate(**inputs, max_new_tokens=max_size, do_sample=False, 
                                output_scores=True, return_dict_in_generate=True)
        
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
    checkpoint_callback = None
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
        
    Returns:
        Tuple of (glitch_tokens, glitch_token_ids)
    """
    device = model.device
    
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
    assistant_prefill = ' Sure, the string is: "«'
    system_message = ''
    
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
    
    for iteration in range(num_iterations):
        if verbose:
            print(f"Iteration: {iteration}" if language == "ENG" else f"迭代次数: {iteration}")
        
        # Step 1: Calculate entropy and gradient for current token
        token = tokenizer.decode([current_token_id])
        user_prompt = f'Please repeat the string: "«{token}»"'
        formatted_user = user_format.format(content=user_prompt)
        formatted_input = (system_format.format(content=system_message) if system_format else "") + formatted_user + assistant_prefill
        
        # Find position of the token in the input
        input_ids = tokenizer.encode(formatted_input)
        quote_id = tokenizer.encode('"«', add_special_tokens=False)[-1]
        current_token_position = input_ids.index(quote_id) + 1
        
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
            
            for token_id in batch_token_ids:
                token = tokenizer.decode([token_id.item()])
                user_prompt = f'Please repeat the string: "«{token}»"'
                formatted_input = (system_format.format(content=system_message) if system_format else "") + user_format.format(content=user_prompt) + assistant_prefill
                input_ids = tokenizer(formatted_input, return_tensors="pt").to(device)
                
                outputs = model(input_ids=input_ids.input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                entropy_value = entropy(next_token_probs)
                batch_entropies.append(entropy_value.item())
                
                max_prob_token_id = next_token_probs.argmax().item()
                is_glitch = max_prob_token_id != token_id.item()
                total_tokens_checked += 1
                
                if is_glitch:
                    glitch_tokens_count += 1
                    glitch_tokens.append(token)
                    glitch_token_ids.append(token_id.item())
                
                if verbose:
                    print_str = f"  Current token: '{token}', token id: {token_id.item()}, is glitch token: {'Yes' if is_glitch else 'No'}, entropy: {entropy_value.item():.4f}" if language == "ENG" else f"  当前token: '{token}', token id: {token_id.item()}, 是否为glitch token: {'是' if is_glitch else '否'}, 熵值: {entropy_value.item():.4f}"
                    print(print_str)
        
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
    
    # Finalize
    glitch_frequency = glitch_tokens_count / total_tokens_checked if total_tokens_checked > 0 else 0
    
    if verbose:
        print(f"Total tokens checked: {total_tokens_checked}" if language == "ENG" else f"检测的总token数: {total_tokens_checked}")
        print(f"Number of glitch tokens: {glitch_tokens_count}" if language == "ENG" else f"glitch token数: {glitch_tokens_count}")
        print(f"Frequency of glitch tokens: {glitch_frequency:.4f}" if language == "ENG" else f"glitch token出现的频率: {glitch_frequency:.4f}")
    
    return glitch_tokens, glitch_token_ids