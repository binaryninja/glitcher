#!/usr/bin/env python3
"""
Model-agnostic assistant response extraction.

Supports multiple chat template formats (Llama, Qwen/ChatML, Mistral, etc.)
by trying known assistant markers in order and falling back to input-length
based extraction.
"""

import re

from .logging_utils import get_logger

_logger = get_logger("response_utils")

# (marker_text, end_token_or_pattern) pairs, tried in order.
# The first match wins.  Most-specific markers come first.
_ASSISTANT_MARKERS = [
    # Llama 3.x
    (
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
    ),
    # Qwen / ChatML
    (
        "<|im_start|>assistant\n",
        "<|im_end|>",
    ),
    # Gemma
    (
        "<start_of_turn>model\n",
        "<end_of_turn>",
    ),
    # Mistral v3+ instruct
    (
        "[/INST]",
        "</s>",
    ),
]


def _strip_think_block(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (e.g. Qwen3).

    Some models emit a chain-of-thought block before the actual response.
    This strips it so downstream JSON parsing sees only the content.
    """
    # Closed think block
    result = re.sub(r'<think>[\s\S]*?</think>\s*', '', text, count=1)
    if result != text:
        return result
    # Unclosed think block (model hit token limit mid-thought)
    if text.startswith('<think>'):
        return ''
    return text


def extract_assistant_response(
    full_output: str,
    formatted_input: str,
    label: str = "",
    logger=None,
) -> str:
    """Extract the assistant portion from a full model generation.

    Tries known chat-template assistant markers in order, then falls
    back to slicing off ``formatted_input`` from the front.

    Args:
        full_output: Complete decoded model output (with special tokens).
        formatted_input: The prompt that was fed to the model.
        label: Human-readable label for log messages.
        logger: Optional logger; uses module-level logger if None.

    Returns:
        The assistant's response text, stripped of surrounding whitespace
        and end-of-turn tokens.
    """
    log = logger or _logger

    # Try each known marker
    for marker, end_token in _ASSISTANT_MARKERS:
        pos = full_output.rfind(marker)
        if pos != -1:
            response = full_output[pos + len(marker):]
            if end_token and response.endswith(end_token):
                response = response[:-len(end_token)]
            response = _strip_think_block(response)
            return response.strip()

    # Fallback: strip the formatted input prefix
    if len(full_output) < len(formatted_input):
        log.warning(
            f"'{label}' - corrupted output "
            f"({len(full_output)} < {len(formatted_input)} chars)"
        )
        response = full_output
    else:
        response = full_output[len(formatted_input):]

    # Try a generic regex: anything after "assistant" role marker
    m = re.search(r'assistant[^a-zA-Z]*\n', full_output)
    if m:
        candidate = full_output[m.end():]
        # Strip common end tokens
        for _, end_token in _ASSISTANT_MARKERS:
            if end_token and candidate.endswith(end_token):
                candidate = candidate[:-len(end_token)]
        candidate = _strip_think_block(candidate)
        return candidate.strip()

    log.warning(f"'{label}' - could not find assistant marker")

    response = _strip_think_block(response).strip()
    if not response and full_output:
        response = full_output.strip()
    return response
