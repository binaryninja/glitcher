"""
Stratified vocabulary scan for glitch token discovery.

Implements the candidate-generation method described in paper §2.4 / §8.2:
sample tokens across (L2 norm quintile, Unicode general category, length
bucket) strata so coverage is not biased toward Latin-script subword
fragments the way entropy-guided mining is. Also exposes a
category-targeted scan (CJK, Arabic, Cyrillic, whitespace, punctuation
blocks) for follow-up runs.

Each candidate is validated with `enhanced_glitch_verify`. The §8.1
stochastic ASR patch is already applied in `enhanced_validation.py` and
activates whenever `num_attempts > 1`, so this scanner inherits it.
"""

from __future__ import annotations

import json
import random
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .enhanced_validation import enhanced_glitch_verify
from .model import get_template_for_model


# --- Unicode category groups for targeted scanning ------------------------------

UNICODE_CATEGORY_GROUPS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # Codepoint ranges; chosen to match the categories the paper called out as
    # disproportionately missed by entropy mining (paper §6.1).
    "cjk": (
        (0x3040, 0x30FF),   # Hiragana / Katakana
        (0x3400, 0x4DBF),   # CJK Ext-A
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0xF900, 0xFAFF),   # CJK Compat Ideographs
        (0x20000, 0x2A6DF), # CJK Ext-B
    ),
    "arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "cyrillic": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
    ),
    "punctuation": (
        (0x2000, 0x206F),  # General Punctuation
        (0x3000, 0x303F),  # CJK Symbols & Punctuation
        (0xFE30, 0xFE4F),  # CJK Compat Forms
    ),
}


def _is_whitespace_token(text: str) -> bool:
    return bool(text) and all((ch.isspace() or ch in "   ") for ch in text)


def _dominant_category(text: str) -> str:
    """
    Pick the modal Unicode general category of a decoded token. Empty / single
    -char tokens map to that char's category; multi-char tokens map to the
    most common, with ties broken alphabetically for determinism.
    """
    if not text:
        return "Cn"  # not assigned
    if _is_whitespace_token(text):
        return "Zs"
    counts = Counter(unicodedata.category(ch) for ch in text)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _length_bucket(text: str) -> str:
    n = len(text)
    if n <= 1:
        return "len1"
    if n <= 3:
        return "len2_3"
    if n <= 6:
        return "len4_6"
    return "len7_plus"


def _has_codepoint_in_ranges(text: str, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(any(lo <= ord(ch) <= hi for ch in text) for lo, hi in ranges)


# --- Stratification -------------------------------------------------------------


def _norm_quintile_thresholds(norms: torch.Tensor) -> List[float]:
    """Return the four cut-points dividing `norms` into 5 quintiles."""
    qs = torch.quantile(
        norms,
        torch.tensor([0.2, 0.4, 0.6, 0.8], device=norms.device, dtype=norms.dtype),
    )
    return [q.item() for q in qs]


def _norm_bucket(value: float, thresholds: Sequence[float]) -> str:
    for i, t in enumerate(thresholds):
        if value < t:
            return f"q{i+1}"
    return f"q{len(thresholds)+1}"


def build_strata(
    tokenizer,
    embeddings: torch.Tensor,
    skip_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[Dict[Tuple[str, str, str], List[int]], List[float]]:
    """
    Build a stratification of the vocabulary keyed by
    (norm quintile, Unicode category, length bucket).

    Returns the strata dict and the norm-quintile thresholds.
    """
    skip = set(skip_token_ids or [])
    vocab_size = embeddings.shape[0]

    norms = torch.linalg.vector_norm(embeddings, dim=1)
    thresholds = _norm_quintile_thresholds(norms)

    strata: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for token_id in range(vocab_size):
        if token_id in skip:
            continue
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue
        if not text:
            continue
        key = (
            _norm_bucket(norms[token_id].item(), thresholds),
            _dominant_category(text),
            _length_bucket(text),
        )
        strata[key].append(token_id)

    return strata, thresholds


def stratified_sample(
    strata: Dict[Tuple[str, str, str], List[int]],
    sample_size: int,
    rng: random.Random,
    min_per_stratum: int = 1,
) -> List[int]:
    """
    Sample `sample_size` token ids from `strata`. Each non-empty stratum
    contributes at least `min_per_stratum` (when available); the rest is
    distributed proportionally to stratum population. (paper §2.4)
    """
    non_empty = [(k, v) for k, v in strata.items() if v]
    if not non_empty:
        return []

    chosen: List[int] = []
    remaining: Dict[Tuple[str, str, str], List[int]] = {}

    # Floor: at least min_per_stratum per stratum (capped at availability).
    for key, ids in non_empty:
        take = min(min_per_stratum, len(ids), sample_size - len(chosen))
        if take > 0:
            picks = rng.sample(ids, take)
            chosen.extend(picks)
            leftovers = [t for t in ids if t not in set(picks)]
        else:
            leftovers = list(ids)
        remaining[key] = leftovers
        if len(chosen) >= sample_size:
            return chosen[:sample_size]

    # Proportional fill of the remainder.
    pool_total = sum(len(v) for v in remaining.values())
    if pool_total == 0 or len(chosen) >= sample_size:
        return chosen[:sample_size]

    quota_left = sample_size - len(chosen)
    raw_quotas = {
        k: (len(v) / pool_total) * quota_left for k, v in remaining.items()
    }
    int_quotas = {k: int(v) for k, v in raw_quotas.items()}
    leftover = quota_left - sum(int_quotas.values())
    # Distribute the rounding leftover to the strata with the largest
    # fractional parts, deterministically.
    fractions = sorted(
        ((k, raw_quotas[k] - int_quotas[k]) for k in int_quotas),
        key=lambda kv: (-kv[1], kv[0]),
    )
    for k, _ in fractions[:leftover]:
        int_quotas[k] += 1

    for key, count in int_quotas.items():
        if count <= 0:
            continue
        ids = remaining[key]
        take = min(count, len(ids))
        chosen.extend(rng.sample(ids, take))

    rng.shuffle(chosen)
    return chosen[:sample_size]


def category_targeted_ids(
    tokenizer,
    group: str,
    skip_token_ids: Optional[Iterable[int]] = None,
) -> List[int]:
    """
    Return all vocabulary tokens whose decoded text contains at least one
    codepoint inside the named UNICODE_CATEGORY_GROUPS group, plus a
    "whitespace" virtual group for whitespace-only tokens.
    """
    skip = set(skip_token_ids or [])
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    out: List[int] = []
    if group == "whitespace":
        for token_id in range(vocab_size):
            if token_id in skip:
                continue
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                continue
            if _is_whitespace_token(text):
                out.append(token_id)
        return out

    if group not in UNICODE_CATEGORY_GROUPS:
        raise ValueError(f"Unknown category group: {group}")
    ranges = UNICODE_CATEGORY_GROUPS[group]
    for token_id in range(vocab_size):
        if token_id in skip:
            continue
        try:
            text = tokenizer.decode([token_id])
        except Exception:
            continue
        if _has_codepoint_in_ranges(text, ranges):
            out.append(token_id)
    return out


# --- Validation driver ----------------------------------------------------------


def stratified_scan(
    model,
    tokenizer,
    sample_size: int = 2000,
    seed: int = 0,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    max_tokens: int = 50,
    num_attempts: int = 5,
    asr_threshold: float = 0.5,
    quiet: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run a stratified vocabulary scan and validate each candidate with
    `enhanced_glitch_verify`. Returns a results dict; if `output_file` is
    provided, also persists the results to disk.

    Defaults align with the paper's §2.4 methodology: 2,000-token sample,
    5 attempts per token (so the §8.1 stochastic loop is meaningful),
    50 generated tokens per attempt, ASR threshold 0.5.
    """
    rng = random.Random(seed)
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)

    embeddings = model.get_input_embeddings().weight.detach()
    strata, thresholds = build_strata(tokenizer, embeddings)
    candidates = stratified_sample(strata, sample_size, rng)

    confirmed: List[Dict[str, Any]] = []
    examined: List[Dict[str, Any]] = []

    started = time.time()
    for idx, token_id in enumerate(candidates):
        try:
            token = tokenizer.decode([token_id])
        except Exception:
            continue
        try:
            is_glitch, asr = enhanced_glitch_verify(
                model, tokenizer, token_id,
                chat_template=chat_template,
                log_file=log_file,
                max_tokens=max_tokens,
                quiet=quiet,
                num_attempts=num_attempts,
                asr_threshold=asr_threshold,
            )
        except Exception as e:
            if log_file:
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "event": "stratified_scan_error",
                        "token_id": token_id,
                        "error": str(e),
                    }) + "\n")
            continue

        record = {"token": token, "token_id": token_id, "asr": asr, "is_glitch": is_glitch}
        examined.append(record)
        if is_glitch:
            confirmed.append(record)
        if progress_callback is not None:
            progress_callback(idx + 1, len(candidates), record)

    results = {
        "model_path": model.config._name_or_path,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(candidates),
        "vocab_size": embeddings.shape[0],
        "norm_quintile_thresholds": thresholds,
        "num_attempts": num_attempts,
        "asr_threshold": asr_threshold,
        "seed": seed,
        "runtime_seconds": time.time() - started,
        "confirmed_glitches": confirmed,
        "examined": examined,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def category_targeted_scan(
    model,
    tokenizer,
    group: str,
    sample_size: Optional[int] = None,
    seed: int = 0,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    max_tokens: int = 50,
    num_attempts: int = 5,
    asr_threshold: float = 0.5,
    quiet: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Validate every (or a sampled subset of) vocabulary tokens that fall in a
    named UNICODE_CATEGORY_GROUPS group.
    """
    chat_template = get_template_for_model(model.config._name_or_path, tokenizer)
    candidates = category_targeted_ids(tokenizer, group)
    if sample_size is not None and sample_size < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, sample_size)

    confirmed: List[Dict[str, Any]] = []
    examined: List[Dict[str, Any]] = []

    started = time.time()
    for idx, token_id in enumerate(candidates):
        try:
            token = tokenizer.decode([token_id])
        except Exception:
            continue
        try:
            is_glitch, asr = enhanced_glitch_verify(
                model, tokenizer, token_id,
                chat_template=chat_template,
                log_file=log_file,
                max_tokens=max_tokens,
                quiet=quiet,
                num_attempts=num_attempts,
                asr_threshold=asr_threshold,
            )
        except Exception as e:
            if log_file:
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "event": "category_scan_error",
                        "token_id": token_id,
                        "group": group,
                        "error": str(e),
                    }) + "\n")
            continue

        record = {"token": token, "token_id": token_id, "asr": asr, "is_glitch": is_glitch}
        examined.append(record)
        if is_glitch:
            confirmed.append(record)
        if progress_callback is not None:
            progress_callback(idx + 1, len(candidates), record)

    results = {
        "model_path": model.config._name_or_path,
        "group": group,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(candidates),
        "num_attempts": num_attempts,
        "asr_threshold": asr_threshold,
        "seed": seed,
        "runtime_seconds": time.time() - started,
        "confirmed_glitches": confirmed,
        "examined": examined,
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
