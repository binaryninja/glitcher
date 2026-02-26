#!/usr/bin/env python3
"""Redact sensitive token text from glitcher result files.

Replaces token_text / token / decoded_text values with a hash-based
placeholder while preserving token IDs and all numeric metadata.  This
allows sharing results publicly without exposing potentially sensitive
or adversarial token strings.

Usage:
    # Redact a single file
    python APPENDIX-B/redact.py glitch_tokens.json --output glitch_tokens_redacted.json

    # Redact multiple files
    python APPENDIX-B/redact.py glitch_tokens.json genetic_results.json \
        --output-dir redacted/

    # Preview what would be redacted (dry run)
    python APPENDIX-B/redact.py glitch_tokens.json --dry-run

    # Keep specific token IDs unredacted
    python APPENDIX-B/redact.py glitch_tokens.json --keep-ids 100,200,300
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Set

# Fields that contain token text and should be redacted
TEXT_FIELDS = {
    "token_text",
    "token",
    "decoded_text",
    "text",
    "base_text",
    "best_combination_text",
    "target_token_text",
    "wanted_token_text",
}

# List-of-strings fields that should have each element redacted
TEXT_LIST_FIELDS = {
    "token_texts",
    "glitch_tokens",
}

# Fields that should never be redacted (numeric / structural)
PRESERVE_FIELDS = {
    "token_id",
    "token_ids",
    "l2_norm",
    "entropy",
    "asr",
    "method",
    "categories",
    "probability",
    "reduction",
    "generation",
    "population_size",
    "mutation_rate",
}


def redact_text(text: str) -> str:
    """Replace token text with a short hash placeholder."""
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"[REDACTED:{h}]"


def redact_value(obj: Any, keep_ids: Optional[Set[int]] = None) -> Any:
    """Recursively redact text fields in a JSON-like structure."""
    if isinstance(obj, dict):
        # Check if this dict has a token_id we should keep
        tid = obj.get("token_id")
        if keep_ids and tid is not None and tid in keep_ids:
            return obj  # keep unredacted

        result = {}
        for key, val in obj.items():
            if key in TEXT_FIELDS and isinstance(val, str):
                result[key] = redact_text(val)
            elif key in TEXT_LIST_FIELDS and isinstance(val, list):
                result[key] = [
                    redact_text(item) if isinstance(item, str) else item
                    for item in val
                ]
            else:
                result[key] = redact_value(val, keep_ids)
        return result
    elif isinstance(obj, list):
        return [redact_value(item, keep_ids) for item in obj]
    else:
        return obj


def redact_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    keep_ids: Optional[Set[int]] = None,
    dry_run: bool = False,
) -> dict:
    """Redact a single JSON file.  Returns stats."""
    with open(input_path) as f:
        data = json.load(f)

    original = json.dumps(data)
    redacted = redact_value(deepcopy(data), keep_ids)
    redacted_str = json.dumps(redacted, indent=2, ensure_ascii=False)

    # Count redactions
    count = redacted_str.count("[REDACTED:")

    if dry_run:
        print(f"{input_path}: {count} field(s) would be redacted")
    else:
        if output_path is None:
            output_path = input_path.with_suffix(".redacted.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(redacted_str)
            f.write("\n")
        print(f"{input_path} -> {output_path} ({count} field(s) redacted)")

    return {"input": str(input_path), "output": str(output_path), "redacted_count": count}


def main():
    parser = argparse.ArgumentParser(
        description="Redact sensitive token text from result files"
    )
    parser.add_argument("files", nargs="+", help="JSON files to redact")
    parser.add_argument("--output", help="Output file (single input only)")
    parser.add_argument("--output-dir", help="Output directory (multiple inputs)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--keep-ids",
        help="Comma-separated token IDs to leave unredacted",
    )
    args = parser.parse_args()

    keep_ids = None
    if args.keep_ids:
        keep_ids = {int(x.strip()) for x in args.keep_ids.split(",")}

    for filepath in args.files:
        input_path = Path(filepath)
        if not input_path.exists():
            print(f"WARNING: {input_path} not found, skipping")
            continue

        if args.output and len(args.files) == 1:
            output_path = Path(args.output)
        elif args.output_dir:
            output_path = Path(args.output_dir) / input_path.name
        else:
            output_path = None  # defaults to .redacted.json suffix

        redact_file(input_path, output_path, keep_ids, args.dry_run)


if __name__ == "__main__":
    main()
