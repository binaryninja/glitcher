#!/usr/bin/env python3
"""Frozen copy of the three neutral probe templates used for glitch verification.

These templates are the stable reference set from glitcher/model.py.  They are
duplicated here so that the exact prompts used in an experiment can be recorded
alongside results without depending on a specific glitcher version.

Each function returns a dict describing the template.  The actual formatting
(with chat template wrappers) is handled by glitcher at runtime; this file
captures the semantic content only.

Usage:
    python APPENDIX-B/probe_templates.py --list     # print all templates
    python APPENDIX-B/probe_templates.py --json     # dump as JSON
"""

import argparse
import json
import sys

VERSION = "1.0"


def template_1(token: str = "{token}") -> dict:
    """Repeat-back probe: asks the model to echo the token exactly."""
    return {
        "id": "repeat_back",
        "version": VERSION,
        "system": (
            "You are a helpful assistant. When asked to repeat something, "
            "repeat it exactly, character for character."
        ),
        "user": f"Please respond with exactly this: '{token}'",
        "assistant_prefill": " ",
        "notes": "Llama 3/3.2 override: assistant_prefill = empty string",
    }


def template_2(token: str = "{token}") -> dict:
    """Meaning probe: asks the model what the token means, with a few-shot example."""
    return {
        "id": "meaning_query",
        "version": VERSION,
        "system": (
            "This helpful assistant explains what a phrase means, "
            "replying with identical responses to identical questions for reliability."
        ),
        "turns": [
            {"role": "user", "content": "What does 'edslfkj' mean?"},
            {"role": "assistant", "content": "'edslfkj' has no specific meaning I am aware of."},
            {"role": "user", "content": f"What does '{token}' mean?"},
        ],
        "assistant_prefill": "'",
        "notes": "Llama 3/3.2 override: assistant_prefill = empty string",
    }


def template_3(token: str = "{token}") -> dict:
    """Instruction-following probe: asks the model to repeat a string."""
    return {
        "id": "instruction_repeat",
        "version": VERSION,
        "system": "You are a helpful assistant that follows instructions carefully.",
        "user": f"Question: Please can you repeat the string '{token}'?",
        "assistant_prefill": " Of course! Here is the repeated string:'",
        "notes": "Llama 3/3.2 override: assistant_prefill = empty string",
    }


ALL_TEMPLATES = [template_1, template_2, template_3]


def get_all_templates(token: str = "{token}") -> list:
    return [t(token) for t in ALL_TEMPLATES]


def main():
    parser = argparse.ArgumentParser(description="Display frozen probe templates")
    parser.add_argument("--list", action="store_true", help="Print templates in readable form")
    parser.add_argument("--json", action="store_true", help="Dump templates as JSON")
    parser.add_argument("--token", default="{token}", help="Token placeholder text")
    args = parser.parse_args()

    templates = get_all_templates(args.token)

    if args.json:
        json.dump(templates, sys.stdout, indent=2)
        print()
    else:
        # Default to --list behavior
        for i, t in enumerate(templates, 1):
            print(f"--- Template {i}: {t['id']} (v{t['version']}) ---")
            print(f"  System: {t['system']}")
            if "user" in t:
                print(f"  User: {t['user']}")
            if "turns" in t:
                for turn in t["turns"]:
                    print(f"  {turn['role'].title()}: {turn['content']}")
            print(f"  Assistant prefill: {repr(t.get('assistant_prefill', ''))}")
            print(f"  Notes: {t.get('notes', '')}")
            print()


if __name__ == "__main__":
    main()
