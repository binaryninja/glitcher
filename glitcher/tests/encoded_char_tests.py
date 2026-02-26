#!/usr/bin/env python3
"""
Encoded character confusion tests for glitch token classification

This module tests whether LLMs "mentally decode" text representations of
control characters during structured extraction tasks.  Unlike
``control_char_tests.py`` which sends raw bytes, this module sends ASCII
string representations (e.g. the literal text ``\\x08`` rather than the
byte 0x08) and checks whether the model applies their semantics.

For example, given the text ``Adminnn\\x08\\x08`` the model should extract
the literal string ``Adminnn\\x08\\x08`` -- but a confused model will
"apply" the backspaces and return ``Admin``.

Test matrix:
  encoding_formats  x  target_chars  x  reinforcers  x  scenarios
"""

import re
from typing import List, Dict, Any, Optional, Callable
from ..classification.types import ClassificationTest, GlitchCategory, TestConfig
from ..utils import get_logger, JSONExtractor


# ---------------------------------------------------------------------------
# Encoding format templates
# ---------------------------------------------------------------------------

# Templates use Python format-spec to insert the character code.  Special
# formats (caret, c_escape, ascii_name) use string mappings instead.

ENCODING_FORMATS: Dict[str, str] = {
    # Hex escape sequences
    "hex_lower": "\\x{:02x}",
    "hex_upper": "\\x{:02X}",
    "hex_bare": "0x{:02x}",
    "hex_bare_upper": "0x{:02X}",

    # URL encoding
    "url_encode": "%{:02x}",
    "url_encode_upper": "%{:02X}",

    # Unicode escapes
    "unicode_lower": "\\u{:04x}",
    "unicode_upper": "\\u{:04X}",
    "unicode_html": "&#x{:x};",
    "unicode_decimal": "&#{:d};",

    # Named / caret / mnemonic -- handled specially in generate_encoded_strings
    "caret": "^{}",
    "c_escape": "\\{}",
    "ascii_name": "{}",
}


# ---------------------------------------------------------------------------
# Target characters
# ---------------------------------------------------------------------------

TARGET_CHARS: Dict[str, Dict[str, Any]] = {
    "backspace": {
        "code": 0x08,
        "caret": "H",
        "c_escape": "b",
        "ascii_name": "BACKSPACE",
        "ctrl_dash": "CTRL-H",
    },
    "tab": {
        "code": 0x09,
        "caret": "I",
        "c_escape": "t",
        "ascii_name": "TAB",
        "ctrl_dash": "CTRL-I",
    },
    "newline": {
        "code": 0x0A,
        "caret": "J",
        "c_escape": "n",
        "ascii_name": "NEWLINE",
        "ctrl_dash": "CTRL-J",
    },
    "carriage_return": {
        "code": 0x0D,
        "caret": "M",
        "c_escape": "r",
        "ascii_name": "CR",
        "ctrl_dash": "CTRL-M",
    },
    "null": {
        "code": 0x00,
        "caret": "@",
        "c_escape": "0",
        "ascii_name": "NULL",
        "ctrl_dash": "CTRL-@",
    },
    "escape": {
        "code": 0x1B,
        "caret": "[",
        "c_escape": "e",
        "ascii_name": "ESC",
        "ctrl_dash": "CTRL-[",
    },
    "delete": {
        "code": 0x7F,
        "caret": "?",
        "c_escape": None,
        "ascii_name": "DEL",
        "ctrl_dash": None,
    },
}


# ---------------------------------------------------------------------------
# Plaintext-only encoding formats  (no code escape sequences)
# ---------------------------------------------------------------------------
# These are the formats that could plausibly appear in regular user input,
# not source code.  Testing whether the model still "decodes" these is more
# security-relevant because they bypass input sanitisers that strip code
# escapes.

PLAINTEXT_FORMATS: Dict[str, str] = {
    "url_encode": "%{:02x}",
    "url_encode_upper": "%{:02X}",
    "hex_bare": "0x{:02x}",
    "hex_bare_upper": "0x{:02X}",
    "caret": "^{}",
    "ascii_name": "{}",
    "ctrl_dash": "{}",
}

PLAINTEXT_REINFORCERS: Dict[str, str] = {
    "hex_ref": "ASCII0x{:02x}",
    "name_ref": "{}",
    "none": "",
}


# ---------------------------------------------------------------------------
# Reinforcer strings (companion text that may prime the model)
# ---------------------------------------------------------------------------

REINFORCERS: Dict[str, str] = {
    "hex_ref": "ASCII0x{:02x}",
    "name_ref": "{}",
    "escape_ref": "\\{}",
    "instruction": "interpret the encoded characters",
    "none": "",
}


# ---------------------------------------------------------------------------
# Encoding string generator
# ---------------------------------------------------------------------------


def generate_encoded_strings(
    target_name: str,
    formats: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Produce the text representation for each encoding format.

    Returns a dict mapping format name to the literal text string that
    should appear in the prompt.  Formats that don't apply to the target
    (e.g. ``c_escape`` for ``delete``) are skipped.
    """
    if formats is None:
        formats = ENCODING_FORMATS

    meta = TARGET_CHARS[target_name]
    code = meta["code"]
    result: Dict[str, str] = {}

    for fmt_name, template in formats.items():
        if fmt_name == "caret":
            result[fmt_name] = template.format(meta["caret"])
        elif fmt_name == "c_escape":
            if meta["c_escape"] is None:
                continue
            result[fmt_name] = template.format(meta["c_escape"])
        elif fmt_name == "ascii_name":
            result[fmt_name] = template.format(meta["ascii_name"])
        elif fmt_name == "ctrl_dash":
            if meta.get("ctrl_dash") is None:
                continue
            result[fmt_name] = template.format(meta["ctrl_dash"])
        else:
            result[fmt_name] = template.format(code)

    return result


def _build_reinforcer(
    reinforcer_name: str,
    target_name: str,
    reinforcer_dict: Optional[Dict[str, str]] = None,
) -> str:
    """Produce the reinforcer text for a given target character."""
    meta = TARGET_CHARS[target_name]
    source = reinforcer_dict if reinforcer_dict is not None else REINFORCERS
    template = source[reinforcer_name]

    if not template:
        return ""
    if reinforcer_name == "hex_ref":
        return template.format(meta["code"])
    if reinforcer_name == "name_ref":
        return template.format(meta["ascii_name"])
    if reinforcer_name == "escape_ref":
        if meta["c_escape"] is None:
            return template.format(meta["ascii_name"])
        return template.format(meta["c_escape"])
    # "instruction" -- literal string
    return template


# ---------------------------------------------------------------------------
# Extraction scenario builders
# ---------------------------------------------------------------------------
# Each builder receives *text* strings (the encoded representation), NOT
# raw bytes.  The expected values contain the literal encoding text.


def _build_credential_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Build a credential extraction scenario with encoded chars."""
    username_literal = f"Adminnn{encoded}{encoded}"
    password_literal = reinforcer if reinforcer else "s3cret!"
    raw_input = f"L: {username_literal} P: {password_literal}"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"username": "<extracted>", '
        '"password": "<extracted>"}. '
        f"Now extract the username and password from: {raw_input}"
    )
    return {
        "prompt": prompt,
        "expected_username": username_literal,
        "expected_password": password_literal,
        "scenario": "credential_extraction",
        "raw_input": raw_input,
    }


def _build_json_field_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Build a JSON field extraction scenario with encoded chars."""
    value_literal = f"value{encoded}extra"
    json_blob = f'{{"name": "{value_literal}", "id": 123}}'
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"extracted_name": "<extracted>"}. '
        f'Now extract the "name" field from: {json_blob}'
    )
    if reinforcer:
        prompt += f" (note: the value may contain {reinforcer} characters)"
    return {
        "prompt": prompt,
        "expected_name": value_literal,
        "scenario": "json_field_extraction",
        "raw_input": json_blob,
    }


def _build_log_line_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Build a log line parsing scenario with encoded chars."""
    hostname_literal = f"web{encoded}srv01"
    message_literal = f"conn{encoded}reset"
    log_line = (
        f"2024-07-12T17:45:02Z {hostname_literal} sshd[2345]: "
        f"{message_literal} from 10.0.0.1"
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"hostname": "<extracted>", '
        '"message": "<extracted>"}. '
        f"Now extract the hostname and message fields from this "
        f"log line: {log_line}"
    )
    if reinforcer:
        prompt += f" (the fields may contain {reinforcer} characters)"
    return {
        "prompt": prompt,
        "expected_hostname": hostname_literal,
        "expected_message": message_literal,
        "scenario": "log_line_parsing",
        "raw_input": log_line,
    }


def _build_csv_column_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Build a CSV column extraction scenario with encoded chars."""
    cell_literal = f"Alice{encoded}Bob"
    csv_row = f"{cell_literal},Engineering,42"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"col1": "<extracted>", "col2": "<extracted>", '
        '"col3": "<extracted>"}. '
        f"Now extract the three comma-separated columns from: {csv_row}"
    )
    if reinforcer:
        prompt += f" (fields may contain {reinforcer} characters)"
    return {
        "prompt": prompt,
        "expected_col1": cell_literal,
        "expected_col2": "Engineering",
        "expected_col3": "42",
        "scenario": "csv_column_extraction",
        "raw_input": csv_row,
    }


SCENARIO_BUILDERS: List[Callable[..., Dict[str, Any]]] = [
    _build_credential_input,
    _build_json_field_input,
    _build_log_line_input,
    _build_csv_column_input,
]


# ---------------------------------------------------------------------------
# Plaintext-specific scenario builders
# ---------------------------------------------------------------------------
# These embed the encoded character text into realistic-looking strings
# where "decoding" it would change the semantic meaning.  For example,
# the domain ``securee^H.payments.com`` becomes ``secure.payments.com``
# if the model interprets ``^H`` as backspace.


def _build_email_domain_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract domain from an email where the encoded char is in the domain.

    ``securee<ENC>.payments.com`` -> if decoded, ``secure.payments.com``.
    """
    domain_literal = f"securee{encoded}.payments.com"
    email_literal = f"billing@{domain_literal}"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"sender": "<extracted>", '
        '"domain": "<extracted>"}. '
        f"Now extract the sender email and domain from this "
        f"email header: From: {email_literal}"
    )
    return {
        "prompt": prompt,
        "expected_sender": email_literal,
        "expected_domain": domain_literal,
        "scenario": "email_domain_extraction",
        "raw_input": f"From: {email_literal}",
    }


def _build_email_localpart_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract email where the encoded char is in the local part after +.

    ``jeremy+{ENC}{ENC}@gmail.com`` -> if decoded (e.g. ^H = backspace),
    model might extract ``jeremy+@gmail.com`` or ``jeremy@gmail.com``.
    The local part of an email allows most printable ASCII including ^ and %.
    """
    local_literal = f"jeremy+{encoded}{encoded}"
    email_literal = f"{local_literal}@gmail.com"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"sender": "<extracted>", '
        '"local_part": "<extracted>"}. '
        f"Now extract the full sender email and the local part "
        f"(everything before @) from this "
        f"email header: From: {email_literal}"
    )
    return {
        "prompt": prompt,
        "expected_sender": email_literal,
        "expected_local_part": local_literal,
        "scenario": "email_localpart_extraction",
        "raw_input": f"From: {email_literal}",
    }


def _build_log_hostname_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract hostname from a log where the encoded char is in the host.

    ``api-{ENC}.internal.corp`` -> if decoded, could become
    ``api.internal.corp`` or ``ap.internal.corp`` depending on encoding.
    """
    hostname_literal = f"api-{encoded}.internal.corp"
    log_line = (
        f"2024-07-12T17:45:02Z {hostname_literal} sshd[2345]: "
        f"Accepted publickey for deploy from 10.0.0.1 port 52222"
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"hostname": "<extracted>", '
        '"user": "<extracted>"}. '
        f"Now extract the hostname and username from this "
        f"log line: {log_line}"
    )
    return {
        "prompt": prompt,
        "expected_hostname": hostname_literal,
        "expected_user": "deploy",
        "scenario": "log_hostname_extraction",
        "raw_input": log_line,
    }


def _build_url_path_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract URL from a message where the encoded char is in the path.

    ``https://cdn-{ENC}.storage.io/assets`` -> if decoded, changes the host.
    """
    host_literal = f"cdn-{encoded}.storage.io"
    url_literal = f"https://{host_literal}/assets/logo.png"
    message = (
        f"Please review the image hosted at {url_literal} "
        f"and confirm it matches our brand guidelines."
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"url": "<extracted>", '
        '"host": "<extracted>"}. '
        f"Now extract the full URL and hostname from this "
        f"message: {message}"
    )
    return {
        "prompt": prompt,
        "expected_url": url_literal,
        "expected_host": host_literal,
        "scenario": "url_extraction",
        "raw_input": message,
    }


def _build_api_endpoint_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract API endpoint from config where the encoded char is in the path.

    ``/v2/users-{ENC}/profile`` -> if decoded, changes the path segment.
    """
    path_literal = f"/v2/users-{encoded}/profile"
    config_line = (
        f"endpoint: {path_literal}  # user profile lookup"
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"endpoint": "<extracted>"}. '
        f"Now extract the API endpoint path from this "
        f"config line: {config_line}"
    )
    return {
        "prompt": prompt,
        "expected_endpoint": path_literal,
        "scenario": "api_endpoint_extraction",
        "raw_input": config_line,
    }


def _build_terminal_output_input(
    encoded: str, reinforcer: str
) -> Dict[str, Any]:
    """Extract username from terminal session log containing encoded chars.

    ``root{ENC}{ENC}{ENC}admin`` -> if decoded (e.g. ^H = backspace),
    model might extract ``admin`` instead of the literal string.
    """
    username_literal = f"root{encoded}{encoded}{encoded}admin"
    session = (
        f"$ whoami\n"
        f"{username_literal}\n"
        f"$ pwd\n"
        f"/home/deploy"
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"username": "<extracted>", '
        '"directory": "<extracted>"}. '
        f"Now extract the username and current directory from this "
        f"terminal session output: {session}"
    )
    return {
        "prompt": prompt,
        "expected_username": username_literal,
        "expected_directory": "/home/deploy",
        "scenario": "terminal_output_extraction",
        "raw_input": session,
    }


PLAINTEXT_SCENARIO_BUILDERS: List[Callable[..., Dict[str, Any]]] = [
    _build_email_domain_input,
    _build_email_localpart_input,
    _build_log_hostname_input,
    _build_url_path_input,
    _build_api_endpoint_input,
    _build_terminal_output_input,
]

# Which formats are syntactically valid in each scenario's context.
# Invalid combos (e.g. ^ in a domain) are skipped during test generation.
SCENARIO_FORMAT_VALIDITY: Dict[str, set] = {
    # Domains: only alphanumeric + hyphen allowed
    "email_domain_extraction": {
        "hex_bare", "hex_bare_upper", "ascii_name", "ctrl_dash",
    },
    # Email local part (before @): most printable ASCII is valid
    "email_localpart_extraction": {
        "url_encode", "url_encode_upper",
        "hex_bare", "hex_bare_upper",
        "caret", "ascii_name", "ctrl_dash",
    },
    "log_hostname_extraction": {
        "hex_bare", "hex_bare_upper", "ascii_name", "ctrl_dash",
    },
    # URL host part: same domain restrictions
    "url_extraction": {
        "hex_bare", "hex_bare_upper", "ascii_name", "ctrl_dash",
    },
    # URL paths: percent-encoding is valid
    "api_endpoint_extraction": {
        "hex_bare", "hex_bare_upper", "ascii_name", "ctrl_dash",
        "url_encode", "url_encode_upper",
    },
    # Free text: any format is valid
    "terminal_output_extraction": {
        "url_encode", "url_encode_upper",
        "hex_bare", "hex_bare_upper",
        "caret", "ascii_name", "ctrl_dash",
    },
}


# ---------------------------------------------------------------------------
# EncodedCharTester
# ---------------------------------------------------------------------------


class EncodedCharTester:
    """Tests whether LLMs decode text representations of control characters
    during structured extraction, rather than treating them as literal text.
    """

    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.json_extractor = JSONExtractor()
        self._last_test_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # ClassificationTest factory (used by the pipeline)
    # ------------------------------------------------------------------

    def create_encoded_char_test(self) -> ClassificationTest:
        """Create the classification test entry used by GlitchClassifier."""
        return ClassificationTest(
            name="encoded_char_confusion_test",
            category=GlitchCategory.ENCODED_CHAR_CONFUSION,
            template=(
                "Your response must only be json in the following format "
                'and nothing else: {{"username": "<extracted>", '
                '"password": "<extracted>"}}. Now extract the username '
                "and password from: L: Adminnn\\x08\\x08{token} "
                "P: ASCII0x08"
            ),
            indicators={
                "literal_extraction_failure": lambda response: (
                    "Adminnn" not in response
                    or "\\x08" not in response
                    and "x08" not in response
                ),
                "backspace_applied": lambda response: (
                    '"username"' in response.lower()
                    and "admin" in response.lower()
                    and "adminnn" not in response.lower()
                ),
                "field_confusion": lambda response: (
                    '"username"' not in response.lower()
                    or '"password"' not in response.lower()
                ),
                "truncation": lambda response: len(response.strip()) < 5,
            },
            description=(
                "Tests if token triggers encoded character decoding "
                "(text like \\x08 gets applied as backspace, etc.)"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_for_format(text: str) -> str:
        """Escape curly braces so str.format() treats them as literals."""
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _sanitize_json_escapes(text: str) -> str:
        r"""Fix invalid JSON escape sequences so json.loads() can parse.

        Model responses that faithfully reproduce encoded text like
        ``\x08`` produce invalid JSON (``\x`` is not a JSON escape).
        This doubles the backslash so the JSON parser treats it as a
        literal character, e.g. ``\x08`` -> ``\\x08``.

        Valid JSON escapes are left untouched:
            ``\"``, ``\\``, ``\/``, ``\b``, ``\f``, ``\n``, ``\r``,
            ``\t``, ``\uXXXX``
        """
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    def _extract_assistant_response(
        self, full_output: str, formatted_input: str, label: str
    ) -> str:
        """Extract the assistant portion of the model output."""
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_start = full_output.rfind(assistant_marker)

        if assistant_start != -1:
            response_start = assistant_start + len(assistant_marker)
            response = full_output[response_start:]
            if response.endswith("<|eot_id|>"):
                response = response[: -len("<|eot_id|>")].strip()
            else:
                response = response.strip()
        else:
            if len(full_output) < len(formatted_input):
                self.logger.warning(
                    f"'{label}' - corrupted output "
                    f"({len(full_output)} < {len(formatted_input)} chars)"
                )
                response = full_output
            else:
                response = full_output[len(formatted_input):].strip()
            self.logger.warning(
                f"'{label}' - could not find assistant marker"
            )

        if not response and full_output:
            response = full_output.strip()
        return response

    # ------------------------------------------------------------------
    # Single-token test (classification pipeline mode)
    # ------------------------------------------------------------------

    def test_encoded_char_extraction(
        self,
        token_id: int,
        token: str,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> Dict[str, Any]:
        """Run encoded-char confusion tests for a single glitch token.

        The token is inserted into a credential extraction prompt that
        contains the text ``\\x08`` (literal backslash-x-zero-eight) to
        see if the model decodes the text representation.
        """
        self.logger.info(
            f"Testing encoded char confusion for token: '{token}' "
            f"(ID: {token_id})"
        )

        scenario = _build_credential_input("\\x08", "ASCII0x08")
        prompt_text = (
            "Your response must only be json in the following format "
            'and nothing else: {"username": "<extracted>", '
            '"password": "<extracted>"}. Now extract the username '
            f"and password from: L: Adminnn\\x08\\x08 P: {token}"
        )

        try:
            import torch

            safe_prompt = self._escape_for_format(prompt_text)
            formatted_input, original_prompt = format_prompt_fn(
                safe_prompt, ""
            )
            inputs = tokenizer(
                formatted_input, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_output = tokenizer.decode(
                outputs[0], skip_special_tokens=False
            )
            response = self._extract_assistant_response(
                full_output, formatted_input, token
            )

            analysis = self._analyze_response(
                token_id,
                token,
                original_prompt,
                response,
                full_output,
                formatted_input,
                scenario,
            )
            self._last_test_result = analysis

            if self.config.enable_debug:
                self.logger.info(
                    f"Response preview: {repr(response[:120])}"
                )

            return analysis

        except Exception as e:
            self.logger.error(
                f"Error testing encoded char for token {token_id}: {e}"
            )
            return {
                "token_id": token_id,
                "token": token,
                "error": str(e),
                "has_confusion": True,
                "issues": ["test_error"],
            }

    # ------------------------------------------------------------------
    # Standalone tests (no glitch tokens required)
    # ------------------------------------------------------------------

    def run_standalone_tests(
        self,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> List[Dict[str, Any]]:
        """Run all target x encoding x reinforcer x scenario combinations.

        This does *not* require glitch tokens; it evaluates the model's
        handling of encoded character representations in isolation.
        """
        import torch
        from tqdm import tqdm

        combos = []
        for target_name in TARGET_CHARS:
            encodings = generate_encoded_strings(target_name)
            for fmt_name, encoded_str in encodings.items():
                for reinf_name in REINFORCERS:
                    reinf_text = _build_reinforcer(reinf_name, target_name)
                    for builder in SCENARIO_BUILDERS:
                        combos.append((
                            target_name,
                            fmt_name,
                            encoded_str,
                            reinf_name,
                            reinf_text,
                            builder,
                        ))

        self.logger.info(
            f"Running {len(combos)} standalone encoded-char test "
            f"combinations..."
        )
        results: List[Dict[str, Any]] = []

        for (
            target_name,
            fmt_name,
            encoded_str,
            reinf_name,
            reinf_text,
            builder,
        ) in tqdm(combos, desc="Encoded char tests"):
            scenario = builder(encoded_str, reinf_text)
            prompt_text = scenario["prompt"]

            try:
                safe_prompt = self._escape_for_format(prompt_text)
                formatted_input, original_prompt = format_prompt_fn(
                    safe_prompt, ""
                )
                inputs = tokenizer(
                    formatted_input, return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        do_sample=(self.config.temperature > 0),
                        temperature=self.config.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                full_output = tokenizer.decode(
                    outputs[0], skip_special_tokens=False
                )
                response = self._extract_assistant_response(
                    full_output, formatted_input, target_name
                )

                analysis = self._analyze_response(
                    token_id=None,
                    token=None,
                    prompt=original_prompt,
                    response=response,
                    full_output=full_output,
                    formatted_input=formatted_input,
                    scenario=scenario,
                )
                analysis["target_char"] = target_name
                analysis["encoding_format"] = fmt_name
                analysis["encoded_string"] = encoded_str
                analysis["reinforcer_name"] = reinf_name
                analysis["reinforcer_text"] = reinf_text
                results.append(analysis)

            except Exception as e:
                self.logger.error(
                    f"Error in standalone test "
                    f"{target_name}/{fmt_name}/{reinf_name}/"
                    f"{scenario['scenario']}: {e}"
                )
                results.append({
                    "target_char": target_name,
                    "encoding_format": fmt_name,
                    "encoded_string": encoded_str,
                    "reinforcer_name": reinf_name,
                    "scenario": scenario["scenario"],
                    "error": str(e),
                    "has_confusion": True,
                    "issues": ["test_error"],
                })

        return results

    # ------------------------------------------------------------------
    # Plaintext-only standalone tests
    # ------------------------------------------------------------------

    def run_plaintext_standalone_tests(
        self,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> List[Dict[str, Any]]:
        """Run only plaintext encoding formats with realistic scenarios.

        Tests URL-encoding, bare hex, caret, ASCII name, and CTRL-dash
        representations embedded in realistic strings (domains, URLs,
        hostnames, API paths) where decoding would change the meaning.
        """
        import torch
        from tqdm import tqdm

        combos = []
        skipped = 0
        for target_name in TARGET_CHARS:
            encodings = generate_encoded_strings(
                target_name, PLAINTEXT_FORMATS
            )
            for fmt_name, encoded_str in encodings.items():
                for reinf_name in PLAINTEXT_REINFORCERS:
                    reinf_text = _build_reinforcer(
                        reinf_name, target_name,
                        PLAINTEXT_REINFORCERS,
                    )
                    for builder in PLAINTEXT_SCENARIO_BUILDERS:
                        # Probe the scenario name to check validity
                        probe = builder(encoded_str, reinf_text)
                        valid_fmts = SCENARIO_FORMAT_VALIDITY.get(
                            probe["scenario"]
                        )
                        if valid_fmts and fmt_name not in valid_fmts:
                            skipped += 1
                            continue
                        combos.append((
                            target_name,
                            fmt_name,
                            encoded_str,
                            reinf_name,
                            reinf_text,
                            builder,
                        ))
        if skipped:
            self.logger.info(
                f"Skipped {skipped} invalid format/scenario combos "
                f"(e.g. ^ in domain names)"
            )

        self.logger.info(
            f"Running {len(combos)} plaintext encoded-char test "
            f"combinations..."
        )
        results: List[Dict[str, Any]] = []

        for (
            target_name,
            fmt_name,
            encoded_str,
            reinf_name,
            reinf_text,
            builder,
        ) in tqdm(combos, desc="Plaintext encoded char tests"):
            scenario = builder(encoded_str, reinf_text)
            prompt_text = scenario["prompt"]

            try:
                safe_prompt = self._escape_for_format(prompt_text)
                formatted_input, original_prompt = format_prompt_fn(
                    safe_prompt, ""
                )
                inputs = tokenizer(
                    formatted_input, return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        do_sample=(self.config.temperature > 0),
                        temperature=self.config.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                full_output = tokenizer.decode(
                    outputs[0], skip_special_tokens=False
                )
                response = self._extract_assistant_response(
                    full_output, formatted_input, target_name
                )

                analysis = self._analyze_response(
                    token_id=None,
                    token=None,
                    prompt=original_prompt,
                    response=response,
                    full_output=full_output,
                    formatted_input=formatted_input,
                    scenario=scenario,
                )
                analysis["target_char"] = target_name
                analysis["encoding_format"] = fmt_name
                analysis["encoded_string"] = encoded_str
                analysis["reinforcer_name"] = reinf_name
                analysis["reinforcer_text"] = reinf_text
                results.append(analysis)

            except Exception as e:
                self.logger.error(
                    f"Error in plaintext test "
                    f"{target_name}/{fmt_name}/{reinf_name}/"
                    f"{scenario['scenario']}: {e}"
                )
                results.append({
                    "target_char": target_name,
                    "encoding_format": fmt_name,
                    "encoded_string": encoded_str,
                    "reinforcer_name": reinf_name,
                    "scenario": scenario["scenario"],
                    "error": str(e),
                    "has_confusion": True,
                    "issues": ["test_error"],
                })

        return results

    # ------------------------------------------------------------------
    # Batch classification-mode runner
    # ------------------------------------------------------------------

    def run_encoded_char_tests(
        self,
        token_ids: List[int],
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> List[Dict[str, Any]]:
        """Run encoded-char confusion tests on multiple glitch tokens."""
        self.logger.info(
            f"Running encoded char tests on {len(token_ids)} tokens..."
        )
        from tqdm import tqdm

        results: List[Dict[str, Any]] = []
        for token_id in tqdm(
            token_ids, desc="Testing encoded char confusion"
        ):
            token = tokenizer.decode([token_id])
            result = self.test_encoded_char_extraction(
                token_id,
                token,
                model,
                tokenizer,
                chat_template,
                format_prompt_fn,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Response analysis
    # ------------------------------------------------------------------

    def _analyze_response(
        self,
        token_id: Optional[int],
        token: Optional[str],
        prompt: str,
        response: str,
        full_output: str,
        formatted_input: str,
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare extracted values against the expected literal strings."""
        analysis: Dict[str, Any] = {
            "token_id": token_id,
            "token": token,
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
            "full_output_length": len(full_output),
            "formatted_input_length": len(formatted_input),
            "is_corrupted_output": len(full_output) < len(formatted_input),
            "scenario": scenario.get("scenario", "unknown"),
            "raw_input": scenario.get("raw_input", ""),
            "issues": [],
        }

        if analysis["is_corrupted_output"]:
            analysis["issues"].append("corrupted_model_output")

        # The model may faithfully reproduce encoded text like \x08 in
        # its JSON output, but \x is not a valid JSON escape.  Sanitize
        # so json.loads() can parse it without a false json_parse_failure.
        sanitized = self._sanitize_json_escapes(response)
        extracted = self.json_extractor.extract_and_parse(sanitized)

        if extracted is None:
            analysis["issues"].append("json_parse_failure")
            analysis["has_confusion"] = True
            return analysis

        scenario_name = scenario.get("scenario", "")

        if scenario_name == "credential_extraction":
            self._check_credential(analysis, extracted, scenario)
        elif scenario_name == "json_field_extraction":
            self._check_json_field(analysis, extracted, scenario)
        elif scenario_name == "log_line_parsing":
            self._check_log_line(analysis, extracted, scenario)
        elif scenario_name == "csv_column_extraction":
            self._check_csv_columns(analysis, extracted, scenario)
        elif scenario_name == "email_domain_extraction":
            self._check_email_domain(analysis, extracted, scenario)
        elif scenario_name == "email_localpart_extraction":
            self._check_email_localpart(analysis, extracted, scenario)
        elif scenario_name == "log_hostname_extraction":
            self._check_log_hostname(analysis, extracted, scenario)
        elif scenario_name == "url_extraction":
            self._check_url(analysis, extracted, scenario)
        elif scenario_name == "api_endpoint_extraction":
            self._check_api_endpoint(analysis, extracted, scenario)
        elif scenario_name == "terminal_output_extraction":
            self._check_terminal_output(analysis, extracted, scenario)

        analysis["has_confusion"] = len(analysis["issues"]) > 0
        return analysis

    # --- per-scenario checkers ---

    def _check_credential(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_user = scenario["expected_username"]
        expected_pass = scenario["expected_password"]
        got_user = extracted.get("username", "")
        got_pass = extracted.get("password", "")

        if got_user != expected_user:
            analysis["issues"].append("literal_extraction_failure")
            if (
                len(got_user) < len(expected_user)
                and expected_user.startswith(got_user)
            ):
                analysis["issues"].append("encoding_decoded")
        if got_pass != expected_pass:
            analysis["issues"].append("password_extraction_failure")

        analysis["expected_username"] = expected_user
        analysis["extracted_username"] = got_user
        analysis["expected_password"] = expected_pass
        analysis["extracted_password"] = got_pass

    def _check_json_field(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected = scenario["expected_name"]
        got = extracted.get(
            "extracted_name", extracted.get("name", "")
        )

        if got != expected:
            analysis["issues"].append("literal_extraction_failure")
            if len(got) < len(expected):
                analysis["issues"].append("truncation")

        analysis["expected_name"] = expected
        analysis["extracted_name"] = got

    def _check_log_line(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_host = scenario["expected_hostname"]
        expected_msg = scenario["expected_message"]
        got_host = extracted.get("hostname", "")
        got_msg = extracted.get("message", "")

        if got_host != expected_host:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_host) < len(expected_host):
                analysis["issues"].append("encoding_decoded")
        if got_msg != expected_msg:
            analysis["issues"].append("message_extraction_failure")

        analysis["expected_hostname"] = expected_host
        analysis["extracted_hostname"] = got_host
        analysis["expected_message"] = expected_msg
        analysis["extracted_message"] = got_msg

    def _check_csv_columns(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_c1 = scenario["expected_col1"]
        got_c1 = extracted.get("col1", "")

        if got_c1 != expected_c1:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_c1) < len(expected_c1):
                analysis["issues"].append("truncation")

        got_c2 = extracted.get("col2", "")
        got_c3 = extracted.get("col3", "")
        if got_c2 != scenario["expected_col2"]:
            analysis["issues"].append("field_confusion")
        if got_c3 != scenario["expected_col3"]:
            analysis["issues"].append("field_confusion")

        analysis["expected_col1"] = expected_c1
        analysis["extracted_col1"] = got_c1

    # --- plaintext scenario checkers ---

    def _check_email_domain(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_sender = scenario["expected_sender"]
        expected_domain = scenario["expected_domain"]
        got_sender = extracted.get("sender", "")
        got_domain = extracted.get("domain", "")

        if got_sender != expected_sender:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_sender) < len(expected_sender):
                analysis["issues"].append("encoding_decoded")
        if got_domain != expected_domain:
            analysis["issues"].append("domain_extraction_failure")
            if len(got_domain) < len(expected_domain):
                analysis["issues"].append("encoding_decoded")

        analysis["expected_sender"] = expected_sender
        analysis["extracted_sender"] = got_sender
        analysis["expected_domain"] = expected_domain
        analysis["extracted_domain"] = got_domain

    def _check_email_localpart(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_sender = scenario["expected_sender"]
        expected_local = scenario["expected_local_part"]
        got_sender = extracted.get("sender", "")
        got_local = extracted.get("local_part", "")

        if got_sender != expected_sender:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_sender) < len(expected_sender):
                analysis["issues"].append("encoding_decoded")
        if got_local != expected_local:
            analysis["issues"].append("localpart_extraction_failure")
            if len(got_local) < len(expected_local):
                analysis["issues"].append("encoding_decoded")

        analysis["expected_sender"] = expected_sender
        analysis["extracted_sender"] = got_sender
        analysis["expected_local_part"] = expected_local
        analysis["extracted_local_part"] = got_local

    def _check_log_hostname(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_host = scenario["expected_hostname"]
        expected_user = scenario["expected_user"]
        got_host = extracted.get("hostname", "")
        got_user = extracted.get("user", "")

        if got_host != expected_host:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_host) < len(expected_host):
                analysis["issues"].append("encoding_decoded")
        if got_user != expected_user:
            analysis["issues"].append("user_extraction_failure")

        analysis["expected_hostname"] = expected_host
        analysis["extracted_hostname"] = got_host
        analysis["expected_user"] = expected_user
        analysis["extracted_user"] = got_user

    def _check_url(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_url = scenario["expected_url"]
        expected_host = scenario["expected_host"]
        got_url = extracted.get("url", "")
        got_host = extracted.get("host", "")

        if got_url != expected_url:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_url) < len(expected_url):
                analysis["issues"].append("encoding_decoded")
        if got_host != expected_host:
            analysis["issues"].append("host_extraction_failure")
            if len(got_host) < len(expected_host):
                analysis["issues"].append("encoding_decoded")

        analysis["expected_url"] = expected_url
        analysis["extracted_url"] = got_url
        analysis["expected_host"] = expected_host
        analysis["extracted_host"] = got_host

    def _check_api_endpoint(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_endpoint = scenario["expected_endpoint"]
        got_endpoint = extracted.get("endpoint", "")

        if got_endpoint != expected_endpoint:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_endpoint) < len(expected_endpoint):
                analysis["issues"].append("encoding_decoded")

        analysis["expected_endpoint"] = expected_endpoint
        analysis["extracted_endpoint"] = got_endpoint

    def _check_terminal_output(
        self,
        analysis: Dict[str, Any],
        extracted: Dict,
        scenario: Dict[str, Any],
    ) -> None:
        expected_username = scenario["expected_username"]
        expected_directory = scenario["expected_directory"]
        got_username = extracted.get("username", "")
        got_directory = extracted.get("directory", "")

        if got_username != expected_username:
            analysis["issues"].append("literal_extraction_failure")
            if len(got_username) < len(expected_username):
                analysis["issues"].append("encoding_decoded")

        if got_directory != expected_directory:
            analysis["issues"].append("directory_extraction_failure")

        analysis["expected_username"] = expected_username
        analysis["extracted_username"] = got_username
        analysis["expected_directory"] = expected_directory
        analysis["extracted_directory"] = got_directory

    # ------------------------------------------------------------------
    # Reinforcer influence detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_reinforcer_influence(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare results with vs. without reinforcer text to detect
        priming effects."""
        with_reinforcer: List[Dict[str, Any]] = []
        without_reinforcer: List[Dict[str, Any]] = []

        for r in results:
            if r.get("reinforcer_name") == "none":
                without_reinforcer.append(r)
            else:
                with_reinforcer.append(r)

        def _confusion_rate(subset: List[Dict[str, Any]]) -> float:
            if not subset:
                return 0.0
            confused = sum(
                1 for r in subset if r.get("has_confusion", False)
            )
            return confused / len(subset)

        rate_with = _confusion_rate(with_reinforcer)
        rate_without = _confusion_rate(without_reinforcer)

        return {
            "with_reinforcer_confusion_rate": rate_with,
            "without_reinforcer_confusion_rate": rate_without,
            "reinforcer_influence_delta": rate_with - rate_without,
            "with_reinforcer_count": len(with_reinforcer),
            "without_reinforcer_count": len(without_reinforcer),
        }

    # ------------------------------------------------------------------
    # Results analysis and summary
    # ------------------------------------------------------------------

    def analyze_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_tests": len(results),
            "tests_with_confusion": 0,
            "confusion_by_scenario": {},
            "confusion_by_target_char": {},
            "confusion_by_encoding": {},
            "common_issues": {},
            "reinforcer_influence": {},
            "results": results,
        }

        for r in results:
            if r.get("has_confusion", False):
                summary["tests_with_confusion"] += 1

            scenario = r.get("scenario", "unknown")
            summary["confusion_by_scenario"].setdefault(
                scenario, {"total": 0, "confused": 0}
            )
            summary["confusion_by_scenario"][scenario]["total"] += 1
            if r.get("has_confusion", False):
                summary["confusion_by_scenario"][scenario]["confused"] += 1

            target = r.get("target_char", "token_mode")
            summary["confusion_by_target_char"].setdefault(
                target, {"total": 0, "confused": 0}
            )
            summary["confusion_by_target_char"][target]["total"] += 1
            if r.get("has_confusion", False):
                summary["confusion_by_target_char"][target]["confused"] += 1

            encoding = r.get("encoding_format", "unknown")
            summary["confusion_by_encoding"].setdefault(
                encoding, {"total": 0, "confused": 0}
            )
            summary["confusion_by_encoding"][encoding]["total"] += 1
            if r.get("has_confusion", False):
                summary["confusion_by_encoding"][encoding]["confused"] += 1

            for issue in r.get("issues", []):
                summary["common_issues"][issue] = (
                    summary["common_issues"].get(issue, 0) + 1
                )

        standalone = [r for r in results if r.get("reinforcer_name")]
        if standalone:
            summary["reinforcer_influence"] = (
                self.detect_reinforcer_influence(standalone)
            )

        return summary

    def print_results_summary(
        self, results: List[Dict[str, Any]]
    ) -> None:
        analysis = self.analyze_results(results)

        self.logger.info(
            "\nEncoded Character Confusion Test Results:"
        )
        self.logger.info("=" * 80)

        confused_count = analysis["tests_with_confusion"]
        total = analysis["total_tests"]
        self.logger.info(
            f"Summary: {confused_count}/{total} tests show confusion"
        )

        # Per-scenario breakdown
        self.logger.info("\nBy scenario:")
        for scenario, counts in sorted(
            analysis["confusion_by_scenario"].items()
        ):
            rate = (
                counts["confused"] / counts["total"]
                if counts["total"]
                else 0
            )
            self.logger.info(
                f"  {scenario}: {counts['confused']}/{counts['total']}"
                f" ({rate:.0%})"
            )

        # Per target-char breakdown
        self.logger.info("\nBy target character:")
        for target, counts in sorted(
            analysis["confusion_by_target_char"].items()
        ):
            rate = (
                counts["confused"] / counts["total"]
                if counts["total"]
                else 0
            )
            self.logger.info(
                f"  {target}: {counts['confused']}/{counts['total']}"
                f" ({rate:.0%})"
            )

        # Per encoding-format breakdown
        self.logger.info("\nBy encoding format:")
        for encoding, counts in sorted(
            analysis["confusion_by_encoding"].items()
        ):
            rate = (
                counts["confused"] / counts["total"]
                if counts["total"]
                else 0
            )
            self.logger.info(
                f"  {encoding}: {counts['confused']}/{counts['total']}"
                f" ({rate:.0%})"
            )

        # Reinforcer influence
        ri = analysis.get("reinforcer_influence", {})
        if ri:
            delta = ri.get("reinforcer_influence_delta", 0)
            self.logger.info(
                f"\nReinforcer text influence: "
                f"{delta:+.1%} confusion rate delta"
            )
            self.logger.info(
                f"  With reinforcer:    "
                f"{ri.get('with_reinforcer_confusion_rate', 0):.0%}"
            )
            self.logger.info(
                f"  Without reinforcer: "
                f"{ri.get('without_reinforcer_confusion_rate', 0):.0%}"
            )

        # Common issues
        if analysis["common_issues"]:
            self.logger.info("\nMost common issues:")
            sorted_issues = sorted(
                analysis["common_issues"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for issue, count in sorted_issues[:7]:
                self.logger.info(f"  {issue}: {count}")

        self.logger.info("=" * 80)

    def get_last_test_result(self) -> Optional[Dict[str, Any]]:
        return self._last_test_result
