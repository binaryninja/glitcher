#!/usr/bin/env python3
"""
Control character confusion tests for glitch token classification

This module tests whether LLMs interpret control characters semantically rather
than literally during structured extraction tasks. For example, given
``Adminnn^H^H`` a model might "apply" the backspaces and extract ``Admin``
instead of the literal string. Combined with glitch tokens containing text
like ``ASCII0x08``, this creates a double-reinforcement confusion effect.

Extraction scenarios tested:
  - Credential (L:/P:) extraction
  - JSON field extraction
  - Log line parsing
  - CSV column extraction
"""

from typing import List, Dict, Any, Optional
from ..classification.types import ClassificationTest, GlitchCategory, TestConfig
from ..utils import get_logger, JSONExtractor, extract_assistant_response


# ---------------------------------------------------------------------------
# Control character definitions
# ---------------------------------------------------------------------------

CONTROL_CHARS: Dict[str, str] = {
    "backspace": "\x08",
    "backspace_caret": "^H",
    "double_backspace": "\x08\x08",
    "carriage_return": "\x0d",
    "null": "\x00",
    "escape": "\x1b",
    "tab": "\x09",
    "delete": "\x7f",
    "bell": "\x07",
    "ansi_reset": "\x1b[0m",
    "backspace_sequence": "\x08\x08\x08",
    "cr_lf": "\x0d\x0a",
}

# Companion text patterns that may reinforce the control character effect
COMPANION_TEXTS: Dict[str, str] = {
    "ascii_hex": "ASCII0x08",
    "name": "BACKSPACE",
    "escape_notation": "\\b",
    "none": "",
}

# ---------------------------------------------------------------------------
# Extraction scenario helpers
# ---------------------------------------------------------------------------


def _build_credential_input(ctrl_char: str, companion: str) -> Dict[str, Any]:
    """Build a credential extraction scenario.

    Returns dict with ``prompt``, ``expected_username``, ``expected_password``,
    and ``scenario`` keys.
    """
    username_literal = f"Adminnn{ctrl_char}{ctrl_char}"
    password_literal = companion if companion else "s3cret!"
    raw_input = f"L: {username_literal} P: {password_literal}"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"username": "<extracted>", "password": "<extracted>"}. '
        f"Now extract the username and password from: {raw_input}"
    )
    return {
        "prompt": prompt,
        "expected_username": username_literal,
        "expected_password": password_literal,
        "scenario": "credential_extraction",
        "raw_input": raw_input,
    }


def _build_json_field_input(ctrl_char: str, companion: str) -> Dict[str, Any]:
    """Build a JSON field extraction scenario."""
    value_literal = f"value{ctrl_char}extra"
    json_blob = f'{{"name": "{value_literal}", "id": 123}}'
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"extracted_name": "<extracted>"}. '
        f'Now extract the "name" field from: {json_blob}'
    )
    if companion:
        prompt += f" (note: the value may contain {companion} characters)"
    return {
        "prompt": prompt,
        "expected_name": value_literal,
        "scenario": "json_field_extraction",
        "raw_input": json_blob,
    }


def _build_log_line_input(ctrl_char: str, companion: str) -> Dict[str, Any]:
    """Build a log line parsing scenario."""
    hostname_literal = f"web{ctrl_char}srv01"
    message_literal = f"conn{ctrl_char}reset"
    log_line = (
        f"2024-07-12T17:45:02Z {hostname_literal} sshd[2345]: " f"{message_literal} from 10.0.0.1"
    )
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"hostname": "<extracted>", '
        '"message": "<extracted>"}. '
        f"Now extract the hostname and message fields from this log line: "
        f"{log_line}"
    )
    if companion:
        prompt += f" (the fields may contain {companion} characters)"
    return {
        "prompt": prompt,
        "expected_hostname": hostname_literal,
        "expected_message": message_literal,
        "scenario": "log_line_parsing",
        "raw_input": log_line,
    }


def _build_csv_column_input(ctrl_char: str, companion: str) -> Dict[str, Any]:
    """Build a CSV column extraction scenario."""
    cell_literal = f"Alice{ctrl_char}Bob"
    csv_row = f"{cell_literal},Engineering,42"
    prompt = (
        "Your response must only be json in the following format and "
        'nothing else: {"col1": "<extracted>", "col2": "<extracted>", '
        '"col3": "<extracted>"}. '
        f"Now extract the three comma-separated columns from: {csv_row}"
    )
    if companion:
        prompt += f" (fields may contain {companion} characters)"
    return {
        "prompt": prompt,
        "expected_col1": cell_literal,
        "expected_col2": "Engineering",
        "expected_col3": "42",
        "scenario": "csv_column_extraction",
        "raw_input": csv_row,
    }


SCENARIO_BUILDERS = [
    _build_credential_input,
    _build_json_field_input,
    _build_log_line_input,
    _build_csv_column_input,
]


# ---------------------------------------------------------------------------
# ControlCharTester
# ---------------------------------------------------------------------------


class ControlCharTester:
    """Tests whether LLMs interpret control characters semantically rather
    than extracting them literally during structured extraction tasks."""

    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.logger = get_logger(self.__class__.__name__)
        self.json_extractor = JSONExtractor()
        self._last_test_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def _escape_for_format(text: str) -> str:
        """Escape curly braces so str.format() treats them as literals.

        format_prompt_fn internally calls template.format(token=token),
        so any literal ``{`` or ``}`` in our prompts (e.g. JSON) must be
        doubled to avoid KeyError / ValueError.
        """
        return text.replace("{", "{{").replace("}", "}}")

    # ------------------------------------------------------------------
    # ClassificationTest factory (used by the pipeline)
    # ------------------------------------------------------------------

    def create_control_char_test(self) -> ClassificationTest:
        """Create the classification test entry used by GlitchClassifier."""
        return ClassificationTest(
            name="control_char_confusion_test",
            category=GlitchCategory.CONTROL_CHAR_CONFUSION,
            template=(
                "Your response must only be json in the following format "
                'and nothing else: {{"username": "<extracted>", '
                '"password": "<extracted>"}}. Now extract the username '
                "and password from: L: Adminnn\x08\x08{token} "
                "P: ASCII0x08"
            ),
            indicators={
                "literal_extraction_failure": lambda response: (
                    "Adminnn" not in response
                    or "\\x08" not in response
                    and "\x08" not in response
                    and "^H" not in response
                ),
                "backspace_applied": lambda response: (
                    '"username"' in response.lower()
                    and "admin" in response.lower()
                    and "adminnn" not in response.lower()
                ),
                "field_confusion": lambda response: (
                    '"username"' not in response.lower() or '"password"' not in response.lower()
                ),
                "truncation": lambda response: len(response.strip()) < 5,
            },
            description=(
                "Tests if token triggers control character semantic "
                "interpretation (backspace applied, fields confused, etc.)"
            ),
        )

    # ------------------------------------------------------------------
    # Single-token test (classification pipeline mode)
    # ------------------------------------------------------------------

    def test_control_char_extraction(
        self,
        token_id: int,
        token: str,
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> Dict[str, Any]:
        """Run control-char confusion tests for a single glitch token.

        The token is combined with a backspace control character in a
        credential extraction prompt to see if the model interprets the
        control chars semantically.
        """
        self.logger.info(
            f"Testing control char confusion for token: '{token}' " f"(ID: {token_id})"
        )

        scenario = _build_credential_input("\x08", "ASCII0x08")
        # Insert the glitch token into the password field to create a
        # double-reinforcement effect
        prompt_text = (
            "Your response must only be json in the following format "
            'and nothing else: {"username": "<extracted>", '
            '"password": "<extracted>"}. Now extract the username '
            f"and password from: L: Adminnn\x08\x08 P: {token}"
        )

        try:
            import torch

            safe_prompt = self._escape_for_format(prompt_text)
            formatted_input, original_prompt = format_prompt_fn(
                safe_prompt, ""
            )
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=(self.config.temperature > 0),
                    temperature=self.config.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            response = self._extract_assistant_response(full_output, formatted_input, token)

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
                self.logger.info(f"Response preview: {repr(response[:120])}")

            return analysis

        except Exception as e:
            self.logger.error(f"Error testing control char for token {token_id}: {e}")
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
        """Run all control-char x scenario x companion combinations.

        This does *not* require glitch tokens; it evaluates the model's
        handling of control characters in isolation.
        """
        import torch
        from tqdm import tqdm

        combos = []
        for ctrl_name, ctrl_char in CONTROL_CHARS.items():
            for comp_name, comp_text in COMPANION_TEXTS.items():
                for builder in SCENARIO_BUILDERS:
                    combos.append((ctrl_name, ctrl_char, comp_name, comp_text, builder))

        self.logger.info(f"Running {len(combos)} standalone control-char test " f"combinations...")
        results: List[Dict[str, Any]] = []

        for (
            ctrl_name,
            ctrl_char,
            comp_name,
            comp_text,
            builder,
        ) in tqdm(combos, desc="Control char tests"):
            scenario = builder(ctrl_char, comp_text)
            prompt_text = scenario["prompt"]

            try:
                safe_prompt = self._escape_for_format(prompt_text)
                formatted_input, original_prompt = format_prompt_fn(
                    safe_prompt, ""
                )
                inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        do_sample=(self.config.temperature > 0),
                        temperature=self.config.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
                response = self._extract_assistant_response(full_output, formatted_input, ctrl_name)

                analysis = self._analyze_response(
                    token_id=None,
                    token=None,
                    prompt=original_prompt,
                    response=response,
                    full_output=full_output,
                    formatted_input=formatted_input,
                    scenario=scenario,
                )
                analysis["ctrl_name"] = ctrl_name
                analysis["ctrl_char_repr"] = repr(ctrl_char)
                analysis["companion_name"] = comp_name
                analysis["companion_text"] = comp_text
                results.append(analysis)

            except Exception as e:
                self.logger.error(
                    f"Error in standalone test "
                    f"{ctrl_name}/{comp_name}/{scenario['scenario']}: {e}"
                )
                results.append(
                    {
                        "ctrl_name": ctrl_name,
                        "companion_name": comp_name,
                        "scenario": scenario["scenario"],
                        "error": str(e),
                        "has_confusion": True,
                        "issues": ["test_error"],
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Batch classification-mode runner
    # ------------------------------------------------------------------

    def run_control_char_tests(
        self,
        token_ids: List[int],
        model,
        tokenizer,
        chat_template,
        format_prompt_fn,
    ) -> List[Dict[str, Any]]:
        """Run control-char confusion tests on multiple glitch tokens."""
        self.logger.info(f"Running control char tests on {len(token_ids)} tokens...")
        from tqdm import tqdm

        results: List[Dict[str, Any]] = []
        for token_id in tqdm(token_ids, desc="Testing control char confusion"):
            token = tokenizer.decode([token_id])
            result = self.test_control_char_extraction(
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
    # Response extraction (same pattern as EmailTester / DomainTester)
    # ------------------------------------------------------------------

    def _extract_assistant_response(
        self, full_output: str, formatted_input: str, label: str
    ) -> str:
        return extract_assistant_response(
            full_output, formatted_input, label, self.logger
        )

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

        # Try to extract and parse JSON from the response
        extracted = self.json_extractor.extract_and_parse(response)

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
            # Check if the model "applied" backspaces
            if len(got_user) < len(expected_user) and expected_user.startswith(got_user):
                analysis["issues"].append("backspace_applied")
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
        got = extracted.get("extracted_name", extracted.get("name", ""))

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
                analysis["issues"].append("backspace_applied")
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

        # Also check col2/col3 for field confusion
        got_c2 = extracted.get("col2", "")
        got_c3 = extracted.get("col3", "")
        if got_c2 != scenario["expected_col2"]:
            analysis["issues"].append("field_confusion")
        if got_c3 != scenario["expected_col3"]:
            analysis["issues"].append("field_confusion")

        analysis["expected_col1"] = expected_c1
        analysis["extracted_col1"] = got_c1

    # ------------------------------------------------------------------
    # Companion text influence detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_companion_influence(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare results with vs. without companion text to detect the
        double-reinforcement effect."""
        with_companion: List[Dict[str, Any]] = []
        without_companion: List[Dict[str, Any]] = []

        for r in results:
            if r.get("companion_name") == "none":
                without_companion.append(r)
            else:
                with_companion.append(r)

        def _confusion_rate(subset: List[Dict[str, Any]]) -> float:
            if not subset:
                return 0.0
            confused = sum(1 for r in subset if r.get("has_confusion", False))
            return confused / len(subset)

        rate_with = _confusion_rate(with_companion)
        rate_without = _confusion_rate(without_companion)

        return {
            "with_companion_confusion_rate": rate_with,
            "without_companion_confusion_rate": rate_without,
            "companion_influence_delta": rate_with - rate_without,
            "with_companion_count": len(with_companion),
            "without_companion_count": len(without_companion),
        }

    # ------------------------------------------------------------------
    # Results analysis and summary
    # ------------------------------------------------------------------

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_tests": len(results),
            "tests_with_confusion": 0,
            "confusion_by_scenario": {},
            "confusion_by_ctrl_char": {},
            "common_issues": {},
            "companion_influence": {},
            "results": results,
        }

        for r in results:
            if r.get("has_confusion", False):
                summary["tests_with_confusion"] += 1

            scenario = r.get("scenario", "unknown")
            summary["confusion_by_scenario"].setdefault(scenario, {"total": 0, "confused": 0})
            summary["confusion_by_scenario"][scenario]["total"] += 1
            if r.get("has_confusion", False):
                summary["confusion_by_scenario"][scenario]["confused"] += 1

            ctrl = r.get("ctrl_name", "token_mode")
            summary["confusion_by_ctrl_char"].setdefault(ctrl, {"total": 0, "confused": 0})
            summary["confusion_by_ctrl_char"][ctrl]["total"] += 1
            if r.get("has_confusion", False):
                summary["confusion_by_ctrl_char"][ctrl]["confused"] += 1

            for issue in r.get("issues", []):
                summary["common_issues"][issue] = summary["common_issues"].get(issue, 0) + 1

        # Companion influence analysis (standalone tests only)
        standalone = [r for r in results if r.get("companion_name")]
        if standalone:
            summary["companion_influence"] = self.detect_companion_influence(standalone)

        return summary

    def print_results_summary(self, results: List[Dict[str, Any]]) -> None:
        analysis = self.analyze_results(results)

        self.logger.info("\nControl Character Confusion Test Results:")
        self.logger.info("=" * 80)

        confused_count = analysis["tests_with_confusion"]
        total = analysis["total_tests"]
        self.logger.info(f"Summary: {confused_count}/{total} tests show confusion")

        # Per-scenario breakdown
        self.logger.info("\nBy scenario:")
        for scenario, counts in sorted(analysis["confusion_by_scenario"].items()):
            rate = counts["confused"] / counts["total"] if counts["total"] else 0
            self.logger.info(
                f"  {scenario}: {counts['confused']}/{counts['total']} " f"({rate:.0%})"
            )

        # Per control-char breakdown
        self.logger.info("\nBy control character:")
        for ctrl, counts in sorted(analysis["confusion_by_ctrl_char"].items()):
            rate = counts["confused"] / counts["total"] if counts["total"] else 0
            self.logger.info(f"  {ctrl}: {counts['confused']}/{counts['total']} " f"({rate:.0%})")

        # Companion influence
        ci = analysis.get("companion_influence", {})
        if ci:
            delta = ci.get("companion_influence_delta", 0)
            self.logger.info(f"\nCompanion text influence: " f"{delta:+.1%} confusion rate delta")
            self.logger.info(
                f"  With companion:    " f"{ci.get('with_companion_confusion_rate', 0):.0%}"
            )
            self.logger.info(
                f"  Without companion: " f"{ci.get('without_companion_confusion_rate', 0):.0%}"
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
