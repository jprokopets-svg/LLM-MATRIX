"""
Response parser for LLM forecasting predictions.

Extracts structured JSON predictions from raw API response text.
Handles both Format A (direct JSON) and Format B (CoT with <JSON> tags).
"""

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Parsed prediction from a single LLM API call.

    Attributes:
        question_id: Which question was asked.
        model: Model identifier (e.g., "claude-sonnet-4-6").
        prompt_format: "direct" or "cot".
        seed: Random seed used for this call.
        predictions: Parsed predictions dict, keyed by "var_horizon".
        parse_success: Whether parsing succeeded.
        parse_error: Error message if parsing failed.
        raw_response: Full response text for debugging.
    """
    question_id: str
    model: str
    prompt_format: str
    seed: int
    predictions: dict = field(default_factory=dict)
    parse_success: bool = False
    parse_error: str | None = None
    raw_response: str = ""


def _extract_json_block(text: str) -> str | None:
    """
    Try to extract a JSON block from response text.

    Strategy (in order):
      1. Look for content between <JSON> and </JSON> tags.
      2. Find the last {...} block in the text.
      3. Try the full text as JSON directly.

    Returns:
        Extracted JSON string, or None if nothing found.
    """
    # Strategy 1: <JSON> tags (Format B / CoT)
    tag_match = re.search(r"<JSON>\s*(.*?)\s*</JSON>", text, re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()

    # Strategy 2: last {...} block
    # Find all top-level brace-delimited blocks
    brace_depth = 0
    block_start = None
    last_block = None
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                block_start = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and block_start is not None:
                last_block = text[block_start:i + 1]

    if last_block:
        return last_block

    return None


def _validate_predictions(
    predictions: dict,
    expected_keys: list[str],
) -> list[str]:
    """
    Validate that parsed predictions have the right structure.

    Checks:
      - All expected keys are present.
      - Values are numeric.
      - ci_low <= point <= ci_high.

    Returns:
        List of validation error strings. Empty = valid.
    """
    errors = []

    for key in expected_keys:
        if key not in predictions:
            errors.append(f"Missing prediction for '{key}'")
            continue

        pred = predictions[key]

        # Check required fields
        for field_name in ["point", "ci_low", "ci_high"]:
            if field_name not in pred:
                errors.append(f"'{key}' missing field '{field_name}'")
                continue
            if not isinstance(pred[field_name], (int, float)):
                errors.append(f"'{key}.{field_name}' is not numeric: {pred[field_name]}")

        # Check ordering
        if all(f in pred for f in ["point", "ci_low", "ci_high"]):
            if isinstance(pred["ci_low"], (int, float)) and isinstance(pred["point"], (int, float)):
                if pred["ci_low"] > pred["point"]:
                    errors.append(
                        f"'{key}': ci_low ({pred['ci_low']}) > point ({pred['point']})"
                    )
            if isinstance(pred["point"], (int, float)) and isinstance(pred["ci_high"], (int, float)):
                if pred["point"] > pred["ci_high"]:
                    errors.append(
                        f"'{key}': point ({pred['point']}) > ci_high ({pred['ci_high']})"
                    )

    return errors


def parse_response(
    raw_response: str,
    question_id: str,
    model: str,
    prompt_format: str,
    seed: int,
    expected_keys: list[str],
) -> PredictionResult:
    """
    Parse a raw LLM response into a structured PredictionResult.

    Args:
        raw_response: Full text returned by the API.
        question_id: Which question this response answers.
        model: Model identifier.
        prompt_format: "direct" or "cot".
        seed: Seed used for this call.
        expected_keys: List of expected prediction keys (e.g., ["y_3", "pi_6"]).

    Returns:
        PredictionResult with parse_success=True if all validations pass.
    """
    result = PredictionResult(
        question_id=question_id,
        model=model,
        prompt_format=prompt_format,
        seed=seed,
        raw_response=raw_response,
    )

    # Step 1: Extract JSON block
    json_str = _extract_json_block(raw_response)
    if json_str is None:
        result.parse_error = "Could not find JSON block in response"
        logger.warning(f"Parse failed for {model}/{question_id}/{prompt_format}: no JSON found")
        return result

    # Step 2: Parse JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        result.parse_error = f"JSON decode error: {exc}"
        logger.warning(f"Parse failed for {model}/{question_id}/{prompt_format}: {exc}")
        return result

    # Step 3: Extract predictions dict
    if "predictions" in parsed:
        predictions = parsed["predictions"]
    else:
        # Maybe the response is the predictions dict directly
        predictions = parsed

    # Step 4: Validate
    validation_errors = _validate_predictions(predictions, expected_keys)
    if validation_errors:
        result.parse_error = "; ".join(validation_errors)
        result.predictions = predictions  # store partial results anyway
        logger.warning(
            f"Validation errors for {model}/{question_id}/{prompt_format}: "
            f"{result.parse_error}"
        )
        return result

    # Success
    result.predictions = predictions
    result.parse_success = True
    return result
