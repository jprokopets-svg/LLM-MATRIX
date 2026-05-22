"""
Prompt templates for the LLMMatrix pilot.

Two formats:
  - Format A (direct): request JSON predictions immediately.
  - Format B (chain-of-thought): ask for step-by-step reasoning first.

Both formats embed the same narrative and question content.
"""

from src.questions import ForecastingQuestion


def _build_prediction_keys(question: ForecastingQuestion) -> list[str]:
    """
    Build the list of prediction keys like ["y_3", "pi_6", ...] from a question.

    Args:
        question: A ForecastingQuestion with target_variables and target_horizons.

    Returns:
        Sorted list of "variable_horizon" strings.
    """
    keys = []
    for var in question.target_variables:
        for horizon in question.target_horizons:
            keys.append(f"{var}_{horizon}")
    keys.sort()
    return keys


def _build_json_example(keys: list[str]) -> str:
    """Build the example JSON structure for the prompt."""
    lines = []
    for key in keys:
        lines.append(f'    "{key}": {{"point": <float>, "ci_low": <float>, "ci_high": <float>}}')
    inner = ",\n".join(lines)
    return '{\n  "predictions": {\n' + inner + "\n  }\n}"


def _build_variable_list(question: ForecastingQuestion) -> str:
    """Build a human-readable list of (variable, horizon) targets."""
    var_names = {
        "y": "output gap (%)",
        "pi": "inflation rate (annual %)",
        "r": "real interest rate (annual %)",
        "e": "real exchange rate (log index)",
        "u": "unemployment rate (%)",
    }
    lines = []
    for var in question.target_variables:
        for horizon in question.target_horizons:
            lines.append(f"- {var_names[var]} at {horizon} quarters ahead")
    return "\n".join(lines)


def format_direct(narrative: str, question: ForecastingQuestion) -> str:
    """
    Format A — Direct prompt. Request JSON predictions immediately.

    Args:
        narrative: The full Vantria economic history text.
        question: The forecasting question to ask.

    Returns:
        Complete prompt string.
    """
    keys = _build_prediction_keys(question)
    json_example = _build_json_example(keys)
    variable_list = _build_variable_list(question)

    prompt = (
        f"You are a macroeconomic forecaster analyzing the Republic of Vantria.\n\n"
        f"{narrative}\n\n"
        f"---\n\n"
        f"Forecasting scenario: {question.shock_description}\n\n"
        f"Predict the following variables at the specified horizons after the shock:\n"
        f"{variable_list}\n\n"
        f"For each prediction, provide:\n"
        f"- point estimate (your best estimate of the variable's value)\n"
        f"- 80% confidence interval (low, high)\n\n"
        f"Respond ONLY with valid JSON in this exact format:\n"
        f"{json_example}"
    )
    return prompt


def format_cot(narrative: str, question: ForecastingQuestion) -> str:
    """
    Format B — Chain-of-thought prompt. Ask for reasoning before predictions.

    Args:
        narrative: The full Vantria economic history text.
        question: The forecasting question to ask.

    Returns:
        Complete prompt string.
    """
    keys = _build_prediction_keys(question)
    json_example = _build_json_example(keys)
    variable_list = _build_variable_list(question)

    prompt = (
        f"You are a macroeconomic forecaster analyzing the Republic of Vantria.\n\n"
        f"{narrative}\n\n"
        f"---\n\n"
        f"Forecasting scenario: {question.shock_description}\n\n"
        f"Predict the following variables at the specified horizons after the shock:\n"
        f"{variable_list}\n\n"
        f"Before answering, reason step-by-step through the causal channels:\n"
        f"1. How does this shock affect the listed variables directly?\n"
        f"2. What second-order effects propagate through monetary policy / "
        f"exchange rate / labor market?\n"
        f"3. How do the dynamics evolve over the forecast horizon?\n\n"
        f"Then provide your final predictions. For each, include:\n"
        f"- point estimate (your best estimate of the variable's value)\n"
        f"- 80% confidence interval (low, high)\n\n"
        f"Respond with your reasoning, followed by your predictions between "
        f"<JSON> and </JSON> tags:\n"
        f"<JSON>\n"
        f"{json_example}\n"
        f"</JSON>"
    )
    return prompt


PROMPT_FORMATTERS = {
    "direct": format_direct,
    "cot": format_cot,
}
