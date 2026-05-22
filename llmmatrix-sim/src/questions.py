"""
Forecasting questions for the LLMMatrix pilot.

Defines 10 hand-written questions covering the four shock types.
Distribution: 3 monetary + 3 demand + 2 cost-push + 2 exchange-rate.

Question details (shock_description, magnitude, target_variables,
target_horizons, expected_direction) are placeholders to be filled
in manually by the project lead before the pilot run.
"""

from dataclasses import dataclass, field


@dataclass
class ForecastingQuestion:
    """
    A single forecasting question for the LLMMatrix pilot.

    Attributes:
        id: Unique identifier ("Q01", "Q02", ...).
        shock_type: One of "monetary", "demand", "cost_push", "exchange_rate".
        shock_description: Human-readable description of the counterfactual shock.
        shock_magnitude: Numerical magnitude for the simulator.
        target_variables: Which variables to forecast (subset of y, pi, r, e, u).
        target_horizons: Forecast horizons in quarters (subset of 3, 6, 12).
        expected_direction: Maps "var_horizon" -> "up" or "down" for balance check.
    """
    id: str
    shock_type: str
    shock_description: str = ""
    shock_magnitude: float = 0.0
    target_variables: list[str] = field(default_factory=list)
    target_horizons: list[int] = field(default_factory=list)
    expected_direction: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Placeholder question list — fill in before running the pilot
# ---------------------------------------------------------------------------

PILOT_QUESTIONS: list[ForecastingQuestion] = [
    # 3 monetary policy questions
    ForecastingQuestion(
        id="Q01",
        shock_type="monetary",
        # TODO: fill in shock_description, shock_magnitude, target_variables,
        #       target_horizons, expected_direction
    ),
    ForecastingQuestion(
        id="Q02",
        shock_type="monetary",
    ),
    ForecastingQuestion(
        id="Q03",
        shock_type="monetary",
    ),

    # 3 demand questions
    ForecastingQuestion(
        id="Q04",
        shock_type="demand",
    ),
    ForecastingQuestion(
        id="Q05",
        shock_type="demand",
    ),
    ForecastingQuestion(
        id="Q06",
        shock_type="demand",
    ),

    # 2 cost-push questions
    ForecastingQuestion(
        id="Q07",
        shock_type="cost_push",
    ),
    ForecastingQuestion(
        id="Q08",
        shock_type="cost_push",
    ),

    # 2 exchange rate questions
    ForecastingQuestion(
        id="Q09",
        shock_type="exchange_rate",
    ),
    ForecastingQuestion(
        id="Q10",
        shock_type="exchange_rate",
    ),
]


def check_direction_balance(questions: list[ForecastingQuestion]) -> dict:
    """
    Verify that expected directions are approximately 50/50 up/down
    for each variable at each horizon across the full question set.

    Args:
        questions: List of ForecastingQuestion objects with expected_direction filled.

    Returns:
        Dict mapping "var_horizon" -> {"up": count, "down": count, "balanced": bool}.
    """
    counts: dict[str, dict[str, int]] = {}

    for q in questions:
        for key, direction in q.expected_direction.items():
            if key not in counts:
                counts[key] = {"up": 0, "down": 0}
            counts[key][direction] = counts[key].get(direction, 0) + 1

    result = {}
    for key, c in counts.items():
        total = c["up"] + c["down"]
        # "Balanced" if neither direction has more than 70% of answers
        balanced = min(c["up"], c["down"]) / total >= 0.3 if total > 0 else True
        result[key] = {**c, "balanced": balanced}

    return result


def validate_questions(questions: list[ForecastingQuestion]) -> list[str]:
    """
    Check that all questions are fully specified and ready for the pilot.

    Returns:
        List of validation error strings. Empty list = all good.
    """
    errors = []

    for q in questions:
        if not q.shock_description:
            errors.append(f"{q.id}: missing shock_description")
        if q.shock_magnitude == 0.0:
            errors.append(f"{q.id}: shock_magnitude is 0")
        if not q.target_variables:
            errors.append(f"{q.id}: no target_variables specified")
        if not q.target_horizons:
            errors.append(f"{q.id}: no target_horizons specified")
        if not q.expected_direction:
            errors.append(f"{q.id}: no expected_direction specified")

        # Validate target_variables
        valid_vars = {"y", "pi", "r", "e", "u"}
        for v in q.target_variables:
            if v not in valid_vars:
                errors.append(f"{q.id}: invalid target variable '{v}'")

        # Validate target_horizons
        valid_horizons = {3, 6, 12}
        for h in q.target_horizons:
            if h not in valid_horizons:
                errors.append(f"{q.id}: invalid horizon {h}")

    return errors
