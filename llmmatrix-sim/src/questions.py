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
    # --- Monetary (3) — tests rates → output, inflation, FX channel ---
    # At h=1: r_override hits r directly, e responds via UIP.
    # IS curve is backward-looking so y/pi/u don't respond until h=2+.
    ForecastingQuestion(
        id="Q01",
        shock_type="monetary",
        shock_description="The Central Bank of Vantria unexpectedly raises its policy rate by 100 basis points to combat persistent inflation, surprising markets that expected a hold.",
        shock_magnitude=1.0,
        target_variables=["y", "pi", "u"],
        target_horizons=[1],
        expected_direction={"y_1": "neutral", "pi_1": "neutral", "u_1": "neutral"},
    ),
    ForecastingQuestion(
        id="Q02",
        shock_type="monetary",
        shock_description="The Central Bank of Vantria cuts its policy rate by 50 basis points in response to softening growth indicators.",
        shock_magnitude=-0.5,
        target_variables=["y", "e", "u"],
        target_horizons=[1],
        expected_direction={"y_1": "neutral", "e_1": "down", "u_1": "neutral"},
    ),
    ForecastingQuestion(
        id="Q03",
        shock_type="monetary",
        shock_description="A surprise 25 basis point rate increase, smaller than the 50bp the market had priced in.",
        shock_magnitude=0.25,
        target_variables=["pi", "e", "r"],
        target_horizons=[1],
        expected_direction={"pi_1": "neutral", "e_1": "neutral", "r_1": "up"},
    ),

    # --- Demand (3) — tests output → inflation → policy response ---
    # At h=1: eps_y hits y directly, r responds via Taylor, e via UIP,
    # u via Okun. Phillips uses lagged y so pi doesn't respond yet.
    ForecastingQuestion(
        id="Q04",
        shock_type="demand",
        shock_description="A surge in business investment driven by improved confidence adds 1.5 percentage points to aggregate demand this quarter.",
        shock_magnitude=1.5,
        target_variables=["y", "pi", "r"],
        target_horizons=[1],
        expected_direction={"y_1": "up", "pi_1": "neutral", "r_1": "up"},
    ),
    ForecastingQuestion(
        id="Q05",
        shock_type="demand",
        shock_description="A consumer confidence collapse reduces aggregate demand by 2 percentage points.",
        shock_magnitude=-2.0,
        target_variables=["y", "u", "r"],
        target_horizons=[1],
        expected_direction={"y_1": "down", "u_1": "up", "r_1": "down"},
    ),
    ForecastingQuestion(
        id="Q06",
        shock_type="demand",
        shock_description="Government infrastructure spending adds a modest 0.75 percentage points to demand.",
        shock_magnitude=0.75,
        target_variables=["y", "pi", "u"],
        target_horizons=[1],
        expected_direction={"y_1": "up", "pi_1": "neutral", "u_1": "down"},
    ),

    # --- Cost-push (2) — tests stagflation reasoning ---
    # At h=1: eps_pi hits pi directly, r responds via Taylor.
    # IS curve is backward-looking so y/u don't respond yet.
    ForecastingQuestion(
        id="Q07",
        shock_type="cost_push",
        shock_description="A global commodity price spike adds 1.5 percentage points to inflation this quarter, unrelated to domestic demand.",
        shock_magnitude=1.5,
        target_variables=["pi", "y", "r"],
        target_horizons=[1],
        expected_direction={"pi_1": "up", "y_1": "neutral", "r_1": "up"},
    ),
    ForecastingQuestion(
        id="Q08",
        shock_type="cost_push",
        shock_description="Easing supply chain pressures reduce inflation by 0.75 percentage points this quarter.",
        shock_magnitude=-0.75,
        target_variables=["pi", "r", "e"],
        target_horizons=[1],
        expected_direction={"pi_1": "down", "r_1": "neutral", "e_1": "neutral"},
    ),

    # --- Exchange rate (2) — tests open-economy channel ---
    # At h=1: eps_e hits e directly. IS/Phillips use lagged e, so
    # y/pi/u don't respond until h=2+.
    ForecastingQuestion(
        id="Q09",
        shock_type="exchange_rate",
        shock_description="A sudden 5 percent depreciation of the vantra following a global risk-off episode.",
        shock_magnitude=-5.0,
        target_variables=["e", "pi", "y"],
        target_horizons=[1],
        expected_direction={"e_1": "down", "pi_1": "neutral", "y_1": "neutral"},
    ),
    ForecastingQuestion(
        id="Q10",
        shock_type="exchange_rate",
        shock_description="A 3 percent appreciation of the vantra following capital inflows.",
        shock_magnitude=3.0,
        target_variables=["e", "pi", "u"],
        target_horizons=[1],
        expected_direction={"e_1": "up", "pi_1": "neutral", "u_1": "neutral"},
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
        valid_horizons = {1, 3, 6, 12}
        for h in q.target_horizons:
            if h not in valid_horizons:
                errors.append(f"{q.id}: invalid horizon {h}")

    return errors
