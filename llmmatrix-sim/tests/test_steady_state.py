"""
Test that the system converges to steady-state targets under no shocks,
and recovers after a one-time perturbation.
"""

import os

from src.simulator import Sim
from src.validation import check_steady_state_recovery


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "ball_baseline.yaml")


def test_no_shock_stays_at_steady_state():
    """With zero shocks and steady-state initial conditions, all variables stay at targets."""
    # Override all shocks to zero
    n_periods = 20
    shock_overrides = {}
    for t in range(1, n_periods + 1):
        for var in ["eps_y", "eps_pi", "eps_e"]:
            shock_overrides[(t, var)] = 0.0

    sim = Sim(CONFIG_PATH, seed=0)
    trajectory = sim.run(n_periods, shock_overrides=shock_overrides)
    df = sim.to_dataframe(trajectory)

    targets = {"y": 0.0, "pi": 2.0, "r": 2.0, "e": 0.0, "u": 5.0}
    tolerance = 1e-10

    for var_name, target in targets.items():
        max_deviation = (df[var_name] - target).abs().max()
        assert max_deviation < tolerance, (
            f"{var_name} deviated from target {target} by {max_deviation:.2e} "
            f"under zero shocks"
        )


def test_recovery_after_demand_shock():
    """System returns to steady state within 50 quarters after a demand shock."""
    result = check_steady_state_recovery(
        CONFIG_PATH,
        shock_variable="eps_y",
        shock_magnitude=3.0,
        recovery_threshold=0.1,
        max_recovery_periods=50,
    )
    assert result["passed"], (
        f"Failed to recover after demand shock. "
        f"Recovery periods: {result['recovery_periods']}. "
        f"Final deviations: {result['final_deviations']}"
    )


def test_recovery_after_cost_push_shock():
    """System returns to steady state within 50 quarters after a cost-push shock."""
    result = check_steady_state_recovery(
        CONFIG_PATH,
        shock_variable="eps_pi",
        shock_magnitude=2.0,
        recovery_threshold=0.1,
        max_recovery_periods=50,
    )
    assert result["passed"], (
        f"Failed to recover after cost-push shock. "
        f"Recovery periods: {result['recovery_periods']}. "
        f"Final deviations: {result['final_deviations']}"
    )


def test_recovery_after_exchange_rate_shock():
    """System returns to steady state within 50 quarters after an exchange rate shock."""
    result = check_steady_state_recovery(
        CONFIG_PATH,
        shock_variable="eps_e",
        shock_magnitude=5.0,
        recovery_threshold=0.1,
        max_recovery_periods=50,
    )
    assert result["passed"], (
        f"Failed to recover after exchange rate shock. "
        f"Recovery periods: {result['recovery_periods']}. "
        f"Final deviations: {result['final_deviations']}"
    )
