"""
Test that impulse responses match standard New Keynesian predictions.

These tests verify the qualitative behavior of the model:
  - Monetary tightening reduces output and inflation, appreciates exchange rate, raises unemployment.
  - Demand shocks raise output and inflation.
  - Cost-push shocks raise inflation without initially raising output.
"""

import os

from src.simulator import Sim
from src.validation import check_impulse_response


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "ball_baseline.yaml")


def _run_deterministic_irf(shock_overrides: dict, n_pre: int = 5, n_post: int = 20) -> dict:
    """
    Helper: run a deterministic simulation with one shock and return IRF data.

    Returns dict mapping variable name -> list of deviations from steady state.
    """
    # Zero out all stochastic shocks
    all_overrides = {}
    for t in range(1, n_pre + n_post + 2):
        for var in ["eps_y", "eps_pi", "eps_e"]:
            all_overrides[(t, var)] = 0.0
    all_overrides.update(shock_overrides)

    sim = Sim(CONFIG_PATH, seed=0)
    trajectory = sim.run(n_pre + n_post, shock_overrides=all_overrides)
    df = sim.to_dataframe(trajectory)

    targets = {"y": 0.0, "pi": sim.pi_star, "r": sim.r_star, "e": sim.e_star, "u": sim.u_star}
    irf = {}
    for var_name, target in targets.items():
        irf[var_name] = (df[var_name] - target).tolist()

    return irf


def test_monetary_tightening_irf():
    """Full check of monetary tightening impulse responses via validation module."""
    result = check_impulse_response(CONFIG_PATH)
    assert result["passed"], (
        f"Impulse response checks failed: "
        + ", ".join(c["name"] for c in result["checks"] if not c["passed"])
    )


def test_demand_shock_raises_output_and_inflation():
    """A positive demand shock should raise y and pi."""
    shock_period = 6
    irf = _run_deterministic_irf({(shock_period, "eps_y"): 2.0})

    # y should jump up at the shock period
    assert irf["y"][shock_period] > 0.1, (
        f"y did not rise on demand shock: {irf['y'][shock_period]:.4f}"
    )

    # pi should rise in the period after (Phillips curve uses lagged y)
    assert irf["pi"][shock_period + 1] > 0.01, (
        f"pi did not rise after demand shock: {irf['pi'][shock_period + 1]:.4f}"
    )


def test_cost_push_shock_raises_inflation():
    """A cost-push shock should raise inflation."""
    shock_period = 6
    irf = _run_deterministic_irf({(shock_period, "eps_pi"): 2.0})

    # pi should jump at the shock period
    assert irf["pi"][shock_period] > 0.1, (
        f"pi did not rise on cost-push shock: {irf['pi'][shock_period]:.4f}"
    )


def test_exchange_rate_shock_effects():
    """A positive exchange rate shock (appreciation) should reduce inflation via passthrough."""
    shock_period = 6
    irf = _run_deterministic_irf({(shock_period, "eps_e"): 5.0})

    # e should appreciate (go up) at shock period
    assert irf["e"][shock_period] > 0.1, (
        f"e did not appreciate on exchange rate shock: {irf['e'][shock_period]:.4f}"
    )

    # pi should fall in the next period (exchange rate passthrough in Phillips curve)
    assert irf["pi"][shock_period + 1] < -0.01, (
        f"pi did not fall after exchange rate appreciation: {irf['pi'][shock_period + 1]:.4f}"
    )
