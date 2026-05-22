"""
Test that baseline forecasters produce valid output and that
the naive and AR(1) baselines run without error.
"""

import os

import pandas as pd

from src.simulator import Sim
from src.baselines import naive_baseline, ar1_baseline, oracle_baseline, FORECAST_VARIABLES
from src.shocks import demand_shock


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "ball_baseline.yaml")

# Number of forward periods to forecast
N_FORWARD = 12


def _generate_history() -> pd.DataFrame:
    """Helper: generate a 60-quarter history for testing."""
    sim = Sim(CONFIG_PATH, seed=42)
    trajectory = sim.run(60)
    return sim.to_dataframe(trajectory)


def test_naive_baseline_shape():
    """Naive baseline produces correct number of rows and columns."""
    history_df = _generate_history()
    result = naive_baseline(history_df, n_forward_periods=N_FORWARD)

    expected_rows = N_FORWARD * len(FORECAST_VARIABLES)
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result)}"
    )

    required_columns = {"period", "variable", "predicted_mean", "predicted_std"}
    assert required_columns.issubset(set(result.columns)), (
        f"Missing columns: {required_columns - set(result.columns)}"
    )


def test_naive_baseline_is_flat():
    """Naive baseline predicts the same value for all horizons."""
    history_df = _generate_history()
    result = naive_baseline(history_df, n_forward_periods=N_FORWARD)

    for var_name in FORECAST_VARIABLES:
        var_preds = result[result["variable"] == var_name]["predicted_mean"]
        # All predictions should be identical (last observed value)
        assert var_preds.nunique() == 1, (
            f"Naive baseline for {var_name} is not flat: {var_preds.tolist()}"
        )


def test_ar1_baseline_shape():
    """AR(1) baseline produces correct output format."""
    history_df = _generate_history()
    result = ar1_baseline(history_df, n_forward_periods=N_FORWARD)

    expected_rows = N_FORWARD * len(FORECAST_VARIABLES)
    assert len(result) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result)}"
    )


def test_ar1_baseline_has_uncertainty():
    """AR(1) baseline should produce non-zero uncertainty estimates."""
    history_df = _generate_history()
    result = ar1_baseline(history_df, n_forward_periods=N_FORWARD)

    for var_name in FORECAST_VARIABLES:
        var_stds = result[result["variable"] == var_name]["predicted_std"]
        assert (var_stds > 0).all(), (
            f"AR(1) baseline for {var_name} has zero uncertainty"
        )


def test_ar1_uncertainty_grows_with_horizon():
    """AR(1) forecast uncertainty should increase with forecast horizon."""
    history_df = _generate_history()
    result = ar1_baseline(history_df, n_forward_periods=N_FORWARD)

    for var_name in FORECAST_VARIABLES:
        var_stds = result[result["variable"] == var_name]["predicted_std"].values
        # Each std should be >= the previous (monotonically non-decreasing)
        for i in range(1, len(var_stds)):
            assert var_stds[i] >= var_stds[i - 1] - 1e-10, (
                f"AR(1) uncertainty for {var_name} decreased at horizon {i}: "
                f"{var_stds[i]:.4f} < {var_stds[i-1]:.4f}"
            )


def test_oracle_baseline_runs():
    """Oracle baseline runs without error and produces valid output."""
    history_df = _generate_history()
    shock_fn = lambda: demand_shock(magnitude=2.0, period=60)

    result = oracle_baseline(
        config_path=CONFIG_PATH,
        history_df=history_df,
        shock_overrides_fn=shock_fn,
        n_forward_periods=N_FORWARD,
        n_paths=100,  # fewer paths for speed in tests
        base_seed=1000,
    )

    expected_rows = N_FORWARD * len(FORECAST_VARIABLES)
    assert len(result) == expected_rows
    assert (result["predicted_std"] > 0).all(), "Oracle should have non-zero uncertainty"
