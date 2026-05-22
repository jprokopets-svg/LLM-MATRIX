"""
Baseline forecasters for benchmark comparison.

Three baselines that will compete with LLM forecasts:
  1. Naive: predict next period = last observed value (random walk).
  2. AR(1): fit AR(1) on each variable independently from history.
  3. Oracle: knows the true model; runs Monte Carlo and outputs mean trajectory.

All baselines return predictions in the same format:
  DataFrame with columns [period, variable, predicted_mean, predicted_std]
  for the 12 forecast periods.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from src.monte_carlo import run_counterfactual


FORECAST_VARIABLES = ["y", "pi", "r", "e", "u"]


def naive_baseline(
    history_df: pd.DataFrame,
    n_forward_periods: int = 12,
) -> pd.DataFrame:
    """
    Naive (random walk) baseline: predict every future period = last observed value.

    No change forecasted. Standard benchmark in time series literature.

    Args:
        history_df: Historical trajectory DataFrame.
        n_forward_periods: Number of quarters to forecast.

    Returns:
        DataFrame with columns [period, variable, predicted_mean, predicted_std].
        predicted_std is 0 because the naive baseline has no uncertainty estimate.
    """
    last_row = history_df.iloc[-1]
    start_period = int(last_row["period"])

    rows = []
    for t in range(1, n_forward_periods + 1):
        period = start_period + t
        for var_name in FORECAST_VARIABLES:
            rows.append({
                "period": period,
                "variable": var_name,
                "predicted_mean": float(last_row[var_name]),
                "predicted_std": 0.0,
            })

    return pd.DataFrame(rows)


def ar1_baseline(
    history_df: pd.DataFrame,
    n_forward_periods: int = 12,
) -> pd.DataFrame:
    """
    AR(1) baseline: fit an AR(1) model on each variable independently, then forecast.

    Uses statsmodels AutoReg with lag=1. Each variable is modeled in isolation,
    ignoring cross-variable dynamics. This is the key baseline the benchmark
    must beat — if AR(1) matches oracle, the system has no extractable structure.

    Args:
        history_df: Historical trajectory DataFrame.
        n_forward_periods: Number of quarters to forecast.

    Returns:
        DataFrame with columns [period, variable, predicted_mean, predicted_std].
    """
    last_period = int(history_df.iloc[-1]["period"])

    rows = []
    for var_name in FORECAST_VARIABLES:
        series = history_df[var_name].values

        # Fit AR(1) model
        model = AutoReg(series, lags=1)
        fitted = model.fit()

        # Forecast forward
        forecasts = fitted.predict(
            start=len(series),
            end=len(series) + n_forward_periods - 1,
        )

        # Estimate forecast std from residuals, growing with sqrt(horizon)
        residual_std = float(np.std(fitted.resid))

        for t_index in range(n_forward_periods):
            period = last_period + t_index + 1
            # Uncertainty grows with forecast horizon for AR processes
            horizon_std = residual_std * np.sqrt(t_index + 1)
            rows.append({
                "period": period,
                "variable": var_name,
                "predicted_mean": float(forecasts[t_index]),
                "predicted_std": horizon_std,
            })

    return pd.DataFrame(rows)


def oracle_baseline(
    config_path: str,
    history_df: pd.DataFrame,
    shock_overrides_fn: callable,
    n_forward_periods: int = 12,
    n_paths: int = 1000,
    base_seed: int = 1000,
) -> pd.DataFrame:
    """
    Oracle baseline: knows the true model equations and parameters.

    Runs Monte Carlo simulation with the actual model and outputs
    the mean and std of the resulting distribution. This is the
    upper bound on forecasting accuracy — no forecaster should
    consistently beat the oracle.

    Args:
        config_path: Path to YAML config file.
        history_df: Historical trajectory DataFrame.
        shock_overrides_fn: Callable returning shock override dict.
        n_forward_periods: Number of quarters to simulate forward.
        n_paths: Number of Monte Carlo paths.
        base_seed: Starting seed for Monte Carlo.

    Returns:
        DataFrame with columns [period, variable, predicted_mean, predicted_std].
    """
    # Run Monte Carlo to get the true distribution
    paths = run_counterfactual(
        config_path=config_path,
        history_df=history_df,
        shock_fn=shock_overrides_fn,
        n_paths=n_paths,
        n_forward_periods=n_forward_periods,
        base_seed=base_seed,
    )

    last_period = int(history_df.iloc[-1]["period"])

    rows = []
    for t_index in range(n_forward_periods):
        period = last_period + t_index + 1
        for v_index, var_name in enumerate(FORECAST_VARIABLES):
            values = paths[:, t_index, v_index]
            rows.append({
                "period": period,
                "variable": var_name,
                "predicted_mean": float(np.mean(values)),
                "predicted_std": float(np.std(values)),
            })

    return pd.DataFrame(rows)
