"""
Scoring functions for the LLMMatrix pilot.

Two metrics:
  - MAE: mean absolute error of point predictions vs Monte Carlo truth mean.
  - CRPS: continuous ranked probability score using predicted Gaussian
    (inferred from 80% CI) vs empirical truth distribution.

Both include bootstrap 95% confidence intervals (1000 resamples).
"""

import logging

import numpy as np
import pandas as pd
from properscoring import crps_gaussian

logger = logging.getLogger(__name__)

# The 80% CI spans from the 10th to 90th percentile of a Gaussian,
# which is +/- 1.28 standard deviations. So width = 2 * 1.28 * sigma.
Z_80 = 1.28  # z-score for 80% CI (each tail = 10%)
CI_WIDTH_TO_STD = 2.0 * Z_80  # = 2.56


def mae_score(
    predictions: list[float],
    truth_means: list[float],
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """
    Compute MAE of point predictions vs the true mean values.

    The "truth" is the Monte Carlo mean of the ground-truth distribution
    at each horizon — the best possible point forecast.

    Args:
        predictions: List of point predictions from the model.
        truth_means: List of true mean values (from MC distribution).
        n_bootstrap: Number of bootstrap resamples for CI.
        rng_seed: Seed for bootstrap reproducibility.

    Returns:
        Dict with keys: mae, ci_low, ci_high.
    """
    predictions_arr = np.array(predictions)
    truth_arr = np.array(truth_means)
    errors = np.abs(predictions_arr - truth_arr)

    mae = float(np.mean(errors))

    # Bootstrap CI
    rng = np.random.default_rng(rng_seed)
    n = len(errors)
    bootstrap_maes = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        bootstrap_maes[i] = np.mean(errors[sample_idx])

    ci_low = float(np.percentile(bootstrap_maes, 2.5))
    ci_high = float(np.percentile(bootstrap_maes, 97.5))

    return {"mae": mae, "ci_low": ci_low, "ci_high": ci_high}


def crps_score(
    predictions: list[dict],
    truth_samples: np.ndarray,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """
    Compute CRPS of predicted Gaussian distributions vs empirical truth.

    Each prediction is a dict with keys: point, ci_low, ci_high.
    We infer a Gaussian: mean = point, std = (ci_high - ci_low) / 2.56
    (since 80% CI = +/- 1.28 sigma).

    Uses properscoring.crps_gaussian for the actual computation.

    Assumption documented: we treat the model's 80% CI as implying a
    Gaussian predictive distribution. This is a simplification — real
    predictive distributions may be skewed or heavy-tailed.

    Args:
        predictions: List of dicts with point, ci_low, ci_high.
        truth_samples: 1D array of Monte Carlo samples at this (variable, horizon).
        n_bootstrap: Number of bootstrap resamples for CI.
        rng_seed: Seed for bootstrap reproducibility.

    Returns:
        Dict with keys: crps, ci_low, ci_high.
    """
    n_preds = len(predictions)
    crps_values = np.zeros(n_preds)

    for i, pred in enumerate(predictions):
        mu = pred["point"]
        ci_width = pred["ci_high"] - pred["ci_low"]
        # Avoid division by zero: if CI width is 0, use a small std
        sigma = max(ci_width / CI_WIDTH_TO_STD, 0.01)

        # crps_gaussian computes CRPS for each observation against Gaussian(mu, sigma)
        # We average over all MC truth samples for this prediction
        scores = crps_gaussian(truth_samples, mu=mu, sig=sigma)
        crps_values[i] = float(np.mean(scores))

    crps_mean = float(np.mean(crps_values))

    # Bootstrap CI
    rng = np.random.default_rng(rng_seed)
    bootstrap_crps = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_preds, size=n_preds)
        bootstrap_crps[i] = np.mean(crps_values[sample_idx])

    ci_low = float(np.percentile(bootstrap_crps, 2.5))
    ci_high = float(np.percentile(bootstrap_crps, 97.5))

    return {"crps": crps_mean, "ci_low": ci_low, "ci_high": ci_high}


def score_baseline_naive(
    history_df: pd.DataFrame,
    truth_paths: np.ndarray,
    variable_index: int,
    horizons: list[int],
) -> dict:
    """
    Score the naive baseline (random walk) against MC truth.

    Naive prediction: last observed value for all horizons.
    Uncertainty: historical standard deviation of that variable.

    Args:
        history_df: Historical trajectory DataFrame.
        truth_paths: 3D array (n_paths, n_periods, 5) from Monte Carlo.
        variable_index: Index into the 5 variables [y, pi, r, e, u].
        horizons: List of horizon indices (0-based into truth_paths).

    Returns:
        Dict with mae and crps scores.
    """
    var_names = ["y", "pi", "r", "e", "u"]
    var_name = var_names[variable_index]

    last_value = float(history_df.iloc[-1][var_name])
    historical_std = float(history_df[var_name].std())

    # Point predictions: all the same (last value)
    predictions_points = [last_value] * len(horizons)

    # Truth means at each horizon
    truth_means = [float(np.mean(truth_paths[:, h, variable_index])) for h in horizons]

    mae_result = mae_score(predictions_points, truth_means)

    # CRPS: naive uses historical std as uncertainty
    pred_dicts = [
        {"point": last_value, "ci_low": last_value - Z_80 * historical_std,
         "ci_high": last_value + Z_80 * historical_std}
        for _ in horizons
    ]
    # For CRPS, use truth samples from each horizon
    crps_values = []
    for h_idx, h in enumerate(horizons):
        samples = truth_paths[:, h, variable_index]
        scores = crps_gaussian(samples, mu=last_value, sig=historical_std)
        crps_values.append(float(np.mean(scores)))

    crps_mean = float(np.mean(crps_values))

    return {
        "mae": mae_result,
        "crps": {"crps": crps_mean, "ci_low": None, "ci_high": None},
    }


def score_baseline_oracle(
    truth_paths: np.ndarray,
    variable_index: int,
    horizons: list[int],
    scoring_paths: np.ndarray,
) -> dict:
    """
    Score the oracle baseline against independent MC truth samples.

    Oracle prediction: MC mean and std from its own simulation.
    Scored against a separate set of MC samples.

    Args:
        truth_paths: MC paths used by oracle for its predictions.
        variable_index: Index into the 5 variables.
        horizons: List of horizon indices.
        scoring_paths: Independent MC paths for scoring.

    Returns:
        Dict with mae and crps scores.
    """
    # Oracle point predictions = mean of its MC paths
    predictions_points = [
        float(np.mean(truth_paths[:, h, variable_index])) for h in horizons
    ]

    # Scoring truth = mean of independent paths
    truth_means = [
        float(np.mean(scoring_paths[:, h, variable_index])) for h in horizons
    ]

    mae_result = mae_score(predictions_points, truth_means)

    # Oracle CRPS: uses MC std as its uncertainty estimate
    crps_values = []
    for h_idx, h in enumerate(horizons):
        mu = float(np.mean(truth_paths[:, h, variable_index]))
        sigma = float(np.std(truth_paths[:, h, variable_index]))
        sigma = max(sigma, 0.01)
        samples = scoring_paths[:, h, variable_index]
        scores = crps_gaussian(samples, mu=mu, sig=sigma)
        crps_values.append(float(np.mean(scores)))

    crps_mean = float(np.mean(crps_values))

    return {
        "mae": mae_result,
        "crps": {"crps": crps_mean, "ci_low": None, "ci_high": None},
    }
