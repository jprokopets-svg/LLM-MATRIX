"""
Validate that question expected_directions agree with simulator ground truth.

For each question:
  1. Run Monte Carlo with the question's specific shock type and magnitude.
  2. For each (variable, horizon), classify the simulator outcome as
     "up", "down", or "neutral" using a 0.5-std-dev threshold.
  3. Compare to the question's expected_direction.

Exit code 0 if agreement >= 80%, exit code 1 otherwise.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.questions import PILOT_QUESTIONS
from src.monte_carlo import run_counterfactual
from src.shocks import (
    monetary_tightening_shock,
    demand_shock,
    cost_push_shock,
    exchange_rate_shock,
)

HISTORY_PATH = os.path.join(project_root, "data", "history.csv")
SIM_CONFIG_PATH = os.path.join(project_root, "config", "ball_baseline.yaml")

VARIABLE_NAMES = ["y", "pi", "r", "e", "u"]
VARIABLE_INDEX = {name: i for i, name in enumerate(VARIABLE_NAMES)}

# Map shock_type to the shock function factory
SHOCK_FN_MAP = {
    "monetary": monetary_tightening_shock,
    "demand": demand_shock,
    "cost_push": cost_push_shock,
    "exchange_rate": exchange_rate_shock,
}

# Horizon in quarters -> 0-based index into the 12-period forward array
HORIZON_INDEX = {1: 0, 3: 2, 6: 5, 12: 11}


def classify_direction(
    mc_values: np.ndarray,
    baseline_values: np.ndarray,
    pre_shock_std: float,
) -> tuple[str, float, float]:
    """
    Classify the shock's effect as "up", "down", or "neutral" relative
    to a no-shock counterfactual baseline.

    The comparison is shock-mean vs no-shock-mean, not vs pre-shock level.
    This isolates the causal effect of the shock from natural mean-reversion.

    Uses a 0.5 standard deviation threshold.

    Args:
        mc_values: 1D array of MC path values WITH the shock applied.
        baseline_values: 1D array of MC path values WITHOUT any shock.
        pre_shock_std: Historical standard deviation of the variable.

    Returns:
        Tuple of (direction, shock_mean, baseline_mean).
    """
    shock_mean = float(np.mean(mc_values))
    baseline_mean = float(np.mean(baseline_values))
    threshold = 0.5 * pre_shock_std
    deviation = shock_mean - baseline_mean

    if deviation > threshold:
        direction = "up"
    elif deviation < -threshold:
        direction = "down"
    else:
        direction = "neutral"

    return direction, shock_mean, baseline_mean


def main() -> None:
    # Load history
    history_df = pd.read_csv(HISTORY_PATH)
    last_period = int(history_df.iloc[-1]["period"])

    # Compute pre-shock levels and historical std devs
    pre_shock = {}
    historical_std = {}
    for var_name in VARIABLE_NAMES:
        pre_shock[var_name] = float(history_df.iloc[-1][var_name])
        # Use std of all periods (excluding period 0 initial conditions)
        historical_std[var_name] = float(history_df[history_df["period"] > 0][var_name].std())

    print("Pre-shock levels and historical std devs:")
    print(f"  {'Var':<4} {'Level':>8} {'Std':>8}")
    for var_name in VARIABLE_NAMES:
        print(f"  {var_name:<4} {pre_shock[var_name]:>8.3f} {historical_std[var_name]:>8.3f}")
    print()

    # Generate no-shock baseline MC paths (same seeds, no shock applied)
    print("Generating no-shock baseline (1000 paths)...")
    no_shock_fn = lambda: {}  # empty overrides = no shock
    baseline_paths = run_counterfactual(
        config_path=SIM_CONFIG_PATH,
        history_df=history_df,
        shock_fn=no_shock_fn,
        n_paths=1000,
        n_forward_periods=12,
        base_seed=7000,  # same seed block as shocked paths for clean comparison
    )
    print()

    # Track results
    all_comparisons = []
    total_checks = 0
    total_agree = 0

    for question in PILOT_QUESTIONS:
        print(f"--- {question.id}: {question.shock_type} (magnitude={question.shock_magnitude}) ---")

        # Generate MC paths for this specific question
        shock_fn_factory = SHOCK_FN_MAP[question.shock_type]
        # Shock must target last_period + 1 (the first forward quarter),
        # because run_counterfactual starts from period=last_period and
        # the first step() produces period=last_period + 1.
        shock_fn = lambda mag=question.shock_magnitude: shock_fn_factory(
            magnitude=mag, period=last_period + 1,
        )

        paths = run_counterfactual(
            config_path=SIM_CONFIG_PATH,
            history_df=history_df,
            shock_fn=shock_fn,
            n_paths=1000,
            n_forward_periods=12,
            base_seed=7000,
        )

        # Check each (variable, horizon) pair
        for var_name in question.target_variables:
            v_idx = VARIABLE_INDEX[var_name]
            for horizon in question.target_horizons:
                h_idx = HORIZON_INDEX[horizon]
                key = f"{var_name}_{horizon}"

                mc_values = paths[:, h_idx, v_idx]
                baseline_values = baseline_paths[:, h_idx, v_idx]

                actual_dir, shock_mean, baseline_mean = classify_direction(
                    mc_values, baseline_values, historical_std[var_name],
                )

                expected_dir = question.expected_direction.get(key, "???")
                agrees = actual_dir == expected_dir

                deviation = shock_mean - baseline_mean
                threshold = 0.5 * historical_std[var_name]

                total_checks += 1
                if agrees:
                    total_agree += 1

                all_comparisons.append({
                    "question": question.id,
                    "key": key,
                    "expected": expected_dir,
                    "actual": actual_dir,
                    "agrees": agrees,
                    "shock_mean": shock_mean,
                    "baseline_mean": baseline_mean,
                    "deviation": deviation,
                    "threshold": threshold,
                })

    # Print results table
    print()
    print("=" * 100)
    print(f"{'Q':<5} {'Key':<8} {'Expected':<10} {'Actual':<10} {'Match':<7} "
          f"{'ShockMn':>9} {'BaseMn':>9} {'Dev':>8} {'Thresh':>8}")
    print("-" * 100)

    disagreements = []
    for c in all_comparisons:
        match_str = "OK" if c["agrees"] else "MISMATCH"
        print(
            f"{c['question']:<5} {c['key']:<8} {c['expected']:<10} {c['actual']:<10} "
            f"{match_str:<7} {c['shock_mean']:>9.3f} {c['baseline_mean']:>9.3f} "
            f"{c['deviation']:>8.3f} {c['threshold']:>8.3f}"
        )
        if not c["agrees"]:
            disagreements.append(c)

    # Summary
    agreement_rate = total_agree / total_checks if total_checks > 0 else 0
    print()
    print("=" * 90)
    print(f"Total checks: {total_checks}")
    print(f"Agreements:   {total_agree}")
    print(f"Disagreements: {len(disagreements)}")
    print(f"Agreement rate: {agreement_rate:.1%}")

    if disagreements:
        print()
        print("DISAGREEMENTS:")
        for d in disagreements:
            print(
                f"  {d['question']} {d['key']}: "
                f"expected={d['expected']}, actual={d['actual']} "
                f"(shock-baseline={d['deviation']:+.3f}, threshold=+/-{d['threshold']:.3f})"
            )

    if agreement_rate >= 0.80:
        print(f"\nPASS: agreement rate {agreement_rate:.1%} >= 80%")
        sys.exit(0)
    else:
        print(f"\nFAIL: agreement rate {agreement_rate:.1%} < 80%")
        sys.exit(1)


if __name__ == "__main__":
    main()
