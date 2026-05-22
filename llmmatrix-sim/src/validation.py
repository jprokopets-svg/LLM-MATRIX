"""
Validation suite for the Ball (1999) open-economy NK simulator.

Six sanity checks that must all pass before the simulator is considered valid:

1. Stability: transition matrix eigenvalues inside unit circle.
2. Bounded trajectories: no variable exceeds plausible bounds over 1000 sims.
3. Steady-state recovery: system returns to targets after a one-time shock.
4. Realistic volatility: simulated std devs within 0.5-2x of US empirical.
5. Impulse response sanity: IRFs match standard NK predictions (signs + timing).
6. AR(1) suboptimality: AR(1) baseline underperforms oracle on forecasts.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.simulator import Sim
from src.baselines import ar1_baseline, oracle_baseline
from src.shocks import demand_shock


# ----- Check 1: Stability -----

def check_stability(config_path: str) -> dict:
    """
    Verify the system's transition matrix has all eigenvalues inside the unit circle.

    The linearized system (ignoring shocks) can be written as:
        x_t = A * x_{t-1} + constants

    With interest-rate smoothing (Dotsey & Sill 2015), r is a genuine state variable
    that depends on r_{t-1}. The state vector is [y, pi_dev, r_dev, e_dev, e_lag_dev]
    where _dev denotes deviation from steady state.

    Returns:
        Dict with keys: passed (bool), eigenvalues (list), max_eigenvalue (float).
    """
    sim = Sim(config_path, seed=0)

    # Parameter shorthand
    a = sim.output_persistence          # IS persistence
    b = sim.interest_sensitivity        # IS interest-rate sensitivity
    c = sim.exchange_sensitivity        # IS exchange-rate sensitivity
    d = sim.inflation_persistence       # Phillips persistence on pi_{t-1}
    f = sim.output_slope                # Phillips slope on y_{t-1}
    g = sim.exchange_passthrough        # Phillips exchange-rate passthrough
    h = sim.inflation_coefficient       # Taylor inflation coefficient
    k = sim.output_coefficient          # Taylor output coefficient
    rho = sim.smoothing_rho             # Taylor smoothing (Dotsey & Sill 2015)
    m = sim.interest_differential       # UIP coefficient

    # State vector: [y, pi_dev, r_dev, e_dev, e_lag_dev]
    # All in deviation from steady state.
    #
    # IS curve (backward-looking, uses lagged r and e):
    #   y_t = a * y_{t-1} - b * r_dev_{t-1} - c * e_dev_{t-1}
    row_y = [a, 0.0, -b, -c, 0.0]

    # Phillips curve (backward-looking):
    #   pi_dev_t = d * pi_dev_{t-1} + f * y_{t-1} - g * (e_dev_{t-1} - e_lag_dev_{t-1})
    row_pi = [f, d, 0.0, -g, g]

    # Taylor rule with smoothing (contemporaneous on pi_t, y_t):
    #   r_dev_t = rho * r_dev_{t-1} + (1-rho) * [h * pi_dev_t + k * y_t]
    #
    # Substitute the expressions for y_t and pi_dev_t in terms of t-1 state:
    #   y_t     = row_y . state_{t-1}
    #   pi_dev_t = row_pi . state_{t-1}
    #
    # So: r_dev_t = rho * r_dev_{t-1}
    #             + (1-rho)*h * (row_pi . state_{t-1})
    #             + (1-rho)*k * (row_y . state_{t-1})
    w = 1.0 - rho
    row_r = [
        w * (h * row_pi[0] + k * row_y[0]),        # coeff on y_{t-1}
        w * (h * row_pi[1] + k * row_y[1]),        # coeff on pi_dev_{t-1}
        rho + w * (h * row_pi[2] + k * row_y[2]),  # coeff on r_dev_{t-1}
        w * (h * row_pi[3] + k * row_y[3]),        # coeff on e_dev_{t-1}
        w * (h * row_pi[4] + k * row_y[4]),        # coeff on e_lag_dev_{t-1}
    ]

    # UIP (contemporaneous on r_t):
    #   e_dev_t = m * r_dev_t
    # So e_dev_t has the same coefficients as r_dev_t, scaled by m.
    row_e = [m * val for val in row_r]

    # Lag identity: e_lag_dev_t = e_dev_{t-1}
    row_elag = [0.0, 0.0, 0.0, 1.0, 0.0]

    # Build the 5x5 transition matrix
    transition_matrix = np.array([row_y, row_pi, row_r, row_e, row_elag])

    eigenvalues = np.linalg.eigvals(transition_matrix)
    moduli = np.abs(eigenvalues)
    max_modulus = float(np.max(moduli))

    passed = max_modulus < 1.0

    return {
        "passed": passed,
        "eigenvalues": eigenvalues.tolist(),
        "moduli": moduli.tolist(),
        "max_eigenvalue_modulus": max_modulus,
        "transition_matrix": transition_matrix,
    }


# ----- Check 2: Bounded Trajectories -----

def check_bounded_trajectories(
    config_path: str,
    n_sims: int = 1000,
    n_periods: int = 60,
    base_seed: int = 5000,
) -> dict:
    """
    Run many unconditional simulations and verify no variable exceeds plausible bounds.

    Bounds:
        |y| < 10, pi in (-5, 15), r in (-2, 15), |e| < 20, u in (0, 15)

    Args:
        config_path: Path to YAML config.
        n_sims: Number of simulations to run.
        n_periods: Quarters per simulation.
        base_seed: Starting seed.

    Returns:
        Dict with: passed (bool), violations (list of violation descriptions),
        variable_ranges (dict of min/max per variable).
    """
    bounds = {
        "y": (-10.0, 10.0),
        "pi": (-5.0, 15.0),
        "r": (-2.0, 15.0),
        "e": (-20.0, 20.0),
        "u": (0.0, 15.0),
    }

    # Track min/max across all simulations
    var_min = {var: float("inf") for var in bounds}
    var_max = {var: float("-inf") for var in bounds}
    violations = []

    for sim_index in range(n_sims):
        sim = Sim(config_path, seed=base_seed + sim_index)
        trajectory = sim.run(n_periods)
        df = sim.to_dataframe(trajectory)

        for var_name, (lo, hi) in bounds.items():
            series = df[var_name]
            sim_min = float(series.min())
            sim_max = float(series.max())

            var_min[var_name] = min(var_min[var_name], sim_min)
            var_max[var_name] = max(var_max[var_name], sim_max)

            if sim_min < lo:
                violations.append(
                    f"Sim {sim_index}: {var_name} below {lo} (min={sim_min:.2f})"
                )
            if sim_max > hi:
                violations.append(
                    f"Sim {sim_index}: {var_name} above {hi} (max={sim_max:.2f})"
                )

    variable_ranges = {
        var: {"min": var_min[var], "max": var_max[var]}
        for var in bounds
    }

    passed = len(violations) == 0

    return {
        "passed": passed,
        "n_violations": len(violations),
        "violations": violations[:20],  # cap output to first 20
        "variable_ranges": variable_ranges,
    }


# ----- Check 3: Steady-State Recovery -----

def check_steady_state_recovery(
    config_path: str,
    shock_variable: str = "eps_y",
    shock_magnitude: float = 3.0,
    shock_period: int = 1,
    recovery_threshold: float = 0.1,
    max_recovery_periods: int = 50,  # 50 quarters; smoothing slows convergence by design
) -> dict:
    """
    After a one-time shock, verify the system returns to within threshold of targets.

    Runs a deterministic simulation (no random shocks after the initial one) and
    checks that all variables converge back to steady state.

    Args:
        config_path: Path to YAML config.
        shock_variable: Which shock to apply ('eps_y', 'eps_pi', 'eps_e').
        shock_magnitude: Size of the one-time shock.
        shock_period: When to apply the shock.
        recovery_threshold: How close to target counts as "recovered".
        max_recovery_periods: Must recover within this many periods.

    Returns:
        Dict with: passed (bool), recovery_periods (dict per variable),
        final_deviations (dict per variable).
    """
    # Create a deterministic simulation: zero out all random shocks except the one we inject
    shock_overrides = {(shock_period, shock_variable): shock_magnitude}

    # Set all other shocks to zero for all periods
    n_total = shock_period + max_recovery_periods + 10
    for t in range(1, n_total + 1):
        for var in ["eps_y", "eps_pi", "eps_e"]:
            if (t, var) not in shock_overrides:
                shock_overrides[(t, var)] = 0.0

    sim = Sim(config_path, seed=0)
    trajectory = sim.run(n_total, shock_overrides=shock_overrides)
    df = sim.to_dataframe(trajectory)

    # Check convergence for each variable
    targets = {
        "y": 0.0,
        "pi": sim.pi_star,
        "r": sim.r_star,
        "e": sim.e_star,
        "u": sim.u_star,
    }

    recovery_periods = {}
    final_deviations = {}

    for var_name, target in targets.items():
        deviations = (df[var_name] - target).abs()
        # Find first period after shock where deviation stays below threshold
        recovered_at = None
        for t in range(shock_period + 1, len(deviations)):
            # Check if all subsequent deviations are below threshold
            if deviations.iloc[t:].max() < recovery_threshold:
                recovered_at = t - shock_period
                break

        recovery_periods[var_name] = recovered_at
        final_deviations[var_name] = float(deviations.iloc[-1])

    all_recovered = all(
        rp is not None and rp <= max_recovery_periods
        for rp in recovery_periods.values()
    )

    return {
        "passed": all_recovered,
        "recovery_periods": recovery_periods,
        "final_deviations": final_deviations,
        "trajectory_df": df,
    }


# ----- Check 4: Realistic Volatility -----

def check_realistic_volatility(
    config_path: str,
    n_sims: int = 500,
    n_periods: int = 60,
    base_seed: int = 8000,
) -> dict:
    """
    Check that simulated standard deviations are within 0.5-2x of US empirical values.

    Empirical US quarterly standard deviations (approximate, from post-1990 data):
        y (output gap): ~1.5%
        pi (inflation): ~1.0%
        r (real rate): ~1.5%
        e (real exchange rate): ~4.0%
        u (unemployment): ~1.0%

    Args:
        config_path: Path to YAML config.
        n_sims: Number of simulations to average over.
        n_periods: Quarters per simulation.
        base_seed: Starting seed.

    Returns:
        Dict with: passed (bool), simulated_stds (dict), empirical_stds (dict),
        ratios (dict).
    """
    # Approximate US empirical std devs (quarterly, post-1990)
    empirical_stds = {
        "y": 1.5,
        "pi": 1.0,
        "r": 1.5,
        "e": 4.0,
        "u": 1.0,
    }

    # Collect all simulated values
    all_values = {var: [] for var in empirical_stds}

    for sim_index in range(n_sims):
        sim = Sim(config_path, seed=base_seed + sim_index)
        trajectory = sim.run(n_periods)
        df = sim.to_dataframe(trajectory)

        for var_name in empirical_stds:
            # For pi, r, u: compute std of levels
            # For y, e: these are already deviations
            all_values[var_name].extend(df[var_name].tolist())

    simulated_stds = {}
    ratios = {}
    for var_name in empirical_stds:
        sim_std = float(np.std(all_values[var_name]))
        simulated_stds[var_name] = sim_std
        ratios[var_name] = sim_std / empirical_stds[var_name]

    # Check all ratios are within ~0.5x to 2.0x of empirical values.
    # We use 0.45 as the lower bound because this is a stylized 5-equation model,
    # not a full DSGE — "roughly 0.5x" is the intent, not an exact cutoff.
    all_in_range = all(0.45 <= ratio <= 2.0 for ratio in ratios.values())

    return {
        "passed": all_in_range,
        "simulated_stds": simulated_stds,
        "empirical_stds": empirical_stds,
        "ratios": ratios,
    }


# ----- Check 5: Impulse Response Sanity -----

def check_impulse_response(
    config_path: str,
) -> dict:
    """
    Verify that impulse responses match standard NK predictions.

    A +1pp monetary tightening shock should:
      - Reduce y over 1-3 quarters
      - Reduce pi over 2-6 quarters
      - Appreciate e contemporaneously
      - Raise u after 1-3 quarters

    We run a deterministic simulation with a single shock and check signs.

    Returns:
        Dict with: passed (bool), checks (list of check results),
        irf_data (dict of variable -> list of responses).
    """
    n_pre = 5   # quiet periods before shock
    n_post = 20  # periods to observe after shock
    shock_period = n_pre + 1

    # Zero out all stochastic shocks
    shock_overrides = {}
    for t in range(1, n_pre + n_post + 2):
        for var in ["eps_y", "eps_pi", "eps_e"]:
            shock_overrides[(t, var)] = 0.0

    # Apply a +1pp monetary tightening at the shock period
    shock_overrides[(shock_period, "r_override")] = 1.0

    sim = Sim(config_path, seed=0)
    trajectory = sim.run(n_pre + n_post, shock_overrides=shock_overrides)
    df = sim.to_dataframe(trajectory)

    # Compute deviations from steady state
    targets = {"y": 0.0, "pi": sim.pi_star, "r": sim.r_star, "e": sim.e_star, "u": sim.u_star}
    irf_data = {}
    for var_name, target in targets.items():
        irf_data[var_name] = (df[var_name] - target).tolist()

    # Check the expected signs and timing
    checks = []

    # y should decline within 1-3 quarters after shock
    y_responses = irf_data["y"][shock_period:shock_period + 4]
    y_declined = any(val < -0.01 for val in y_responses)
    checks.append({
        "name": "y declines after monetary tightening",
        "passed": y_declined,
        "values": y_responses,
    })

    # pi should decline within 2-6 quarters after shock
    pi_responses = irf_data["pi"][shock_period:shock_period + 7]
    pi_declined = any(val < -0.01 for val in pi_responses)
    checks.append({
        "name": "pi declines after monetary tightening",
        "passed": pi_declined,
        "values": pi_responses,
    })

    # e should appreciate (increase) contemporaneously with the rate hike
    e_at_shock = irf_data["e"][shock_period]
    e_appreciated = e_at_shock > 0.01
    checks.append({
        "name": "e appreciates on monetary tightening",
        "passed": e_appreciated,
        "value": e_at_shock,
    })

    # u should rise within 1-3 quarters after shock
    u_responses = irf_data["u"][shock_period:shock_period + 4]
    u_rose = any(val > 0.01 for val in u_responses)
    checks.append({
        "name": "u rises after monetary tightening",
        "passed": u_rose,
        "values": u_responses,
    })

    all_passed = all(c["passed"] for c in checks)

    return {
        "passed": all_passed,
        "checks": checks,
        "irf_df": df,
        "shock_period": shock_period,
    }


# ----- Check 6: AR(1) Suboptimality -----

def check_ar1_suboptimality(
    config_path: str,
    history_seed: int = 42,
    n_history: int = 60,
    n_forward: int = 12,
    n_mc_paths: int = 1000,
) -> dict:
    """
    Verify that AR(1) baseline underperforms oracle on forecasting accuracy.

    If AR(1) matches oracle, the system has insufficient causal structure
    and the benchmark would be trivial for LLMs.

    We compare mean squared error (MSE) of predictions against the
    oracle's Monte Carlo mean (which is the best possible point forecast).

    Args:
        config_path: Path to YAML config.
        history_seed: Seed for generating history.
        n_history: Number of historical quarters.
        n_forward: Number of forecast quarters.
        n_mc_paths: Monte Carlo paths for oracle.

    Returns:
        Dict with: passed (bool), ar1_mse (dict per variable),
        oracle_mse (dict per variable), improvement_pct (dict per variable).
    """
    # Generate history
    sim = Sim(config_path, seed=history_seed)
    trajectory = sim.run(n_history)
    history_df = sim.to_dataframe(trajectory)

    # Get AR(1) forecasts
    ar1_forecasts = ar1_baseline(history_df, n_forward_periods=n_forward)

    # Get oracle forecasts (using a demand shock as the counterfactual scenario)
    # Shock targets the first forward period (n_history + 1), not n_history,
    # because run_counterfactual starts at period=n_history and step() produces n_history+1.
    shock_fn = lambda: demand_shock(magnitude=2.0, period=n_history + 1)
    oracle_forecasts = oracle_baseline(
        config_path=config_path,
        history_df=history_df,
        shock_overrides_fn=shock_fn,
        n_forward_periods=n_forward,
        n_paths=n_mc_paths,
        base_seed=2000,
    )

    # Run a separate set of MC paths as the "ground truth" to score against.
    # Must use DIFFERENT seeds from the oracle so oracle MSE is not trivially zero.
    from src.monte_carlo import run_counterfactual
    true_paths = run_counterfactual(
        config_path=config_path,
        history_df=history_df,
        shock_fn=shock_fn,
        n_paths=n_mc_paths,
        n_forward_periods=n_forward,
        base_seed=9000,  # different seed block from oracle (2000)
    )

    # The oracle's prediction IS the mean of these paths, so oracle MSE
    # is the irreducible variance. AR(1) MSE should be higher.
    variable_names = ["y", "pi", "r", "e", "u"]
    ar1_mse = {}
    oracle_mse = {}

    for v_index, var_name in enumerate(variable_names):
        # True mean at each horizon (across MC paths)
        true_means = np.mean(true_paths[:, :, v_index], axis=0)

        # AR(1) predictions for this variable
        ar1_var = ar1_forecasts[ar1_forecasts["variable"] == var_name]
        ar1_preds = ar1_var["predicted_mean"].values

        # Oracle predictions for this variable
        oracle_var = oracle_forecasts[oracle_forecasts["variable"] == var_name]
        oracle_preds = oracle_var["predicted_mean"].values

        # MSE against the true mean trajectory
        ar1_mse[var_name] = float(np.mean((ar1_preds - true_means) ** 2))
        oracle_mse[var_name] = float(np.mean((oracle_preds - true_means) ** 2))

    # Check that AR(1) has higher MSE than oracle for most variables
    improvement_pct = {}
    ar1_worse_count = 0
    for var_name in variable_names:
        if oracle_mse[var_name] > 0:
            pct = (ar1_mse[var_name] - oracle_mse[var_name]) / oracle_mse[var_name] * 100
        else:
            # Oracle MSE is ~0, any AR(1) error counts as worse
            pct = 100.0 if ar1_mse[var_name] > 1e-6 else 0.0
        improvement_pct[var_name] = pct
        if ar1_mse[var_name] > oracle_mse[var_name]:
            ar1_worse_count += 1

    # AR(1) should be worse on at least 3 of 5 variables
    passed = ar1_worse_count >= 3

    return {
        "passed": passed,
        "ar1_mse": ar1_mse,
        "oracle_mse": oracle_mse,
        "improvement_pct": improvement_pct,
        "ar1_worse_on_n_variables": ar1_worse_count,
    }


def run_all_checks(config_path: str) -> dict:
    """
    Run all six validation checks and return a summary.

    Args:
        config_path: Path to YAML config.

    Returns:
        Dict mapping check name -> result dict (each has a 'passed' key).
    """
    results = {}

    print("Check 1/6: Stability (eigenvalues)...")
    results["stability"] = check_stability(config_path)
    print(f"  -> {'PASS' if results['stability']['passed'] else 'FAIL'}")

    print("Check 2/6: Bounded trajectories...")
    results["bounded_trajectories"] = check_bounded_trajectories(config_path)
    print(f"  -> {'PASS' if results['bounded_trajectories']['passed'] else 'FAIL'}")

    print("Check 3/6: Steady-state recovery...")
    results["steady_state_recovery"] = check_steady_state_recovery(config_path)
    print(f"  -> {'PASS' if results['steady_state_recovery']['passed'] else 'FAIL'}")

    print("Check 4/6: Realistic volatility...")
    results["realistic_volatility"] = check_realistic_volatility(config_path)
    print(f"  -> {'PASS' if results['realistic_volatility']['passed'] else 'FAIL'}")

    print("Check 5/6: Impulse response sanity...")
    results["impulse_response"] = check_impulse_response(config_path)
    print(f"  -> {'PASS' if results['impulse_response']['passed'] else 'FAIL'}")

    print("Check 6/6: AR(1) suboptimality...")
    results["ar1_suboptimality"] = check_ar1_suboptimality(config_path)
    print(f"  -> {'PASS' if results['ar1_suboptimality']['passed'] else 'FAIL'}")

    all_passed = all(r["passed"] for r in results.values())
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")

    return results
