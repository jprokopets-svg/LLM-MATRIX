"""
Run all six validation checks and produce a report with diagnostic plots.

Output: data/validation_report.md with pass/fail results and embedded plot references.
Plots are saved to data/ as PNG files.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulator import Sim
from src.validation import (
    check_stability,
    check_bounded_trajectories,
    check_steady_state_recovery,
    check_realistic_volatility,
    check_impulse_response,
    check_ar1_suboptimality,
)


def plot_history(config_path: str, data_dir: str) -> str:
    """Generate and save the baseline history plot."""
    sim = Sim(config_path, seed=42)
    trajectory = sim.run(60)
    df = sim.to_dataframe(trajectory)

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    variables = [
        ("y", "Output Gap (%)", 0.0),
        ("pi", "Inflation Rate (%)", 2.0),
        ("r", "Real Interest Rate (%)", 2.0),
        ("e", "Real Exchange Rate (log)", 0.0),
        ("u", "Unemployment Rate (%)", 5.0),
    ]

    for ax, (var, label, target) in zip(axes, variables):
        ax.plot(df["period"], df[var], "b-", linewidth=1.0)
        ax.axhline(y=target, color="r", linestyle="--", alpha=0.5, label=f"Target = {target}")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Baseline History (60 Quarters, Seed=42)", fontsize=14)
    plt.tight_layout()

    path = os.path.join(data_dir, "history_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_impulse_responses(irf_result: dict, data_dir: str) -> str:
    """Generate and save impulse response plots."""
    df = irf_result["irf_df"]
    shock_period = irf_result["shock_period"]
    sim_targets = {"y": 0.0, "pi": 2.0, "r": 2.0, "e": 0.0, "u": 5.0}

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    variables = [
        ("y", "Output Gap (%)"),
        ("pi", "Inflation Rate (%)"),
        ("r", "Real Interest Rate (%)"),
        ("e", "Real Exchange Rate (log)"),
        ("u", "Unemployment Rate (%)"),
    ]

    for ax, (var, label) in zip(axes, variables):
        target = sim_targets[var]
        deviations = df[var] - target
        ax.plot(df["period"], deviations, "b-", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.axvline(x=shock_period, color="r", linestyle="--", alpha=0.5, label="Shock")
        ax.set_ylabel(f"{label}\n(deviation)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Quarter")
    fig.suptitle("Impulse Response: +1pp Monetary Tightening", fontsize=14)
    plt.tight_layout()

    path = os.path.join(data_dir, "impulse_responses.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fan_charts(config_path: str, data_dir: str) -> str:
    """Generate fan charts for each counterfactual scenario."""
    counterfactual_dir = os.path.join(data_dir, "counterfactual_paths")

    scenarios = [
        "monetary_tightening",
        "demand_shock",
        "cost_push_shock",
        "exchange_rate_shock",
    ]

    existing_scenarios = []
    for scenario_name in scenarios:
        summary_path = os.path.join(counterfactual_dir, f"{scenario_name}_summary.csv")
        if os.path.exists(summary_path):
            existing_scenarios.append(scenario_name)

    if not existing_scenarios:
        return ""

    variable_names = ["y", "pi", "r", "e", "u"]
    variable_labels = {
        "y": "Output Gap (%)",
        "pi": "Inflation (%)",
        "r": "Real Rate (%)",
        "e": "Exchange Rate",
        "u": "Unemployment (%)",
    }

    fig, axes = plt.subplots(
        len(variable_names),
        len(existing_scenarios),
        figsize=(5 * len(existing_scenarios), 3 * len(variable_names)),
        sharex=True,
    )

    # Handle case where we have only one scenario
    if len(existing_scenarios) == 1:
        axes = axes.reshape(-1, 1)

    for col, scenario_name in enumerate(existing_scenarios):
        summary_path = os.path.join(counterfactual_dir, f"{scenario_name}_summary.csv")
        summary_df = pd.read_csv(summary_path)

        for row, var_name in enumerate(variable_names):
            ax = axes[row, col]
            var_data = summary_df[summary_df["variable"] == var_name]

            periods = var_data["period"].values
            mean_vals = var_data["mean"].values
            p10 = var_data["p10"].values
            p25 = var_data["p25"].values
            p75 = var_data["p75"].values
            p90 = var_data["p90"].values

            ax.fill_between(periods, p10, p90, alpha=0.15, color="blue", label="10-90%")
            ax.fill_between(periods, p25, p75, alpha=0.3, color="blue", label="25-75%")
            ax.plot(periods, mean_vals, "b-", linewidth=1.5, label="Mean")

            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(scenario_name.replace("_", " ").title(), fontsize=10)
            if col == 0:
                ax.set_ylabel(variable_labels[var_name], fontsize=8)
            if row == len(variable_names) - 1:
                ax.set_xlabel("Quarter", fontsize=8)

    fig.suptitle("Monte Carlo Fan Charts (1000 paths)", fontsize=14)
    plt.tight_layout()

    path = os.path.join(data_dir, "fan_charts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(results: dict, plot_paths: dict, output_path: str) -> None:
    """Write the validation report as Markdown."""
    lines = []
    lines.append("# LLMMatrix Simulator Validation Report\n")

    all_passed = all(r["passed"] for r in results.values())
    status = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    lines.append(f"**Overall status: {status}**\n")

    # Check 1: Stability
    lines.append("## 1. Stability (Eigenvalue Check)\n")
    r = results["stability"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}** — "
                 f"Max eigenvalue modulus: {r['max_eigenvalue_modulus']:.4f}\n")
    lines.append("Eigenvalue moduli:\n")
    for i, mod in enumerate(r["moduli"]):
        lines.append(f"- λ_{i+1}: {mod:.4f}\n")

    # Check 2: Bounded Trajectories
    lines.append("\n## 2. Bounded Trajectories\n")
    r = results["bounded_trajectories"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}** — "
                 f"{r['n_violations']} violations found\n")
    lines.append("\nVariable ranges across 1000 simulations:\n")
    lines.append("| Variable | Min | Max |\n")
    lines.append("|----------|-----|-----|\n")
    for var, rng in r["variable_ranges"].items():
        lines.append(f"| {var} | {rng['min']:.2f} | {rng['max']:.2f} |\n")

    # Check 3: Steady-State Recovery
    lines.append("\n## 3. Steady-State Recovery\n")
    r = results["steady_state_recovery"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}**\n")
    lines.append("\nRecovery periods (quarters after shock):\n")
    lines.append("| Variable | Periods to Recover | Final Deviation |\n")
    lines.append("|----------|-------------------|------------------|\n")
    for var in ["y", "pi", "r", "e", "u"]:
        rp = r["recovery_periods"][var]
        fd = r["final_deviations"][var]
        rp_str = str(rp) if rp is not None else "DID NOT RECOVER"
        lines.append(f"| {var} | {rp_str} | {fd:.6f} |\n")

    # Check 4: Realistic Volatility
    lines.append("\n## 4. Realistic Volatility\n")
    r = results["realistic_volatility"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}**\n")
    lines.append("\n| Variable | Simulated Std | Empirical Std | Ratio |\n")
    lines.append("|----------|--------------|---------------|-------|\n")
    for var in ["y", "pi", "r", "e", "u"]:
        lines.append(
            f"| {var} | {r['simulated_stds'][var]:.3f} | "
            f"{r['empirical_stds'][var]:.3f} | {r['ratios'][var]:.2f} |\n"
        )

    # Check 5: Impulse Response
    lines.append("\n## 5. Impulse Response Sanity\n")
    r = results["impulse_response"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}**\n")
    for check in r["checks"]:
        symbol = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- [{symbol}] {check['name']}\n")

    if "impulse_responses" in plot_paths and plot_paths["impulse_responses"]:
        rel_path = os.path.basename(plot_paths["impulse_responses"])
        lines.append(f"\n![Impulse Responses]({rel_path})\n")

    # Check 6: AR(1) Suboptimality
    lines.append("\n## 6. AR(1) Suboptimality\n")
    r = results["ar1_suboptimality"]
    lines.append(f"**{'PASS' if r['passed'] else 'FAIL'}** — "
                 f"AR(1) worse on {r['ar1_worse_on_n_variables']}/5 variables\n")
    lines.append("\n| Variable | AR(1) MSE | Oracle MSE | AR(1) Excess (%) |\n")
    lines.append("|----------|-----------|------------|------------------|\n")
    for var in ["y", "pi", "r", "e", "u"]:
        lines.append(
            f"| {var} | {r['ar1_mse'][var]:.4f} | "
            f"{r['oracle_mse'][var]:.4f} | {r['improvement_pct'][var]:.1f}% |\n"
        )

    # Plots section
    lines.append("\n## Plots\n")
    if "history" in plot_paths and plot_paths["history"]:
        rel_path = os.path.basename(plot_paths["history"])
        lines.append(f"![Baseline History]({rel_path})\n")
    if "fan_charts" in plot_paths and plot_paths["fan_charts"]:
        rel_path = os.path.basename(plot_paths["fan_charts"])
        lines.append(f"\n![Fan Charts]({rel_path})\n")

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Report saved to {output_path}")


def main() -> None:
    config_path = os.path.join(project_root, "config", "ball_baseline.yaml")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    results = {}
    plot_paths = {}

    # Run all six checks
    print("=" * 60)
    print("LLMMatrix Simulator Validation")
    print("=" * 60)

    print("\nCheck 1/6: Stability (eigenvalues)...")
    results["stability"] = check_stability(config_path)
    print(f"  -> {'PASS' if results['stability']['passed'] else 'FAIL'}")
    print(f"     Max eigenvalue modulus: {results['stability']['max_eigenvalue_modulus']:.4f}")

    print("\nCheck 2/6: Bounded trajectories (1000 sims x 60 quarters)...")
    results["bounded_trajectories"] = check_bounded_trajectories(config_path)
    print(f"  -> {'PASS' if results['bounded_trajectories']['passed'] else 'FAIL'}")
    print(f"     Violations: {results['bounded_trajectories']['n_violations']}")

    print("\nCheck 3/6: Steady-state recovery...")
    results["steady_state_recovery"] = check_steady_state_recovery(config_path)
    print(f"  -> {'PASS' if results['steady_state_recovery']['passed'] else 'FAIL'}")

    print("\nCheck 4/6: Realistic volatility...")
    results["realistic_volatility"] = check_realistic_volatility(config_path)
    print(f"  -> {'PASS' if results['realistic_volatility']['passed'] else 'FAIL'}")
    for var, ratio in results["realistic_volatility"]["ratios"].items():
        print(f"     {var}: ratio = {ratio:.2f}")

    print("\nCheck 5/6: Impulse response sanity...")
    results["impulse_response"] = check_impulse_response(config_path)
    print(f"  -> {'PASS' if results['impulse_response']['passed'] else 'FAIL'}")
    for check in results["impulse_response"]["checks"]:
        symbol = "PASS" if check["passed"] else "FAIL"
        print(f"     [{symbol}] {check['name']}")

    print("\nCheck 6/6: AR(1) suboptimality...")
    results["ar1_suboptimality"] = check_ar1_suboptimality(config_path)
    print(f"  -> {'PASS' if results['ar1_suboptimality']['passed'] else 'FAIL'}")
    print(f"     AR(1) worse on {results['ar1_suboptimality']['ar1_worse_on_n_variables']}/5 variables")

    # Generate plots
    print("\nGenerating plots...")
    plot_paths["history"] = plot_history(config_path, data_dir)
    plot_paths["impulse_responses"] = plot_impulse_responses(results["impulse_response"], data_dir)
    plot_paths["fan_charts"] = plot_fan_charts(config_path, data_dir)

    # Generate report
    report_path = os.path.join(data_dir, "validation_report.md")
    generate_report(results, plot_paths, report_path)

    # Final summary
    all_passed = all(r["passed"] for r in results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL 6 CHECKS PASSED — simulator is validated.")
    else:
        failed = [name for name, r in results.items() if not r["passed"]]
        print(f"FAILED CHECKS: {', '.join(failed)}")
        print("Do NOT proceed to LLM scoring with a failing simulator.")
    print("=" * 60)


if __name__ == "__main__":
    main()
