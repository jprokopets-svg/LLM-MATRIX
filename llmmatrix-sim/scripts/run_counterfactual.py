"""
Run counterfactual Monte Carlo simulations for all four shock scenarios.

For each scenario:
  1. Load the canonical history from data/history.csv.
  2. Apply the shock at the end of history (period 60).
  3. Run 1000 forward paths of 12 quarters each.
  4. Save results as Parquet files in data/counterfactual_paths/.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.monte_carlo import run_counterfactual, summarize_paths
from src.shocks import (
    monetary_tightening_shock,
    demand_shock,
    cost_push_shock,
    exchange_rate_shock,
)


def main() -> None:
    config_path = os.path.join(project_root, "config", "ball_baseline.yaml")
    history_path = os.path.join(project_root, "data", "history.csv")
    output_dir = os.path.join(project_root, "data", "counterfactual_paths")

    # Load history
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found. Run generate_history.py first.")
        sys.exit(1)

    history_df = pd.read_csv(history_path)
    print(f"Loaded history: {len(history_df)} periods")

    os.makedirs(output_dir, exist_ok=True)

    # Define the four scenarios
    scenarios = {
        "monetary_tightening": lambda: monetary_tightening_shock(magnitude=2.0, period=60),
        "demand_shock": lambda: demand_shock(magnitude=2.0, period=60),
        "cost_push_shock": lambda: cost_push_shock(magnitude=2.0, period=60),
        "exchange_rate_shock": lambda: exchange_rate_shock(magnitude=5.0, period=60),
    }

    n_paths = 1000
    n_forward = 12

    for scenario_name, shock_fn in scenarios.items():
        print(f"\nRunning {scenario_name} ({n_paths} paths, {n_forward} quarters)...")

        paths = run_counterfactual(
            config_path=config_path,
            history_df=history_df,
            shock_fn=shock_fn,
            n_paths=n_paths,
            n_forward_periods=n_forward,
            base_seed=3000,
        )

        # Save the raw 3D array as a reshaped DataFrame in Parquet format
        # Reshape: (n_paths, n_forward, 5) -> flat DataFrame
        variable_names = ["y", "pi", "r", "e", "u"]
        rows = []
        for path_index in range(n_paths):
            for t_index in range(n_forward):
                row = {
                    "path": path_index,
                    "period": 61 + t_index,
                }
                for v_index, var_name in enumerate(variable_names):
                    row[var_name] = paths[path_index, t_index, v_index]
                rows.append(row)

        paths_df = pd.DataFrame(rows)
        parquet_path = os.path.join(output_dir, f"{scenario_name}.parquet")
        paths_df.to_parquet(parquet_path, index=False)
        print(f"  Saved to {parquet_path} ({len(paths_df)} rows)")

        # Also save a summary
        summary = summarize_paths(paths, start_period=60)
        summary_path = os.path.join(output_dir, f"{scenario_name}_summary.csv")
        summary.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"  Summary saved to {summary_path}")

    print("\nAll counterfactual simulations complete.")


if __name__ == "__main__":
    main()
