"""
Generate the baseline 60-quarter history and save to data/history.csv.

Uses seed=42 for reproducibility. This is the canonical history
that all counterfactual scenarios branch from.
"""

import os
import sys

# Add project root to path so we can import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.simulator import Sim


def main() -> None:
    config_path = os.path.join(project_root, "config", "ball_baseline.yaml")
    output_path = os.path.join(project_root, "data", "history.csv")

    print("Generating 60-quarter baseline history (seed=42)...")

    sim = Sim(config_path, seed=42)
    trajectory = sim.run(n_periods=60)
    df = sim.to_dataframe(trajectory)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"History saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFinal state (period 60):")
    print(df.iloc[-1].to_string())


if __name__ == "__main__":
    main()
