"""
Test that the system's transition matrix has all eigenvalues inside the unit circle.

A stable system means shocks die out over time rather than growing without bound.
This is a necessary condition for the simulator to produce meaningful dynamics.
"""

import os

from src.validation import check_stability


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "ball_baseline.yaml")


def test_all_eigenvalues_inside_unit_circle():
    """All eigenvalue moduli must be strictly less than 1."""
    result = check_stability(CONFIG_PATH)
    assert result["passed"], (
        f"System is unstable! Max eigenvalue modulus = {result['max_eigenvalue_modulus']:.4f}. "
        f"Eigenvalues: {result['eigenvalues']}"
    )


def test_max_eigenvalue_modulus_has_margin():
    """
    The largest eigenvalue modulus should be well below 1, not just barely stable.
    With interest-rate smoothing (rho=0.85), eigenvalues naturally sit closer to 1
    because the policy response is deliberately spread over time. A threshold of 0.98
    ensures stability while accommodating the persistent dynamics that smoothing creates.
    """
    result = check_stability(CONFIG_PATH)
    assert result["max_eigenvalue_modulus"] < 0.98, (
        f"System is barely stable — max eigenvalue modulus = {result['max_eigenvalue_modulus']:.4f}. "
        f"Consider adjusting parameters for more robust stability."
    )
