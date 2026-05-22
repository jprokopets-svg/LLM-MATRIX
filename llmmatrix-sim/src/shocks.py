"""
Counterfactual shock scenarios for the LLMMatrix simulator.

Each function returns a dict mapping (period, variable) -> shock_value,
which can be passed directly to Sim.run(shock_overrides=...).

Four scenarios for the prototype:
  1. Monetary tightening (discretionary rate hike above Taylor rule)
  2. Demand shock (positive eps_y)
  3. Cost-push shock (positive eps_pi)
  4. Exchange rate shock (positive eps_e = depreciation pressure)
"""


def monetary_tightening_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Discretionary monetary tightening: the central bank raises the real rate
    by `magnitude` percentage points above what the Taylor rule prescribes.

    This overrides r_t at the shock period. The Taylor rule still operates
    in subsequent periods, so the tightening is a one-time deviation.

    Args:
        magnitude: Percentage points above Taylor-rule rate.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "r_override"): magnitude}


def demand_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Positive demand shock: an unexpected boost to output gap.

    Sets eps_y to `magnitude` at the shock period. Could represent
    a fiscal stimulus, a surge in consumer confidence, etc.

    Args:
        magnitude: Size of the demand shock in percentage points of output gap.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_y"): magnitude}


def cost_push_shock(
    magnitude: float = 2.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Cost-push shock: an unexpected increase in inflation not driven by demand.

    Sets eps_pi to `magnitude` at the shock period. Could represent
    an oil price spike, supply chain disruption, etc.

    Args:
        magnitude: Size of the cost-push shock in percentage points of inflation.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_pi"): magnitude}


def exchange_rate_shock(
    magnitude: float = 5.0,
    period: int = 60,
) -> dict[tuple[int, str], float]:
    """
    Exchange rate shock: an unexpected appreciation of the real exchange rate.

    Sets eps_e to `magnitude` at the shock period. Could represent
    a capital flow surge, terms-of-trade shift, etc. Positive = appreciation.

    Args:
        magnitude: Size of the exchange rate shock in log-index units.
        period: Quarter in which the shock hits.

    Returns:
        Shock override dict for Sim.run().
    """
    return {(period, "eps_e"): magnitude}
