# LLMMatrix Simulator

Synthetic open-economy macroeconomic simulator for the LLMMatrix benchmark. Produces internally consistent dynamics from a five-equation backward-looking New Keynesian model. The simulator is a **measurement instrument** -- its job is to produce known-by-construction dynamics for evaluating LLM forecasting capability, not to forecast the real world.

## Model

Five-equation backward-looking open-economy NK system:

| Equation | Source |
|----------|--------|
| IS curve (output gap) | Ball (1999) |
| Phillips curve (inflation) | Ball (1999); Gali (2008); Stock & Watson (1999) |
| Taylor rule with interest-rate smoothing | Taylor (1993); Dotsey & Sill (2015); FRB/US model |
| UIP condition (exchange rate) | Standard small open economy calibration |
| Okun's law (unemployment) | Okun (1962) |

### Phillips curve

Inflation persistence is calibrated to 0.95 (deviation-from-target form):

```
(pi_t - pi_star) = 0.95 * (pi_{t-1} - pi_star) + 0.4 * y_{t-1} - 0.2 * (e_{t-1} - e_{t-2}) + eps_pi
```

Modern empirical Phillips curve estimates consistently find inflation persistence below 1.0 -- typically 0.85-0.95 in US post-Volcker data (Gali 2008; Stock & Watson 1999). The original unit-root specification (persistence = 1.0) is the Carlin-Soskice teaching version; modern empirical work has moved away from it. Using 0.95 ensures inflation naturally reverts to target rather than drifting.

### Interest-rate smoothing

The Taylor rule includes a smoothing parameter rho = 0.85:

```
r_t = rho * r_{t-1} + (1 - rho) * [r_star + 1.5*(pi_t - pi_star) + 2.0*y_t]
```

This follows Dotsey & Sill (2015) and the Federal Reserve's FRB/US model, which estimates rho around 0.85. Interest-rate smoothing is standard in modern central bank practice and estimated DSGE models -- the central bank adjusts rates gradually rather than jumping to the Taylor-rule prescription each quarter.

With smoothing, the output gap coefficient is raised to k = 2.0 (from Taylor's original 0.5) to ensure stability of the backward-looking system. This compensates for the reduced per-period policy response: effective within-quarter output response is (1 - rho) * k = 0.3. Estimated Taylor rules with high smoothing routinely find elevated output coefficients (Clarida, Gali & Gertler 2000; Orphanides 2003). Note: k = 0.5 is mathematically incompatible with rho = 0.85 in this backward-looking model -- the binding constraint is the IS-Taylor-UIP feedback loop, not the Phillips curve.

## Variables

| Symbol | Variable | Units | Target |
|--------|----------|-------|--------|
| y | Output gap | % deviation from potential | 0 |
| pi | Inflation rate | annual % | 2 |
| r | Real interest rate | annual % | 2 |
| e | Real exchange rate | log index (higher = appreciation) | 0 |
| u | Unemployment rate | % | 5 |

## Parameter citations

- **IS curve persistence (0.8)** -- Ball (1999); standard AR(1) for output gap.
- **IS interest-rate sensitivity (0.6)** -- Ball (1999) original calibration.
- **IS exchange-rate sensitivity (0.2)** -- Ball (1999) open-economy extension.
- **Phillips inflation persistence (0.95)** -- Gali (2008); Stock & Watson (1999); within standard post-Volcker empirical range (0.85-0.95).
- **Phillips slope (0.4)** -- Ball (1999) for US-like economies.
- **Phillips exchange-rate passthrough (0.2)** -- McCarthy (1999, 2000).
- **Taylor inflation coefficient (1.5)** -- Taylor (1993); satisfies the Taylor principle.
- **Taylor output coefficient (2.0)** -- elevated from Taylor (1993) original 0.5 to compensate for smoothing in backward-looking model; consistent with Clarida, Gali & Gertler (2000) estimates.
- **Taylor smoothing rho (0.85)** -- Dotsey & Sill (2015); FRB/US model calibration.
- **UIP coefficient (2.0)** -- standard small open economy calibration.
- **Okun coefficient (0.5)** -- Okun (1962); long-standing US empirical estimate.

### Calibration notes

This calibration satisfies four constraints: backward-looking dynamics (Carlin-Soskice 2005), open-economy UIP channel (Ball 1999), interest-rate smoothing per modern central bank practice (Dotsey & Sill 2015), and empirically-grounded Phillips persistence (Gali 2008; Stock & Watson 1999). These constraints jointly require the Taylor output coefficient k=2.0; standard k=0.5 (Taylor 1993) produces unstable systems under any empirically defensible parameter set in this model class. Calibration sensitivity analysis is available in data/validation_report.md.

## Quickstart

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Generate history, counterfactuals, and validation report
python scripts/generate_history.py
python scripts/run_counterfactual.py
python scripts/validate_sim.py
```

## Validation

Six sanity checks must all pass:

1. **Stability** -- transition matrix eigenvalues inside unit circle (max modulus: 0.93).
2. **Bounded trajectories** -- no variable exceeds plausible bounds over 1000 simulations.
3. **Steady-state recovery** -- system returns to targets within 50 quarters after a one-time shock.
4. **Realistic volatility** -- simulated std devs within 0.45-2.0x of US empirical values.
5. **Impulse response sanity** -- IRFs match standard NK predictions (signs and timing).
6. **AR(1) suboptimality** -- AR(1) baseline underperforms oracle by >3300% MSE on all variables.

## Environment variables

No API keys required for the simulator. All parameters are in `config/ball_baseline.yaml`.

## File structure

```
llmmatrix-sim/
├── config/ball_baseline.yaml    # All model parameters
├── src/
│   ├── simulator.py             # Core Sim class
│   ├── shocks.py                # Counterfactual shock scenarios
│   ├── monte_carlo.py           # MC forward simulation
│   ├── baselines.py             # Naive, AR(1), Oracle forecasters
│   └── validation.py            # Six sanity checks
├── tests/                       # pytest test suite
├── scripts/                     # CLI scripts
├── data/                        # Generated outputs
└── requirements.txt
```

## References

- Ball, L. (1999). Policy rules for open economies. In J.B. Taylor (Ed.), *Monetary Policy Rules*. University of Chicago Press.
- Clarida, R., Gali, J., & Gertler, M. (2000). Monetary policy rules and macroeconomic stability: Evidence and some theory. *Quarterly Journal of Economics*, 115(1), 147-180.
- Dotsey, M. & Sill, K. (2015). Interest-rate rules and macroeconomic stability. Federal Reserve Bank of Philadelphia.
- Gali, J. (2008). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press.
- McCarthy, J. (1999). Pass-through of exchange rates and import prices to domestic inflation in some industrialized economies. Federal Reserve Bank of New York Staff Reports.
- Okun, A.M. (1962). Potential GNP: Its measurement and significance. *Proceedings of the Business and Economic Statistics Section*, American Statistical Association.
- Orphanides, A. (2003). Historical monetary policy analysis and the Taylor rule. *Journal of Monetary Economics*, 50(5), 983-1022.
- Stock, J.H. & Watson, M.W. (1999). Forecasting inflation. *Journal of Monetary Economics*, 44(2), 293-335.
- Taylor, J.B. (1993). Discretion versus policy rules in practice. *Carnegie-Rochester Conference Series on Public Policy*, 39, 195-214.
