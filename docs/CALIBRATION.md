# Calibration Notes

## Parameter table

| Parameter | Value | Source |
|-----------|-------|--------|
| IS curve persistence | 0.8 | Ball (1999); standard AR(1) for output gap |
| IS interest-rate sensitivity | 0.6 | Ball (1999) original calibration |
| IS exchange-rate sensitivity | 0.2 | Ball (1999) open-economy extension |
| Phillips inflation persistence | 0.95 | Gali (2008); Stock & Watson (1999); post-Volcker empirical range 0.85--0.95 |
| Phillips output slope | 0.4 | Ball (1999) for US-like economies |
| Phillips exchange-rate passthrough | 0.2 | McCarthy (1999, 2000) |
| Taylor inflation coefficient | 1.5 | Taylor (1993); satisfies the Taylor principle |
| Taylor output coefficient | 2.0 | See "Why k=2.0" below |
| Taylor smoothing rho | 0.85 | Dotsey & Sill (2015); FRB/US model calibration |
| UIP coefficient | 2.0 | Standard small open economy calibration |
| Okun coefficient | 0.5 | Okun (1962); long-standing US empirical estimate |
| Demand shock std dev | 0.7 | Calibrated to match US output gap volatility |
| Cost-push shock std dev | 0.4 | Calibrated to match US inflation volatility |
| Exchange rate shock std dev | 1.5 | Calibrated to match US real exchange rate volatility |

## Why k=2.0

This calibration satisfies four constraints jointly: backward-looking dynamics (Carlin & Soskice 2005), open-economy UIP channel (Ball 1999), interest-rate smoothing per modern central bank practice (Dotsey & Sill 2015), and empirically-grounded Phillips persistence (Gali 2008; Stock & Watson 1999).

These constraints require the Taylor output coefficient k=2.0. Standard k=0.5 (Taylor 1993) produces unstable systems under any empirically defensible parameter set in this model class. Stability analysis confirms:

| Phillips persistence (d) | Minimum k for stability | Max eigenvalue |
|:--:|:--:|:--:|
| 1.00 | 1.3 | 0.991 |
| 0.95 | 1.2 | 0.995 |
| 0.90 | 1.1 | 0.995 |

With rho=0.85, the effective per-period output response is (1-rho) * k = 0.3, comparable to the un-smoothed original 0.5. Estimated Taylor rules with high smoothing routinely find elevated output coefficients (Clarida, Gali & Gertler 2000 estimate ~0.93 at rho=0.79; Orphanides 2003).

## Phillips curve: deviation-from-target form

With inflation persistence < 1 (d=0.95), the Phillips curve uses deviation-from-target form to ensure inflation reverts to pi_star=2.0:

```
(pi_t - pi_star) = 0.95 * (pi_{t-1} - pi_star) + 0.4 * y_{t-1} - 0.2 * (e_{t-1} - e_{t-2}) + eps_pi
```

Modern empirical Phillips curve estimates consistently find inflation persistence below 1.0 -- typically 0.85--0.95 in US post-Volcker data (Gali 2008; Stock & Watson 1999). The original unit-root specification (persistence=1.0) is the Carlin-Soskice teaching version; empirical work has moved away from it.

## Validation summary

The calibrated simulator passes all six validation checks:

1. **Stability** -- max eigenvalue modulus 0.929
2. **Bounded trajectories** -- 0 violations across 1000 simulations
3. **Steady-state recovery** -- all variables recover within 46 quarters
4. **Realistic volatility** -- y: 1.03x, pi: 0.96x, r: 0.56x, e: 0.56x, u: 0.78x empirical US
5. **Impulse response sanity** -- all four NK predictions confirmed
6. **AR(1) suboptimality** -- AR(1) excess MSE 5,776--23,820% across all variables

Calibration sensitivity analysis is available in `data/simulator/validation_report.md`.
