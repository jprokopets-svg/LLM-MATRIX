# Methodology

## The contamination problem

Existing benchmarks for LLM economic reasoning (e.g., financial question-answering, earnings call analysis) use historical data that may appear in training corpora. Models can score well by memorizing patterns rather than reasoning about causal mechanisms. Lopez-Lira et al. (2025) document this problem for financial benchmarks.

## The synthetic-world solution

LLMMatrix constructs a synthetic macroeconomy whose dynamics are **known by construction** but have never appeared in any training corpus. The simulator produces a fictional economy (the "Republic of Vantria") with internally consistent quarterly data. LLMs are given a narrative description of the economy's history and asked to forecast outcomes after counterfactual shocks.

Because the data-generating process is known, we can compute exact ground-truth distributions via Monte Carlo simulation. LLM predictions are scored against these distributions, not against historical outcomes.

## Model

Five-equation backward-looking open-economy New Keynesian system:

| Equation | Description | Source |
|----------|-------------|--------|
| IS curve | Output gap dynamics | Ball (1999) |
| Phillips curve | Inflation dynamics with exchange rate passthrough | Ball (1999); Gali (2008) |
| Taylor rule | Monetary policy with interest-rate smoothing | Taylor (1993); Dotsey & Sill (2015) |
| UIP condition | Exchange rate determination | Standard open-economy calibration |
| Okun's law | Output-unemployment relationship | Okun (1962) |

### Equations

```
y_t    = 0.8 * y_{t-1} - 0.6 * (r_{t-1} - r*) - 0.2 * (e_{t-1} - e*) + eps_y     [IS]
pi_t   = pi* + 0.95 * (pi_{t-1} - pi*) + 0.4 * y_{t-1} - 0.2 * de_{t-1} + eps_pi  [Phillips]
r_t    = 0.85 * r_{t-1} + 0.15 * [r* + 1.5 * (pi_t - pi*) + 2.0 * y_t]             [Taylor]
e_t    = e* + 2.0 * (r_t - r_world) + eps_e                                         [UIP]
u_t    = u* - 0.5 * y_t                                                              [Okun]
```

All parameters are loaded from `config/ball_baseline.yaml`. Justifications are in [CALIBRATION.md](CALIBRATION.md).

## Evaluation protocol

### Narrative generation

The 60-quarter simulated history is converted to analyst-style prose (~3,000 tokens) using rule-based templates. The narrative includes explicit numerical anchors (inflation, policy rate, unemployment for every quarter) and qualitative descriptions of output gap and exchange rate trends. A regex sanity check verifies 100% of quarterly values are recoverable from the text.

### Forecasting questions

Ten questions spanning four shock types (monetary, demand, cost-push, exchange rate) with varying magnitudes. Each question specifies target variables and the h=1 forecast horizon.

### Prompt formats

- **Direct**: request JSON predictions immediately
- **Chain-of-thought (CoT)**: ask for step-by-step causal reasoning before predictions

### Ground truth

For each question, 1,000 Monte Carlo paths are simulated forward from the end of history with the specified shock applied. The mean of the MC distribution at each (variable, horizon) is the point-estimate ground truth.

### Scoring

- **MAE**: mean absolute error of point predictions against MC mean
- **CRPS**: continuous ranked probability score using predicted 80% CI (assumed Gaussian) against empirical MC distribution
- **Bootstrap 95% CIs**: 1,000 resamples for both metrics

### Contamination control

- Fictional economy ("Republic of Vantria") with synthetic data
- Contamination probe before each evaluation session
- All models confirmed non-familiarity in v0.1 pilot

## Evaluation horizon

The v0.1 pilot evaluates at h=1 (one-quarter-ahead), the canonical nowcasting horizon (Marcellino, Stock & Watson 2006; Tashman 2000). Multi-horizon evaluation (h=3, 6, 12) is deferred to v0.2 pending calibration of impulse response dynamics (Jorda 2005 local projection framework).

## References

- Ball, L. (1999). Policy rules for open economies. In J.B. Taylor (Ed.), *Monetary Policy Rules*. University of Chicago Press.
- Clarida, R., Gali, J., & Gertler, M. (2000). Monetary policy rules and macroeconomic stability. *QJE*, 115(1), 147--180.
- Dotsey, M. & Sill, K. (2015). Interest-rate rules and macroeconomic stability. Federal Reserve Bank of Philadelphia.
- Gali, J. (2008). *Monetary Policy, Inflation, and the Business Cycle*. Princeton University Press.
- Jorda, O. (2005). Estimation and inference of impulse responses by local projections. *AER*, 95(1), 161--182.
- Lopez-Lira, A. et al. (2025). Contamination in financial NLP benchmarks. Working paper.
- Marcellino, M., Stock, J.H., & Watson, M.W. (2006). A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series. *Journal of Econometrics*, 135(1--2), 499--526.
- McCarthy, J. (1999). Pass-through of exchange rates and import prices to domestic inflation. Federal Reserve Bank of New York Staff Reports.
- Okun, A.M. (1962). Potential GNP: Its measurement and significance. ASA Proceedings.
- Stock, J.H. & Watson, M.W. (1999). Forecasting inflation. *Journal of Monetary Economics*, 44(2), 293--335.
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *IJF*, 16(4), 437--450.
- Taylor, J.B. (1993). Discretion versus policy rules in practice. *Carnegie-Rochester Conference Series*, 39, 195--214.
