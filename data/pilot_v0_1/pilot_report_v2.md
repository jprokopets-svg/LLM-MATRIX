# LLMMatrix v0.1 Pilot: Findings Report

**Date:** 2026-05-22
**Version:** v0.1 pilot (pre-publication internal report)

## Methodology

Ten forecasting questions (3 monetary, 3 demand, 2 cost-push, 2 exchange-rate shocks) were administered to four LLMs across two prompt formats (direct JSON, chain-of-thought) with three random seeds each. Models evaluated:

| Tier | Anthropic | DeepSeek |
|------|-----------|----------|
| Lightweight | Claude Haiku 4.5 | DeepSeek V4 Flash |
| Heavyweight | Claude Sonnet 4.6 | DeepSeek V4 Reasoner |

Evaluation uses the **h=1 (one-quarter-ahead) nowcasting horizon**, the canonical short-horizon evaluation point in macroeconomic forecasting (Marcellino, Stock & Watson 2006; Tashman 2000). Ground truth is a 1000-path Monte Carlo distribution from a validated Ball (1999) open-economy NK simulator with interest-rate smoothing (Dotsey & Sill 2015). The simulator produces internally consistent dynamics that are known by construction and have never appeared in any training corpus -- addressing contamination concerns raised by Lopez-Lira et al. (2025) for financial benchmarks built on historical data. All models confirmed non-familiarity with the fictional economy ("Republic of Vantria") via contamination probes.

Primary metric: mean absolute error (MAE) of point predictions against the Monte Carlo mean, with bootstrap 95% confidence intervals (1000 resamples).

## Three Findings

### Finding 1: Benchmark discriminates at single-question granularity

The benchmark is sensitive enough to detect model differences at the level of individual questions. Q09 (5% currency depreciation) is the highest-dispersion question in the pilot, with cross-model standard deviation 2.6x the average across all ten questions. It cleanly discriminates between Anthropic models and DeepSeek Flash:

| Model | e_1 prediction (3 seeds) | Direction | Truth |
|-------|--------------------------|-----------|-------|
| Claude Sonnet 4.6 | -4.80, -4.80, -4.80 | DOWN (correct) | -6.17 |
| Claude Haiku 4.5 | -5.00, -5.00, -5.00 | DOWN (correct) | -6.17 |
| DeepSeek V4 Flash | -0.05, -0.05, -0.05 | UP (wrong) | -6.17 |

Both Anthropic models correctly predict the exchange rate moves sharply down (depreciation) across all seeds. DeepSeek Flash systematically predicts near-zero movement -- an effective sign error on the depreciation shock, consistent across all three seeds.

This sensitivity is desirable: it means the benchmark can detect real capability differences on specific causal channels. It also means that n=10 questions is insufficient for stable cross-model rankings. Removing Q09 alone changes the rank ordering:

| Rank | All 10 questions | Without Q09 |
|:----:|:----------------:|:-----------:|
| 1 | Claude Sonnet 4.6 (0.688) | Claude Sonnet 4.6 (0.668) |
| 2 | Claude Haiku 4.5 (0.749) | Claude Haiku 4.5 (0.775) |
| 3 | Naive baseline (0.835) | DeepSeek V4 Flash (0.823) |
| 4 | DeepSeek V4 Flash (0.935) | Naive baseline (0.835) |

Claude Sonnet holds rank 1 with and without Q09. The Anthropic advantage is not driven by a single outlier -- it persists across question types (monetary, demand, exchange rate), with DeepSeek winning only on cost-push shocks. Scaling to 50+ questions across multiple world variants will further stabilize these rankings.

### Finding 2: CoT x architecture interaction (preliminary)

Chain-of-thought prompting effects depend on model architecture:

| Model | Direct MAE | CoT MAE | CoT effect |
|-------|:----------:|:-------:|:----------:|
| Claude Sonnet 4.6 | 0.688 | 0.824 | +19.7% (hurts) |
| Claude Haiku 4.5 | 0.749 | 1.898 | +153.2% (hurts badly) |
| DeepSeek V4 Flash | 0.935 | 1.230 | +31.5% (hurts) |
| DeepSeek V4 Reasoner | 0.685 | 0.555 | -18.9% (helps)* |

*Reasoner scores are unreliable due to low parse rate (see Finding 3).

CoT hurts all three standard models. Mechanistic inspection of CoT failures reveals a consistent pattern: models substitute cached macroeconomic textbook intuitions for the specific numerical scenario, producing sign errors on exchange rate and interest rate variables. In the worst cases, the reasoning trace correctly identifies the shock direction in prose -- then inverts the sign in the final numerical prediction.

This finding is preliminary. n=10 questions per model is insufficient for stable conclusions about CoT effects by architecture. The full benchmark tests this interaction across 50+ questions and 6+ models with pre-registered hypotheses.

### Finding 3: Reasoning-specialized models cannot reliably produce structured outputs

DeepSeek V4 Reasoner had a **62% parse failure rate** on structured JSON output requirements (8 failures out of 13 non-empty responses; 47 of 60 total responses were empty). The model's extended internal reasoning chain frequently consumed the response budget, leaving the final structured output empty or malformed.

The few successfully parsed Reasoner responses covered only a small subset of questions, making within-model comparisons statistically unreliable. The apparent "CoT helps Reasoner" signal cannot be distinguished from question-composition confounding.

This is a methodology finding with implications beyond LLMMatrix. Reasoning-specialized models that route through extended chain-of-thought tokens before answering may be fundamentally incompatible with structured-output benchmarks as currently designed. The full benchmark should test workarounds: a two-step protocol (reasoning call followed by a separate formatting call), explicit JSON-mode API parameters where available, and structured output schemas that constrain the response format at the API level.

## Weight-Class Matrix

Direct format, all 10 questions:

|                | Lightweight | Heavyweight |
|----------------|:-----------:|:-----------:|
| **Anthropic**  | Haiku 4.5: **0.749** | Sonnet 4.6: **0.688** |
| **DeepSeek**   | V4 Flash: **0.935** | V4 Reasoner: **0.685*** |
| **Naive baseline** | | **0.835** |

*Reasoner: only 5/13 responses parsed. Score unreliable.

At the lightweight tier, Anthropic (Haiku 0.749) outperforms DeepSeek (Flash 0.935). At the heavyweight tier, comparison is confounded by Reasoner's format failures.

## Limitations

- **n=10 questions.** Insufficient for stable cross-model rankings at the aggregate level, though sufficient to demonstrate benchmark sensitivity at the question level.
- **Single world variant.** The "ball_baseline" calibration tests one macro regime.
- **Single horizon (h=1).** Nowcasting only.
- **Two models per provider.** Only one per capability tier.
- **No temperature sweep.** Fixed temperature=0.7.
- **GPT-5 mini excluded.** OpenAI API quota exceeded during both runs.

## What the Full Benchmark Adds

The pilot validates the pipeline and surfaces three real findings. The full benchmark addresses each limitation: 760 questions across 76 O*NET primitives, multiple world variants, multi-horizon evaluation (h=1, 3, 6, 12), 10+ models, reasoning-model workaround protocols, and monthly re-runs.

## Operational Statistics

| Metric | Value |
|--------|-------|
| Total API calls (Phase 1 + Phase 2) | ~240 |
| Total cost | ~$3.36 |
| Parse failure rate (non-Reasoner models) | 0.0% (180/180) |
| Parse failure rate (DeepSeek Reasoner) | 62% (8/13 non-empty) |
| Contamination detected | None |
| Models successfully evaluated | 3 of 4 (Reasoner excluded from headline comparisons) |
| Naive baseline MAE | 0.835 |
| Best model MAE (direct) | 0.688 (Claude Sonnet 4.6) |
