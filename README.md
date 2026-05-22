# LLMMatrix

**Synthetic-world benchmark for contamination-free evaluation of LLM macroeconomic forecasting.**

![Version](https://img.shields.io/badge/version-0.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The problem

Existing benchmarks for LLM economic reasoning use historical data that may appear in training corpora. Models can score well by memorizing patterns rather than reasoning about causal mechanisms (Lopez-Lira et al. 2025). You cannot tell if a model *understands* macroeconomics or *remembers* the answer.

## The solution

LLMMatrix constructs a synthetic macroeconomy --- the "Republic of Vantria" --- whose dynamics are **known by construction** and have never appeared in any training corpus. A validated simulator produces 60 quarters of economic history. LLMs read a narrative description and forecast outcomes after counterfactual shocks. Predictions are scored against the simulator's true Monte Carlo distributions.

The simulator is a measurement instrument, not a model of reality. Its job is to produce internally consistent, defensible dynamics that test whether LLMs can reason about causal mechanisms in macroeconomics.

## v0.1 pilot findings

Three findings from the pilot run (4 models, 10 questions, h=1 nowcasting):

| Finding | Detail |
|---------|--------|
| **Benchmark discriminates** | Detects real capability differences at single-question granularity. Removing one question (Q09, exchange rate shock) reverses the rank ordering between Claude Sonnet and DeepSeek Flash --- demonstrating sensitivity but also that n=10 is insufficient for stable rankings. |
| **CoT x architecture interaction** | Chain-of-thought hurts standard models (+9% to +78% MAE) but helps the lightweight model (-6%). Models substitute textbook intuitions for specific numerical scenarios, producing sign errors. |
| **Reasoning models fail structured output** | DeepSeek Reasoner had 74% parse failure rate on JSON output requirements. Reasoning-specialized models may be incompatible with structured-output benchmarks as currently designed. |

Full results: [`data/pilot_v0_1/pilot_report_v2.md`](data/pilot_v0_1/pilot_report_v2.md)

## Impulse responses

The simulator produces textbook New Keynesian dynamics. A +1pp monetary tightening:

![Impulse Response: +1pp Monetary Tightening](data/simulator/plots/impulse_responses.png)

## Quickstart

```bash
git clone https://github.com/jprokopets-svg/LLM-MATRIX.git
cd LLM-MATRIX

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run simulator validation (all 6 checks)
python scripts/validate_sim.py

# Run tests
pytest tests/ -v
```

To run the pilot (requires API keys):
```bash
cp .env.example .env   # fill in your API keys
python scripts/run_pilot.py
```

## Repository layout

```
LLM-MATRIX/
├── README.md                        # you are here
├── LICENSE                          # MIT
├── CITATION.cff                     # cite this project
├── pyproject.toml                   # pip install -e .
├── requirements.txt                 # pinned dependencies
│
├── config/
│   ├── ball_baseline.yaml           # all simulator parameters
│   └── pilot_config.yaml            # pilot run settings (models, seeds)
│
├── src/llmmatrix/                   # Python package
│   ├── simulator.py                 # core Sim class (step/run)
│   ├── shocks.py                    # 4 counterfactual shock scenarios
│   ├── monte_carlo.py               # MC forward simulation
│   ├── baselines.py                 # naive, AR(1), oracle forecasters
│   ├── validation.py                # 6 simulator sanity checks
│   ├── narrative.py                 # rule-based narrative generation
│   ├── questions.py                 # 10 pilot forecasting questions
│   ├── prompts.py                   # direct + CoT prompt templates
│   ├── model_runner.py              # API wrapper (Anthropic/OpenAI/DeepSeek)
│   ├── parser.py                    # JSON prediction extractor
│   └── pilot_scoring.py             # MAE + CRPS with bootstrap CIs
│
├── scripts/
│   ├── generate_history.py          # produce 60-quarter baseline history
│   ├── run_counterfactual.py        # MC paths for each shock scenario
│   ├── validate_sim.py              # run all 6 validation checks + plots
│   ├── validate_questions.py        # verify question directions vs simulator
│   └── run_pilot.py                 # full pilot orchestration
│
├── tests/                           # pytest suite
│   ├── test_stability.py            # eigenvalue check
│   ├── test_steady_state.py         # convergence to targets
│   ├── test_impulse_response.py     # IRF signs and timing
│   └── test_baselines.py            # baseline forecaster validity
│
├── data/
│   ├── simulator/                   # generated simulator outputs
│   │   ├── history.csv
│   │   ├── validation_report.md
│   │   ├── counterfactual_paths/
│   │   └── plots/
│   └── pilot_v0_1/                  # pilot results (immutable)
│       ├── pilot_report_v2.md
│       ├── raw_responses/
│       └── bias_investigation/
│
└── docs/
    ├── METHODOLOGY.md               # equations, evaluation protocol, references
    ├── CALIBRATION.md               # parameter choices and justifications
    └── ROADMAP.md                   # v0.1 -> v0.2 -> v0.3 plan
```

## Methodology

Five-equation backward-looking open-economy NK model based on Ball (1999), with interest-rate smoothing (Dotsey & Sill 2015) and empirically-grounded Phillips persistence (Gali 2008). All parameters cite established literature.

The simulator passes six validation checks: stability (eigenvalues), bounded trajectories, steady-state recovery, realistic volatility (~1x US empirical for output gap and inflation), impulse response sanity, and AR(1) suboptimality (5,776--23,820% excess MSE).

Full methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Parameter justifications: [`docs/CALIBRATION.md`](docs/CALIBRATION.md)

## Citation

```bibtex
@software{prokopets2026llmmatrix,
  title  = {LLMMatrix: Synthetic-World Benchmark for Contamination-Free LLM Forecasting},
  author = {Prokopets, Jake},
  year   = {2026},
  url    = {https://github.com/jprokopets-svg/LLM-MATRIX},
  note   = {Version 0.1}
}
```

## Roadmap

| Version | Status | Scope |
|---------|--------|-------|
| **v0.1** | Complete | Simulator + pilot (10 questions, 4 models, h=1) |
| **v0.2** | Grant-funded | 760 questions, 10+ models, multi-horizon, agentic scaffolds |
| **v0.3** | Planned | Occupation-level composites, public leaderboard |

Details: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Acknowledgments

- Lopez-Lira et al. (2025) for documenting the contamination problem in financial benchmarks
- Ball (1999), Taylor (1993), Dotsey & Sill (2015) for the underlying macroeconomic model
- Marcellino, Stock & Watson (2006) and Tashman (2000) for nowcasting evaluation standards

## Contact

Jake Prokopets -- jprokopets@gmail.com
