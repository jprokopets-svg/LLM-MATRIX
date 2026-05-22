# Roadmap

## v0.1 -- Pilot (complete)

Validated simulator + first LLM forecasting results.

- Five-equation Ball (1999) open-economy NK simulator with interest-rate smoothing
- Six validation checks passing (stability, bounded trajectories, steady-state recovery, realistic volatility, impulse response sanity, AR(1) suboptimality)
- Rule-based narrative generation for "Republic of Vantria"
- 10 forecasting questions across 4 shock types
- Pilot run: 4 models (Claude Sonnet 4.6, Claude Haiku 4.5, DeepSeek V4 Flash, DeepSeek V4 Reasoner), 2 prompt formats, 3 seeds
- Three findings: benchmark discriminates at question level, CoT x architecture interaction, reasoning-model format compliance issues
- Total pilot cost: $3.95

## v0.2 -- Grant-funded full benchmark

Scale from pilot to publication-grade benchmark.

- **760 questions** across 76 O\*NET elemental skills (10 tasks per primitive)
- **Multiple world variants**: monetary-only, trade-focused, commodity-exporter, fixed exchange rate
- **Multi-horizon evaluation**: h=1, 3, 6, 12 with calibrated impulse responses (Jorda 2005 local projection framework)
- **10+ models** across all major providers (Anthropic, OpenAI, Google, DeepSeek, open-weight)
- **Agentic scaffolds**: three standardized scaffolds (custom, Claude SDK, OpenAI SDK) for tool-augmented forecasting
- **Reasoning-model protocol**: two-step (reason then format) and API-level structured output
- **Monthly re-runs** for longitudinal tracking
- **Temperature sweeps** for robustness analysis
- **Employer jury validation** of task relevance

## v0.3 -- Longer-term ambitions

- Occupation-level composite scores mapping model capabilities to O\*NET occupations
- Forward-looking expectations variant (rational expectations, learning models)
- Multi-country / multi-sector extensions
- LLM-generated narratives (test narrative quality as a separate variable)
- Public leaderboard at llmmatrix.ai
- Integration with OCB (Occupational Capability Benchmark) for cross-benchmark validation
