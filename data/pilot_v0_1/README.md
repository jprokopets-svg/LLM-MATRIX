# Pilot v0.1 Artifacts

This directory contains the full audit trail for the LLMMatrix v0.1 pilot run. All files are immutable once published -- corrections appear in subsequent pilot versions, not edits to these files.

## How to verify these results

```bash
# 1. Regenerate ground truth (deterministic, seed=42)
python scripts/generate_history.py
python scripts/run_counterfactual.py

# 2. Verify question directions match simulator
python scripts/validate_questions.py   # should print 30/30, 100% agreement

# 3. Re-score the existing predictions against fresh ground truth
# Load all_results.json and score with src/llmmatrix/pilot_scoring.py
```

The raw API responses cannot be reproduced exactly (model outputs are stochastic), but the scoring pipeline is fully deterministic given the predictions and the simulator seed.

## File inventory

### Top-level results

| File | Description |
|------|-------------|
| `pilot_report_v2.md` | Final findings report with three findings, weight-class matrix, and methodology |
| `all_results.json` | Parsed predictions for all 120 Phase 1 API calls (Sonnet 4.6 + V4 Flash, 10 questions x 2 formats x 3 seeds each) |
| `narrative.txt` | The ~3,000-token economic history narrative shown to all models |

### `raw_responses/`

The actual text returned by each API call. 120 files, named `{model}_{question}_{format}_seed{n}.txt`.

- `claude-sonnet-4-6_Q01_direct_seed1.txt` -- Claude Sonnet 4.6, question Q01, direct format, seed 1
- `deepseek-chat_Q01_cot_seed1.txt` -- DeepSeek V4 Flash, question Q01, chain-of-thought, seed 1
- etc.

These are the audit trail. Anyone can re-parse these files using `src/llmmatrix/parser.py` and verify the predictions in `all_results.json`.

### `bias_investigation/`

Phase 1 diagnostic analysis outputs:

| File | Description |
|------|-------------|
| `format_analysis.json` | Per-response format compliance data (strict JSON, parser-rescued, markdown-wrapped, token count) |
| `per_question_mae.json` | MAE breakdown by question, model, and format |

### `weight_class/`

Phase 2 weight-class balanced run:

| File | Description |
|------|-------------|
| `phase2_results.json` | Parsed predictions for Claude Haiku 4.5 and DeepSeek V4 Reasoner |
| `raw_responses/` | Raw API responses for Phase 2 models |

DeepSeek Reasoner had a high parse failure rate (62%). Only 5 of 13 non-empty responses were successfully parsed. See Finding 3 in `pilot_report_v2.md`.

## Models evaluated

| Model | Provider | API model ID | Phase | Parse rate |
|-------|----------|-------------|-------|:----------:|
| Claude Sonnet 4.6 | Anthropic | `claude-sonnet-4-6` | 1 | 100% |
| DeepSeek V4 Flash | DeepSeek | `deepseek-chat` | 1 | 100% |
| Claude Haiku 4.5 | Anthropic | `claude-haiku-4-5-20251001` | 2 | 100% |
| DeepSeek V4 Reasoner | DeepSeek | `deepseek-reasoner` | 2 | 38% |
| GPT-5 mini | OpenAI | `gpt-5-mini` | 1 (failed) | N/A (quota) |

## Cost

- Phase 1: ~$1.55 (120 calls: 60 Sonnet + 60 Flash)
- Phase 2: ~$1.81 (120 calls: 60 Haiku + 60 Reasoner, many Reasoner responses empty)
- Total: ~$3.36
