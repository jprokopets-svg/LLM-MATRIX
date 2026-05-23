"""
Phase 2: Weight-class balanced re-run.

Runs Claude Haiku 4.5 and DeepSeek Reasoner on the same 10 questions,
2 prompt formats, 3 seeds. Saves raw responses and parsed results.
"""

import json
import logging
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if value.strip():
                    os.environ[key.strip()] = value.strip()

sys.path.insert(0, os.path.join(project_root, "src"))

from llmmatrix.questions import PILOT_QUESTIONS
from llmmatrix.prompts import PROMPT_FORMATTERS, _build_prediction_keys
from llmmatrix.model_runner import ModelRunner
from llmmatrix.parser import parse_response
from llmmatrix.narrative import generate_narrative
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(project_root, "data", "pilot_v0_1", "weight_class")
RAW_DIR = os.path.join(RESULTS_DIR, "raw_responses")
os.makedirs(RAW_DIR, exist_ok=True)

PHASE2_MODELS = [
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
    {"provider": "deepseek", "model": "deepseek-reasoner", "display_name": "DeepSeek Reasoner"},
]

SEEDS = [1, 2, 3]
FORMATS = ["direct", "cot"]


def main() -> None:
    history_df = pd.read_csv(os.path.join(project_root, "data", "simulator", "history.csv"))
    narrative = generate_narrative(history_df)

    runner = ModelRunner(cost_cap_usd=10.0)
    all_results = []

    total_calls = len(PHASE2_MODELS) * len(PILOT_QUESTIONS) * len(FORMATS) * len(SEEDS)
    call_count = 0

    for model_cfg in PHASE2_MODELS:
        for question in PILOT_QUESTIONS:
            expected_keys = _build_prediction_keys(question)
            for fmt in FORMATS:
                formatter = PROMPT_FORMATTERS[fmt]
                prompt = formatter(narrative, question)
                for seed in SEEDS:
                    call_count += 1
                    logger.info(
                        f"[{call_count}/{total_calls}] "
                        f"{model_cfg['display_name']} / {question.id} / {fmt} / seed={seed}"
                    )
                    try:
                        response = runner.run_model(
                            provider=model_cfg["provider"],
                            model=model_cfg["model"],
                            prompt=prompt, temperature=0.7, max_tokens=4000,
                        )
                    except Exception as exc:
                        logger.error(f"API call failed: {exc}")
                        all_results.append({
                            "question_id": question.id,
                            "model": model_cfg["display_name"],
                            "prompt_format": fmt, "seed": seed,
                            "error": str(exc),
                        })
                        continue

                    raw_path = os.path.join(
                        RAW_DIR, f"{model_cfg['model']}_{question.id}_{fmt}_seed{seed}.txt",
                    )
                    with open(raw_path, "w") as f:
                        f.write(response["response_text"])

                    parsed = parse_response(
                        raw_response=response["response_text"],
                        question_id=question.id,
                        model=model_cfg["display_name"],
                        prompt_format=fmt, seed=seed,
                        expected_keys=expected_keys,
                    )
                    all_results.append({
                        "question_id": question.id,
                        "model": model_cfg["display_name"],
                        "prompt_format": fmt, "seed": seed,
                        "predictions": parsed.predictions,
                        "parse_success": parsed.parse_success,
                        "parse_error": parsed.parse_error,
                        "cost_usd": response["cost_usd"],
                        "latency_s": response["latency_s"],
                    })

        mr = [r for r in all_results if r.get("model") == model_cfg["display_name"]]
        total = len(mr)
        fails = sum(1 for r in mr if not r.get("parse_success", False) and "error" not in r)
        errors = sum(1 for r in mr if "error" in r)
        logger.info(f"Stats for {model_cfg['display_name']}: {total} calls, {fails} parse failures, {errors} API errors")

    results_path = os.path.join(RESULTS_DIR, "phase2_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Phase 2 complete. Total cost: ${runner.total_cost_usd:.2f}")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
