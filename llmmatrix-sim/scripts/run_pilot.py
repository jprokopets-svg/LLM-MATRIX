"""
Orchestration script for the LLMMatrix pilot run.

Workflow:
  1. Load narrative, questions, ground truth.
  2. Run contamination probes.
  3. Main run: (model x question x prompt_format x seed) API calls.
  4. Score predictions against MC truth + baselines.
  5. Generate pilot_report.md.

IMPORTANT: Do not run the full pilot autonomously. Build and validate
infrastructure first, then wait for sign-off before spending on API calls.
"""

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.narrative import save_narrative
from src.questions import PILOT_QUESTIONS, validate_questions
from src.prompts import PROMPT_FORMATTERS, _build_prediction_keys
from src.model_runner import ModelRunner, CostCapExceeded
from src.parser import parse_response
from src.pilot_scoring import mae_score, crps_score
from src.monte_carlo import run_counterfactual
from src.shocks import (
    monetary_tightening_shock,
    demand_shock,
    cost_push_shock,
    exchange_rate_shock,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(project_root, "data", "pilot_results")
CONFIG_PATH = os.path.join(project_root, "pilot_config.yaml")
HISTORY_PATH = os.path.join(project_root, "data", "history.csv")
SIM_CONFIG_PATH = os.path.join(project_root, "config", "ball_baseline.yaml")


# ---------------------------------------------------------------------------
# Shock type → shock function mapping
# ---------------------------------------------------------------------------

SHOCK_FN_MAP = {
    "monetary": monetary_tightening_shock,
    "demand": demand_shock,
    "cost_push": cost_push_shock,
    "exchange_rate": exchange_rate_shock,
}


def load_config() -> dict:
    """Load pilot configuration from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def ensure_narrative() -> str:
    """Generate or load the narrative text."""
    narrative_path = os.path.join(RESULTS_DIR, "narrative.txt")
    if not os.path.exists(narrative_path):
        logger.info("Generating narrative...")
        save_narrative(HISTORY_PATH, narrative_path)

    with open(narrative_path, "r") as f:
        return f.read()


def run_contamination_probes(
    config: dict,
    runner: ModelRunner,
) -> list[dict]:
    """
    Send the contamination probe to each model and log responses.

    Returns:
        List of probe results.
    """
    probe_prompt = config["contamination_probe"]["prompt"]
    results = []

    for model_cfg in config["models"]:
        logger.info(f"Contamination probe: {model_cfg['display_name']}...")
        try:
            response = runner.run_model(
                provider=model_cfg["provider"],
                model=model_cfg["model"],
                prompt=probe_prompt,
                temperature=0.0,
                max_tokens=500,
            )
            results.append({
                "model": model_cfg["display_name"],
                "response": response["response_text"],
                "cost_usd": response["cost_usd"],
            })
            logger.info(f"  Response: {response['response_text'][:200]}")
        except Exception as exc:
            logger.error(f"  Probe failed for {model_cfg['display_name']}: {exc}")
            results.append({
                "model": model_cfg["display_name"],
                "response": f"ERROR: {exc}",
                "cost_usd": 0.0,
            })

    return results


def get_ground_truth(
    question_id: str,
    shock_type: str,
    shock_magnitude: float,
    history_df: pd.DataFrame,
) -> np.ndarray:
    """
    Get MC truth paths for a specific question's shock scenario.

    Args:
        question_id: For logging.
        shock_type: "monetary", "demand", "cost_push", "exchange_rate".
        shock_magnitude: Shock size.
        history_df: Historical trajectory.

    Returns:
        3D array (n_paths, 12, 5) of forward paths.
    """
    last_period = int(history_df.iloc[-1]["period"])
    shock_fn_factory = SHOCK_FN_MAP[shock_type]
    # Shock targets last_period + 1 (the first forward quarter)
    shock_fn = lambda: shock_fn_factory(magnitude=shock_magnitude, period=last_period + 1)

    paths = run_counterfactual(
        config_path=SIM_CONFIG_PATH,
        history_df=history_df,
        shock_fn=shock_fn,
        n_paths=1000,
        n_forward_periods=12,
        base_seed=5000,
    )
    return paths


def run_main_loop(
    config: dict,
    narrative: str,
    runner: ModelRunner,
) -> list[dict]:
    """
    Run all (model x question x prompt x seed) combinations.

    Returns:
        List of result dicts with predictions and metadata.
    """
    all_results = []
    questions = PILOT_QUESTIONS

    total_calls = (
        len(config["models"])
        * len(questions)
        * len(config["prompt_formats"])
        * len(config["seeds"])
    )
    call_count = 0

    for model_cfg in config["models"]:
        parse_failures = 0
        total_for_model = 0

        for question in questions:
            expected_keys = _build_prediction_keys(question)

            for fmt in config["prompt_formats"]:
                formatter = PROMPT_FORMATTERS[fmt]
                prompt = formatter(narrative, question)

                for seed in config["seeds"]:
                    call_count += 1
                    total_for_model += 1

                    logger.info(
                        f"[{call_count}/{total_calls}] "
                        f"{model_cfg['display_name']} / {question.id} / {fmt} / seed={seed}"
                    )

                    try:
                        response = runner.run_model(
                            provider=model_cfg["provider"],
                            model=model_cfg["model"],
                            prompt=prompt,
                            temperature=config["temperature"],
                            seed=seed if model_cfg["provider"] == "openai" else None,
                            max_tokens=config["max_tokens"],
                        )
                    except CostCapExceeded:
                        logger.error("Cost cap exceeded — aborting pilot run.")
                        return all_results
                    except Exception as exc:
                        logger.error(f"API call failed: {exc}")
                        all_results.append({
                            "question_id": question.id,
                            "model": model_cfg["display_name"],
                            "prompt_format": fmt,
                            "seed": seed,
                            "error": str(exc),
                        })
                        continue

                    # Parse response
                    parsed = parse_response(
                        raw_response=response["response_text"],
                        question_id=question.id,
                        model=model_cfg["display_name"],
                        prompt_format=fmt,
                        seed=seed,
                        expected_keys=expected_keys,
                    )

                    if not parsed.parse_success:
                        parse_failures += 1

                    result = {
                        "question_id": question.id,
                        "model": model_cfg["display_name"],
                        "prompt_format": fmt,
                        "seed": seed,
                        "predictions": parsed.predictions,
                        "parse_success": parsed.parse_success,
                        "parse_error": parsed.parse_error,
                        "tokens_in": response["tokens_in"],
                        "tokens_out": response["tokens_out"],
                        "cost_usd": response["cost_usd"],
                        "latency_s": response["latency_s"],
                    }
                    all_results.append(result)

                    # Save raw response to disk
                    raw_dir = os.path.join(RESULTS_DIR, "raw_responses")
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(
                        raw_dir,
                        f"{model_cfg['model']}_{question.id}_{fmt}_seed{seed}.txt",
                    )
                    with open(raw_path, "w") as f:
                        f.write(response["response_text"])

        # Check parse failure rate
        if total_for_model > 0:
            failure_rate = parse_failures / total_for_model
            logger.info(
                f"Parse failure rate for {model_cfg['display_name']}: "
                f"{failure_rate:.1%} ({parse_failures}/{total_for_model})"
            )
            if failure_rate > 0.20:
                logger.error(
                    f"Parse failure rate > 20% for {model_cfg['display_name']}. "
                    f"Investigate before continuing."
                )

    return all_results


def generate_report(
    config: dict,
    probe_results: list[dict],
    all_results: list[dict],
    runner: ModelRunner,
) -> str:
    """
    Generate the pilot report as Markdown.

    Returns:
        Report text.
    """
    lines = []
    lines.append("# LLMMatrix Pilot Run Report\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Pilot name:** {config['pilot_name']}\n")

    # Run summary
    lines.append("## Methodology\n")
    lines.append(
        "This pilot evaluates LLM forecasting at the **h=1 (one-quarter-ahead) "
        "nowcasting horizon**. This is the canonical short-horizon evaluation point "
        "in macroeconomic forecasting (Marcellino, Stock & Watson 2006; Tashman 2000). "
        "Single-horizon evaluation isolates the model's ability to identify "
        "contemporaneous causal channels without confounding from multi-step "
        "oscillatory dynamics. The ground truth is a 1000-path Monte Carlo "
        "distribution from a validated Ball (1999) open-economy NK simulator. "
        "Multi-horizon analysis (h=3, 6, 12) is deferred to v0.2 pending "
        "calibration of impulse response dynamics (Jorda 2005 local projection "
        "framework).\n"
    )

    lines.append("\n## Run Summary\n")
    model_names = [m["display_name"] for m in config["models"]]
    lines.append(f"- **Models:** {', '.join(model_names)}\n")
    lines.append(f"- **Questions:** {len(PILOT_QUESTIONS)}\n")
    lines.append(f"- **Forecast horizon:** h=1 (one quarter ahead)\n")
    lines.append(f"- **Prompt formats:** {', '.join(config['prompt_formats'])}\n")
    lines.append(f"- **Seeds:** {config['seeds']}\n")
    lines.append(f"- **Total API calls:** {len(all_results)}\n")
    lines.append(f"- **Total cost:** ${runner.total_cost_usd:.2f}\n")

    # Contamination probes
    lines.append("\n## Contamination Probe Results\n")
    for probe in probe_results:
        lines.append(f"### {probe['model']}\n")
        lines.append(f"```\n{probe['response'][:500]}\n```\n")

    # Parse failure rates
    lines.append("\n## Parse Failure Rates\n")
    lines.append("| Model | Total Calls | Failures | Rate |\n")
    lines.append("|-------|------------|----------|------|\n")
    for model_cfg in config["models"]:
        model_name = model_cfg["display_name"]
        model_results = [r for r in all_results if r.get("model") == model_name]
        total = len(model_results)
        failures = sum(1 for r in model_results if not r.get("parse_success", False))
        rate = failures / total if total > 0 else 0
        lines.append(f"| {model_name} | {total} | {failures} | {rate:.1%} |\n")

    # Per-model costs
    lines.append("\n## Cost Breakdown\n")
    lines.append("| Model | Total Cost | Avg Latency |\n")
    lines.append("|-------|-----------|-------------|\n")
    for model_cfg in config["models"]:
        model_name = model_cfg["display_name"]
        model_results = [r for r in all_results if r.get("model") == model_name]
        total_cost = sum(r.get("cost_usd", 0) for r in model_results)
        latencies = [r.get("latency_s", 0) for r in model_results if "latency_s" in r]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        lines.append(f"| {model_name} | ${total_cost:.2f} | {avg_latency:.1f}s |\n")

    # Scoring placeholder (populated when questions are filled in)
    lines.append("\n## Scoring Results\n")
    lines.append("*Scoring will be computed once questions are fully specified.*\n")

    return "".join(lines)


def main() -> None:
    """Run the full pilot pipeline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 0: Load config and validate questions
    config = load_config()
    logger.info(f"Pilot: {config['pilot_name']}")

    question_errors = validate_questions(PILOT_QUESTIONS)
    if question_errors:
        logger.error("Questions are not fully specified yet:")
        for err in question_errors:
            logger.error(f"  {err}")
        logger.error("Fill in questions.py before running the pilot.")
        sys.exit(1)

    # Step 1: Ensure narrative exists
    narrative = ensure_narrative()
    logger.info(f"Narrative loaded: {len(narrative)} chars")

    # Step 2: Initialize runner
    runner = ModelRunner(cost_cap_usd=config["cost_cap_usd"])

    # Step 3: Contamination probes
    probe_results = []
    if config["contamination_probe"]["enabled"]:
        probe_results = run_contamination_probes(config, runner)

    # Step 4: Main run
    all_results = run_main_loop(config, narrative, runner)

    # Step 5: Save results
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(results_path, "w") as f:
        # Convert to JSON-safe format (predictions may have nested dicts)
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Step 6: Generate report
    report = generate_report(config, probe_results, all_results, runner)
    report_path = os.path.join(RESULTS_DIR, "pilot_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Summary
    logger.info(f"Pilot complete. Total cost: ${runner.total_cost_usd:.2f}")
    logger.info(f"Results: {len(all_results)} calls logged")


if __name__ == "__main__":
    main()
