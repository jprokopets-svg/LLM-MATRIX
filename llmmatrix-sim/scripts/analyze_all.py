"""
Combined analysis: Phase 2 weight-class matrix + Q09 sensitivity check.

Reads original pilot results + Phase 2 results, computes the 2x2 capability
matrix, and re-runs all comparisons with Q09 excluded.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.questions import PILOT_QUESTIONS
from src.prompts import _build_prediction_keys
from src.pilot_scoring import mae_score
from src.monte_carlo import run_counterfactual
from src.shocks import monetary_tightening_shock, demand_shock, cost_push_shock, exchange_rate_shock

config_path = os.path.join(project_root, "config", "ball_baseline.yaml")
history_df = pd.read_csv(os.path.join(project_root, "data", "history.csv"))
last_period = int(history_df.iloc[-1]["period"])

SHOCK_FN_MAP = {
    "monetary": monetary_tightening_shock,
    "demand": demand_shock,
    "cost_push": cost_push_shock,
    "exchange_rate": exchange_rate_shock,
}
var_idx = {v: i for i, v in enumerate(["y", "pi", "r", "e", "u"])}
q_map = {q.id: q for q in PILOT_QUESTIONS}


def load_all_results():
    """Load original pilot + Phase 2 results."""
    results = []

    # Original pilot
    orig_path = os.path.join(project_root, "data", "pilot_results", "all_results.json")
    with open(orig_path) as f:
        results.extend(json.load(f))

    # Phase 2
    p2_path = os.path.join(project_root, "data", "pilot_results", "weight_class", "phase2_results.json")
    if os.path.exists(p2_path):
        with open(p2_path) as f:
            results.extend(json.load(f))

    return results


def compute_mae_for_subset(results, model, fmt, questions, gt_cache):
    """Compute MAE for a specific model/format on a subset of questions."""
    preds, truths = [], []
    mr = [r for r in results if r.get("model") == model
          and r.get("prompt_format") == fmt and r.get("parse_success", False)]

    valid_qids = {q.id for q in questions}
    for r in mr:
        if r["question_id"] not in valid_qids:
            continue
        q = q_map[r["question_id"]]
        gt = gt_cache[r["question_id"]]
        for var in q.target_variables:
            key = f"{var}_1"
            if key in r.get("predictions", {}):
                preds.append(r["predictions"][key]["point"])
                truths.append(float(np.mean(gt[:, 0, var_idx[var]])))

    if not preds:
        return None
    return mae_score(preds, truths)


def main():
    # Generate ground truth
    print("Generating ground truth MC paths...")
    gt_cache = {}
    for q in PILOT_QUESTIONS:
        shock_fn_factory = SHOCK_FN_MAP[q.shock_type]
        shock_fn = lambda mag=q.shock_magnitude: shock_fn_factory(
            magnitude=mag, period=last_period + 1
        )
        gt_cache[q.id] = run_counterfactual(
            config_path, history_df, shock_fn, 1000, 12, 9000
        )

    all_results = load_all_results()

    # Check what models we have
    models_found = set()
    for r in all_results:
        if r.get("parse_success"):
            models_found.add(r["model"])
    print(f"Models with parsed results: {sorted(models_found)}")

    # Parse stats per model
    print("\nParse stats:")
    for model in sorted(models_found):
        mr = [r for r in all_results if r.get("model") == model]
        ok = sum(1 for r in mr if r.get("parse_success"))
        err = sum(1 for r in mr if "error" in r)
        pfail = len(mr) - ok - err
        print(f"  {model}: {len(mr)} total, {ok} parsed, {err} API errors, {pfail} parse fails")

    all_questions = PILOT_QUESTIONS
    questions_no_q09 = [q for q in PILOT_QUESTIONS if q.id != "Q09"]

    # Naive baseline
    naive_preds_all, naive_truths_all = [], []
    naive_preds_noq09, naive_truths_noq09 = [], []
    for q in all_questions:
        gt = gt_cache[q.id]
        for var in q.target_variables:
            last_val = float(history_df.iloc[-1][var])
            truth_mean = float(np.mean(gt[:, 0, var_idx[var]]))
            naive_preds_all.append(last_val)
            naive_truths_all.append(truth_mean)
            if q.id != "Q09":
                naive_preds_noq09.append(last_val)
                naive_truths_noq09.append(truth_mean)

    naive_all = mae_score(naive_preds_all, naive_truths_all)
    naive_noq09 = mae_score(naive_preds_noq09, naive_truths_noq09)

    # ============================================================
    # 2x2 Weight-class matrix (direct format only)
    # ============================================================
    print("\n" + "=" * 80)
    print("WEIGHT-CLASS MATRIX (direct format, all 10 questions)")
    print("=" * 80)

    matrix_models = [
        ("Claude Haiku 4.5", "Lightweight"),
        ("Claude Sonnet 4.6", "Heavyweight"),
        ("DeepSeek V4 Flash", "Lightweight"),
        ("DeepSeek Reasoner", "Heavyweight"),
    ]

    print(f"\n{'Model':<24} {'Tier':<14} {'MAE':>7} {'95% CI':>20}")
    print("-" * 70)
    for model, tier in matrix_models:
        result = compute_mae_for_subset(all_results, model, "direct", all_questions, gt_cache)
        if result:
            print(f"{model:<24} {tier:<14} {result['mae']:>7.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
        else:
            print(f"{model:<24} {tier:<14}     N/A")

    print(f"{'Naive baseline':<24} {'—':<14} {naive_all['mae']:>7.3f}")

    # Formatted 2x2 grid
    print("\n2x2 Grid:")
    print(f"{'':>20} {'Lightweight':>15} {'Heavyweight':>15}")
    for provider in ["Anthropic", "DeepSeek"]:
        if provider == "Anthropic":
            lw = compute_mae_for_subset(all_results, "Claude Haiku 4.5", "direct", all_questions, gt_cache)
            hw = compute_mae_for_subset(all_results, "Claude Sonnet 4.6", "direct", all_questions, gt_cache)
        else:
            lw = compute_mae_for_subset(all_results, "DeepSeek V4 Flash", "direct", all_questions, gt_cache)
            hw = compute_mae_for_subset(all_results, "DeepSeek Reasoner", "direct", all_questions, gt_cache)
        lw_str = f"{lw['mae']:.3f}" if lw else "N/A"
        hw_str = f"{hw['mae']:.3f}" if hw else "N/A"
        print(f"{provider:>20} {lw_str:>15} {hw_str:>15}")

    # ============================================================
    # Q09 Sensitivity Check
    # ============================================================
    print("\n" + "=" * 80)
    print("Q09 SENSITIVITY CHECK (direct format, Q09 excluded)")
    print("=" * 80)

    all_model_names = sorted(models_found)
    print(f"\n{'Model':<24} {'All Qs MAE':>10} {'No Q09 MAE':>11} {'Change':>8}")
    print("-" * 60)
    for model in all_model_names:
        mae_all = compute_mae_for_subset(all_results, model, "direct", all_questions, gt_cache)
        mae_noq09 = compute_mae_for_subset(all_results, model, "direct", questions_no_q09, gt_cache)
        if mae_all and mae_noq09:
            change = mae_noq09["mae"] - mae_all["mae"]
            print(f"{model:<24} {mae_all['mae']:>10.3f} {mae_noq09['mae']:>11.3f} {change:>+8.3f}")
        elif mae_all:
            print(f"{model:<24} {mae_all['mae']:>10.3f}         N/A")

    print(f"{'Naive baseline':<24} {naive_all['mae']:>10.3f} {naive_noq09['mae']:>11.3f} {naive_noq09['mae']-naive_all['mae']:>+8.3f}")

    # Rank ordering with and without Q09
    print("\nRank ordering (direct format):")
    print("  All 10 questions:")
    ranks_all = []
    for model in all_model_names:
        r = compute_mae_for_subset(all_results, model, "direct", all_questions, gt_cache)
        if r:
            ranks_all.append((model, r["mae"], r["ci_low"], r["ci_high"]))
    ranks_all.sort(key=lambda x: x[1])
    for i, (model, mae, lo, hi) in enumerate(ranks_all, 1):
        print(f"    {i}. {model}: {mae:.3f} [{lo:.3f}, {hi:.3f}]")

    print("  Without Q09:")
    ranks_noq09 = []
    for model in all_model_names:
        r = compute_mae_for_subset(all_results, model, "direct", questions_no_q09, gt_cache)
        if r:
            ranks_noq09.append((model, r["mae"], r["ci_low"], r["ci_high"]))
    ranks_noq09.sort(key=lambda x: x[1])
    for i, (model, mae, lo, hi) in enumerate(ranks_noq09, 1):
        print(f"    {i}. {model}: {mae:.3f} [{lo:.3f}, {hi:.3f}]")

    # Check if rank ordering changed
    order_all = [m for m, _, _, _ in ranks_all]
    order_noq09 = [m for m, _, _, _ in ranks_noq09]
    if order_all == order_noq09:
        print("\n  Rank ordering: UNCHANGED after excluding Q09")
    else:
        print(f"\n  Rank ordering CHANGED:")
        print(f"    All Qs:  {' > '.join(order_all)}")
        print(f"    No Q09:  {' > '.join(order_noq09)}")

    # Also do CoT comparison with Q09 excluded
    print("\n" + "=" * 80)
    print("COT COMPARISON (Q09 excluded)")
    print("=" * 80)
    print(f"\n{'Model':<24} {'Direct':>8} {'CoT':>8} {'CoT Effect':>11}")
    print("-" * 55)
    for model in all_model_names:
        d = compute_mae_for_subset(all_results, model, "direct", questions_no_q09, gt_cache)
        c = compute_mae_for_subset(all_results, model, "cot", questions_no_q09, gt_cache)
        if d and c:
            effect = (c["mae"] - d["mae"]) / d["mae"] * 100
            print(f"{model:<24} {d['mae']:>8.3f} {c['mae']:>8.3f} {effect:>+10.1f}%")


if __name__ == "__main__":
    main()
