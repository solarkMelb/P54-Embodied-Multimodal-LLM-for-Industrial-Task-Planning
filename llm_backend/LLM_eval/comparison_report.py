"""
comparison_report.py
--------------------
Generates the full comparative analysis report across GPT-4o, Gemini,
DeepSeek, and Ollama. Combines evaluation results, metrics, and per-case
analysis into a readable terminal report and a CSV export.

Run this file directly to execute the full evaluation pipeline:
    python comparison_report.py
    python -m llm_backend.LLM_eval.comparison_report

Or import and call programmatically:
    from llm_backend.LLM_eval.comparison_report import run_comparison_report

    run_comparison_report(models=["openai", "gemini"])
"""

import sys
import os
import json
from typing import Optional

from .evaluator import run_evaluation, EvalResult
from .metrics import compute_metrics, print_metrics_table, export_metrics_csv
from .model_registry import get_available_models, MODEL_DISPLAY_NAMES
from .test_cases import TEST_CASES, get_all_categories


# ── Section printers ──────────────────────────────────────────────────────────

def _header(title: str, width: int = 70) -> None:
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def _section(title: str, width: int = 70) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def print_per_case_comparison(results: list[EvalResult], models: list[str]) -> None:
    """Print a side-by-side result for every test case across all models."""
    _section("PER-CASE COMPARISON")

    # Group results by case_id then model
    by_case: dict[str, dict[str, EvalResult]] = {}
    for r in results:
        if r.case_id not in by_case:
            by_case[r.case_id] = {}
        by_case[r.case_id][r.model] = r

    col_w = 16
    model_labels = {m: MODEL_DISPLAY_NAMES.get(m, m)[:col_w] for m in models}

    # Header row
    header = f"  {'ID':<5} {'Category':<12} {'Instruction':<35}"
    for m in models:
        header += f" {model_labels[m]:>{col_w}}"
    print(header)
    print(f"  {'─'*5} {'─'*12} {'─'*35}" + " " + " ".join("─"*col_w for _ in models))

    for case in TEST_CASES:
        case_results = by_case.get(case.id, {})
        instruction_short = case.instruction[:33] + ".." if len(case.instruction) > 35 else case.instruction

        row = f"  {case.id:<5} {case.category:<12} {instruction_short:<35}"
        for m in models:
            r = case_results.get(m)
            if r is None:
                row += f" {'N/A':>{col_w}}"
            elif not r.parse_success:
                row += f" {'FAIL':>{col_w}}"
            else:
                symbol = "✓ CORRECT" if r.fully_correct else "✗ PARTIAL"
                row += f" {symbol:>{col_w}}"
        print(row)


def print_field_accuracy_breakdown(results: list[EvalResult], models: list[str]) -> None:
    """Print field-level accuracy per model."""
    _section("FIELD-LEVEL ACCURACY BREAKDOWN")

    fields = [
        ("action_correct",      "Action"),
        ("object_correct",      "Object"),
        ("destination_correct", "Destination"),
        ("spatial_correct",     "Spatial Relation"),
        ("confidence_correct",  "Confidence Level"),
    ]

    col_w = 18
    header = f"  {'Field':<22}"
    for m in models:
        header += f" {MODEL_DISPLAY_NAMES.get(m, m)[:col_w]:>{col_w}}"
    print(header)
    print("  " + "─"*22 + " " + " ".join("─"*col_w for _ in models))

    by_model = {m: [r for r in results if r.model == m] for m in models}

    for attr, label in fields:
        row = f"  {label:<22}"
        for m in models:
            model_results = by_model[m]
            relevant = [r for r in model_results if getattr(r, attr) is not None]
            correct  = sum(getattr(r, attr) for r in relevant if getattr(r, attr))
            pct = f"{100*correct/len(relevant):.1f}%" if relevant else "—"
            row += f" {pct:>{col_w}}"
        print(row)


def print_latency_analysis(results: list[EvalResult], models: list[str]) -> None:
    """Print latency statistics per model and category."""
    import statistics

    _section("LATENCY ANALYSIS (ms)")

    col_w = 16
    header = f"  {'Category':<18} {'Metric':<10}"
    for m in models:
        header += f" {MODEL_DISPLAY_NAMES.get(m, m)[:col_w]:>{col_w}}"
    print(header)
    print("  " + "─"*18 + " " + "─"*10 + " " + " ".join("─"*col_w for _ in models))

    categories = ["OVERALL"] + sorted(get_all_categories())

    for cat in categories:
        for metric_label, fn in [
            ("mean", lambda vals: statistics.mean(vals) if vals else None),
            ("p95",  lambda vals: sorted(vals)[int(0.95*len(vals))-1] if len(vals) >= 2 else (vals[0] if vals else None)),
            ("max",  lambda vals: max(vals) if vals else None),
        ]:
            row = f"  {cat:<18} {metric_label:<10}"
            for m in models:
                if cat == "OVERALL":
                    vals = [r.latency_ms for r in results if r.model == m and r.parse_success]
                else:
                    vals = [r.latency_ms for r in results if r.model == m and r.category == cat and r.parse_success]

                val = fn(vals)
                display = f"{val:.0f}" if val is not None else "—"
                row += f" {display:>{col_w}}"
            print(row)

        print()


def print_failure_analysis(results: list[EvalResult], models: list[str]) -> None:
    """Print a breakdown of failed and partially correct cases."""
    _section("FAILURE ANALYSIS")

    for m in models:
        model_results = [r for r in results if r.model == m]
        failures  = [r for r in model_results if not r.parse_success]
        partials  = [r for r in model_results if r.parse_success and not r.fully_correct]

        print(f"\n  {MODEL_DISPLAY_NAMES.get(m, m)}")

        if failures:
            print(f"    Complete failures ({len(failures)}):")
            for r in failures:
                print(f"      [{r.case_id}] {r.instruction[:50]}")
                if r.error:
                    print(f"             Error: {r.error[:80]}")
        else:
            print(f"    Complete failures: none")

        if partials:
            print(f"    Partial failures ({len(partials)}):")
            for r in partials:
                wrong_fields = []
                if r.action_correct is False:      wrong_fields.append("action")
                if r.object_correct is False:      wrong_fields.append("object")
                if r.destination_correct is False: wrong_fields.append("destination")
                if r.spatial_correct is False:     wrong_fields.append("spatial")
                print(f"      [{r.case_id}] {r.instruction[:45]:<45}  wrong: {', '.join(wrong_fields)}")
        else:
            print(f"    Partial failures: none")


def print_use_case_evaluation(results: list[EvalResult], models: list[str]) -> None:
    """Print evaluation strategy results organised by use case category."""
    _section("EVALUATION BY USE CASE CATEGORY")

    categories = sorted(get_all_categories())
    category_descriptions = {
        "simple":     "Basic single-action instructions (pick, place, move, locate)",
        "spatial":    "Instructions with positional relationships (left of, near, on top of)",
        "synonym":    "Non-standard action words that map to valid actions",
        "multi_step": "Instructions implying a sequence of two actions",
        "ambiguous":  "Underspecified or vague instructions — should return low confidence",
        "edge_case":  "Unknown objects, formatting variations, boundary conditions",
    }

    col_w = 16
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        n_cases = len(set(r.case_id for r in cat_results))
        desc = category_descriptions.get(cat, "")

        print(f"\n  ▸ {cat.upper():<14}  {desc}")
        print(f"    {'─'*65}")

        header = f"    {'Model':<30} {'Accuracy':>10} {'Avg ms':>8} {'Errors':>8}"
        print(header)

        for m in models:
            m_cat = [r for r in cat_results if r.model == m]
            if not m_cat:
                continue

            accuracy = 100 * sum(r.fully_correct for r in m_cat) / len(m_cat)
            avg_lat  = sum(r.latency_ms for r in m_cat if r.parse_success) / max(sum(r.parse_success for r in m_cat), 1)
            errors   = sum(not r.parse_success for r in m_cat)

            name = MODEL_DISPLAY_NAMES.get(m, m)[:28]
            print(f"    {name:<30} {accuracy:>9.1f}% {avg_lat:>7.0f}ms {errors:>7}")

        print(f"    ({n_cases} test cases)")


def print_recommendation(metrics: dict) -> None:
    """Print a model recommendation based on the metrics."""
    _section("MODEL RECOMMENDATION")

    models = metrics["models"]
    if not models:
        return

    scores: dict[str, float] = {}
    for m in models:
        overall = metrics["by_model"][m]["overall"]
        # Composite score: weight accuracy heavily, penalise latency and errors
        score = (
            overall.get("instruction_accuracy", 0) * 0.5 +
            overall.get("parse_success_rate", 0)   * 0.3 +
            (100 - overall.get("error_rate", 0))    * 0.1 +
            max(0, 100 - overall.get("avg_latency_ms", 9999) / 100) * 0.1
        )
        scores[m] = score

    best = max(scores, key=scores.get)
    best_name = MODEL_DISPLAY_NAMES.get(best, best)

    print(f"\n  Based on instruction accuracy, parse success rate, latency,")
    print(f"  and error rate, the recommended model for this pipeline is:\n")
    print(f"    ★  {best_name}  (composite score: {scores[best]:.1f}/100)")
    print()

    for m in sorted(scores, key=scores.get, reverse=True):
        name = MODEL_DISPLAY_NAMES.get(m, m)
        overall = metrics["by_model"][m]["overall"]
        print(f"    {name:<35}  score: {scores[m]:5.1f}  "
              f"accuracy: {overall.get('instruction_accuracy',0):5.1f}%  "
              f"latency: {overall.get('avg_latency_ms',0):6.0f}ms")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_comparison_report(
    models: Optional[list[str]] = None,
    export_csv: bool = True,
    export_json: bool = True,
    csv_path: str = "evaluation_metrics.csv",
    json_path: str = "evaluation_results.json",
) -> None:
    """
    Run the full multi-model comparative evaluation and print a complete report.

    Args:
        models:      Models to evaluate. Defaults to all models with API keys set.
        export_csv:  Write metrics to a CSV file.
        export_json: Write raw results to a JSON file.
        csv_path:    Path for the CSV output.
        json_path:   Path for the JSON output.
    """

    if models is None:
        models = get_available_models()

    if not models:
        print("No models available. Set at least one API key in your .env file.")
        print("  OPENAI_API_KEY   for GPT-4o")
        print("  GEMINI_API_KEY   for Gemini")
        print("  DEEPSEEK_API_KEY for DeepSeek")
        return

    _header("MULTIMODAL LLM — COMPARATIVE EVALUATION REPORT")
    print(f"  Project: P54 Embodied Multimodal LLM for Industrial Task Planning")
    print(f"  Models:  {', '.join(MODEL_DISPLAY_NAMES.get(m, m) for m in models)}")
    print(f"  Cases:   {len(TEST_CASES)} test instructions across {len(get_all_categories())} categories")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    results = run_evaluation(models=models, verbose=True)

    if not results:
        print("No results returned. Check API keys and connectivity.")
        return

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(results)

    # ── Print all report sections ──────────────────────────────────────────────
    print_metrics_table(metrics)
    print_field_accuracy_breakdown(results, models)
    print_latency_analysis(results, models)
    print_use_case_evaluation(results, models)
    print_per_case_comparison(results, models)
    print_failure_analysis(results, models)
    print_recommendation(metrics)

    # ── Export ─────────────────────────────────────────────────────────────────
    if export_csv:
        export_metrics_csv(metrics, csv_path)

    if export_json:
        raw = []
        for r in results:
            raw.append({
                "model":              r.model,
                "model_display":      r.model_display,
                "case_id":            r.case_id,
                "category":           r.category,
                "instruction":        r.instruction,
                "parse_success":      r.parse_success,
                "fully_correct":      r.fully_correct,
                "action_correct":     r.action_correct,
                "object_correct":     r.object_correct,
                "destination_correct": r.destination_correct,
                "spatial_correct":    r.spatial_correct,
                "confidence_correct": r.confidence_correct,
                "latency_ms":         r.latency_ms,
                "error":              r.error,
                "parsed":             r.parsed.model_dump(mode="json") if r.parsed else None,
            })
        with open(json_path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"Raw results exported to {json_path}")

    _header("REPORT COMPLETE")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run LLM comparative evaluation")
    ap.add_argument("--models", nargs="+", choices=["openai", "gemini", "deepseek"],
                    help="Models to evaluate (default: all with API keys set)")
    ap.add_argument("--no-csv",  action="store_true", help="Skip CSV export")
    ap.add_argument("--no-json", action="store_true", help="Skip JSON export")
    args = ap.parse_args()

    run_comparison_report(
        models=args.models,
        export_csv=not args.no_csv,
        export_json=not args.no_json,
    )