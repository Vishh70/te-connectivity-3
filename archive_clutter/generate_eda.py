from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
RAW_ANALYSIS_PATH = DOCS_DIR / "raw_data_analysis.json"
FEB_REPORT_PATH = PROJECT_ROOT / "metrics" / "feb_current_model_report.json"
FINAL_REPORT_PATH = PROJECT_ROOT / "metrics" / "final_model_report_v5.json"
OUTPUT_PATH = DOCS_DIR / "EDA_REPORT.md"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_pct(value) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


def _summarize_sources(raw_analysis: dict) -> list[str]:
    lines = []
    sources = raw_analysis.get("sources", {}) or {}
    for source_name, machine_map in sources.items():
        machines = sorted(machine_map.keys())
        lines.append(f"- `{source_name}`: {', '.join(machines) if machines else 'no sampled machines'}")
    return lines


def _summarize_processed_master(raw_analysis: dict) -> list[str]:
    pm = raw_analysis.get("processed_master", {}) or {}
    rows = pm.get("total_rows", "N/A")
    cols = pm.get("n_columns", "N/A")
    date_range = pm.get("date_range", {})
    rows_per_machine = pm.get("rows_per_machine", {})
    scrap_rates = pm.get("scrap_rate_per_machine", {})
    dead_cols = pm.get("dead_cols", [])
    roc_count = pm.get("roc_feature_count", 0)

    lines = [
        f"- Rows: `{rows:,}`" if isinstance(rows, int) else f"- Rows: `{rows}`",
        f"- Columns: `{cols}`",
        f"- Date range: `{date_range.get('min', 'N/A')} to {date_range.get('max', 'N/A')}`",
        f"- ROC features: `{roc_count}`",
        f"- Dead columns: {', '.join(dead_cols) if dead_cols else 'none detected'}",
    ]

    if rows_per_machine:
        machine_rows = ", ".join(f"{machine}: {count:,}" for machine, count in rows_per_machine.items())
        lines.append(f"- Rows per machine: {machine_rows}")

    if scrap_rates:
        machine_rates = ", ".join(f"{machine}: {_format_pct(rate)}" for machine, rate in scrap_rates.items())
        lines.append(f"- Scrap rate per machine: {machine_rates}")

    return lines


def _summarize_feb_report(feb_report: dict) -> list[str]:
    feb = feb_report.get("feb_test", {}) or {}
    baseline = feb.get("baseline", {}) or {}
    tuned = feb.get("tuned", {}) or {}
    precision_target = feb.get("precision_target_threshold", {}) or {}
    per_machine = feb.get("per_machine", []) or []

    lines = [
        f"- Evaluation rows: `{feb.get('rows', 'N/A')}`",
        f"- Baseline precision / recall / AUC: {_format_pct(baseline.get('precision', 0) * 100)} / {_format_pct(baseline.get('recall', 0) * 100)} / {baseline.get('auc', 'N/A'):.3f}" if baseline else "- Baseline metrics unavailable",
        f"- Tuned precision / recall / AUC: {_format_pct(tuned.get('precision', 0) * 100)} / {_format_pct(tuned.get('recall', 0) * 100)} / {tuned.get('auc', 'N/A'):.3f}" if tuned else "- Tuned metrics unavailable",
        f"- Precision-target threshold: `{precision_target.get('threshold', 'N/A')}`",
    ]

    best_machine = None
    best_precision = -1.0
    for item in per_machine:
        tuned_metrics = item.get("tuned", {}) or {}
        precision = tuned_metrics.get("precision")
        if isinstance(precision, (int, float)) and precision > best_precision:
            best_precision = float(precision)
            best_machine = item

    if best_machine:
        tuned_metrics = best_machine.get("tuned", {}) or {}
        lines.append(
            "- Best per-machine tuned precision: "
            f"`{best_machine.get('machine', 'N/A')}` at {_format_pct(tuned_metrics.get('precision', 0) * 100)} "
            f"(recall {_format_pct(tuned_metrics.get('recall', 0) * 100)})"
        )

    return lines


def build_report() -> str:
    raw_analysis = _load_json(RAW_ANALYSIS_PATH)
    feb_report = _load_json(FEB_REPORT_PATH)
    final_report = _load_json(FINAL_REPORT_PATH)

    lines = [
        "# EDA Report",
        "",
        "## Scope",
        "- Data sources: old raw data, new raw data, new data, MES files, and the processed master dataset.",
        "- Goal: document the current training data shape, quality, and the model evaluation snapshot used for the control room.",
        "",
        "## Source Coverage",
        *(_summarize_sources(raw_analysis) or ["- No raw analysis snapshot available."]),
        "",
        "## Processed Master Summary",
        *(_summarize_processed_master(raw_analysis) or ["- No processed master summary available."]),
        "",
        "## Evaluation Snapshot",
        *(_summarize_feb_report(feb_report) or ["- No FEB evaluation snapshot available."]),
        "",
        "## Model Reference",
        f"- Training feature count: `{final_report.get('training_feature_count', 'N/A')}`",
        f"- Label horizon: `{final_report.get('label_horizon_minutes', 'N/A')}` minutes" if final_report else "- Label horizon unavailable",
        "",
        "## Notes",
        "- The dashboard now consumes root-cause payloads from the backend control-room contract.",
        "- The current blocker is not wiring; it is model ceiling and future iteration on feature quality.",
    ]

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    OUTPUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
