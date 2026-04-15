from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
FEATURES_DIR = PROJECT_ROOT / "features"
ARCHIVE_ROOT = METRICS_DIR / "horizon_sweeps"

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_model.py"
RECOMPUTE_SCRIPT = PROJECT_ROOT / "scripts" / "recompute_feb_metrics.py"

ARTIFACTS = {
    "model": MODEL_DIR / "scrap_risk_model_v5.pkl",
    "features": MODEL_DIR / "model_features_v5.pkl",
    "calibrator": MODEL_DIR / "scrap_risk_calibrator_v5.pkl",
    "categorical_map": METRICS_DIR / "categorical_encodings_v5.json",
    "machine_registry": METRICS_DIR / "machine_registry_v5.json",
    "thresholds": METRICS_DIR / "machine_thresholds_v5.json",
    "training_metrics": METRICS_DIR / "training_metrics_v5.json",
    "final_report": METRICS_DIR / "final_model_report_v5.json",
    "feb_report": METRICS_DIR / "feb_current_model_report.json",
    "feature_importance": FEATURES_DIR / "feature_importance_v5.csv",
}


def _parse_horizons() -> list[int]:
    raw = os.getenv("SWEEP_HORIZONS", "15,20,25,30,45")
    horizons: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            horizon = int(token)
        except ValueError:
            continue
        if horizon > 0 and horizon not in horizons:
            horizons.append(horizon)
    if not horizons:
        horizons = [30]
    return horizons


def _copy_artifacts(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, src in ARTIFACTS.items():
        if src.exists():
            shutil.copy2(src, destination / src.name)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _load_report() -> dict:
    report_path = ARTIFACTS["feb_report"]
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def _score_candidate(report: dict) -> tuple[float, float, float]:
    feb = report.get("feb_test", {})
    tuned = feb.get("tuned", {})
    precision = float(tuned.get("precision", 0.0) or 0.0)
    recall = float(tuned.get("recall", 0.0) or 0.0)
    f1 = float(tuned.get("f1", 0.0) or 0.0)
    return f1, precision, recall


def main() -> None:
    horizons = _parse_horizons()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    sweep_summary: list[dict] = []
    best_entry: dict | None = None

    base_env = os.environ.copy()
    base_env["TRAIN_USE_RAW_FILES"] = os.getenv("SWEEP_USE_RAW_FILES", "0")
    base_env["TRAIN_SKIP_FEB_EVAL"] = "1"

    for horizon in horizons:
        print(f"\n=== Horizon {horizon} minutes ===")
        run_env = base_env.copy()
        run_env["TRAIN_LABEL_HORIZON_MINUTES"] = str(horizon)

        _run([sys.executable, str(TRAIN_SCRIPT)], run_env)
        _run([sys.executable, str(RECOMPUTE_SCRIPT)], run_env)

        report = _load_report()
        feb = report.get("feb_test", {})
        candidate = {
            "horizon_minutes": horizon,
            "score": list(_score_candidate(report)),
            "report": report,
        }
        sweep_summary.append(
            {
                "horizon_minutes": horizon,
                "label_horizon_minutes": report.get("label_horizon_minutes", horizon),
                "training_feature_count": report.get("training_feature_count"),
                "baseline": feb.get("baseline", {}),
                "tuned": feb.get("tuned", {}),
                "precision_target_threshold": feb.get("precision_target_threshold", {}),
                "per_machine": feb.get("per_machine", []),
            }
        )

        archive_dir = ARCHIVE_ROOT / f"h{horizon}"
        _copy_artifacts(archive_dir)

        if best_entry is None or candidate["score"] > best_entry["score"]:
            best_entry = candidate

    summary_path = ARCHIVE_ROOT / "summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    if best_entry is None:
        print("No successful horizon run completed.")
        return

    best_horizon = int(best_entry["horizon_minutes"])
    best_archive = ARCHIVE_ROOT / f"h{best_horizon}"
    print(f"\nBest horizon: {best_horizon} minutes")
    print(f"Best score tuple (F1, precision, recall): {best_entry['score']}")

    for src_name in ARTIFACTS.values():
        archived = best_archive / src_name.name
        if archived.exists():
            shutil.copy2(archived, src_name)

    print(f"Restored best horizon artifacts from {best_archive}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
