from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COMMON_ENV = {
    "TRAIN_LABEL_HORIZON_MINUTES": "15",
    "TRAIN_MAX_ROWS_PER_MACHINE": "15000",
    "TRAIN_BIG_DATASET_ROWS_PER_MACHINE": "60000",
    "TRAIN_SKIP_FEB_EVAL": "1",
}

BRANCHES = [
    ("big_dataset", {"TRAIN_USE_RAW_FILES": "0"}),
    ("raw_files", {"TRAIN_USE_RAW_FILES": "1"}),
]

ARTIFACTS = [
    Path("models/scrap_risk_model_v5.pkl"),
    Path("models/model_features_v5.pkl"),
    Path("models/scrap_risk_calibrator_v5.pkl"),
    Path("metrics/training_metrics_v5.json"),
    Path("metrics/final_model_report_v5.json"),
    Path("metrics/feb_current_model_report.json"),
    Path("metrics/machine_thresholds_v5.json"),
    Path("metrics/machine_registry_v5.json"),
    Path("metrics/categorical_encodings_v5.json"),
    Path("features/feature_importance_v5.csv"),
]

SNAPSHOT_ROOT = PROJECT_ROOT / "artifacts" / "stage1_runs"
SUMMARY_PATH = SNAPSHOT_ROOT / "stage1_dual_retrain_summary.json"


def _run(command: list[str], extra_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(extra_env)
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=True)


def _load_feb_summary() -> dict:
    report_path = PROJECT_ROOT / "metrics" / "feb_current_model_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing FEB report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _score(summary: dict) -> tuple[float, float, float, float]:
    feb = summary.get("feb_test", {})
    tuned = feb.get("tuned", {})
    return (
        float(tuned.get("f1", 0.0)),
        float(tuned.get("recall", 0.0)),
        float(tuned.get("precision", 0.0)),
        float(tuned.get("auc", 0.0)),
    )


def _snapshot_branch(branch_name: str) -> Path:
    branch_dir = SNAPSHOT_ROOT / branch_name
    for relative_path in ARTIFACTS:
        src = PROJECT_ROOT / relative_path
        if not src.exists():
            continue
        dest = branch_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    summary = _load_feb_summary()
    (branch_dir / "feb_current_model_report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return branch_dir


def _restore_snapshot(branch_dir: Path) -> None:
    for relative_path in ARTIFACTS:
        src = branch_dir / relative_path
        if not src.exists():
            continue
        dest = PROJECT_ROOT / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _run_realtime_check() -> dict[str, dict]:
    from backend.run_realtime_check import run

    machines = ["M231", "M356", "M471", "M607", "M612"]
    results: dict[str, dict] = {}
    for machine in machines:
        result = run(machine)
        results[machine] = result
        print(
            "{}: {} risk={:.4f} threshold={:.4f}".format(
                machine,
                result.get("alert_level", "UNKNOWN"),
                float(result.get("ml_risk_probability", 0.0)),
                float(result.get("calibrated_threshold", 0.0)),
            )
        )
    return results


def main() -> None:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)

    branch_results: list[dict] = []
    for branch_name, branch_env in BRANCHES:
        print(f"\n=== Stage 1 branch: {branch_name} ===")
        _run([PYTHON, "scripts/train_model.py"], branch_env)
        _run([PYTHON, "scripts/recompute_feb_metrics.py"], branch_env)
        summary = _load_feb_summary()
        branch_dir = _snapshot_branch(branch_name)
        branch_results.append(
            {
                "branch": branch_name,
                "snapshot_dir": str(branch_dir),
                "summary": summary,
                "score": _score(summary),
            }
        )
        print(
            "Branch {} FEB tuned: precision={:.4f} recall={:.4f} f1={:.4f} auc={:.4f}".format(
                branch_name,
                summary["feb_test"]["tuned"]["precision"],
                summary["feb_test"]["tuned"]["recall"],
                summary["feb_test"]["tuned"]["f1"],
                summary["feb_test"]["tuned"]["auc"],
            )
        )

    winner = max(branch_results, key=lambda item: item["score"])
    winner_dir = Path(winner["snapshot_dir"])
    _restore_snapshot(winner_dir)

    print(
        "\nWinner: {} | tuned precision={:.4f} recall={:.4f} f1={:.4f} auc={:.4f}".format(
            winner["branch"],
            winner["summary"]["feb_test"]["tuned"]["precision"],
            winner["summary"]["feb_test"]["tuned"]["recall"],
            winner["summary"]["feb_test"]["tuned"]["f1"],
            winner["summary"]["feb_test"]["tuned"]["auc"],
        )
    )

    realtime_results = _run_realtime_check()
    summary_payload = {
        "branches": branch_results,
        "winner": winner["branch"],
        "realtime": realtime_results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nStage 1 dual retrain summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
