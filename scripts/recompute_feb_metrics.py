from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

from train_model import (
    _best_threshold_for_precision,
    _build_machine_code_map,
    _build_machine_metadata,
    _build_reconstructed_frame,
    _compute_future_scrap_label,
    _encode_categorical_columns,
    _load_available_columns,
    _load_existing_machine_codes,
    _load_reference_feature_columns,
    _machine_sort_key,
    _normalize_machine_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
HORIZON_MINUTES = int(os.getenv("TRAIN_LABEL_HORIZON_MINUTES", "30"))
FEB_PATH = PROJECT_ROOT / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "scrap_risk_model_v5.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features_v5.pkl"
CAT_MAP_PATH = PROJECT_ROOT / "metrics" / "categorical_encodings_v5.json"
REGISTRY_PATH = PROJECT_ROOT / "metrics" / "machine_registry_v5.json"
OUTPUT_PATH = PROJECT_ROOT / "metrics" / "feb_current_model_report.json"
THRESHOLDS_PATH = PROJECT_ROOT / "metrics" / "machine_thresholds_v5.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_map_from_frame(
    frame: pd.DataFrame,
    score_col: str,
    label_col: str,
    group_col: str,
    target_precision: float = 0.55,
) -> dict[str, float]:
    if frame.empty or score_col not in frame.columns or label_col not in frame.columns or group_col not in frame.columns:
        return {}

    thresholds: dict[str, float] = {}
    for group_name, group_df in frame.groupby(group_col, sort=True):
        y_group = group_df[label_col].to_numpy(dtype=np.uint8, copy=True)
        score_group = group_df[score_col].to_numpy(dtype=np.float32, copy=True)
        if len(y_group) == 0:
            continue
        if len(np.unique(y_group)) < 2:
            thresholds[str(group_name)] = 0.5
            continue
        best = _best_threshold_for_precision(y_group, score_group, target_precision=target_precision)
        thresholds[str(group_name)] = float(best["threshold"])
    return thresholds


def main() -> None:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "rb") as f:
        model_feature_cols = list(pickle.load(f))
    category_payload = _load_json(CAT_MAP_PATH)
    registry_payload = _load_json(REGISTRY_PATH)

    machine_codes = registry_payload.get("machine_codes") or {}
    machine_defs = {}
    machine_ids: list[str] = []
    for machine in registry_payload.get("machines", []):
        if not isinstance(machine, dict):
            continue
        machine_id = str(machine.get("machine_id_normalized", "")).strip().upper()
        if not machine_id:
            continue
        machine_ids.append(machine_id)
        machine_defs[machine_id] = str(machine.get("type", "UNKNOWN")).upper()

    machine_ids = sorted(set(machine_ids), key=_machine_sort_key)
    machine_code_map = _build_machine_code_map(machine_ids, seed_map=machine_codes)
    machine_definitions_by_id = {
        str(machine.get("machine_id_normalized", "")).strip().upper(): str(machine.get("type", "UNKNOWN")).upper()
        for machine in registry_payload.get("machines", [])
        if isinstance(machine, dict) and str(machine.get("machine_id_normalized", "")).strip()
    }

    base_feature_cols = _load_reference_feature_columns(PROJECT_ROOT, include_machine_context=True)
    feb_available = _load_available_columns(FEB_PATH)
    feb_required = [
        c
        for c in ("timestamp", "machine_id_normalized", "is_scrap_actual", "scrap_probability", "predicted_scrap", "machine_definition")
        if c in feb_available
    ]
    feb_base_cols = [c for c in base_feature_cols if c in feb_available]
    common_cols = list(dict.fromkeys(feb_required + feb_base_cols))

    machine_id_map = category_payload.get("machine_id_code", {})
    machine_definition_map = category_payload.get("machine_definition_code", {})
    tool_id_map = category_payload.get("tool_id_code", {})
    machine_type_map = category_payload.get("machine_type_code", {})
    part_number_map = category_payload.get("part_number_code", {})
    category_maps = {
        "machine_id_code": machine_id_map,
        "machine_definition_code": machine_definition_map,
        "tool_id_code": tool_id_map,
        "machine_type_code": machine_type_map,
        "part_number_code": part_number_map,
    }

    all_rows = []
    per_machine = []

    for machine in machine_ids:
        machine_df = pd.read_parquet(
            FEB_PATH,
            columns=common_cols,
            filters=[("machine_id_normalized", "==", machine)],
        )
        if machine_df.empty:
            continue

        machine_df["timestamp"] = pd.to_datetime(machine_df["timestamp"], errors="coerce", utc=True)
        machine_df = machine_df.loc[machine_df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
        machine_df["machine_id_normalized"] = machine_df["machine_id_normalized"].astype(str).map(_normalize_machine_id)
        machine_df["machine_definition"] = machine_df["machine_id_normalized"].map(
            lambda m: machine_definitions_by_id.get(m, "UNKNOWN")
        )

        feb_features = _build_reconstructed_frame(machine_df, model_feature_cols, machine_code_map)
        feb_features["future_scrap"] = _compute_future_scrap_label(
            machine_df,
            horizon_minutes=HORIZON_MINUTES,
            event_col="is_scrap_actual",
        )
        feb_features["machine_id_normalized"] = machine_df["machine_id_normalized"].values
        feb_features["machine_definition"] = machine_df["machine_definition"].values
        meta = feb_features["machine_id_normalized"].map(lambda m: _build_machine_metadata(m))
        feb_features["tool_id"] = meta.map(lambda item: item["tool_id"])
        feb_features["machine_type"] = meta.map(lambda item: item["type"])
        feb_features["part_number"] = meta.map(lambda item: item["part_number"])
        feb_features = _encode_categorical_columns(feb_features, category_maps, machine_definitions_by_id)
        for code_col in ("machine_id_code", "machine_definition_code", "tool_id_code", "machine_type_code", "part_number_code"):
            feb_features[code_col] = pd.to_numeric(feb_features[code_col], errors="coerce").fillna(0).astype(np.int32)

        for feat in model_feature_cols:
            if feat not in feb_features.columns:
                feb_features[feat] = 0.0

        feb_X = feb_features[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
        feb_y = feb_features["future_scrap"].to_numpy(dtype=np.uint8, copy=True)
        feb_proba = model.predict(feb_X, num_iteration=getattr(model, "best_iteration", None))
        feb_pred = (feb_proba >= 0.5).astype(int)
        threshold = _best_threshold_for_precision(feb_y, feb_proba, target_precision=0.55)
        tuned_pred = (feb_proba >= threshold["threshold"]).astype(int)

        if "predicted_scrap" in machine_df.columns:
            baseline_pred = machine_df["predicted_scrap"].astype(int).to_numpy(copy=True)
        else:
            baseline_pred = feb_pred

        per_machine.append(
            {
                "machine": machine,
                "rows": int(len(feb_y)),
                "positives": int(feb_y.sum()),
                "baseline": {
                    "accuracy": float(accuracy_score(feb_y, baseline_pred)),
                    "precision": float(precision_score(feb_y, baseline_pred, zero_division=0)),
                    "recall": float(recall_score(feb_y, baseline_pred, zero_division=0)),
                    "f1": float(f1_score(feb_y, baseline_pred, zero_division=0)),
                    "predicted_positives": int(baseline_pred.sum()),
                },
                "tuned": {
                    "threshold": float(threshold["threshold"]),
                    "accuracy": float(accuracy_score(feb_y, tuned_pred)),
                    "precision": float(precision_score(feb_y, tuned_pred, zero_division=0)),
                    "recall": float(recall_score(feb_y, tuned_pred, zero_division=0)),
                    "f1": float(f1_score(feb_y, tuned_pred, zero_division=0)),
                    "predicted_positives": int(tuned_pred.sum()),
                },
                "auc": float(roc_auc_score(feb_y, feb_proba)) if len(np.unique(feb_y)) > 1 else None,
            }
        )

        all_rows.append(
            pd.DataFrame(
                {
                    "machine_id_normalized": feb_features["machine_id_normalized"].values,
                    "y": feb_y,
                    "proba": feb_proba,
                    "baseline_pred": baseline_pred,
                    "tuned_pred": tuned_pred,
                    "tool_id": feb_features["tool_id"].values,
                    "machine_type": feb_features["machine_type"].values,
                }
            )
        )

    all_df = pd.concat(all_rows, ignore_index=True)
    y = all_df["y"].astype(int)
    proba = all_df["proba"].astype(float)
    baseline_pred = all_df["baseline_pred"].astype(int)
    tuned_pred = all_df["tuned_pred"].astype(int)

    overall = {
        "baseline": {
            "accuracy": float(accuracy_score(y, baseline_pred)),
            "precision": float(precision_score(y, baseline_pred, zero_division=0)),
            "recall": float(recall_score(y, baseline_pred, zero_division=0)),
            "f1": float(f1_score(y, baseline_pred, zero_division=0)),
            "auc": float(roc_auc_score(y, proba)),
            "predicted_positives": int(baseline_pred.sum()),
        },
        "tuned": {
            "accuracy": float(accuracy_score(y, tuned_pred)),
            "precision": float(precision_score(y, tuned_pred, zero_division=0)),
            "recall": float(recall_score(y, tuned_pred, zero_division=0)),
            "f1": float(f1_score(y, tuned_pred, zero_division=0)),
            "auc": float(roc_auc_score(y, proba)),
            "predicted_positives": int(tuned_pred.sum()),
        },
        "precision_target_threshold": _best_threshold_for_precision(y.to_numpy(dtype=np.uint8), proba.to_numpy(dtype=np.float32), target_precision=0.55),
    }

    overall_threshold = float(overall["precision_target_threshold"]["threshold"])
    threshold_payload = {
        "precision_target": 0.55,
        "default_threshold": overall_threshold,
        "overall_threshold": overall_threshold,
        "per_machine_thresholds": {
            str(item["machine"]): float(item["tuned"]["threshold"])
            for item in per_machine
            if item.get("tuned", {}).get("threshold") is not None
        },
        "per_tool_thresholds": _threshold_map_from_frame(all_df, "proba", "y", "tool_id", target_precision=0.55),
        "per_type_thresholds": _threshold_map_from_frame(all_df, "proba", "y", "machine_type", target_precision=0.55),
    }

    THRESHOLDS_PATH.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

    payload = {
        "version": category_payload.get("version", "unknown"),
        "training_feature_count": len(model_feature_cols),
        "feb_test": {
            "rows": int(len(all_df)),
            **overall,
            "per_machine": per_machine,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote FEB thresholds to {THRESHOLDS_PATH}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
