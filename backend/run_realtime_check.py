import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
from datetime import datetime
from backend.config_limits import ML_THRESHOLDS, SAFE_LIMITS
from backend.data_access import (
    build_realtime_model_vector,
    get_recent_window,
    _get_machine_threshold,
    unified_predict_scrap,
    unified_get_model_and_features,
)
from backend.root_cause_analyzer import compute_root_causes


def _resolve_sensor_column(row, param: str):
    """Find the best matching sensor column for a configured parameter."""
    if param in row.index:
        return param

    candidates = [c for c in row.index if c.startswith(param + "__")]
    if candidates:
        return candidates[0]

    return None


def run(machine_id: str):
    """Return the current safety and ML risk status for one machine."""
    try:
        # 1. Get the latest data window.
        df = get_recent_window(machine_id, minutes=60)

        if df.empty:
            return {
                "machine_id": machine_id,
                "timestamp": None,
                "ml_risk_probability": 0.0,
                "alert_level": "LOW",
                "decision_reason": "NO DATA",
                "violations": [],
            }

        latest_row = df.iloc[-1]
        timestamp = (
            str(latest_row["event_timestamp"])
            if "event_timestamp" in latest_row.index
            else str(latest_row.name)
        )

        # 2. Check hard safety violations against configured limits.
        violations = []
        for param, limits in SAFE_LIMITS.items():
            col_name = _resolve_sensor_column(latest_row, param)
            if not col_name:
                continue

            try:
                current_val = float(latest_row[col_name])
            except (TypeError, ValueError):
                continue

            if "max" in limits and current_val > limits["max"]:
                violations.append(
                    {
                        "parameter": param,
                        "current": round(current_val, 2),
                        "limit": limits["max"],
                        "unit": limits.get("unit", ""),
                        "deviation": round(current_val - limits["max"], 2),
                        "direction": "above",
                    }
                )

            if "min" in limits and current_val < limits["min"]:
                violations.append(
                    {
                        "parameter": param,
                        "current": round(current_val, 2),
                        "limit": limits["min"],
                        "unit": limits.get("unit", ""),
                        "deviation": round(limits["min"] - current_val, 2),
                        "direction": "below",
                    }
                )

        # 3. Run ML inference using the unified engine.
        # strict=False: missing features default to 0.0 instead of raising RuntimeError,
        # which prevents fallback machines (M607/M612 narrow columns) from tripping CRITICAL.
        machine_norm = str(latest_row.get("machine_id_normalized", "")).upper()
        if not machine_norm or machine_norm == "M":
            from backend.data_access import normalize_machine_id
            machine_norm = normalize_machine_id(machine_id)
        sensor_input = build_realtime_model_vector(df, machine_norm=machine_norm, strict=False)
        ml_risk = float(unified_predict_scrap(machine_norm, sensor_input))
        calibrated_threshold = float(_get_machine_threshold(machine_norm))

        # 4. Explainable AI: Compute Root Causes if risk is significant
        root_causes = []
        if ml_risk > 0.05:
            try:
                model, feature_names = unified_get_model_and_features(machine_norm)
                if model:
                    # Format as a single row for the analyzer
                    feature_row = np.array([sensor_input.get(f, 0.0) for f in feature_names]).reshape(1, -1)
                    root_causes = compute_root_causes(model, feature_row, feature_names)
            except Exception as ex:
                print(f"[XAI] Failed to compute root causes for {machine_id}: {ex}")

        if violations:
            status = "CRITICAL"
            reason = "SAFETY VIOLATION"
        elif ml_risk >= calibrated_threshold:
            status = "HIGH"
            reason = "AI RISK PREDICTION (CALIBRATED)"
        elif ml_risk >= min(ML_THRESHOLDS["MEDIUM"], calibrated_threshold * 0.85):
            status = "MEDIUM"
            reason = "AI RISK PREDICTION (CALIBRATED)"
        elif ml_risk >= min(ML_THRESHOLDS["LOW"], calibrated_threshold * 0.60):
            status = "LOW"
            reason = "AI WARNING"
        else:
            status = "NORMAL"
            reason = "OPTIMAL"

        return {
            "machine_id": machine_id,
            "timestamp": timestamp,
            "ml_risk_probability": ml_risk,
            "calibrated_threshold": calibrated_threshold,
            "alert_level": status,
            "decision_reason": reason,
            "violations": violations,
            "root_causes": root_causes,
            "meta": {
                "processed_at": datetime.now().isoformat()
            }
        }

    except FileNotFoundError as e:
        # Don't spam the console if the user hasn't uploaded data for this machine yet.
        # Just return OFFLINE cleanly.
        return {
            "machine_id": machine_id,
            "timestamp": None,
            "ml_risk_probability": 0.0,
            "alert_level": "OFFLINE",
            "decision_reason": f"INFERENCE ERROR: File missing ({str(e).split(' at ')[0]})",
            "violations": [],
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[Checker Error] {machine_id}: {e}\n{tb}")
        return {
            "machine_id": machine_id,
            "timestamp": None,
            "ml_risk_probability": 0.0,
            "alert_level": "OFFLINE",
            "decision_reason": f"INFERENCE ERROR: {str(e)[:120]}",
            "violations": [],
        }
