import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Senior Pro V6 Paths
MODEL_PATH = PROJECT_ROOT / "models" / "scrap_risk_model_v6.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features_v5.pkl" # Reuse feature list registry
NORMALIZATION_STATS_PATH = PROJECT_ROOT / "models" / "machine_normalization_v5.json"

# Triple-Oracle Domain Constants (from processed/safe)
TOLERANCES = {
    "Cushion": 0.5,
    "Injection_time": 0.03,
    "Dosage_time": 1.0,
    "Injection_pressure": 100.0,
    "Switch_pressure": 100.0,
    "Cyl_tmp_z1": 5.0, "Cyl_tmp_z2": 5.0, "Cyl_tmp_z3": 5.0, "Cyl_tmp_z4": 5.0,
    "Cyl_tmp_z5": 5.0, "Cyl_tmp_z6": 5.0, "Cyl_tmp_z7": 5.0, "Cyl_tmp_z8": 5.0,
    "Switch_position": 0.05
}

_MODEL_CACHE = {}
_MACHINE_BUFFERS = {} # {machine_id: {sensor: deque(maxlen=20)}}
_BUFFER_LOCK = threading.Lock()

def _get_ml_components():
    global _MODEL_CACHE
    if not _MODEL_CACHE:
        try:
            print(f"[v6-pro] Initializing Triple-Oracle Inference Engine...")
            model = joblib.load(MODEL_PATH)
            # Use model features if available, fallback to existing registry
            features = model.feature_name() if hasattr(model, 'feature_name') else joblib.load(FEATURES_PATH)
            
            with open(NORMALIZATION_STATS_PATH, "r") as f:
                stats = json.load(f)
                
            _MODEL_CACHE = {
                'model': model,
                'features': features,
                'stats': stats
            }
            print(f"[v6-pro] Senior Pro V6 online. Utilizing {len(features)} Triple-Oracle features.")
        except Exception as e:
            print(f"[v6-pro] FATAL error loading V6 components: {e}")
            return None, [], {}
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('features'), _MODEL_CACHE.get('stats')

def _update_buffer_and_get_stats(machine_id: str, sensor_row: dict):
    """Maintains a 20-point state buffer for every machine to calculate real-time tolerance deviations."""
    global _MACHINE_BUFFERS
    with _BUFFER_LOCK:
        if machine_id not in _MACHINE_BUFFERS:
            _MACHINE_BUFFERS[machine_id] = {s: deque(maxlen=20) for s in TOLERANCES.keys()}
        
        buffers = _MACHINE_BUFFERS[machine_id]
        stats = {}
        for sensor in TOLERANCES.keys():
            if sensor in sensor_row:
                try:
                    val = float(sensor_row[sensor])
                    buffers[sensor].append(val)
                except:
                    pass
            
            # Calculate rolling mean as 'Set Value' reference
            if buffers[sensor]:
                stats[f"{sensor}_rolling_ref"] = float(np.mean(buffers[sensor]))
            else:
                stats[f"{sensor}_rolling_ref"] = 0.0
        return stats

def predict_scrap_probability_v6(machine_id: str, sensor_row: dict):
    """
    Stateful V6 Triple-Oracle Inference.
    - Updates machine history buffer.
    - Calculates real-time engineering tolerance violations.
    - Normalizes and predicts with LightGBM.
    """
    model, features, norm_stats = _get_ml_components()
    if model is None: return 0.0

    # 1. Update State Buffer and get reference values
    ref_stats = _update_buffer_and_get_stats(machine_id, sensor_row)
    
    # 2. Domain Engineering (Tolerance Oracle)
    enriched_row = sensor_row.copy()
    instability_count = 0
    for sensor, tol in TOLERANCES.items():
        if sensor in sensor_row:
            try:
                val = float(sensor_row[sensor])
                ref = ref_stats.get(f"{sensor}_rolling_ref", val)
                diff = abs(val - ref)
                
                violation = 1.0 if diff > tol else 0.0
                deviation = diff / max(tol, 1e-6)
                
                enriched_row[f"{sensor}_tol_violation"] = violation
                enriched_row[f"{sensor}_tol_deviation"] = deviation
                instability_count += violation
            except:
                pass
    enriched_row["process_instability_index"] = float(instability_count)

    # 3. Hardware Normalization (Z-Score)
    m_stats = norm_stats.get(machine_id, {})
    row_data = {}
    for feat in features:
        # Priority: Enriched Domain Features -> Normalized Sensor Features
        if feat in enriched_row:
            val = enriched_row[feat]
            # If it's a raw sensor needing normalization
            if feat in m_stats:
                mean = m_stats[feat]["mean"]
                std = max(m_stats[feat]["std"], 1e-6)
                val = (float(val) - mean) / std
            row_data[feat] = float(val)
        else:
            row_data[feat] = 0.0

    # 4. Predict
    try:
        X = pd.DataFrame([row_data])[list(features)].fillna(0)
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[:, 1][0])
        return float(model.predict(X)[0])
    except Exception as e:
        # print(f"[v6-pro] Inference error: {e}")
        return 0.0
