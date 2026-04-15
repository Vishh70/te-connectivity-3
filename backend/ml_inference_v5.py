import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Senior Pro V5 Paths
MODEL_PATH = PROJECT_ROOT / "models" / "scrap_risk_model_v5.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "model_features_v5.pkl"
NORMALIZATION_STATS_PATH = PROJECT_ROOT / "models" / "machine_normalization_v5.json"

_MODEL_CACHE = {}
_GLOBAL_STATS_CACHE = None
_LOGGED_FALLBACKS = set()

def _get_ml_components():
    """Internal lazy loader for model, features, and normalization stats."""
    global _MODEL_CACHE
    if not _MODEL_CACHE:
        t_start = time.perf_counter()
        print(f"[v5-pro] Initializing Senior Pro Inference Engine...")
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"V5 Model not found at {MODEL_PATH}")
                
            model = joblib.load(MODEL_PATH)
            features = joblib.load(FEATURES_PATH)
            
            _MODEL_CACHE['model'] = model
            _MODEL_CACHE['features'] = features
            
            stats = {}
            if NORMALIZATION_STATS_PATH.exists():
                with open(NORMALIZATION_STATS_PATH, "r") as f:
                    stats = json.load(f)
                _MODEL_CACHE['stats'] = stats
                print(f"[v5-pro] Loaded normalization stats for {len(stats)} machines.")
            else:
                print(f"[v5-pro] WARNING: Normalization stats missing. Using raw data fallback.")
                _MODEL_CACHE['stats'] = {}
            
            # Senior Pro Performance: Warm up the Fleet-Wide stats cache on startup
            _get_global_stats(stats)
            
            t_end = time.perf_counter()
            print(f"[v5-pro] Model and {_MODEL_CACHE['features'].shape[0] if hasattr(_MODEL_CACHE['features'], 'shape') else len(_MODEL_CACHE['features'])} features loaded in {t_end - t_start:.2f}s.")
        except Exception as e:
            print(f"[v5-pro] ERROR loading ML components: {str(e)}")
            _MODEL_CACHE['model'] = None
            _MODEL_CACHE['features'] = []
            _MODEL_CACHE['stats'] = {}
            
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('features'), _MODEL_CACHE.get('stats')


def _get_global_stats(stats: dict) -> dict:
    """
    Computes a 'Fleet Average' normalization for unknown machines.
    Aggregates the mean and std across all known assets using vectorized operations.
    """
    global _GLOBAL_STATS_CACHE
    if _GLOBAL_STATS_CACHE is not None:
        return _GLOBAL_STATS_CACHE
        
    if not stats:
        return {}
        
    t_start = time.perf_counter()
    print(f"[v5-pro] Computing Fleet-Wide normalization baseline from {len(stats)} known assets...")
    
    # Senior Pro Fix: Vectorized aggregation using a single pass dictionary conversion
    # This avoids nested list comprehensions which can hang during large-scale startup.
    data_points = []
    for machine_id, features in stats.items():
        for sensor, values in features.items():
            data_points.append({
                'sensor': sensor,
                'mean': values['mean'],
                'std': values['std']
            })
            
    if not data_points:
        _GLOBAL_STATS_CACHE = {}
        return {}
        
    df = pd.DataFrame(data_points)
    # Group by sensor and take the mean of means/stds across machines
    fleet_stats = df.groupby('sensor').agg({'mean': 'mean', 'std': 'mean'}).to_dict('index')
    
    _GLOBAL_STATS_CACHE = fleet_stats
    t_end = time.perf_counter()
    print(f"[v5-pro] Fleet-Wide baseline cached for {len(fleet_stats)} sensors in {t_end - t_start:.2f}s.")
    return fleet_stats


def _apply_machine_normalization(machine_id: str, sensor_row: dict, stats: dict) -> dict:
    """
    Applies Senior-level Z-score normalization.
    Uses Machine-Specific stats if available, otherwise falls back to Fleet-Wide averages.
    """
    m_stats = stats.get(machine_id)
    if not m_stats:
        # Senior Upgrade: Fallback to Fleet Average instead of giving up
        m_stats = _get_global_stats(stats)
        if not m_stats:
            return sensor_row
        
        # Log suppression: only log once per machine session
        if machine_id not in _LOGGED_FALLBACKS:
            print(f"[v5-pro] Applying Fleet-Wide fallback normalization for unknown machine {machine_id}")
            _LOGGED_FALLBACKS.add(machine_id)
        
    normalized_row = sensor_row.copy()
    
    for feat, val in sensor_row.items():
        if feat in m_stats:
            mean = m_stats[feat]["mean"]
            std = m_stats[feat]["std"]
            try:
                # Value - Mean / Std
                normalized_row[feat] = (float(val) - mean) / std
            except (ValueError, TypeError):
                continue
                
    return normalized_row


def predict_scrap_probability_v5(machine_id: str, sensor_row: dict):
    """
    Senior Pro V5 Inference Entry Point.
    - Normalizes data based on machine context.
    - Runs LightGBM prediction.
    """
    model, model_features, stats = _get_ml_components()
    if model is None:
        return 0.0

    # 1. Normalize based on hardware context
    normalized_row = _apply_machine_normalization(machine_id, sensor_row, stats)

    # 2. Build the exact feature vector expected by the model
    if hasattr(model, "feature_name"):
        expected_features = model.feature_name() if callable(model.feature_name) else model.feature_name
    elif hasattr(model, "feature_name_"):
        expected_features = model.feature_name_
    else:
        expected_features = model_features

    # Fallback to 0.0 for missing features (e.g. machine-specific ones)
    row_data = {f: normalized_row.get(f, 0.0) for f in expected_features}
    
    # 3. Predict
    X = pd.DataFrame([row_data])[list(expected_features)].fillna(0)
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[:, 1][0])
        else:
            prob = float(model.predict(X)[0])
        return prob
    except Exception as e:
        print(f"[v5-pro] Prediction error for {machine_id}: {e}")
        return 0.0
