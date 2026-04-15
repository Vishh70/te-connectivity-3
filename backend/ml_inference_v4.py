import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH_V5 = PROJECT_ROOT / "models" / "scrap_risk_model_v5.pkl"
FEATURES_PATH_V5 = PROJECT_ROOT / "models" / "model_features_v5.pkl"
CALIBRATOR_PATH_V5 = PROJECT_ROOT / "models" / "scrap_risk_calibrator_v5.pkl"
MODEL_PATH_V4 = PROJECT_ROOT / "models" / "scrap_risk_model_v4.pkl"
FEATURES_PATH_V4 = PROJECT_ROOT / "models" / "model_features_v4.pkl"

MODEL_PATH = MODEL_PATH_V5 if MODEL_PATH_V5.exists() else MODEL_PATH_V4
FEATURES_PATH = FEATURES_PATH_V5 if FEATURES_PATH_V5.exists() else FEATURES_PATH_V4
CALIBRATOR_PATH = CALIBRATOR_PATH_V5 if CALIBRATOR_PATH_V5.exists() else None

_MODEL_CACHE = {}

def _get_ml_components():
    """Internal lazy loader for model, features, and calibrator."""
    global _MODEL_CACHE
    if not _MODEL_CACHE:
        print(f"[v4] Loading model components from: {MODEL_PATH}")
        try:
            _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
            _MODEL_CACHE['features'] = joblib.load(FEATURES_PATH)
            _MODEL_CACHE['calibrator'] = joblib.load(CALIBRATOR_PATH) if CALIBRATOR_PATH else None
            print(f"[v4] Model and {len(_MODEL_CACHE['features'])} features loaded into cache.")
        except Exception as e:
            print(f"[v4] ERROR loading ML components: {str(e)}")
            _MODEL_CACHE['model'] = None
            _MODEL_CACHE['features'] = []
            _MODEL_CACHE['calibrator'] = None
    return _MODEL_CACHE.get('model'), _MODEL_CACHE.get('features'), _MODEL_CACHE.get('calibrator')


def _predict_raw_probability(model, X: pd.DataFrame) -> float:
    """Return the model's raw probability or score before any calibration."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[:, 1][0])
    return float(model.predict(X)[0])

def __getattr__(name):
    """Module-level lazy loader for backward compatibility with direct imports."""
    if name in ('model', 'model_features', 'calibrator'):
        components = _get_ml_components()
        if name == 'model': return components[0]
        if name == 'model_features': return components[1]
        if name == 'calibrator': return components[2]
    raise AttributeError(f"module {__name__} has no attribute {name}")

def predict_scrap_probability(sensor_row: dict):
    model, model_features, calibrator = _get_ml_components()
    if model is None:
        return 0.0

    # Determine the exact features the model expects
    if hasattr(model, "feature_name"):
        expected_features = model.feature_name() if callable(model.feature_name) else model.feature_name
    elif hasattr(model, "feature_name_"):
        expected_features = model.feature_name_
    else:
        expected_features = model_features

    # Build the row with fallback to 0.0 for missing features
    row = {f: sensor_row.get(f, 0.0) for f in expected_features}
    
    # Create DataFrame and ensure column order matches model expectations exactly
    X = pd.DataFrame([row])[list(expected_features)].fillna(0)

    try:
        # Use the raw LightGBM score for the dashboard. The existing calibrator
        # is currently collapsing most scores to ~1.0, which makes the control
        # room look permanently critical even when the model output is normal.
        prob = _predict_raw_probability(model, X)
    except Exception as e:
        print(f"[v4] Prediction error: {e}")
        # Log the shape mismatch if it happens
        if "number of features" in str(e).lower():
            print(f"[v4] SHAPE MISMATCH: Model expected {len(expected_features)} features, but got {X.shape[1]}")
        return 0.0

    return float(prob)
