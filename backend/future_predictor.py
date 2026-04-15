from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "future_models"

MODEL_FILES = {
    "5m": "model_scrap_5m.pkl",
    "10m": "model_scrap_10m.pkl",
    "15m": "model_scrap_15m.pkl",
    "20m": "model_scrap_20m.pkl",
    "25m": "model_scrap_25m.pkl",
    "30m": "model_scrap_30m.pkl",
}


@lru_cache(maxsize=1)
def _load_future_models():
    models = {}
    for horizon, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            models[horizon] = joblib.load(model_path)
    return models


def predict_future_risk(feature_row: dict, feature_columns: list):
    models = _load_future_models()
    if not models:
        return {}

    results = {}
    for horizon, model in models.items():
        # Determine features for this specific model
        if hasattr(model, "feature_name"):
            m_features = model.feature_name() if callable(model.feature_name) else model.feature_name
        elif hasattr(model, "feature_name_"):
            m_features = model.feature_name_
        else:
            m_features = feature_columns

        # Align input row to this specific model's schema exactly
        row = {f: feature_row.get(f, 0.0) for f in m_features}
        X = pd.DataFrame([row])[list(m_features)].fillna(0)
        
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = model.predict(X)[0]
            results[horizon] = float(prob)
        except Exception as e:
            print(f"[future] Model {horizon} prediction failed: {e}")
            results[horizon] = 0.0

    return results
