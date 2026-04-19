from functools import lru_cache
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "future_models"
MODEL_MANIFEST_PATH = MODEL_DIR / "future_model_manifest.json"

MODEL_FILES = {
    "5m": "model_scrap_5m.pkl",
    "10m": "model_scrap_10m.pkl",
    "15m": "model_scrap_15m.pkl",
    "20m": "model_scrap_20m.pkl",
    "25m": "model_scrap_25m.pkl",
    "30m": "model_scrap_30m.pkl",
}


def _unwrap_future_artifact(artifact):
    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("estimator") or artifact.get("forecaster")
        calibrator = artifact.get("calibrator") or artifact.get("calibration")
        feature_names = (
            artifact.get("feature_names")
            or artifact.get("features")
            or artifact.get("feature_columns")
            or []
        )
        if isinstance(feature_names, tuple):
            feature_names = list(feature_names)
        elif not isinstance(feature_names, list):
            feature_names = list(feature_names) if feature_names else []
        return model or artifact, calibrator, feature_names
    return artifact, None, []


@lru_cache(maxsize=1)
def _load_future_model_manifest():
    if not MODEL_MANIFEST_PATH.exists():
        return {}

    try:
        return json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(
            f"Unable to read future model manifest at {MODEL_MANIFEST_PATH}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}


def _feature_names_for_model(model) -> list[str]:
    if isinstance(model, dict):
        _, _, feature_names = _unwrap_future_artifact(model)
        if feature_names:
            return [str(name) for name in feature_names]

    feature_names = None

    if hasattr(model, "feature_name"):
        try:
            feature_names = model.feature_name() if callable(model.feature_name) else model.feature_name
        except Exception:
            feature_names = None

    if feature_names is None and hasattr(model, "feature_name_"):
        feature_names = model.feature_name_

    if feature_names is None:
        return []

    if isinstance(feature_names, str):
        return [feature_names]

    return [str(name) for name in list(feature_names)]


def _coerce_input_frame(feature_row_or_df) -> tuple[pd.DataFrame, bool]:
    if isinstance(feature_row_or_df, pd.DataFrame):
        return feature_row_or_df.copy(), True
    if isinstance(feature_row_or_df, pd.Series):
        return pd.DataFrame([feature_row_or_df.to_dict()]), False
    if isinstance(feature_row_or_df, dict):
        return pd.DataFrame([feature_row_or_df]), False
    raise TypeError(
        "feature_row_or_df must be a dict, pandas Series, or pandas DataFrame"
    )


def _align_frame_to_features(
    df_input: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    # SENIOR PRO: Axis Sanitization
    # Reindexing fails when there are duplicate labels (columns or index).
    # We strip them here to ensure inference stability under high load.
    sanitized = df_input.loc[~df_input.index.duplicated(keep="last")]
    sanitized = sanitized.loc[:, ~sanitized.columns.duplicated(keep="last")]
    
    # SENIOR PRO: Strict Feature Alignment
    # LightGBM crashes if we provide more features than it was trained on (729 vs 550).
    # We select ONLY the features the model expects.
    aligned = sanitized.reindex(columns=feature_names, fill_value=0.0)
    aligned = aligned[feature_names] # Force exact order and count
    
    aligned = aligned.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return aligned


@lru_cache(maxsize=1)
def _load_future_models():
    models = {}
    for horizon, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            artifact = joblib.load(model_path)
            model, _, _ = _unwrap_future_artifact(artifact)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(verbose=-1)
                except Exception:
                    pass
            models[horizon] = artifact
    return models


def predict_future_risk(feature_row_or_df, feature_columns: list):
    """
    Predicts scrap risk for one or more rows.
    If input is a dict, returns a dict of horizons -> prob.
    If input is a DataFrame, returns a DataFrame of rows x horizons.
    """
    models = _load_future_models()
    if not models:
        if isinstance(feature_row_or_df, pd.DataFrame):
            return pd.DataFrame(index=feature_row_or_df.index)
        return {}

    df_input, is_batch = _coerce_input_frame(feature_row_or_df)
    fallback_features = list(feature_columns or [])
    manifest = _load_future_model_manifest()

    batch_results = pd.DataFrame(index=df_input.index)

    for horizon, model in models.items():
        base_model, calibrator, artifact_features = _unwrap_future_artifact(model)
        # Determine the exact feature order for this horizon.
        m_features = artifact_features or _feature_names_for_model(base_model)
        if not m_features:
            horizon_manifest = (manifest.get("models") or {}).get(horizon) or {}
            m_features = list(horizon_manifest.get("feature_names") or [])

        if not m_features:
            m_features = fallback_features

        X = _align_frame_to_features(df_input, m_features)

        try:
            if hasattr(base_model, "predict_proba"):
                raw_probs = base_model.predict_proba(X)
                if getattr(raw_probs, "ndim", 1) > 1 and raw_probs.shape[1] > 1:
                    probs = raw_probs[:, 1]
                else:
                    probs = np.asarray(raw_probs, dtype=float).reshape(-1)
            else:
                probs = base_model.predict(X)

            probs = np.asarray(probs, dtype=float).reshape(-1)
            if probs.size == 1 and len(X.index) > 1:
                probs = np.repeat(float(probs[0]), len(X.index))
            elif probs.size != len(X.index):
                raise ValueError(
                    f"prediction length {probs.size} does not match input rows {len(X.index)}"
                )

            if calibrator is not None and hasattr(calibrator, "predict"):
                try:
                    probs = np.asarray(calibrator.predict(probs), dtype=float).reshape(-1)
                except Exception:
                    pass

            batch_results[horizon] = np.clip(probs, 0.0, 1.0)
        except Exception as e:
            warnings.warn(
                f"Future model {horizon} prediction failed: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            batch_results[horizon] = 0.0

    if not is_batch:
        return batch_results.iloc[0].to_dict()

    return batch_results
