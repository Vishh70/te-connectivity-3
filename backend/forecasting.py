from __future__ import annotations

from functools import lru_cache
from typing import Sequence
import warnings

import numpy as np
import pandas as pd

from backend.data_access import _infer_step_seconds, _load_sensor_forecaster, prepare_tcn_input_window


DEFAULT_FUTURE_POINTS = 10
DEFAULT_FUTURE_MINUTES = 60
DEFAULT_FALLBACK_LAGS = 30


class SensorForecasterWrapper:
    """
    Wraps a MultiOutputRegressor (one LightGBM per sensor).
    Accepts a 3-D tensor  (1, num_lags, num_sensors)  and returns
    an autoregressive forecast  (num_points, num_sensors).
    """

    def __init__(
        self,
        model: any,
        sensor_columns: list[str],
        num_lags: int,
        num_points: int = DEFAULT_FUTURE_POINTS,
    ):
        self.model = model
        self.sensor_columns = list(sensor_columns)
        self.num_lags = int(num_lags)
        self.num_points = int(num_points)
        # Pre-build flat feature names for downstream inspection
        self.input_features = [f"{s}_lag_{lag}" for lag in range(1, num_lags + 1) for s in sensor_columns]

    def __repr__(self) -> str:
        return f"<SensorForecasterWrapper sensors={len(self.sensor_columns)} lags={self.num_lags} points={self.num_points}>"

    def predict(self, tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        tensor : shape (1, num_lags, num_sensors)  — or anything broadcastable
        returns: shape (num_points, num_sensors)
        """
        arr = np.asarray(tensor, dtype=np.float32)

        # Accept (1, L, F) or (L, F) or (1, L*F)
        if arr.ndim == 3:
            window = arr[0]  # (L, F)
        elif arr.ndim == 2 and arr.shape[0] == 1:
            # Flattened input
            window = arr[0].reshape(self.num_lags, len(self.sensor_columns))
        elif arr.ndim == 2:
            window = arr  # (L, F)
        else:
            raise ValueError(f"Unexpected tensor shape: {arr.shape}")

        window = window[-self.num_lags :]  # Clip to window length
        n_sensors = len(self.sensor_columns)

        # Pad if needed
        if len(window) < self.num_lags:
            pad = np.tile(window[[0]], (self.num_lags - len(window), 1))
            window = np.vstack([pad, window])

        future = np.zeros((self.num_points, n_sensors), dtype=np.float32)
        buf = window.copy()  # Rolling buffer (L, F)

        for step in range(self.num_points):
            flat = buf.reshape(1, -1)  # (1, L * F)
            # Handle potential model.predict multi-output varieties
            pred = np.asarray(self.model.predict(flat), dtype=np.float32)
            pred = pred.flatten()[:n_sensors]
            future[step] = pred
            # Advance buffer: drop oldest, append prediction
            buf = np.vstack([buf[1:], pred[np.newaxis, :]])

        return future  # (num_points, num_sensors)


def _as_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = pd.to_numeric(df.get(column, pd.Series(dtype=float)), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _pick_feature_columns(history_df: pd.DataFrame, parameter_name: str, configured_columns: Sequence[str]) -> list[str]:
    if configured_columns:
        feature_columns = list(dict.fromkeys([c for c in configured_columns if c in history_df.columns]))
    else:
        feature_columns = [
            c
            for c in history_df.columns
            if c not in {"timestamp", "type", "event_timestamp"}
            and pd.api.types.is_numeric_dtype(history_df[c])
        ]

    if parameter_name in history_df.columns and parameter_name not in feature_columns:
        feature_columns.append(parameter_name)

    if not feature_columns and parameter_name in history_df.columns:
        feature_columns = [parameter_name]

    return feature_columns


def _weighted_quadratic_projection(target_values: np.ndarray, horizon_steps: np.ndarray) -> np.ndarray:
    if target_values.size == 0:
        return np.zeros_like(horizon_steps, dtype=float)

    if target_values.size == 1:
        return np.repeat(float(target_values[-1]), len(horizon_steps))

    x = np.arange(target_values.size, dtype=float)
    weights = np.linspace(0.35, 1.0, target_values.size, dtype=float) ** 2
    degree = 2 if target_values.size >= 3 else 1

    try:
        coeffs = np.polyfit(x, target_values, deg=degree, w=weights)
        polynomial = np.poly1d(coeffs)
        x_future = (target_values.size - 1) + horizon_steps
        return polynomial(x_future).astype(float)
    except Exception:
        last = float(target_values[-1])
        slope = float(np.mean(np.diff(target_values[-min(5, target_values.size):])))
        curvature = 0.0
        if target_values.size >= 4:
            curvature = float(np.mean(np.diff(target_values, n=2)[-min(4, target_values.size - 2):]))
        
        raw_projection = last + slope * horizon_steps + 0.5 * curvature * (horizon_steps ** 2)
        # Prevent exponential explosions outside of standard physical boundaries
        bound = 10.0 * float(np.std(target_values)) if target_values.size > 1 else abs(last * 0.5)
        return np.clip(raw_projection, last - bound, last + bound).astype(float)


def _contextual_adjustment(window_df: pd.DataFrame, target_column: str, feature_columns: Sequence[str]) -> float:
    if target_column not in window_df.columns:
        return 0.0

    target = _as_numeric_series(window_df, target_column).to_numpy(dtype=float)
    if target.size < 3 or np.nanstd(target) < 1e-9:
        return 0.0

    target_std = float(np.nanstd(target))
    adjustment = 0.0
    context_columns = [c for c in feature_columns if c != target_column and c in window_df.columns]

    for column in context_columns:
        series = _as_numeric_series(window_df, column).to_numpy(dtype=float)
        if series.size < 3 or np.nanstd(series) < 1e-9:
            continue

        corr = np.corrcoef(target, series)[0, 1]
        if not np.isfinite(corr) or abs(corr) < 0.12:
            continue

        recent_delta = float(series[-1] - np.mean(series[-min(6, series.size):-1] if series.size > 1 else series[-1:]))
        series_std = float(np.nanstd(series))
        normalized_delta = recent_delta / max(series_std, 1e-6)
        adjustment += corr * normalized_delta * target_std * 0.08

    return float(np.tanh(adjustment / max(target_std, 1e-6)) * target_std * 0.75)


def _build_future_timestamps(last_timestamp: pd.Timestamp, step_seconds: int, future_minutes: int, num_points: int) -> list[pd.Timestamp]:
    if pd.isna(last_timestamp):
        return []
    step_minutes = np.linspace(future_minutes / num_points, future_minutes, num_points)
    return [last_timestamp + pd.Timedelta(minutes=float(minutes)) for minutes in step_minutes]


class TCNForecaster:
    """
    Lightweight runtime adapter for a serialized TCN or lag-window surrogate.

    The adapter accepts the artifact shape used by the workspace and gracefully
    falls back to a deterministic multivariate lag-based extrapolator when the
    serialized model is absent.
    """

    def __init__(self):
        model, sensor_columns, input_features, num_lags, hydra_features = _load_sensor_forecaster()
        self.model = model
        self.sensor_columns = list(sensor_columns or [])
        self.input_features = list(input_features or [])
        self.hydra_features = list(hydra_features or [])
        self.num_lags = int(num_lags or DEFAULT_FALLBACK_LAGS)
        self.window_size = max(self.num_lags, DEFAULT_FALLBACK_LAGS)
        self.available = self.model is not None
        self._warned_fallback = False

    def _warn_once(self, message: str):
        if not self._warned_fallback:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            self._warned_fallback = True

    def _model_predict(self, tensor: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None

        try:
            if hasattr(self.model, "predict"):
                try:
                    prediction = self.model.predict(tensor, verbose=0)
                except TypeError:
                    prediction = self.model.predict(tensor)
            else:
                prediction = self.model(tensor)
            return np.asarray(prediction, dtype=float)
        except Exception as exc:
            self._warn_once(f"TCN forecaster prediction failed, using surrogate fallback: {exc}")
            return None

    @staticmethod
    def _coerce_prediction_sequence(
        prediction: np.ndarray | None,
        target_column: str,
        feature_columns: Sequence[str],
        num_points: int,
    ) -> np.ndarray | None:
        if prediction is None:
            return None

        arr = np.asarray(prediction, dtype=float)
        if arr.size == 0:
            return None

        if arr.ndim == 0:
            return np.repeat(float(arr), num_points)

        if arr.ndim >= 3:
            arr = np.squeeze(arr)

        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            elif target_column in feature_columns and arr.shape[1] == len(feature_columns):
                target_idx = list(feature_columns).index(target_column)
                arr = arr[:, target_idx]
            else:
                arr = arr.reshape(-1)

        if arr.ndim == 1:
            if target_column in feature_columns and arr.size == len(feature_columns) and arr.size != num_points:
                target_idx = list(feature_columns).index(target_column)
                return np.repeat(float(arr[target_idx]), num_points)
            if arr.size >= num_points:
                return arr[:num_points].astype(float)
            if arr.size == 1:
                return np.repeat(float(arr[0]), num_points)
            padded = np.pad(arr, (0, num_points - arr.size), mode="edge")
            return padded.astype(float)

        return None

    def _surrogate_forecast(
        self,
        window_df: pd.DataFrame,
        target_column: str,
        feature_columns: Sequence[str],
        future_minutes: int,
        num_points: int,
    ) -> np.ndarray:
        target_values = _as_numeric_series(window_df, target_column).to_numpy(dtype=float)
        if target_values.size == 0:
            return np.zeros(num_points, dtype=float)

        step_seconds = _infer_step_seconds(window_df if "timestamp" in window_df.columns else pd.DataFrame())
        step_seconds = max(int(step_seconds), 1)
        horizon_seconds = np.linspace(
            future_minutes * 60 / num_points,
            future_minutes * 60,
            num_points,
            dtype=float,
        )
        horizon_steps = horizon_seconds / step_seconds

        base_curve = _weighted_quadratic_projection(target_values, horizon_steps)
        context_shift = _contextual_adjustment(window_df, target_column, feature_columns)

        recent_values = target_values[-min(5, target_values.size):]
        if recent_values.size >= 2:
            recent_slope = float(np.mean(np.diff(recent_values)))
        else:
            recent_slope = 0.0

        decay = 1.0 - np.exp(-np.maximum(horizon_steps, 0.0) / 3.0)
        ripple = 0.12 * recent_slope * np.sin(np.minimum(horizon_steps, 10.0) / 2.5)
        forecast = base_curve + context_shift * decay + ripple
        return np.nan_to_num(forecast, nan=float(target_values[-1]), posinf=float(target_values[-1]), neginf=float(target_values[-1]))

    def forecast(
        self,
        history_df: pd.DataFrame,
        parameter_name: str,
        future_minutes: int = DEFAULT_FUTURE_MINUTES,
        num_points: int = DEFAULT_FUTURE_POINTS,
    ) -> pd.DataFrame:
        if history_df is None or history_df.empty:
            return pd.DataFrame(columns=["timestamp", "value", "type"])

        if "timestamp" not in history_df.columns:
            raise ValueError("history_df must include a 'timestamp' column")

        df = history_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "value", "type"])

        # 1. Determine forecasting mode and feature requirements
        is_supported_sensor = self.available and (parameter_name in self.sensor_columns)
        
        if is_supported_sensor:
            # Model path: strictly use the model's trained sensor columns
            feature_columns = list(self.sensor_columns)
        else:
            # Surrogate path: pick contextually relevant features
            feature_columns = _pick_feature_columns(df, parameter_name, self.input_features or self.sensor_columns)

        if parameter_name not in feature_columns:
            feature_columns.append(parameter_name)

        # 2. Build input window
        window = prepare_tcn_input_window(
            df,
            sensor_columns=feature_columns,
            num_lags=self.num_lags,
            target_column=parameter_name,
        )
        window_df = window["window_df"]
        tensor = window["tensor"]
        
        # Ensure tensor matches model shape if using model
        if is_supported_sensor:
            tensor_features = window["feature_columns"]
            if len(tensor_features) != len(self.sensor_columns):
                model_idxs = [tensor_features.index(s) for s in self.sensor_columns if s in tensor_features]
                if len(model_idxs) == len(self.sensor_columns):
                    tensor = tensor[:, :, model_idxs]
                else:
                    is_supported_sensor = False 

        future_values = None
        if is_supported_sensor:
            model_prediction = self._model_predict(tensor)
            future_values = self._coerce_prediction_sequence(
                model_prediction,
                target_column=parameter_name,
                feature_columns=self.sensor_columns,
                num_points=num_points,
            )

        if future_values is None:
            if self.available:
                self._warn_once(
                    "Sensor forecaster artifact is missing or incompatible; using lag-window surrogate forecast."
                )
            future_values = self._surrogate_forecast(
                window_df=window_df,
                target_column=parameter_name,
                feature_columns=window["feature_columns"],
                future_minutes=future_minutes,
                num_points=num_points,
            )

        target_history = _as_numeric_series(df, parameter_name)
        history_out = df[["timestamp"]].copy()
        history_out["value"] = target_history.astype(float)
        history_out["type"] = "history"

        last_timestamp = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce")
        future_timestamps = _build_future_timestamps(last_timestamp, window["step_seconds"], future_minutes, num_points)
        future_df = pd.DataFrame({
            "timestamp": future_timestamps,
            "value": np.asarray(future_values, dtype=float),
            "type": "prediction",
        })

        full_df = pd.concat([history_out, future_df], ignore_index=True)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
        return full_df


@lru_cache(maxsize=1)
def _get_tcn_forecaster() -> TCNForecaster:
    return TCNForecaster()


def generate_forecast(history_df, parameter_name, future_minutes=DEFAULT_FUTURE_MINUTES):
    """
    Forecast a sensor trend with a TCN-aware runtime adapter.

    The return shape remains compatible with the frontend:
    columns = timestamp, value, type
    """
    forecaster = _get_tcn_forecaster()
    return forecaster.forecast(history_df, parameter_name, future_minutes=future_minutes, num_points=DEFAULT_FUTURE_POINTS)
