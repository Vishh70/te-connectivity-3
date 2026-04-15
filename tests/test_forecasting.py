import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
if SITE_PACKAGES.exists() and str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))

from backend import data_access  # noqa: E402
from backend.forecasting import generate_forecast  # noqa: E402


def _sample_history(rows: int = 8) -> pd.DataFrame:
    timestamps = pd.date_range("2026-04-13 00:00:00", periods=rows, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": np.linspace(10.0, 18.0, rows),
            "pressure": np.linspace(100.0, 112.0, rows),
        }
    )


def test_prepare_tcn_input_window_shapes_and_padding():
    history = _sample_history(rows=3)
    window = data_access.prepare_tcn_input_window(
        history,
        input_features=["temp", "pressure"],
        num_lags=5,
        target_column="temp",
    )

    assert window["matrix"].shape == (5, 2)
    assert window["tensor"].shape == (1, 5, 2)
    assert window["flat"].shape == (1, 10)
    assert window["feature_columns"] == ["temp", "pressure"]
    assert window["window_df"].iloc[-1]["temp"] == history.iloc[-1]["temp"]


def test_missing_sensor_forecaster_is_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(data_access, "FORECASTER_MODEL_PATH", tmp_path / "missing.pkl")
    data_access._load_sensor_forecaster.cache_clear()

    with pytest.warns(RuntimeWarning, match="Sensor forecaster not found"):
        model, sensor_columns, input_features, num_lags, hydra_features = data_access._load_sensor_forecaster()

    assert model is None
    assert sensor_columns == []
    assert input_features == []
    assert num_lags == 0
    assert hydra_features == []


def test_generate_forecast_returns_history_and_predictions():
    history = _sample_history(rows=6)
    forecast = generate_forecast(history, "temp", future_minutes=30)

    assert list(forecast.columns) == ["timestamp", "value", "type"]
    assert len(forecast) == 16
    assert forecast.iloc[:6]["type"].eq("history").all()
    assert forecast.iloc[6:]["type"].eq("prediction").all()
    assert pd.api.types.is_datetime64_any_dtype(forecast["timestamp"])
