"""
Smoke tests for the TCN-ready forecasting pipeline.
Run with:  python tests/smoke_test_tcn.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def _sample_history(rows: int = 8) -> pd.DataFrame:
    ts = pd.date_range("2026-04-13", periods=rows, freq="min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temp": np.linspace(10.0, 18.0, rows),
            "pressure": np.linspace(100.0, 112.0, rows),
        }
    )


# ── Test 1: prepare_tcn_input_window shapes & padding ─────────────────────
def test_window_shapes_and_padding():
    from backend.data_access import prepare_tcn_input_window

    hist = _sample_history(rows=3)
    w = prepare_tcn_input_window(
        hist,
        input_features=["temp", "pressure"],
        num_lags=5,
        target_column="temp",
    )

    assert w["matrix"].shape == (5, 2), f"matrix shape wrong: {w['matrix'].shape}"
    assert w["tensor"].shape == (1, 5, 2), f"tensor shape wrong: {w['tensor'].shape}"
    assert w["flat"].shape == (1, 10), f"flat shape wrong: {w['flat'].shape}"
    assert w["feature_columns"] == ["temp", "pressure"], (
        f"feature_columns wrong: {w['feature_columns']}"
    )
    assert w["window_df"].iloc[-1]["temp"] == hist.iloc[-1]["temp"], (
        "Last row of window_df must match the last real observation"
    )
    print("  PASS  test_window_shapes_and_padding")


# ── Test 2: missing model artifact is graceful ────────────────────────────
def test_missing_model_graceful():
    import backend.data_access as da

    real_path = da.FORECASTER_MODEL_PATH
    da.FORECASTER_MODEL_PATH = Path("/nonexistent/missing.pkl")
    da._load_sensor_forecaster.cache_clear()

    try:
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            model, sc, inf, nl, hf = da._load_sensor_forecaster()

        assert model is None, "model should be None when file is missing"
        assert sc == [], "sensor_columns should be empty list"
        assert inf == [], "input_features should be empty list"
        assert nl == 0, "num_lags should be 0"
        assert hf == [], "hydra_features should be empty list"
        assert any("surrogate" in str(x.message).lower() for x in w_list), (
            "RuntimeWarning mentioning 'surrogate' expected"
        )
    finally:
        da.FORECASTER_MODEL_PATH = real_path
        da._load_sensor_forecaster.cache_clear()

    print("  PASS  test_missing_model_graceful")


# ── Test 3: generate_forecast output shape & dtype ────────────────────────
def test_generate_forecast_output():
    from backend.forecasting import generate_forecast, _get_tcn_forecaster

    _get_tcn_forecaster.cache_clear()

    hist = _sample_history(rows=6)
    fc = generate_forecast(hist, "temp", future_minutes=30)

    assert list(fc.columns) == ["timestamp", "value", "type"], (
        f"Unexpected columns: {list(fc.columns)}"
    )
    assert len(fc) == 16, f"Expected 16 rows (6 history + 10 predictions), got {len(fc)}"
    assert fc.iloc[:6]["type"].eq("history").all(), "First 6 rows should be 'history'"
    assert fc.iloc[6:]["type"].eq("prediction").all(), "Last 10 rows should be 'prediction'"
    assert pd.api.types.is_datetime64_any_dtype(fc["timestamp"]), (
        "timestamp column must be datetime64"
    )
    assert fc["value"].notna().all(), "No NaN values expected in forecast output"
    print("  PASS  test_generate_forecast_output")


if __name__ == "__main__":
    print("Running TCN forecasting smoke tests...\n")
    test_window_shapes_and_padding()
    test_missing_model_graceful()
    test_generate_forecast_output()
    print("\nAll smoke tests PASSED ✓")
