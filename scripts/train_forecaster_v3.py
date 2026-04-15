"""
scripts/train_forecaster_v3.py
──────────────────────────────
Trains a lag-window sensor forecaster that the TCNForecaster adapter
can load from  models/sensor_forecaster_lagged.pkl.

Architecture
────────────
• Input : flat lag-window of shape  (1, num_lags × num_sensors)
• Output: next-step values for all sensors  (num_sensors,)
• Inference: autoregressive rolling → (num_points, num_sensors)
             so the adapter's _coerce_prediction_sequence can extract
             any target sensor column by index.

Saved artifact format (dict):
{
    "model"          : SensorForecasterWrapper  (has .predict(tensor) method)
    "sensor_columns" : list[str]                raw sensor column names
    "input_features" : list[str]                flat lag feature names
    "num_lags"       : int                      look-back window length
    "hydra_features" : []                       reserved, kept as empty list
}
"""

from __future__ import annotations

import gc
import json
import time
import warnings
import sys
from pathlib import Path
from typing import Sequence

# ─── Path Hack for local import ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from backend.forecasting import SensorForecasterWrapper

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "new_processed_data" / "FINAL_TRAINING_MASTER_V3.parquet"
OUT_PATH     = PROJECT_ROOT / "models" / "sensor_forecaster_lagged.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "forecaster_metrics_v3.json"

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_LAGS       = 30        # look-back window length (adjustable)
NUM_POINTS     = 10        # forecast horizon during training evaluation
MAX_ROWS_PER_MACHINE = 20_000   # samples per machine — keeps peak RAM ≈ 200 MB
RANDOM_STATE   = 42

SENSOR_COLUMNS: list[str] = [
    "Cushion", "Cycle_time", "Cyl_tmp_z1", "Cyl_tmp_z2",
    "Cyl_tmp_z3", "Cyl_tmp_z4", "Cyl_tmp_z5", "Cyl_tmp_z6",
    "Cyl_tmp_z7", "Cyl_tmp_z8", "Dosage_time",
    "Ejector_fix_deviation_torque", "Extruder_start_position",
    "Extruder_torque", "Injection_pressure", "Injection_time",
    "Machine_status", "Peak_pressure_position", "Peak_pressure_time",
    "Scrap_counter", "Shot_counter", "Shot_size",
    "Switch_position", "Switch_pressure",
]

LGB_PARAMS = {
    "n_estimators"  : 300,
    "learning_rate" : 0.05,
    "num_leaves"    : 64,
    "max_depth"     : 7,
    "min_child_samples": 30,
    "subsample"     : 0.8,
    "colsample_bytree": 0.8,
    "random_state"  : RANDOM_STATE,
    "n_jobs"        : -1,
    "verbose"       : -1,
}


# SensorForecasterWrapper is now imported from backend.forecasting


# ─── Dataset helpers ──────────────────────────────────────────────────────────
def _build_feature_names(sensors: Sequence[str], num_lags: int) -> list[str]:
    """Returns flat feature names: sensor_lag_1, sensor_lag_2, …"""
    return [f"{s}_lag_{lag}" for lag in range(1, num_lags + 1) for s in sensors]


def _build_lag_dataset(
    df: pd.DataFrame,
    sensors: list[str],
    num_lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X  (num_samples, num_lags × num_sensors)
         y  (num_samples, num_sensors)
    from a time-ordered dataframe.
    """
    data = df[sensors].to_numpy(dtype=np.float32, na_value=0.0)
    n_rows, n_sensors = data.shape
    min_rows = num_lags + 1

    if n_rows < min_rows:
        return np.empty((0, num_lags * n_sensors)), np.empty((0, n_sensors))

    # sliding window: row i uses data[i : i+num_lags] as input,
    # data[i+num_lags] as target
    X_list, y_list = [], []
    for i in range(num_lags, n_rows):
        window = data[i - num_lags : i].flatten()   # (L*F,)
        target = data[i]                             # (F,)
        X_list.append(window)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─── Training evaluation ─────────────────────────────────────────────────────
def _evaluate_autoregressive(
    wrapper: SensorForecasterWrapper,
    df_test: pd.DataFrame,
    sensors: list[str],
    num_lags: int,
    num_points: int,
) -> dict:
    """Run autoregressive forecast on test set; return per-sensor MAE."""
    data = df_test[sensors].to_numpy(dtype=np.float32, na_value=0.0)
    n_rows = len(data)

    if n_rows < num_lags + num_points:
        return {"mae_mean": None, "per_sensor_mae": {}}

    maes: dict[str, list[float]] = {s: [] for s in sensors}
    n_eval = min(200, n_rows - num_lags - num_points)   # limit eval samples

    indices = np.linspace(num_lags, n_rows - num_points - 1, n_eval, dtype=int)
    for idx in indices:
        window = data[idx - num_lags : idx]              # (L, F)
        tensor = window[np.newaxis, :, :]                # (1, L, F)
        future = wrapper.predict(tensor)                 # (num_points, F)
        truth  = data[idx : idx + num_points]            # (num_points, F)
        for j, s in enumerate(sensors):
            mae = float(np.mean(np.abs(future[:, j] - truth[:, j])))
            maes[s].append(mae)

    per_sensor = {s: float(np.mean(v)) for s, v in maes.items() if v}
    mean_mae   = float(np.mean(list(per_sensor.values()))) if per_sensor else None
    return {"mae_mean": mean_mae, "per_sensor_mae": per_sensor}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    np.random.seed(RANDOM_STATE)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Sensor Forecaster v3 — Training Pipeline")
    print(f"{'='*60}\n")
    print(f"  Data  : {DATA_PATH}")
    print(f"  Output: {OUT_PATH}")
    print(f"  Lags  : {NUM_LAGS}  |  Sensors: {len(SENSOR_COLUMNS)}")
    print(f"  Max rows per machine: {MAX_ROWS_PER_MACHINE}\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("[1/5] Loading data …")
    drop_cols = ["future_scrap", "scrap_weight", "scrap_quantity"]
    raw = pd.read_parquet(DATA_PATH, engine="pyarrow")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    # Keep only sensor columns that actually exist
    sensors = [s for s in SENSOR_COLUMNS if s in raw.columns]
    if not sensors:
        raise RuntimeError("None of the SENSOR_COLUMNS found in dataset!")
    print(f"  Found {len(sensors)}/{len(SENSOR_COLUMNS)} sensor columns in dataset.")

    # ── 2. Handle missing values ──────────────────────────────────────────────
    print("[2/5] Cleaning …")
    raw[sensors] = raw[sensors].apply(pd.to_numeric, errors="coerce")
    # Per-machine forward/backward fill to preserve temporal dynamics
    raw[sensors] = (
        raw.groupby("machine_id", group_keys=False)[sensors]
        .apply(lambda g: g.ffill().bfill())
    )
    raw[sensors] = raw[sensors].fillna(0.0)

    # ── 3. Build lag dataset ──────────────────────────────────────────────────
    print("[3/5] Building lag dataset (sampling per machine to stay within RAM) …")
    rng = np.random.default_rng(RANDOM_STATE)
    X_all, y_all = [], []
    for machine_id, group in raw.groupby("machine_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        Xi, yi = _build_lag_dataset(group, sensors, NUM_LAGS)
        if len(Xi) == 0:
            continue
        # Time-based sample: keep last MAX_ROWS_PER_MACHINE samples
        # (most recent cycles are most representative)
        if len(Xi) > MAX_ROWS_PER_MACHINE:
            Xi = Xi[-MAX_ROWS_PER_MACHINE:]
            yi = yi[-MAX_ROWS_PER_MACHINE:]
        X_all.append(Xi)
        y_all.append(yi)
        print(f"  Machine {machine_id}: {len(Xi):,} samples used")

    X = np.vstack(X_all)
    y = np.vstack(y_all)
    del X_all, y_all, raw
    gc.collect()

    total_mb = X.nbytes / 1_048_576
    print(f"\n  Total samples: {len(X):,}  |  Features: {X.shape[1]}  |  Targets: {y.shape[1]}")
    print(f"  RAM used by X: {total_mb:.0f} MB")

    # Shuffle (samples from different machines would be ordered otherwise)
    shuffle_idx = rng.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Train / test split (80/20 random after shuffle)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[4/5] Training MultiOutputRegressor (LightGBM) …")
    print("  (One LGBMRegressor per sensor target — this may take a few minutes)")

    base_model = lgb.LGBMRegressor(**LGB_PARAMS)
    multi_model = MultiOutputRegressor(estimator=base_model, n_jobs=1)
    multi_model.fit(X_train, y_train)

    wall = time.time() - t0
    print(f"  Training complete in {wall:.1f}s")

    # Quick train-set check
    y_pred_train = multi_model.predict(X_train[:1000])
    mae_train = np.mean(np.abs(y_pred_train - y_train[:1000]))
    print(f"  Train MAE (first 1000 samples): {mae_train:.4f}")

    # ── 5. Evaluate autoregressive forecast ───────────────────────────────────
    print("\n[5/5] Evaluating autoregressive forecast …")

    # We need a small test dataframe for evaluation
    # Reconstruct a minimal time-ordered slice from test indices
    # (Use last 500 rows of X_test to build a pseudo-sequence)
    test_flat = X_test[-500:]
    # Unroll the last sensor values from the flat lags (last lag = most recent)
    last_vals = test_flat[:, -len(sensors):]  # (500, n_sensors)
    test_df = pd.DataFrame(last_vals, columns=sensors)

    wrapper = SensorForecasterWrapper(
        model=multi_model,
        sensor_columns=sensors,
        num_lags=NUM_LAGS,
        num_points=NUM_POINTS,
    )

    metrics = _evaluate_autoregressive(
        wrapper=wrapper,
        df_test=test_df,
        sensors=sensors,
        num_lags=NUM_LAGS,
        num_points=NUM_POINTS,
    )

    print(f"\n  Mean MAE across all sensors: {metrics['mae_mean']}")
    print("  Per-sensor MAE (top 10 by error):")
    sorted_mae = sorted(metrics["per_sensor_mae"].items(), key=lambda x: x[1], reverse=True)
    for s, v in sorted_mae[:10]:
        print(f"    {s:<38}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving artifact → {OUT_PATH} …")
    artifact = {
        "model"         : wrapper,
        "sensor_columns": sensors,
        "input_features": wrapper.input_features,
        "num_lags"      : NUM_LAGS,
        "hydra_features": [],
    }
    joblib.dump(artifact, OUT_PATH, compress=3)
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"  Saved ({size_mb:.1f} MB)")

    # Save metrics
    full_metrics = {
        "version"       : "v3-lgb-multioutput",
        "num_lags"      : NUM_LAGS,
        "num_sensors"   : len(sensors),
        "sensor_columns": sensors,
        "train_samples" : int(len(X_train)),
        "test_samples"  : int(len(X_test)),
        "train_wall_sec": round(wall, 1),
        "mae_mean"      : metrics["mae_mean"],
        "per_sensor_mae": metrics["per_sensor_mae"],
        "artifact_mb"   : round(size_mb, 2),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Metrics saved → {METRICS_PATH}")

    total_time = time.time() - t0
    print(f"\n✓ Done in {total_time:.1f}s")
    print("  Drop models/sensor_forecaster_lagged.pkl → restart backend → TCN path is live.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
