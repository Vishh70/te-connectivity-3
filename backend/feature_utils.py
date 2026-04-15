from __future__ import annotations

import numpy as np
import pandas as pd

from backend.config_limits import SAFE_LIMITS


SAFETY_SIGNAL_SENSORS = tuple(SAFE_LIMITS.keys())
TEMPORAL_WINDOWS = (5, 10, 20)
TEMPORAL_LAGS = (1, 3, 5)
SAFETY_SIGNAL_FEATURES: tuple[str, ...] = tuple(
    [f"{sensor}_safe_utilization" for sensor in SAFETY_SIGNAL_SENSORS]
    + [f"{sensor}_safe_margin_norm" for sensor in SAFETY_SIGNAL_SENSORS]
    + [f"{sensor}_safe_violation" for sensor in SAFETY_SIGNAL_SENSORS]
    + [
        "safety_violation_count",
        "safety_margin_min",
        "safety_margin_mean",
        "safety_utilization_mean",
        "Cycle_Dosage_Ratio",
        "Thermal_Stability_Index",
        "Pressure_Impulse",
        "Injection_Efficiency",
        "Pressure_Volatility",
        "Cushion_Consistency_Score",
        "Thermal_Sync_Error",
    ]
)
TEMPORAL_SIGNAL_SOURCES: tuple[str, ...] = tuple(
    list(SAFETY_SIGNAL_SENSORS)
)
TEMPORAL_SIGNAL_FEATURES: tuple[str, ...] = tuple(
    [f"{source}_delta_1" for source in TEMPORAL_SIGNAL_SOURCES]
    + [f"{source}_lag_{lag}" for source in TEMPORAL_SIGNAL_SOURCES for lag in TEMPORAL_LAGS]
    + [f"{source}_rolling_mean_{window}" for source in TEMPORAL_SIGNAL_SOURCES for window in TEMPORAL_WINDOWS]
    + [f"{source}_rolling_std_{window}" for source in TEMPORAL_SIGNAL_SOURCES for window in TEMPORAL_WINDOWS]
    + [f"{source}_rolling_min_5" for source in TEMPORAL_SIGNAL_SOURCES]
    + [f"{source}_rolling_max_5" for source in TEMPORAL_SIGNAL_SOURCES]
    + [
        "safety_violation_count_delta_1",
        "safety_violation_count_lag_1",
        "safety_violation_count_lag_3",
        "safety_violation_count_lag_5",
        "safety_violation_count_rolling_mean_5",
        "safety_violation_count_rolling_std_5",
        "safety_violation_count_rolling_min_5",
        "safety_violation_count_rolling_max_5",
        "safety_margin_min_lag_1",
        "safety_margin_min_lag_3",
        "safety_margin_min_lag_5",
        "safety_margin_min_rolling_mean_5",
        "safety_margin_min_rolling_std_5",
        "safety_margin_min_rolling_min_5",
        "safety_margin_min_rolling_max_5",
        "safety_margin_mean_lag_1",
        "safety_margin_mean_lag_3",
        "safety_margin_mean_lag_5",
        "safety_margin_mean_rolling_mean_5",
        "safety_margin_mean_rolling_std_5",
        "safety_margin_mean_rolling_min_5",
        "safety_margin_mean_rolling_max_5",
        "safety_utilization_mean_lag_1",
        "safety_utilization_mean_lag_3",
        "safety_utilization_mean_lag_5",
        "safety_utilization_mean_rolling_mean_5",
        "safety_utilization_mean_rolling_std_5",
        "safety_utilization_mean_rolling_min_5",
        "safety_utilization_mean_rolling_max_5",
        "safety_violation_count_rolling_mean_10",
        "safety_violation_count_rolling_std_10",
        "safety_violation_count_rolling_mean_20",
        "safety_violation_count_rolling_std_20",
        "safety_margin_min_delta_1",
        "safety_margin_min_rolling_mean_10",
        "safety_margin_min_rolling_std_10",
        "safety_margin_min_rolling_mean_20",
        "safety_margin_min_rolling_std_20",
        "safety_margin_mean_delta_1",
        "safety_margin_mean_rolling_mean_10",
        "safety_margin_mean_rolling_std_10",
        "safety_margin_mean_rolling_mean_20",
        "safety_margin_mean_rolling_std_20",
        "safety_utilization_mean_delta_1",
        "safety_utilization_mean_rolling_mean_10",
        "safety_utilization_mean_rolling_std_10",
        "safety_utilization_mean_rolling_mean_20",
        "safety_utilization_mean_rolling_std_20",
    ]
    + [f"{s}_tol_violation" for s in ("Cushion", "Injection_time", "Dosage_time", "Injection_pressure", "Switch_pressure", "Cyl_tmp_z1", "Cyl_tmp_z2", "Cyl_tmp_z3", "Cyl_tmp_z4", "Cyl_tmp_z5", "Cyl_tmp_z6", "Cyl_tmp_z7", "Cyl_tmp_z8", "Switch_position")]
    + [f"{s}_tol_deviation" for s in ("Cushion", "Injection_time", "Dosage_time", "Injection_pressure", "Switch_pressure", "Cyl_tmp_z1", "Cyl_tmp_z2", "Cyl_tmp_z3", "Cyl_tmp_z4", "Cyl_tmp_z5", "Cyl_tmp_z6", "Cyl_tmp_z7", "Cyl_tmp_z8", "Switch_position")]
    + ["process_instability_index"]
)


def augment_safety_signal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple boundary-distance features for the core safety-limited sensors.
    These features are cheap to compute and capture how close the machine is to
    the operating envelope, which the rolling statistics alone do not express.
    """
    if frame is None or frame.empty:
        return frame

    result = frame.copy(deep=False)
    util_cols: list[str] = []
    margin_cols: list[str] = []
    violation_cols: list[str] = []

    for sensor, limits in SAFE_LIMITS.items():
        if sensor not in result.columns:
            continue

        lower = limits.get("min")
        upper = limits.get("max")
        if lower is None or upper is None:
            continue

        try:
            lower_f = float(lower)
            upper_f = float(upper)
        except (TypeError, ValueError):
            continue

        span = max(upper_f - lower_f, 1e-6)
        values = pd.to_numeric(result[sensor], errors="coerce").astype("float32")

        utilization = ((values - lower_f) / span).astype("float32")
        margin = np.minimum(values - lower_f, upper_f - values) / span
        margin = pd.Series(margin, index=result.index, dtype="float32")
        violation = ((values < lower_f) | (values > upper_f)).astype("float32")

        util_col = f"{sensor}_safe_utilization"
        margin_col = f"{sensor}_safe_margin_norm"
        violation_col = f"{sensor}_safe_violation"

        result[util_col] = utilization
        result[margin_col] = margin
        result[violation_col] = violation

        util_cols.append(util_col)
        margin_cols.append(margin_col)
        violation_cols.append(violation_col)

    if util_cols:
        result["safety_violation_count"] = result[violation_cols].sum(axis=1).astype("float32")
        result["safety_margin_min"] = result[margin_cols].min(axis=1).astype("float32")
        result["safety_margin_mean"] = result[margin_cols].mean(axis=1).astype("float32")
        result["safety_utilization_mean"] = result[util_cols].mean(axis=1).astype("float32")

    # SENIOR INTERACTION FEATURES
    # 1. Cycle Dosage Ratio: Captures dosage efficiency
    if "Cycle_time" in result.columns and "Dosage_time" in result.columns:
        cycle = pd.to_numeric(result["Cycle_time"], errors="coerce").astype("float32")
        dosage = pd.to_numeric(result["Dosage_time"], errors="coerce").astype("float32")
        result["Cycle_Dosage_Ratio"] = (cycle / dosage.replace(0, 1.0)).fillna(0.0).astype("float32")
    
    # 2. Thermal Stability Index: Variance across all heat zones
    temp_zones = [f"Cyl_tmp_z{i}" for i in range(1, 9)]
    available_zones = [z for z in temp_zones if z in result.columns]
    if available_zones:
        temps = result[available_zones].apply(pd.to_numeric, errors="coerce").astype("float32")
        result["Thermal_Stability_Index"] = temps.std(axis=1).fillna(0.0).astype("float32")
    
    if "Injection_pressure" in result.columns and "Injection_time" in result.columns:
        press = pd.to_numeric(result["Injection_pressure"], errors="coerce").astype("float32")
        itime = pd.to_numeric(result["Injection_time"], errors="coerce").astype("float32")
        result["Pressure_Impulse"] = (press * itime).fillna(0.0).astype("float32")
        result["Injection_Efficiency"] = (press / itime.replace(0, 1.0)).fillna(0.0).astype("float32")

    # 4. Senior Pro V7: Volatility & Consistency
    if "Injection_pressure" in result.columns:
        press = pd.to_numeric(result["Injection_pressure"], errors="coerce").astype("float32")
        result["Pressure_Volatility"] = press.diff().abs().rolling(window=10).mean().fillna(0.0).astype("float32")

    if "Cushion" in result.columns:
        cushion = pd.to_numeric(result["Cushion"], errors="coerce").astype("float32")
        result["Cushion_Consistency_Score"] = 1.0 / (cushion.rolling(window=10).std().fillna(0.1) + 1e-6)

    # 5. Thermal Sync Error: Deviation from baseline thermal profile
    if available_zones:
        temps = result[available_zones].apply(pd.to_numeric, errors="coerce").astype("float32")
        avg_temp = temps.mean(axis=1)
        result["Thermal_Sync_Error"] = (temps.sub(avg_temp, axis=0).abs().mean(axis=1)).fillna(0.0).astype("float32")

    return result


def augment_temporal_signal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight trend features for core process sensors and counter-style
    signals. These capture drift and short-term movement, which are often more
    predictive than the raw values alone.
    """
    if frame is None or frame.empty:
        return frame

    result = augment_safety_signal_features(frame)
    temporal_sources = TEMPORAL_SIGNAL_SOURCES

    def _to_series(column: str) -> pd.Series:
        if column not in result.columns:
            return pd.Series(0.0, index=result.index, dtype="float32")
        return pd.to_numeric(result[column], errors="coerce").astype("float32")

    new_columns: dict[str, pd.Series] = {}

    for source in temporal_sources:
        if source not in result.columns:
            continue

        series = _to_series(source)
        new_columns[f"{source}_delta_1"] = series.diff(1).fillna(0.0).astype("float32")
        for lag in TEMPORAL_LAGS:
            new_columns[f"{source}_lag_{lag}"] = series.shift(lag).fillna(0.0).astype("float32")
        new_columns[f"{source}_rolling_mean_5"] = series.rolling(window=5, min_periods=1).mean().astype("float32")
        new_columns[f"{source}_rolling_std_5"] = series.rolling(window=5, min_periods=1).std().fillna(0.0).astype("float32")
        new_columns[f"{source}_rolling_min_5"] = series.rolling(window=5, min_periods=1).min().astype("float32")
        new_columns[f"{source}_rolling_max_5"] = series.rolling(window=5, min_periods=1).max().astype("float32")
        for window in TEMPORAL_WINDOWS:
            if window == 5:
                continue
            new_columns[f"{source}_rolling_mean_{window}"] = series.rolling(window=window, min_periods=1).mean().astype("float32")
            new_columns[f"{source}_rolling_std_{window}"] = (
                series.rolling(window=window, min_periods=1).std().fillna(0.0).astype("float32")
            )

    summary_sources = (
        "safety_violation_count",
        "safety_margin_min",
        "safety_margin_mean",
        "safety_utilization_mean",
    )
    for source in summary_sources:
        if source not in result.columns:
            continue

        series = _to_series(source)
        new_columns[f"{source}_delta_1"] = series.diff(1).fillna(0.0).astype("float32")
        new_columns[f"{source}_lag_1"] = series.shift(1).fillna(0.0).astype("float32")
        new_columns[f"{source}_lag_3"] = series.shift(3).fillna(0.0).astype("float32")
        new_columns[f"{source}_lag_5"] = series.shift(5).fillna(0.0).astype("float32")
        new_columns[f"{source}_rolling_mean_5"] = series.rolling(window=5, min_periods=1).mean().astype("float32")
        new_columns[f"{source}_rolling_std_5"] = series.rolling(window=5, min_periods=1).std().fillna(0.0).astype("float32")
        new_columns[f"{source}_rolling_min_5"] = series.rolling(window=5, min_periods=1).min().astype("float32")
        new_columns[f"{source}_rolling_max_5"] = series.rolling(window=5, min_periods=1).max().astype("float32")
        for window in TEMPORAL_WINDOWS:
            if window == 5:
                continue
            new_columns[f"{source}_rolling_mean_{window}"] = series.rolling(window=window, min_periods=1).mean().astype("float32")
            new_columns[f"{source}_rolling_std_{window}"] = (
                series.rolling(window=window, min_periods=1).std().fillna(0.0).astype("float32")
            )

    if new_columns:
        result = pd.concat([result, pd.DataFrame(new_columns, index=result.index)], axis=1)

    # SENIOR PRO v26: DOMAIN TOLERANCE ORACLE
    # Values extracted from processed/safe/AI_cup_parameter_info_cleaned.csv
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
    
    domain_cols: dict[str, pd.Series] = {}
    for sensor, tol in TOLERANCES.items():
        if sensor in result.columns:
            val = pd.to_numeric(result[sensor], errors="coerce").ffill().fillna(0).astype("float32")
            # Use rolling mean as proxy for 'Set Value'
            ref = val.rolling(window=20, min_periods=1).mean().astype("float32")
            diff = (val - ref).abs()
            
            domain_cols[f"{sensor}_tol_violation"] = (diff > tol).astype("float32")
            domain_cols[f"{sensor}_tol_deviation"] = (diff / max(tol, 1e-6)).astype("float32")
            
    if domain_cols:
        result = pd.concat([result, pd.DataFrame(domain_cols, index=result.index)], axis=1)
        # Global process health index
        violation_cols = [c for c in result.columns if "_tol_violation" in c]
        result["process_instability_index"] = result[violation_cols].sum(axis=1).astype("float32")

    return result
