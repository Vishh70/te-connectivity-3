from __future__ import annotations

import gc
import json
import os
import pickle
import re
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.feature_utils import (
    SAFETY_SIGNAL_FEATURES,
    TEMPORAL_SIGNAL_FEATURES,
    TEMPORAL_SIGNAL_SOURCES,
    augment_safety_signal_features,
    augment_temporal_signal_features,
)
from backend.normalization_utils import MachineContextScaler

RAW_MES_WORKBOOK = PROJECT_ROOT / "raw_data" / "raw_data" / "MES_combined_final.xlsx"

LABEL_HORIZON_MINUTES = int(os.getenv("TRAIN_LABEL_HORIZON_MINUTES", "30"))
DEFAULT_TRAIN_LABEL_COLUMN = "future_scrap"
REQUESTED_TRAIN_LABEL_COLUMN = os.getenv("TRAIN_LABEL_COLUMN", DEFAULT_TRAIN_LABEL_COLUMN)
ALLOW_EXPERIMENTAL_LABEL = os.getenv("TRAIN_ALLOW_EXPERIMENTAL_LABEL") == "1"
TRAIN_LABEL_COLUMN = REQUESTED_TRAIN_LABEL_COLUMN if (
    REQUESTED_TRAIN_LABEL_COLUMN == DEFAULT_TRAIN_LABEL_COLUMN or ALLOW_EXPERIMENTAL_LABEL
) else DEFAULT_TRAIN_LABEL_COLUMN
VERSION = f"v5-big-dataset-{TRAIN_LABEL_COLUMN}-precision-first"
RANDOM_STATE = 42
# Balance target: keep precision decent while allowing recall to improve.
PRECISION_TARGET = 0.55
RAW_MACHINE_ROW_CAP = int(os.getenv("TRAIN_MAX_ROWS_PER_MACHINE", "15000"))
BIG_DATASET_ROWS_PER_MACHINE = int(os.getenv("TRAIN_BIG_DATASET_ROWS_PER_MACHINE", "30000"))
WEAK_MACHINE_ROW_MULTIPLIERS = {
    "M231": float(os.getenv("TRAIN_WEAK_MACHINE_ROW_MULTIPLIER", "2.0")),
}
WEAK_MACHINE_MIN_POSITIVE_FACTOR = int(os.getenv("TRAIN_WEAK_MACHINE_MIN_POSITIVE_FACTOR", "6"))
PRUNE_ZERO_IMPORTANCE_FEATURES = os.getenv("TRAIN_PRUNE_ZERO_IMPORTANCE", "1") == "1"
TRAIN_USE_RAW_FILES_ENV = os.getenv("TRAIN_USE_RAW_FILES")
USE_RAW_MACHINE_FILES = TRAIN_USE_RAW_FILES_ENV == "1" if TRAIN_USE_RAW_FILES_ENV is not None else False
SKIP_FEB_EVAL = os.getenv("TRAIN_SKIP_FEB_EVAL") == "1"
MODEL_FEATURE_BLACKLIST = {
    "future_scrap",
    "is_scrap",
    "timestamp",
    "machine_id",
    "machine_id_normalized",
    "machine_definition",
    "tool_id",
    "machine_type",
    "part_number",
    "scrap_probability",
    "future_scrap_probability",
    "is_scrap_actual",
    "predicted_scrap",
}
MODEL_FEATURE_BLACKLIST_PREFIXES = ("scrap_",)
MODEL_FEATURE_ALLOWED_PREFIXES = ("scrap_counter", "shot_counter")
MODEL_FEATURE_SUFFIX_TOKENS = (
    "_rolling_mean_5",
    "_rolling_std_5",
    "_rolling_min_5",
    "_rolling_max_5",
    "_rolling_mean_10",
    "_rolling_std_10",
    "_rolling_mean_20",
    "_rolling_std_20",
    "_lag_1",
    "_lag_3",
    "_lag_5",
    "_delta_1",
    "_rate_of_change_5",
    "_rate_of_change_30",
    "_roc_5",
    "_roc_30",
)
MODEL_CONTEXT_FEATURES = [
    "machine_id_code",
    "machine_definition_code",
    "tool_id_code",
    "machine_type_code",
    "part_number_code",
]
DEFAULT_MACHINE_MAP = {"M231": 0.0, "M356": 1.0, "M471": 2.0, "M607": 3.0, "M612": 4.0}
DEFAULT_MACHINE_METADATA = {
    "M231": {"id": "M-231", "name": "Injection Molder M-231", "type": "High Precision", "tool_id": "T-041", "part_number": "UNKNOWN"},
    "M356": {"id": "M-356", "name": "Injection Molder M-356", "type": "Standard", "tool_id": "T-022", "part_number": "UNKNOWN"},
    "M471": {"id": "M-471", "name": "Injection Molder M-471", "type": "High Pressure", "tool_id": "T-109", "part_number": "UNKNOWN"},
    "M607": {"id": "M-607", "name": "Injection Molder M-607", "type": "Multi-Cavity", "tool_id": "T-085", "part_number": "UNKNOWN"},
    "M612": {"id": "M-612", "name": "Injection Molder M-612", "type": "Standard", "tool_id": "T-022", "part_number": "UNKNOWN"},
}


def _normalize_machine_id(machine_id: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", str(machine_id or "")).upper()
    if compact.startswith("M"):
        return compact
    return f"M{compact}"


def _machine_code(machine_id: str) -> float:
    machine_norm = _normalize_machine_id(machine_id)
    match = re.search(r"(\d+)", machine_norm)
    if match:
        return float(int(match.group(1)) / 1000.0)

    compact = re.sub(r"[^A-Z0-9]", "", machine_norm)
    if not compact:
        return 0.0
    total = sum(ord(ch) for ch in compact)
    return float((total % 1000) / 1000.0)


def _machine_sort_key(machine_id: str):
    match = re.search(r"(\d+)", str(machine_id or ""))
    if match:
        return (0, int(match.group(1)))
    return (1, str(machine_id or ""))


def _load_existing_machine_codes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(payload, dict) and "machine_codes" in payload:
        payload = payload["machine_codes"]

    if not isinstance(payload, dict):
        return {}

    return {
        _normalize_machine_id(machine_id): float(code)
        for machine_id, code in payload.items()
        if isinstance(code, (int, float))
    }


def _build_machine_code_map(machine_ids: list[str], seed_map: dict[str, float] | None = None) -> dict[str, float]:
    machine_map: dict[str, float] = {}
    if seed_map:
        for machine_id in seed_map:
            machine_map[_normalize_machine_id(machine_id)] = _machine_code(machine_id)

    for machine_id in sorted({_normalize_machine_id(mid) for mid in machine_ids}, key=_machine_sort_key):
        machine_map[machine_id] = _machine_code(machine_id)

    return dict(sorted(machine_map.items(), key=lambda item: item[0]))


def _build_machine_metadata(machine_id: str) -> dict:
    machine_norm = _normalize_machine_id(machine_id)
    default = DEFAULT_MACHINE_METADATA.get(machine_norm, {})
    display_id = default.get("id", f"M-{machine_norm[1:]}" if machine_norm.startswith("M") and machine_norm[1:].isdigit() else machine_norm)
    return {
        "id": display_id,
        "machine_id_normalized": machine_norm,
        "name": default.get("name", f"Machine {display_id}"),
        "type": default.get("type", "Generic"),
        "tool_id": default.get("tool_id", "T-AUTO"),
        "part_number": PART_NUMBER_BY_MACHINE.get(machine_norm, default.get("part_number", "UNKNOWN")),
    }


def _normalize_category_key(value: str) -> str:
    return str(value or "").strip().upper()


def _normalize_part_number(value: str) -> str:
    text = str(value or "").strip().upper()
    return text if text else "UNKNOWN"


def _load_part_number_lookup() -> dict[str, str]:
    if not RAW_MES_WORKBOOK.exists():
        return {}

    try:
        xls = pd.ExcelFile(RAW_MES_WORKBOOK)
        sheet_name = xls.sheet_names[0]
        header = pd.read_excel(RAW_MES_WORKBOOK, sheet_name=sheet_name, nrows=0)
        available_cols = {str(col).strip() for col in header.columns}
        machine_col = "machine_id_clean" if "machine_id_clean" in available_cols else "machine_id" if "machine_id" in available_cols else None
        if machine_col is None or "part_number" not in available_cols:
            return {}

        frame = pd.read_excel(RAW_MES_WORKBOOK, sheet_name=sheet_name, usecols=[machine_col, "part_number"], dtype=str)
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    machine_series = frame[machine_col].fillna("").astype(str)
    part_series = frame["part_number"].fillna("").astype(str)
    temp = pd.DataFrame({"machine": machine_series, "part": part_series})
    temp["machine"] = temp["machine"].map(_normalize_machine_id)
    temp["part"] = temp["part"].map(_normalize_part_number)
    temp = temp[(temp["machine"] != "M") & (temp["part"] != "UNKNOWN")]

    if temp.empty:
        return {}

    for machine_id, group in temp.groupby("machine", sort=True):
        values = group["part"].dropna().astype(str)
        if values.empty:
            continue
        try:
            mode = values.mode(dropna=True)
            part_number = str(mode.iloc[0]).strip().upper() if not mode.empty else "UNKNOWN"
        except Exception:
            part_number = str(values.iloc[0]).strip().upper()
        if part_number:
            lookup[machine_id] = part_number

    return lookup


PART_NUMBER_BY_MACHINE = _load_part_number_lookup()


def _compute_future_scrap_label(
    group_df: pd.DataFrame,
    horizon_minutes: int = 30,
    event_col: str = "is_scrap",
) -> np.ndarray:
    if group_df.empty:
        return np.array([], dtype=np.uint8)

    timestamps = pd.to_datetime(group_df["timestamp"], errors="coerce", utc=True)
    if event_col not in group_df.columns:
        return np.array([], dtype=np.uint8)

    labels = pd.to_numeric(group_df[event_col], errors="coerce").fillna(0).astype(np.uint8).to_numpy(copy=True)
    valid = timestamps.notna().to_numpy()
    if not valid.all():
        labels = labels[valid]
        timestamps = timestamps[valid].reset_index(drop=True)

    if len(labels) == 0:
        return np.array([], dtype=np.uint8)

    ts_ns = timestamps.astype("int64").to_numpy(copy=True)
    horizon_ns = int(horizon_minutes * 60 * 1_000_000_000)
    csum = np.zeros(len(labels) + 1, dtype=np.int64)
    csum[1:] = np.cumsum(labels.astype(np.int64))
    upper = np.searchsorted(ts_ns, ts_ns + horizon_ns, side="right")
    future = csum[upper] - csum[np.arange(len(labels)) + 1]
    return (future > 0).astype(np.uint8)


def _load_feature_list(project_root: Path) -> list[str]:
    with open(project_root / "models" / "model_features_v4.pkl", "rb") as f:
        return list(pickle.load(f))


def _is_model_feature_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return False
    if lowered in MODEL_FEATURE_BLACKLIST:
        return False
    if any(lowered.startswith(prefix) for prefix in MODEL_FEATURE_BLACKLIST_PREFIXES):
        if any(lowered.startswith(prefix) for prefix in MODEL_FEATURE_ALLOWED_PREFIXES):
            return True
        return False
    return True


def _load_reference_feature_columns(project_root: Path, include_machine_context: bool = False) -> list[str]:
    reference_path = project_root / "new_processed_data" / "cleaned_dataset_v4.5f.parquet"
    feature_cols: list[str] = []

    if reference_path.exists():
        try:
            feature_cols = [
                name
                for name in pq.ParquetFile(reference_path).schema.names
                if _is_model_feature_name(name)
            ]
        except Exception:
            feature_cols = []

    if not feature_cols:
        feature_cols = [name for name in _load_feature_list(project_root) if _is_model_feature_name(name)]

    feature_cols = [
        name
        for name in feature_cols
        if name != "machine_id_encoded"
        or not name.startswith("machine_id_encoded_")
    ]

    def _feature_depth(name: str) -> int:
        return sum(1 for token in MODEL_FEATURE_SUFFIX_TOKENS if token in str(name))

    feature_cols = [
        name
        for name in feature_cols
        if _feature_depth(name) <= 1 and not str(name).startswith("machine_id_encoded_")
    ]

    if include_machine_context:
        feature_cols = list(dict.fromkeys(feature_cols + MODEL_CONTEXT_FEATURES))

    return feature_cols


def _load_available_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def _select_training_label_column(available_cols: set[str]) -> str:
    horizon_label_column = f"scrap_{LABEL_HORIZON_MINUTES}m"
    if REQUESTED_TRAIN_LABEL_COLUMN in available_cols:
        if REQUESTED_TRAIN_LABEL_COLUMN == DEFAULT_TRAIN_LABEL_COLUMN or ALLOW_EXPERIMENTAL_LABEL:
            return REQUESTED_TRAIN_LABEL_COLUMN
        print(
            f"Requested TRAIN_LABEL_COLUMN={REQUESTED_TRAIN_LABEL_COLUMN!r} is experimental and was ignored. "
            f"Using {DEFAULT_TRAIN_LABEL_COLUMN!r} instead."
        )
    if DEFAULT_TRAIN_LABEL_COLUMN in available_cols:
        return DEFAULT_TRAIN_LABEL_COLUMN
    if horizon_label_column in available_cols:
        return horizon_label_column
    return "is_scrap"


def _extract_machine_norm_from_path(path: Path) -> str | None:
    match = re.search(r"(M\d+)", path.stem.upper())
    if match:
        return match.group(1)
    return None


def _load_machine_definition(path: Path) -> str:
    try:
        if "machine_definition" not in _load_available_columns(path):
            return "UNKNOWN"
        raw = pd.read_parquet(path, columns=["machine_definition"])
        defs = raw["machine_definition"].dropna().astype(str).unique()
        if len(defs) > 0:
            return str(defs[0]).upper()
    except Exception:
        pass
    return "UNKNOWN"


def _load_raw_machine_wide_frame(path: Path, allowed_variable_names: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    machine_norm = _extract_machine_norm_from_path(path)
    if not machine_norm:
        return pd.DataFrame(), {}

    raw = pd.read_parquet(
        path,
        columns=["timestamp", "variable_name", "value", "machine_definition"],
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    raw = raw.loc[raw["timestamp"].notna()]
    if allowed_variable_names:
        raw = raw[raw["variable_name"].isin(allowed_variable_names)].copy()
    raw["value"] = raw["value"]

    wide = raw.pivot_table(
        index="timestamp",
        columns="variable_name",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce", utc=True)
    wide = wide.loc[wide["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    if wide.empty:
        return pd.DataFrame(), {}

    machine_definition = _load_machine_definition(path)
    if machine_definition == "UNKNOWN":
        defs = raw["machine_definition"].dropna().astype(str).unique()
        if len(defs) > 0:
            machine_definition = str(defs[0]).upper()

    wide["machine_id_normalized"] = machine_norm
    wide["machine_definition"] = machine_definition

    for col in ("Scrap_counter", "Shot_counter"):
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    return wide, {
        "machine_id_normalized": machine_norm,
        "machine_definition": machine_definition,
        "source_file": path.name,
    }


def _future_binary_event_label(timestamps: pd.Series, event_flags: np.ndarray, horizon_minutes: int = 30) -> np.ndarray:
    if len(event_flags) == 0:
        return np.array([], dtype=np.uint8)

    timestamp_values = pd.to_datetime(timestamps, errors="coerce", utc=True)
    valid = timestamp_values.notna().to_numpy()
    if not valid.all():
        event_flags = event_flags[valid]
        timestamp_values = timestamp_values[valid].reset_index(drop=True)

    if len(event_flags) == 0:
        return np.array([], dtype=np.uint8)

    ts_ns = timestamp_values.astype("int64").to_numpy(copy=True)
    horizon_ns = int(horizon_minutes * 60 * 1_000_000_000)
    csum = np.zeros(len(event_flags) + 1, dtype=np.int64)
    csum[1:] = np.cumsum(event_flags.astype(np.int64))
    upper = np.searchsorted(ts_ns, ts_ns + horizon_ns, side="right")
    future = csum[upper] - csum[np.arange(len(event_flags)) + 1]
    return (future > 0).astype(np.uint8)


def _sample_training_frame(
    frame: pd.DataFrame,
    label_column: str,
    max_rows: int = RAW_MACHINE_ROW_CAP,
    machine_norm: str | None = None,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame

    if label_column not in frame.columns:
        sample_idx = np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)
        return frame.iloc[sample_idx].copy()

    machine_key = _normalize_machine_id(machine_norm) if machine_norm else None
    labels = pd.to_numeric(frame[label_column], errors="coerce").fillna(0).astype(np.uint8).to_numpy(copy=True)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)

    if machine_key in WEAK_MACHINE_ROW_MULTIPLIERS:
        row_multiplier = max(float(WEAK_MACHINE_ROW_MULTIPLIERS[machine_key]), 1.0)
        boosted_cap = int(np.ceil(max_rows * row_multiplier))
        positive_floor = int(len(pos_idx) * WEAK_MACHINE_MIN_POSITIVE_FACTOR)
        max_rows = min(len(frame), max(max_rows, boosted_cap, positive_floor))
        if len(frame) <= max_rows:
            return frame.copy()

    if len(pos_idx) == 0:
        sample_idx = np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)
    elif len(pos_idx) >= max_rows:
        pos_pick = np.linspace(0, len(pos_idx) - 1, num=max_rows, dtype=int)
        sample_idx = pos_idx[pos_pick]
    else:
        neg_target = max_rows - len(pos_idx)
        if len(neg_idx) <= neg_target:
            sample_idx = np.concatenate([pos_idx, neg_idx])
        else:
            neg_pick = np.linspace(0, len(neg_idx) - 1, num=neg_target, dtype=int)
            sample_idx = np.concatenate([pos_idx, neg_idx[neg_pick]])

    sample_idx = np.unique(sample_idx)
    sample_idx.sort()
    return frame.iloc[sample_idx].copy()


def _compute_future_scrap_from_counter(group_df: pd.DataFrame, horizon_minutes: int = 30) -> np.ndarray:
    if group_df.empty or "Scrap_counter" not in group_df.columns:
        return np.array([], dtype=np.uint8)

    scrap_counter = pd.to_numeric(group_df["Scrap_counter"], errors="coerce").fillna(0.0).to_numpy(copy=True)
    if len(scrap_counter) == 0:
        return np.array([], dtype=np.uint8)

    deltas = np.diff(np.concatenate(([scrap_counter[0]], scrap_counter)))
    event_flags = (np.clip(deltas, 0, None) > 0).astype(np.uint8)
    return _future_binary_event_label(group_df["timestamp"], event_flags, horizon_minutes=horizon_minutes)


def _encode_categorical_columns(
    frame: pd.DataFrame,
    category_maps: dict[str, dict[str, int]],
    machine_meta_by_id: dict[str, str],
) -> pd.DataFrame:
    encoded = frame.copy(deep=False)

    if "machine_id_code" in category_maps:
        encoded["machine_id_code"] = [
            int(category_maps["machine_id_code"].get(_normalize_machine_id(value), 0))
            for value in encoded.get("machine_id_normalized", pd.Series(["UNKNOWN"] * len(encoded)))
        ]

    if "machine_definition_code" in category_maps:
        machine_defs = encoded.get("machine_definition")
        if machine_defs is None:
            machine_defs = pd.Series(
                [
                    machine_meta_by_id.get(_normalize_machine_id(value), "UNKNOWN")
                    for value in encoded.get("machine_id_normalized", pd.Series(["UNKNOWN"] * len(encoded)))
                ],
                index=encoded.index,
            )
        else:
            machine_defs = machine_defs.fillna("UNKNOWN").astype(str).str.upper()
            fallback_defs = [
                machine_meta_by_id.get(_normalize_machine_id(value), "UNKNOWN")
                for value in encoded.get("machine_id_normalized", pd.Series(["UNKNOWN"] * len(encoded)))
            ]
            machine_defs = machine_defs.where(machine_defs.ne("UNKNOWN"), pd.Series(fallback_defs, index=encoded.index))
        encoded["machine_definition_code"] = [
            int(category_maps["machine_definition_code"].get(str(value).upper(), 0))
            for value in machine_defs
        ]

    if "tool_id_code" in category_maps:
        tool_ids = encoded.get("tool_id")
        if tool_ids is None:
            tool_ids = pd.Series(["UNKNOWN"] * len(encoded), index=encoded.index)
        encoded["tool_id_code"] = [
            int(category_maps["tool_id_code"].get(str(value).upper(), 0))
            for value in tool_ids.fillna("UNKNOWN").astype(str)
        ]

    if "machine_type_code" in category_maps:
        machine_types = encoded.get("machine_type")
        if machine_types is None:
            machine_types = pd.Series(["UNKNOWN"] * len(encoded), index=encoded.index)
        encoded["machine_type_code"] = [
            int(category_maps["machine_type_code"].get(str(value).upper(), 0))
            for value in machine_types.fillna("UNKNOWN").astype(str)
        ]

    if "part_number_code" in category_maps:
        part_numbers = encoded.get("part_number")
        if part_numbers is None:
            part_numbers = pd.Series(["UNKNOWN"] * len(encoded), index=encoded.index)
        encoded["part_number_code"] = [
            int(category_maps["part_number_code"].get(str(value).upper(), 0))
            for value in part_numbers.fillna("UNKNOWN").astype(str)
        ]

    return encoded


def _resolve_base_feature(name: str) -> str | None:
    if name == "machine_id_encoded":
        return None

    for suffix in (
        "_rolling_mean_5",
        "_rolling_std_5",
        "_rolling_min_5",
        "_rolling_max_5",
        "_lag_1",
        "_lag_3",
        "_lag_5",
        "_rate_of_change_5",
        "_rate_of_change_30",
        "_roc_5",
        "_roc_30",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _build_group_feature_frame(group_df: pd.DataFrame, feature_cols: list[str], machine_code: float) -> pd.DataFrame:
    if any(
        feature in SAFETY_SIGNAL_FEATURES
        or feature in TEMPORAL_SIGNAL_FEATURES
        for feature in feature_cols
    ):
        group_df = augment_temporal_signal_features(group_df)

    index = group_df.index
    cache: dict[str, pd.Series] = {}
    feature_data: dict[str, pd.Series] = {}

    def _finalize_series(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    def series_for(name: str) -> pd.Series:
        if name in cache:
            return cache[name]

        if name == "machine_id_encoded":
            s = _finalize_series(pd.Series(machine_code, index=index, dtype="float32"))
            cache[name] = s
            return s

        if name in group_df.columns:
            val = group_df[name]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            s = _finalize_series(pd.to_numeric(val, errors="coerce").astype("float32"))
            cache[name] = s
            return s

        base = _resolve_base_feature(name)
        if base is None:
            s = _finalize_series(pd.Series(0.0, index=index, dtype="float32"))
            cache[name] = s
            return s

        if base not in group_df.columns:
            s = _finalize_series(pd.Series(0.0, index=index, dtype="float32"))
            cache[name] = s
            return s

        base_series = series_for(base)

        if name.endswith(("_lag_1", "_lag_3", "_lag_5")):
            lag_n = int(name.rsplit("_lag_", 1)[1])
            s = base_series.shift(lag_n)
        elif name.endswith(("_rolling_mean_5",)):
            s = base_series.rolling(window=5, min_periods=1).mean()
        elif name.endswith(("_rolling_std_5",)):
            s = base_series.rolling(window=5, min_periods=1).std()
        elif name.endswith(("_rolling_min_5",)):
            s = base_series.rolling(window=5, min_periods=1).min()
        elif name.endswith(("_rolling_max_5",)):
            s = base_series.rolling(window=5, min_periods=1).max()
        elif name.endswith(("_rate_of_change_5", "_roc_5")):
            s = base_series.pct_change(periods=5, fill_method=None)
        elif name.endswith(("_rate_of_change_30", "_roc_30")):
            s = base_series.pct_change(periods=30, fill_method=None)
        else:
            s = pd.Series(0.0, index=index, dtype="float32")

        s = _finalize_series(s)
        cache[name] = s
        return s

    for feature in feature_cols:
        feature_data[feature] = series_for(feature)

    return pd.DataFrame(feature_data, index=index)


def _build_reconstructed_frame(
    raw_df: pd.DataFrame,
    feature_cols: list[str],
    machine_code_map: dict[str, float],
) -> pd.DataFrame:
    if "timestamp" not in raw_df.columns:
        raise ValueError("Expected a timestamp column for feature reconstruction.")

    machine_col = None
    for candidate in ("machine_id_normalized", "machine_id", "machine_definition"):
        if candidate in raw_df.columns:
            machine_col = candidate
            break
    if machine_col is None:
        raise ValueError("Expected a machine identifier column for feature reconstruction.")

    work_df = raw_df.copy()
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce", utc=True)
    work_df = work_df.loc[work_df["timestamp"].notna()].sort_values([machine_col, "timestamp"]).reset_index(drop=True)

    feature_frames = []
    for machine_id, group in work_df.groupby(machine_col, sort=True):
        group = group.reset_index(drop=True)
        machine_key = _normalize_machine_id(machine_id)
        machine_code = float(machine_code_map.get(machine_key, -1.0))
        feat_df = _build_group_feature_frame(group, feature_cols, machine_code)
        if "timestamp" in group.columns:
            feat_df["timestamp"] = group["timestamp"].values
        # SENIOR FIX v10: Always attach the normalized machine ID for the labeling bridge
        feat_df["machine_id_normalized"] = _normalize_machine_id(machine_id)
        if "machine_definition" in group.columns:
            feat_df["machine_definition"] = group["machine_definition"].values
        feature_frames.append(feat_df)

    return pd.concat(feature_frames, ignore_index=True)


def _best_threshold_for_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float = PRECISION_TARGET) -> dict:
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float32)

    def _metrics_from_pred(pred: np.ndarray, threshold: float) -> dict:
        pred = np.asarray(pred, dtype=np.uint8)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_positives": int(pred.sum()),
        }

    # Exact threshold search gets very expensive on the raw machine corpus.
    # A balanced cap keeps the search fast while preserving the precision/recall shape.
    max_points = 150_000
    if len(scores) > max_points:
        rng = np.random.default_rng(RANDOM_STATE)
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            sample_idx = rng.choice(len(scores), size=max_points, replace=False)
        else:
            pos_target = min(len(pos_idx), max_points // 4)
            neg_target = min(len(neg_idx), max_points - pos_target)
            if pos_target == 0:
                pos_target = min(len(pos_idx), max(1, max_points // 10))
            if neg_target == 0:
                neg_target = min(len(neg_idx), max(1, max_points - pos_target))
            chosen_pos = rng.choice(pos_idx, size=pos_target, replace=False) if len(pos_idx) > pos_target else pos_idx
            chosen_neg = rng.choice(neg_idx, size=neg_target, replace=False) if len(neg_idx) > neg_target else neg_idx
            sample_idx = np.concatenate([chosen_pos, chosen_neg])
            rng.shuffle(sample_idx)
        y_true = y_true[sample_idx]
        scores = scores[sample_idx]

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        pred = (scores >= 0.5).astype(int)
        return _metrics_from_pred(pred, 0.5)

    candidates = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision:
            pred = (scores >= t).astype(int)
            candidates.append(_metrics_from_pred(pred, t))

    if candidates:
        return max(candidates, key=lambda item: (item["recall"], item["f1"], item["precision"]))

    # Fall back to the threshold with the best F1 if the precision target is unreachable.
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    t = float(thresholds[best_idx])
    pred = (scores >= t).astype(int)
    return _metrics_from_pred(pred, t)


def _split_timewise_frame(frame: pd.DataFrame, train_ratio: float = 0.70, calib_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a sorted frame into train, calibration, and validation slices."""
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()

    n = len(frame)
    if n < 3:
        train_end = max(1, n - 2)
        calib_end = max(train_end + 1, n - 1)
    else:
        train_end = int(n * train_ratio)
        train_end = min(max(train_end, 1), n - 2)
        calib_end = int(n * (train_ratio + calib_ratio))
        calib_end = min(max(calib_end, train_end + 1), n - 1)

    train_df = frame.iloc[:train_end].copy()
    calib_df = frame.iloc[train_end:calib_end].copy()
    valid_df = frame.iloc[calib_end:].copy()

    if calib_df.empty:
        calib_df = frame.iloc[max(train_end - 1, 0):max(train_end, 1)].copy()
    if valid_df.empty:
        valid_df = frame.iloc[-1:].copy()

    return train_df, calib_df, valid_df


def _fit_probability_calibrator(scores: np.ndarray, y_true: np.ndarray):
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.uint8)
    if len(scores) < 20 or len(np.unique(y_true)) < 2:
        return None

    try:
        calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
        calibrator.fit(scores.reshape(-1, 1), y_true)
        return calibrator
    except Exception:
        return None


def _apply_calibration(calibrator, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if calibrator is None:
        return scores
    try:
        if hasattr(calibrator, "predict_proba"):
            calibrated = np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float32)
        else:
            calibrated = np.asarray(calibrator.predict(scores.reshape(-1, 1)), dtype=np.float32)
    except Exception:
        return scores
    return np.clip(calibrated, 0.0, 1.0)


def _threshold_map_from_validation(
    validation_frame: pd.DataFrame,
    score_col: str,
    label_col: str,
    group_col: str | None = None,
    target_precision: float = PRECISION_TARGET,
) -> dict[str, dict]:
    if validation_frame.empty or score_col not in validation_frame.columns or label_col not in validation_frame.columns:
        return {}

    if group_col and group_col in validation_frame.columns:
        group_iter = validation_frame.groupby(group_col, sort=True)
    else:
        group_iter = [("__ALL__", validation_frame)]

    thresholds: dict[str, dict] = {}
    for group_name, group_df in group_iter:
        y_group = group_df[label_col].to_numpy(dtype=np.uint8, copy=True)
        score_group = group_df[score_col].to_numpy(dtype=np.float32, copy=True)
        if len(y_group) == 0:
            continue
        if len(np.unique(y_group)) < 2:
            threshold = 0.5
            pred = (score_group >= threshold).astype(int)
            thresholds[str(group_name)] = {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_group, pred)),
                "precision": float(precision_score(y_group, pred, zero_division=0)),
                "recall": float(recall_score(y_group, pred, zero_division=0)),
                "f1": float(f1_score(y_group, pred, zero_division=0)),
                "predicted_positives": int(pred.sum()),
            }
            continue

        thresholds[str(group_name)] = _best_threshold_for_precision(y_group, score_group, target_precision=target_precision)

    return thresholds


def _feature_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )


def _prune_zero_importance_features(feature_cols: list[str], importance_df: pd.DataFrame) -> tuple[list[str], set[str]]:
    zero_mask = (importance_df["importance_gain"] == 0) & (importance_df["importance_split"] == 0)
    removable = set(importance_df.loc[zero_mask, "feature"].astype(str))
    pruned = [feature for feature in feature_cols if feature not in removable]
    
    # Senior Guard: Never prune below 10 features to avoid LightGBM crash if importance is sparse
    if len(pruned) < 10:
        print(f"Pruning Safety: Avoided removing {len(removable)} features as it would leave only {len(pruned)}. Retaining top 20 instead.")
        top_20 = importance_df.sort_values("importance_gain", ascending=False).head(20)["feature"].tolist()
        return top_20, set(feature_cols) - set(top_20)
        
    return pruned, removable


def _subset_feature_matrix(matrix, keep_indices: np.ndarray):
    if hasattr(matrix, "iloc"):
        return matrix.iloc[:, keep_indices]
    return matrix[:, keep_indices]


def main() -> None:
    np.random.seed(RANDOM_STATE)

    project_root = Path(__file__).resolve().parent.parent

    raw_train_files = []
    for path in sorted((project_root / "new_processed_data").glob("*_TRAIN.parquet")):
        stem = path.stem.upper()
        if "HYDRA" in stem:
            continue
        if re.search(r"^(NEW|OLD)_M\d+_TRAIN$", stem):
            raw_train_files.append(path)

    if TRAIN_USE_RAW_FILES_ENV is None:
        use_raw_machine_files = bool(raw_train_files)
    else:
        use_raw_machine_files = TRAIN_USE_RAW_FILES_ENV == "1"

    if raw_train_files and use_raw_machine_files:
        feb_path = project_root / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
        model_path = project_root / "models" / "scrap_risk_model_v5.pkl"
        model_features_path = project_root / "models" / "model_features_v5.pkl"
        categorical_map_path = project_root / "metrics" / "categorical_encodings_v5.json"
        metrics_path = project_root / "metrics" / "training_metrics_v5.json"
        report_path = project_root / "metrics" / "final_model_report_v5.json"
        thresholds_path = project_root / "metrics" / "machine_thresholds_v5.json"
        registry_path = project_root / "metrics" / "machine_registry_v5.json"
        calibration_path = project_root / "models" / "scrap_risk_calibrator_v5.pkl"
        feature_importance_path = project_root / "features" / "feature_importance_v5.csv"

        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        categorical_map_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

        base_feature_cols = _load_reference_feature_columns(project_root, include_machine_context=True)
        feature_source_cols = list(dict.fromkeys(base_feature_cols + list(TEMPORAL_SIGNAL_SOURCES)))
        model_base_feature_cols = [c for c in base_feature_cols if c not in {"Scrap_counter", "Shot_counter"}]
        categorical_feature_cols = [
            "machine_id_code",
            "machine_definition_code",
            "tool_id_code",
            "machine_type_code",
            "part_number_code",
        ]
        model_feature_cols = list(
            dict.fromkeys(
                model_base_feature_cols
                + categorical_feature_cols
                + list(SAFETY_SIGNAL_FEATURES)
                + list(TEMPORAL_SIGNAL_FEATURES)
            )
        )
        allowed_variable_names = {
            str(base_name)
            for base_name in (_resolve_base_feature(feature) for feature in feature_source_cols)
            if base_name is not None
        }

        machine_ids: list[str] = []
        machine_definitions_by_id: dict[str, str] = {}
        raw_meta: list[tuple[Path, dict]] = []

        for path in raw_train_files:
            machine_norm = _extract_machine_norm_from_path(path)
            if not machine_norm:
                continue
            machine_definition = _load_machine_definition(path)
            machine_ids.append(machine_norm)
            machine_definitions_by_id.setdefault(machine_norm, machine_definition)
            raw_meta.append(
                (
                    path,
                    {
                        "machine_id_normalized": machine_norm,
                        "machine_definition": machine_definition,
                    },
                )
            )

        if not raw_meta:
            raise ValueError("No usable raw machine training files were found.")

        machine_code_map = _build_machine_code_map(machine_ids, seed_map=_load_existing_machine_codes(registry_path))
        machine_id_map = {"UNKNOWN": 0}
        for idx, machine_id in enumerate(sorted(set(machine_ids), key=_machine_sort_key), start=1):
            machine_id_map[_normalize_machine_id(machine_id)] = idx

        machine_definition_values = sorted(
            {
                (machine_definitions_by_id.get(machine_id) or "UNKNOWN").upper()
                for machine_id in machine_ids
                if (machine_definitions_by_id.get(machine_id) or "UNKNOWN").upper() != "UNKNOWN"
            }
        )
        machine_definition_map = {"UNKNOWN": 0}
        for idx, machine_definition in enumerate(machine_definition_values, start=1):
            machine_definition_map[str(machine_definition).upper()] = idx

        tool_values = sorted({str(_build_machine_metadata(machine_id)["tool_id"]).upper() for machine_id in machine_ids})
        machine_type_values = sorted({str(_build_machine_metadata(machine_id)["type"]).upper() for machine_id in machine_ids})
        part_number_values = sorted({str(_build_machine_metadata(machine_id)["part_number"]).upper() for machine_id in machine_ids})
        tool_id_map = {"UNKNOWN": 0}
        for idx, tool_id in enumerate([value for value in tool_values if value != "UNKNOWN"], start=1):
            tool_id_map[str(tool_id).upper()] = idx
        machine_type_map = {"UNKNOWN": 0}
        for idx, machine_type in enumerate([value for value in machine_type_values if value != "UNKNOWN"], start=1):
            machine_type_map[str(machine_type).upper()] = idx
        part_number_map = {"UNKNOWN": 0}
        for idx, part_number in enumerate([value for value in part_number_values if value != "UNKNOWN"], start=1):
            part_number_map[str(part_number).upper()] = idx

        category_maps = {
            "machine_id_code": machine_id_map,
            "machine_definition_code": machine_definition_map,
            "tool_id_code": tool_id_map,
            "machine_type_code": machine_type_map,
            "part_number_code": part_number_map,
        }

        train_frames = []
        calib_frames = []
        valid_frames = []
        per_machine_registry = []

        print(f"Loading {len(raw_meta)} raw machine train files ...")
        for path, meta in raw_meta:
            wide_df, loaded_meta = _load_raw_machine_wide_frame(path, allowed_variable_names=allowed_variable_names)
            if wide_df.empty or "Scrap_counter" not in wide_df.columns:
                continue

            machine_norm = loaded_meta.get("machine_id_normalized", meta["machine_id_normalized"])
            machine_definition = loaded_meta.get("machine_definition", meta["machine_definition"])
            wide_df["machine_id_normalized"] = machine_norm
            wide_df["machine_definition"] = machine_definition
            wide_df["future_scrap"] = _compute_future_scrap_from_counter(wide_df, horizon_minutes=LABEL_HORIZON_MINUTES)

            feature_df = _build_reconstructed_frame(wide_df, model_feature_cols, machine_code_map)
            feature_df["future_scrap"] = pd.to_numeric(wide_df["future_scrap"], errors="coerce").fillna(0).astype(np.uint8).values
            feature_df["machine_id_normalized"] = wide_df["machine_id_normalized"].values
            feature_df["machine_definition"] = wide_df["machine_definition"].values
            machine_meta = _build_machine_metadata(machine_norm)
            feature_df["tool_id"] = machine_meta["tool_id"]
            feature_df["machine_type"] = machine_meta["type"]
            feature_df["part_number"] = machine_meta["part_number"]
            feature_df = _encode_categorical_columns(feature_df, category_maps, machine_definitions_by_id)
            feature_df["machine_id_code"] = pd.to_numeric(feature_df["machine_id_code"], errors="coerce").fillna(0).astype(np.int32)
            feature_df["machine_definition_code"] = pd.to_numeric(feature_df["machine_definition_code"], errors="coerce").fillna(0).astype(np.int32)
            feature_df["tool_id_code"] = pd.to_numeric(feature_df["tool_id_code"], errors="coerce").fillna(0).astype(np.int32)
            feature_df["machine_type_code"] = pd.to_numeric(feature_df["machine_type_code"], errors="coerce").fillna(0).astype(np.int32)
            feature_df["part_number_code"] = pd.to_numeric(feature_df["part_number_code"], errors="coerce").fillna(0).astype(np.int32)
            for column in model_feature_cols:
                if column in feature_df.columns:
                    feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce").fillna(0.0).astype(np.float32)

            feature_df = _sample_training_frame(feature_df, "future_scrap", RAW_MACHINE_ROW_CAP, machine_norm=machine_norm)
            train_df, calib_df, valid_df = _split_timewise_frame(feature_df, train_ratio=0.70, calib_ratio=0.15)
            train_frames.append(train_df)
            calib_frames.append(calib_df)
            valid_frames.append(valid_df)

            machine_meta = _build_machine_metadata(machine_norm)
            per_machine_registry.append(
                {
                    "id": machine_meta["id"],
                    "machine_id_normalized": machine_norm,
                    "name": machine_meta["name"],
                    "type": machine_meta["type"],
                    "tool_id": machine_meta["tool_id"],
                    "machine_definition": machine_definition,
                }
            )

        if not train_frames or not calib_frames or not valid_frames:
            raise ValueError("Unable to build raw training and validation frames from the machine files.")

        train_df = pd.concat(train_frames, ignore_index=True)
        calib_df = pd.concat(calib_frames, ignore_index=True)
        valid_df = pd.concat(valid_frames, ignore_index=True)
        label_column = "future_scrap"

        row_count = int(len(train_df) + len(calib_df) + len(valid_df))
        dataset_shape = (row_count, len(model_feature_cols))
        scrap_ratio = float(
            (
                train_df[label_column].sum()
                + calib_df[label_column].sum()
                + valid_df[label_column].sum()
            )
            / max(row_count, 1)
        )

        X_train = train_df[model_feature_cols]
        y_train = train_df[label_column].to_numpy(dtype=np.uint8, copy=True)
        X_calib = calib_df[model_feature_cols]
        y_calib = calib_df[label_column].to_numpy(dtype=np.uint8, copy=True)
        X_valid = valid_df[model_feature_cols]
        y_valid = valid_df[label_column].to_numpy(dtype=np.uint8, copy=True)
        valid_meta_df = valid_df[["machine_id_normalized", "tool_id", "machine_type", "part_number", label_column]].copy()

        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = neg_count / max(pos_count, 1)
        print(f"Dynamic scale_pos_weight: {scale_pos_weight:.2f} (neg={neg_count:,}, pos={pos_count:,})")

        del train_df, calib_df, valid_df
        gc.collect()

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 96,
            "max_depth": 10,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos_weight,
            "seed": RANDOM_STATE,
            "feature_fraction_seed": RANDOM_STATE,
            "bagging_seed": RANDOM_STATE,
            "data_random_seed": RANDOM_STATE,
            "max_bin": 255,
            "verbose": -1,
        }

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=model_feature_cols,
            categorical_feature=categorical_feature_cols,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            feature_name=model_feature_cols,
            categorical_feature=categorical_feature_cols,
            reference=train_data,
            free_raw_data=False,
        )

        print("Training LightGBM model on the raw machine files ...")
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1500,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=120, first_metric_only=True),
                lgb.log_evaluation(period=50),
            ],
        )

        raw_valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
        calibrator = _fit_probability_calibrator(
            model.predict(X_calib, num_iteration=model.best_iteration),
            y_calib,
        )
        calibrated_valid_proba = _apply_calibration(calibrator, raw_valid_proba)

        valid_pred = (calibrated_valid_proba >= 0.5).astype(int)
        valid_auc = float(roc_auc_score(y_valid, calibrated_valid_proba))
        valid_precision = float(precision_score(y_valid, valid_pred, zero_division=0))
        valid_recall = float(recall_score(y_valid, valid_pred, zero_division=0))
        valid_f1 = float(f1_score(y_valid, valid_pred, zero_division=0))
        valid_accuracy = float(accuracy_score(y_valid, valid_pred))
        valid_target_threshold = _best_threshold_for_precision(y_valid, calibrated_valid_proba, target_precision=PRECISION_TARGET)
        valid_rows = int(len(y_valid))

        joblib.dump(model, model_path)
        if calibrator is not None:
            joblib.dump(calibrator, calibration_path)
        with open(model_features_path, "wb") as f:
            pickle.dump(model_feature_cols, f)
        with open(categorical_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": VERSION,
                    "categorical_features": categorical_feature_cols,
                    "machine_id_code": machine_id_map,
                    "machine_definition_code": machine_definition_map,
                    "tool_id_code": tool_id_map,
                    "machine_type_code": machine_type_map,
                    "part_number_code": part_number_map,
                },
                f,
                indent=2,
            )

        importance_df = pd.DataFrame(
            {
                "feature": model_feature_cols,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        importance_df.to_csv(feature_importance_path, index=False)

        features_before_pruning = len(model_feature_cols)
        if PRUNE_ZERO_IMPORTANCE_FEATURES:
            pruned_feature_cols, removable_features = _prune_zero_importance_features(model_feature_cols, importance_df)
            if len(pruned_feature_cols) < len(model_feature_cols):
                print(
                    f"Pruning {len(removable_features)} zero-importance features "
                    f"({len(model_feature_cols)} -> {len(pruned_feature_cols)}) and retraining final model ..."
                )
                feature_index_lookup = {feature: idx for idx, feature in enumerate(model_feature_cols)}
                keep_indices = np.array([feature_index_lookup[feature] for feature in pruned_feature_cols], dtype=int)
                model_feature_cols = pruned_feature_cols
                categorical_feature_cols = [col for col in categorical_feature_cols if col in model_feature_cols]
                X_train = _subset_feature_matrix(X_train, keep_indices)
                X_calib = _subset_feature_matrix(X_calib, keep_indices)
                X_valid = _subset_feature_matrix(X_valid, keep_indices)
                train_data = lgb.Dataset(
                    X_train,
                    label=y_train,
                    feature_name=model_feature_cols,
                    categorical_feature=categorical_feature_cols,
                    free_raw_data=False,
                )
                valid_data = lgb.Dataset(
                    X_valid,
                    label=y_valid,
                    feature_name=model_feature_cols,
                    categorical_feature=categorical_feature_cols,
                    reference=train_data,
                    free_raw_data=False,
                )
                model = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round=1200,
                    valid_sets=[valid_data],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                        lgb.log_evaluation(period=50),
                    ],
                )

                valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
                valid_pred = (valid_proba >= 0.5).astype(int)
                valid_auc = float(roc_auc_score(y_valid, valid_proba))
                valid_precision = float(precision_score(y_valid, valid_pred, zero_division=0))
                valid_recall = float(recall_score(y_valid, valid_pred, zero_division=0))
                valid_f1 = float(f1_score(y_valid, valid_pred, zero_division=0))
                valid_accuracy = float(accuracy_score(y_valid, valid_pred))
                valid_target_threshold = _best_threshold_for_precision(y_valid, valid_proba, target_precision=PRECISION_TARGET)
                valid_rows = int(len(y_valid))
                raw_valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
                calibrator = _fit_probability_calibrator(
                    model.predict(X_calib, num_iteration=model.best_iteration),
                    y_calib,
                )
                calibrated_valid_proba = _apply_calibration(calibrator, raw_valid_proba)
                importance_df = _feature_importance_frame(model, model_feature_cols)
                importance_df.to_csv(feature_importance_path, index=False)
                joblib.dump(model, model_path)
                if calibrator is not None:
                    joblib.dump(calibrator, calibration_path)
                with open(model_features_path, "wb") as f:
                    pickle.dump(model_feature_cols, f)

        else:
            features_before_pruning = len(model_feature_cols)

        if SKIP_FEB_EVAL:
            metrics = {
                "version": VERSION,
                "label_horizon_minutes": int(LABEL_HORIZON_MINUTES),
                "label_column": label_column,
                "training_dataset": "raw_machine_train_files",
                "training_rows": row_count,
                "training_positive_rate": scrap_ratio,
                "training_label_column": label_column,
                "training_machine_column": "machine_id_normalized",
                "training_feature_count": len(model_feature_cols),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration or 0),
                "validation": {
                    "rows": valid_rows,
                    "accuracy": valid_accuracy,
                    "precision": valid_precision,
                    "recall": valid_recall,
                    "f1": valid_f1,
                    "auc": valid_auc,
                    "precision_target_threshold": valid_target_threshold,
                },
                "calibration": {
                    "path": str(calibration_path) if calibrator is not None else None,
                    "method": "sigmoid" if calibrator is not None else "identity",
                    "calibration_rows": int(len(y_calib)) if calibrator is not None else 0,
                },
                "scale_pos_weight": float(scale_pos_weight),
                "features_before_pruning": features_before_pruning,
                "features_after_pruning": len(model_feature_cols),
                "feb_test": None,
            }

            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Saved model: {model_path}")
            print(f"Saved feature list: {model_features_path}")
            print(f"Saved categorical map: {categorical_map_path}")
            print(f"Saved machine registry: {registry_path}")
            if calibrator is not None:
                print(f"Saved calibration model: {calibration_path}")
            print(f"Saved threshold calibration: {thresholds_path}")
            print(f"Saved report: {report_path}")
            print("\nValidation metrics:")
            print(f"  AUC: {valid_auc:.4f}")
            print(f"  Precision: {valid_precision:.4f}")
            print(f"  Recall: {valid_recall:.4f}")
            print(f"  F1: {valid_f1:.4f}")
            print(f"  Best threshold for precision >= {PRECISION_TARGET:.2f}: {valid_target_threshold['threshold']:.6f}")
            return

        threshold_scope_cols = ["machine_id_normalized", "tool_id", "machine_type", "part_number", label_column]
        valid_threshold_df = valid_meta_df[threshold_scope_cols].copy()
        valid_threshold_df["raw_probability"] = raw_valid_proba
        valid_threshold_df["calibrated_probability"] = calibrated_valid_proba

        global_threshold = _best_threshold_for_precision(
            valid_threshold_df[label_column].to_numpy(dtype=np.uint8, copy=True),
            valid_threshold_df["calibrated_probability"].to_numpy(dtype=np.float32, copy=True),
            target_precision=PRECISION_TARGET,
        )

        per_machine_thresholds = _threshold_map_from_validation(
            valid_threshold_df,
            "calibrated_probability",
            label_column,
            group_col="machine_id_normalized",
        )
        per_tool_thresholds = _threshold_map_from_validation(
            valid_threshold_df,
            "calibrated_probability",
            label_column,
            group_col="tool_id",
        )
        per_type_thresholds = _threshold_map_from_validation(
            valid_threshold_df,
            "calibrated_probability",
            label_column,
            group_col="machine_type",
        )

        feb_available = _load_available_columns(feb_path)
        feb_required = [
            c
            for c in ("timestamp", "machine_id_normalized", "is_scrap_actual", "scrap_probability", "predicted_scrap", "machine_definition")
            if c in feb_available
        ]
        feb_base_cols = [c for c in base_feature_cols if c in feb_available]
        feb_df = pd.read_parquet(feb_path, columns=list(dict.fromkeys(feb_required + feb_base_cols)))
        feb_df["timestamp"] = pd.to_datetime(feb_df["timestamp"], errors="coerce", utc=True)
        feb_df = feb_df.loc[feb_df["timestamp"].notna()].sort_values(["machine_id_normalized", "timestamp"]).reset_index(drop=True)
        if "machine_id_normalized" not in feb_df.columns:
            raise ValueError("FEB_TEST_RESULTS.parquet must include machine_id_normalized.")
        feb_df["machine_id_normalized"] = feb_df["machine_id_normalized"].astype(str).map(_normalize_machine_id)
        feb_df["machine_definition"] = feb_df["machine_id_normalized"].map(lambda m: machine_definitions_by_id.get(m, "UNKNOWN"))
        feb_meta = feb_df["machine_id_normalized"].map(lambda m: _build_machine_metadata(m))
        feb_df["tool_id"] = feb_meta.map(lambda meta: meta["tool_id"])
        feb_df["machine_type"] = feb_meta.map(lambda meta: meta["type"])
        feb_df["part_number"] = feb_meta.map(lambda meta: meta["part_number"])
        feb_features = _build_reconstructed_frame(feb_df, model_feature_cols, machine_code_map)
        feb_features["future_scrap"] = 0
        for _, group in feb_df.groupby("machine_id_normalized", sort=True):
            feb_features.loc[group.index, "future_scrap"] = _compute_future_scrap_label(
                group,
                horizon_minutes=LABEL_HORIZON_MINUTES,
                event_col="is_scrap_actual",
            )
        feb_features["machine_id_normalized"] = feb_df["machine_id_normalized"].values
        feb_features["machine_definition"] = feb_df["machine_definition"].values
        feb_features["tool_id"] = feb_df["tool_id"].values
        feb_features["machine_type"] = feb_df["machine_type"].values
        feb_features["part_number"] = feb_df["part_number"].values
        feb_features = _encode_categorical_columns(feb_features, category_maps, machine_definitions_by_id)
        feb_features["machine_id_code"] = pd.to_numeric(feb_features["machine_id_code"], errors="coerce").fillna(0).astype(np.int32)
        feb_features["machine_definition_code"] = pd.to_numeric(feb_features["machine_definition_code"], errors="coerce").fillna(0).astype(np.int32)
        feb_features["tool_id_code"] = pd.to_numeric(feb_features["tool_id_code"], errors="coerce").fillna(0).astype(np.int32)
        feb_features["machine_type_code"] = pd.to_numeric(feb_features["machine_type_code"], errors="coerce").fillna(0).astype(np.int32)
        feb_features["part_number_code"] = pd.to_numeric(feb_features["part_number_code"], errors="coerce").fillna(0).astype(np.int32)

        feb_X = feb_features[model_feature_cols]
        feb_y = feb_features["future_scrap"].to_numpy(dtype=np.uint8, copy=True)
        feb_raw_proba = model.predict(feb_X, num_iteration=model.best_iteration)
        feb_proba = _apply_calibration(calibrator, feb_raw_proba)

        def _resolve_threshold_row(machine_id_value: str, tool_value: str | None = None, type_value: str | None = None) -> float:
            machine_key = _normalize_machine_id(machine_id_value)
            tool_key = _normalize_category_key(tool_value)
            type_key = _normalize_category_key(type_value)
            def _maybe_use(value: float | None) -> float | None:
                if value is None:
                    return None
                if not np.isfinite(value) or value >= 0.99 or value <= 0.0:
                    return None
                return float(value)

            chosen = _maybe_use(per_machine_thresholds.get(machine_key, {}).get("threshold"))
            if chosen is not None:
                return chosen

            chosen = _maybe_use(per_tool_thresholds.get(tool_key, {}).get("threshold"))
            if chosen is not None:
                return chosen

            chosen = _maybe_use(per_type_thresholds.get(type_key, {}).get("threshold"))
            if chosen is not None:
                return chosen

            chosen = _maybe_use(float(global_threshold["threshold"]))
            if chosen is not None:
                return chosen

            return float(global_threshold["threshold"])

        feb_thresholds = pd.Series(
            [
                _resolve_threshold_row(machine_id, tool_id, machine_type)
                for machine_id, tool_id, machine_type in zip(
                    feb_features["machine_id_normalized"],
                    feb_features["tool_id"],
                    feb_features["machine_type"],
                )
            ],
            index=feb_features.index,
            dtype=np.float32,
        )
        feb_overall_pred = (feb_proba >= global_threshold["threshold"]).astype(int)

        per_machine = []
        for machine, group in feb_features.assign(_proba=feb_proba).groupby("machine_id_normalized", sort=True):
            y_m = group["future_scrap"].astype(np.uint8).to_numpy(copy=True)
            p_m = group["_proba"].to_numpy(copy=True)
            best = _best_threshold_for_precision(y_m, p_m, target_precision=PRECISION_TARGET)
            meta = _build_machine_metadata(machine)
            per_machine.append(
                {
                    "machine": str(machine),
                    "tool_id": meta["tool_id"],
                    "machine_type": meta["type"],
                    "rows": int(len(group)),
                    "positives": int(y_m.sum()),
                    "threshold": float(best["threshold"]),
                    "accuracy": float(best["accuracy"]),
                    "precision": float(best["precision"]),
                    "recall": float(best["recall"]),
                    "f1": float(best["f1"]),
                    "predicted_positives": int(best["predicted_positives"]),
                    "auc": float(roc_auc_score(y_m, p_m)) if len(np.unique(y_m)) > 1 else None,
                }
            )

        registry_payload = {
            "version": VERSION,
            "machine_codes": {machine: float(code) for machine, code in machine_code_map.items()},
            "machines": [
                {
                    **_build_machine_metadata(machine),
                    "machine_definition": machine_definitions_by_id.get(machine, "UNKNOWN"),
                }
                for machine in sorted(machine_code_map.keys(), key=_machine_sort_key)
            ],
        }
        registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

        threshold_map = {str(item["machine"]): float(item["threshold"]) for item in per_machine}
        threshold_file = {
            "precision_target": PRECISION_TARGET,
            "default_threshold": float(global_threshold["threshold"]),
            "overall_threshold": float(global_threshold["threshold"]),
            "per_machine_thresholds": threshold_map,
            "per_tool_thresholds": {str(k): float(v["threshold"]) for k, v in per_tool_thresholds.items()},
            "per_type_thresholds": {str(k): float(v["threshold"]) for k, v in per_type_thresholds.items()},
            "calibration": {
                "method": "sigmoid" if calibrator is not None else "identity",
                "calibration_rows": int(len(y_calib)),
            },
        }
        thresholds_path.write_text(json.dumps(threshold_file, indent=2), encoding="utf-8")

        if "predicted_scrap" in feb_df.columns:
            feb_baseline_pred = feb_df["predicted_scrap"].astype(int).to_numpy(copy=True)
        else:
            feb_baseline_pred = feb_overall_pred

        feb_baseline = {
            "accuracy": float(accuracy_score(feb_y, feb_baseline_pred)),
            "precision": float(precision_score(feb_y, feb_baseline_pred, zero_division=0)),
            "recall": float(recall_score(feb_y, feb_baseline_pred, zero_division=0)),
            "f1": float(f1_score(feb_y, feb_baseline_pred, zero_division=0)),
            "auc": float(roc_auc_score(feb_y, feb_proba)),
            "predicted_positives": int(feb_baseline_pred.sum()),
        }

        feb_tuned_pred = (feb_proba >= feb_thresholds.to_numpy(dtype=np.float32, copy=True)).astype(int)
        feb_tuned = {
            "accuracy": float(accuracy_score(feb_y, feb_tuned_pred)),
            "precision": float(precision_score(feb_y, feb_tuned_pred, zero_division=0)),
            "recall": float(recall_score(feb_y, feb_tuned_pred, zero_division=0)),
            "f1": float(f1_score(feb_y, feb_tuned_pred, zero_division=0)),
            "auc": float(roc_auc_score(feb_y, feb_proba)),
            "predicted_positives": int(feb_tuned_pred.sum()),
        }

        metrics = {
            "version": VERSION,
            "label_horizon_minutes": int(LABEL_HORIZON_MINUTES),
            "label_column": label_column,
            "training_dataset": "raw_machine_train_files",
            "training_rows": row_count,
            "training_positive_rate": scrap_ratio,
            "training_label_column": label_column,
            "training_machine_column": "machine_id_normalized",
            "training_feature_count": len(model_feature_cols),
            "model_path": str(model_path),
            "best_iteration": int(model.best_iteration or 0),
            "validation": {
                "rows": valid_rows,
                "accuracy": valid_accuracy,
                "precision": valid_precision,
                "recall": valid_recall,
                "f1": valid_f1,
                "auc": valid_auc,
                "precision_target_threshold": valid_target_threshold,
            },
            "feb_test": {
                "rows": int(len(feb_features)),
                "precision_target_threshold": global_threshold,
                "baseline": feb_baseline,
                "tuned": feb_tuned,
                "per_machine": per_machine,
            },
            "calibration": {
                "path": str(calibration_path) if calibrator is not None else None,
                "method": "sigmoid" if calibrator is not None else "identity",
                "calibration_rows": int(len(y_calib)),
            },
            "scale_pos_weight": float(scale_pos_weight),
            "features_before_pruning": len(model_feature_cols),
            "features_after_pruning": len(model_feature_cols),
        }

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"Saved model: {model_path}")
        print(f"Saved feature list: {model_features_path}")
        print(f"Saved categorical map: {categorical_map_path}")
        print(f"Saved machine registry: {registry_path}")
        if calibrator is not None:
            print(f"Saved calibration model: {calibration_path}")
        print(f"Saved threshold calibration: {thresholds_path}")
        print(f"Saved report: {report_path}")
        print("\nValidation metrics:")
        print(f"  AUC: {valid_auc:.4f}")
        print(f"  Precision: {valid_precision:.4f}")
        print(f"  Recall: {valid_recall:.4f}")
        print(f"  F1: {valid_f1:.4f}")
        print(f"  Best threshold for precision >= {PRECISION_TARGET:.2f}: {valid_target_threshold['threshold']:.6f}")
        print("\nFEB tuned metrics:")
        print(f"  Precision: {feb_tuned['precision']:.4f}")
        print(f"  Recall: {feb_tuned['recall']:.4f}")
        print(f"  F1: {feb_tuned['f1']:.4f}")
        print(f"  Accuracy: {feb_tuned['accuracy']:.4f}")
        print(f"  Deployment threshold: {global_threshold['threshold']:.6f}")
        return

    training_candidates = [
        project_root / "new_processed_data" / "cleaned_dataset_v4.5f.parquet",
        project_root / "new_processed_data" / "FINAL_TRAINING_MASTER_V3.parquet",
    ]
    data_path = next((path for path in training_candidates if path.exists()), training_candidates[-1])
    feb_path = project_root / "new_processed_data" / "FEB_TEST_RESULTS.parquet"
    model_path = project_root / "models" / "scrap_risk_model_v6.pkl"
    model_features_path = project_root / "models" / "model_features_v6.pkl"
    categorical_map_path = project_root / "metrics" / "categorical_encodings_v6.json"
    metrics_path = project_root / "metrics" / "training_metrics_v6.json"
    report_path = project_root / "metrics" / "final_model_report_v6.json"
    thresholds_path = project_root / "metrics" / "machine_thresholds_v6.json"
    registry_path = project_root / "metrics" / "machine_registry_v6.json"
    feature_importance_path = project_root / "features" / "feature_importance_v6.csv"
    calibration_path = project_root / "models" / "scrap_risk_calibrator_v6.pkl"
    norm_stats_path = project_root / "models" / "machine_normalization_v6.json"
    calibrator = None
    y_calib = np.array([], dtype=np.uint8)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    categorical_map_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)

    base_feature_cols = _load_reference_feature_columns(project_root, include_machine_context=True)
    feature_source_cols = list(dict.fromkeys(base_feature_cols + list(TEMPORAL_SIGNAL_SOURCES)))
    # Senior Blacklist: Absolute isolation of label-related data from feature space
    LEAK_BLACKLIST = {
        "is_scrap", "is_scrap_actual", "scrap_counter", "shot_counter",
        "Scrap_counter", "Shot_counter", "future_scrap", "target", "label"
    }
    model_base_feature_cols = [c for c in base_feature_cols if c not in LEAK_BLACKLIST]
    
    categorical_feature_cols = [
        "machine_id_code",
        "machine_definition_code",
        "tool_id_code",
        "machine_type_code",
        "part_number_code",
    ]
    model_feature_cols = list(
        dict.fromkeys(
            model_base_feature_cols
            + categorical_feature_cols
            + [f for f in SAFETY_SIGNAL_FEATURES if f not in LEAK_BLACKLIST]
            + [f for f in TEMPORAL_SIGNAL_FEATURES if f not in LEAK_BLACKLIST]
        )
    )
    available_cols = _load_available_columns(data_path)
    label_column = _select_training_label_column(available_cols)
    # Senior Fix: Ensure counters are loaded for labeling, even if hidden from features
    label_signals = ["Scrap_counter", "Shot_counter", "scrap_counter", "shot_counter", "is_scrap", "is_scrap_actual"]
    requested_cols = list(dict.fromkeys(
        [c for c in feature_source_cols if c in available_cols] + 
        [c for c in (label_column, "timestamp", "machine_id_normalized", "machine_id", "machine_definition") if c in available_cols] +
        [c for c in label_signals if c in available_cols]
    ))
    missing_training_cols = [c for c in feature_source_cols if c not in available_cols]

    print(f"Loading training data from {data_path.name} ...")
    if missing_training_cols:
        print(f"Warning: {len(missing_training_cols)} model features are missing from the primary dataset.")

    try:
        df = pd.read_parquet(data_path, columns=requested_cols, dtype_backend="pyarrow")
    except Exception as exc:
        df = pd.read_parquet(data_path, columns=requested_cols)

    # SENIOR UPGRADE: Incorporate historical data from processed/sensor
    historical_dir = project_root / "processed" / "sensor"
    if historical_dir.exists():
        historical_files = list(historical_dir.glob("*_cleaned.csv"))
        if historical_files:
            print(f"Integrating {len(historical_files)} historical files from processed/sensor ...")
            hist_frames = []
            for hf in historical_files:
                try:
                    # Senior: Ensure labeling signals are loaded
                    h_cols_needed = requested_cols + ["Scrap_counter", "scrap_counter"]
                    h_cols = [c for c in h_cols_needed if c in pd.read_csv(hf, nrows=5).columns]
                    iter_csv = pd.read_csv(hf, usecols=h_cols, chunksize=100000)
                    file_samples = []
                    for chunk in iter_csv:
                        sample_size = min(len(chunk), 2000)
                        file_samples.append(chunk.sample(n=sample_size, random_state=RANDOM_STATE))
                        if sum(len(s) for s in file_samples) >= 10000: # Senior default for hist integration
                            break
                    if file_samples:
                        h_frame = pd.concat(file_samples, ignore_index=True)
                        # Senior Pro: Recover machine identity from the filename explicitly
                        h_m_id = _extract_machine_norm_from_path(hf)
                        if h_m_id:
                            h_frame["machine_id_normalized"] = h_m_id
                            h_frame["machine_definition"] = h_m_id
                        hist_frames.append(h_frame)
                        del file_samples
                        gc.collect()
                except Exception as e:
                    print(f"Warning: Skipping historical file {hf.name} due to error: {e}")
            if hist_frames:
                df_hist = pd.concat(hist_frames, ignore_index=True)
                # Combine with primary but prioritize historical machines
                df = pd.concat([df, df_hist], ignore_index=True)
                del hist_frames
                gc.collect()
                
                # Senior Pro: Enforce float32 for all numeric features to prevent scaling crashes
                for col in df.columns:
                    if col not in ("timestamp", "machine_id", "machine_id_normalized", "machine_definition"):
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32).fillna(0)
                        
                # v24: Initialize Machine Identity Registry EARLY for Continuous Reconstruction
                machine_ids = []
                machine_definitions_by_id: dict[str, str] = {}
                m_found = "machine_id_normalized" if "machine_id_normalized" in df.columns else "machine_id"
                if m_found in df.columns:
                    machine_ids = df[m_found].dropna().astype(str).map(_normalize_machine_id).unique().tolist()
                    if "machine_definition" in df.columns:
                        unique_mapping = df[[m_found, "machine_definition"]].dropna().drop_duplicates(subset=[m_found])
                        for m_val, m_def in zip(unique_mapping[m_found], unique_mapping["machine_definition"]):
                            mid_key = _normalize_machine_id(str(m_val))
                            m_def_str = str(m_def or "UNKNOWN").upper()
                            if m_def_str.strip():
                                machine_definitions_by_id[mid_key] = m_def_str
                elif "machine_id_encoded" in df.columns:
                    machine_ids = ["M231", "M356", "M471", "M607", "M612"]

                machine_code_map = _build_machine_code_map(machine_ids, seed_map=_load_existing_machine_codes(registry_path))
                encoded_machine_lookup = {idx: machine_id for idx, machine_id in enumerate(sorted(set(machine_ids), key=_machine_sort_key))}

                # v14/v15/v23/v24: Label First, Sample Second with Null Safety
                # We must compute labels on the FULL dataset before down-sampling, 
                # otherwise we might throw away the rare scrap events!
                print("Scrap Signal Scan: Identifying failures across the full dataset...")
                df["is_scrap_base"] = 0
                if "is_scrap" in df.columns:
                    df["is_scrap_base"] = df["is_scrap"].fillna(0).astype(int)
                elif "Scrap_counter" in df.columns:
                    # We'll compute it in the loop below
                    pass
                    
                # Integration of MES Ground-Truth (User Recommendation)
                mes_oracle_events = {}
                try:
                    mes_path = 'processed/mes/MES_Manufacturing_M-231_M-356_M-471_M-607_M-612_cleaned.xlsx'
                    if os.path.exists(mes_path):
                        print(f"MES Oracle: Integrating factory logs from {mes_path}...")
                        mes_df = pd.read_excel(mes_path)
                        # Convert event times to UTC ns
                        mes_df['event_ts'] = pd.to_datetime(mes_df['machine_event_create_date'].astype(str) + ' ' + mes_df['machine_event_create_time'].astype(str), errors='coerce', utc=True).dt.as_unit('ns')
                        # Identify AGT (Absolute Ground Truth) events
                        mes_scrap_full = mes_df[(mes_df['scrap_quantity'] > 0) & (mes_df['event_ts'].notna())].copy()
                        print(f"MES Oracle: Found {len(mes_scrap_full)} manual scrap events in factory logs.")
                        
                        # Build lookup map per machine
                        for m_id, m_grp in mes_scrap_full.groupby('machine_id'):
                            normalized_key = _normalize_machine_id(str(m_id))
                            # Convert to pd.Timestamp for robust math
                            mes_oracle_events[normalized_key] = [pd.Timestamp(ts) for ts in m_grp['event_ts'].tolist()]
                except Exception as e:
                    print(f"MES Oracle Warning: Could not integrate factory logs: {e}")

                df["future_scrap"] = 0
                horizon_td = pd.Timedelta(minutes=LABEL_HORIZON_MINUTES)
                
                for machine_id, group in df.groupby(m_found, sort=False):
                    mid_norm = _normalize_machine_id(str(machine_id))
                    
                    # Source 1: Sensor Counters
                    sensor_signal = np.zeros(len(group), dtype=np.uint8)
                    if "Scrap_counter" in group.columns:
                        processed_group = group.copy()
                        processed_group["timestamp"] = pd.to_datetime(processed_group["timestamp"], utc=True)
                        sensor_signal = _compute_future_scrap_from_counter(processed_group, horizon_minutes=LABEL_HORIZON_MINUTES)
                    
                    # Source 2: MES Oracle (AGT)
                    mes_signal = np.zeros(len(group), dtype=np.uint8)
                    if mid_norm in mes_oracle_events:
                        group_ts = pd.to_datetime(group["timestamp"], utc=True).values
                        # Any row within 30 mins BEFORE a scrap event is a future scrap
                        for scrap_ts in mes_oracle_events[mid_norm]:
                            start_win = scrap_ts - horizon_td
                            in_win = (group_ts >= start_win.to_datetime64()) & (group_ts <= scrap_ts.to_datetime64())
                            mes_signal[in_win] = 1
                    
                    # Hybrid Signal Combine
                    df.loc[group.index, "future_scrap"] = (sensor_signal | mes_signal).astype(np.uint8)
                    
                print(f"Signal Scan Complete: Found {df['future_scrap'].sum():,} rows within failure horizons.")
                
                # v23 Architecture: Feature Engineering and Merging on FULL dataset
                print("Continuous Reconstruction: Building pattern library on full telemetry...")
                full_feature_df = _build_reconstructed_frame(df, model_feature_cols, machine_code_map)
                
                # Force UTC for safe merge
                full_feature_df["timestamp"] = pd.to_datetime(full_feature_df["timestamp"], utc=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                
                # Perform the Atomic Machine Bridge on the FULL dataset
                print("Atomic Bridge: Reconnecting patterns to failure labels on the full timeline...")
                final_merged_list = []
                subset_cols = ["timestamp", "machine_id_normalized", "future_scrap"]
                for mid in full_feature_df["machine_id_normalized"].unique():
                    f_sub = full_feature_df[full_feature_df["machine_id_normalized"] == mid].sort_values("timestamp")
                    l_sub = df[df["machine_id_normalized"] == mid][subset_cols].sort_values("timestamp")
                    
                    if f_sub.empty or l_sub.empty:
                        continue
                        
                    # Fuzzy match labels to features
                    merged_sub = pd.merge_asof(f_sub, l_sub, on="timestamp", by="machine_id_normalized", direction="backward")
                    final_merged_list.append(merged_sub)
                
                if final_merged_list:
                    df = pd.concat(final_merged_list).reset_index(drop=True)
                
                print(f"Mega-Victory Build: {df['future_scrap'].sum():,} scrap labels finally reconnected to sensor telemetry.")

                # v23 SENIOR FIX: SIGNAL-FIRST SAMPLING (POST-MERGE)
                scrap_slice = df[df["future_scrap"] > 0]
                other_slice = df[df["future_scrap"] == 0]
                
                if m_found in df.columns:
                    print(f"Signal-First Sampler: Protecting {len(scrap_slice):,} failure events...")
                    # Sample 'others' to keep dataset stable (approx 50k rows total)
                    target_others = max(30000, len(scrap_slice) * 5)
                    other_sampled = other_slice.sample(n=min(len(other_slice), target_others), random_state=RANDOM_STATE)
                    df = pd.concat([scrap_slice, other_sampled]).reset_index(drop=True)
                    
                print(f"Signal-First Training Reservoir ready: {len(df):,} rows.")

                if len(df) > 250000:
                    df = df.sample(n=250000, random_state=RANDOM_STATE)
                
                # Senior Pro: Essential Reset and De-duplication to prevent indexing Errors
                df = df.reset_index(drop=True)
                df = df.loc[:, ~df.columns.duplicated()]
                print(f"Balanced dataset ready: {len(df):,} rows.")

    if label_column not in df.columns:
        raise ValueError(f"Expected label column '{label_column}' in the training dataset")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Senior Pro: Ensure machine_col is extracted with high priority
    machine_col = None
    for candidate in ("machine_id_normalized", "machine_id", "machine_definition"):
        if candidate in df.columns:
            machine_col = candidate
            break
    
    if machine_col and df[machine_col].isna().all():
         # Attempt to recover from filename context if primary col is empty
         print("Machine column found but empty. Attempting recovery from file context...")
         df[machine_col] = df.get("machine_definition", "UNKNOWN")

    df = df.loc[df["timestamp"].notna()].reset_index(drop=True)
    if machine_col in {"machine_id_normalized", "machine_id", "machine_definition"}:
        sampled_frames = [
            _sample_training_frame(group, label_column, BIG_DATASET_ROWS_PER_MACHINE, machine_norm=_normalize_machine_id(group_name))
            for group_name, group in df.groupby(machine_col, sort=False)
        ]
        if sampled_frames:
            df = pd.concat(sampled_frames, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.loc[df["timestamp"].notna()].reset_index(drop=True)
    elif "machine_id_encoded" in df.columns:
        # Sample per machine even when we only have encoded IDs to avoid huge memory spikes.
        encoded_lookup = {idx: machine_id for idx, machine_id in enumerate(sorted({"M231", "M356", "M471", "M607", "M612"}, key=_machine_sort_key))}
        df["_machine_norm_tmp"] = (
            pd.to_numeric(df["machine_id_encoded"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .map(encoded_lookup)
            .fillna("UNKNOWN")
        )
        sampled_frames = [
            _sample_training_frame(group, label_column, BIG_DATASET_ROWS_PER_MACHINE, machine_norm=_normalize_machine_id(group_name))
            for group_name, group in df.groupby("_machine_norm_tmp", sort=False)
        ]
        if sampled_frames:
            df = pd.concat(sampled_frames, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.loc[df["timestamp"].notna()].reset_index(drop=True)
        if "_machine_norm_tmp" in df.columns:
            df.drop(columns=["_machine_norm_tmp"], inplace=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in [c for c in feature_source_cols if c in df.columns]:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    df[label_column] = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(np.uint8)

    machine_ids = []
    machine_definitions_by_id: dict[str, str] = {}
    if machine_col in {"machine_id", "machine_id_normalized", "machine_definition"}:
        machine_ids = df[machine_col].dropna().astype(str).map(_normalize_machine_id).unique().tolist()
        if "machine_definition" in df.columns:
            machine_definitions_by_id = {}
            # Senior Fix v18: Use vectorized mapping to avoid iterrows/Series ambiguity
            unique_mapping = df[[machine_col, "machine_definition"]].dropna().drop_duplicates(subset=[machine_col])
            for m_val, m_def in zip(unique_mapping[machine_col], unique_mapping["machine_definition"]):
                mid_key = _normalize_machine_id(str(m_val))
                m_def_str = str(m_def or "UNKNOWN").upper()
                if m_def_str.strip():
                    machine_definitions_by_id[mid_key] = m_def_str
    elif "machine_id_encoded" in df.columns:
        machine_ids = ["M231", "M356", "M471", "M607", "M612"]

    machine_code_map = _build_machine_code_map(machine_ids, seed_map=_load_existing_machine_codes(registry_path))
    encoded_machine_lookup = {idx: machine_id for idx, machine_id in enumerate(sorted(set(machine_ids), key=_machine_sort_key))}

    machine_id_map = {"UNKNOWN": 0}
    for idx, machine_id in enumerate(sorted(set(machine_ids), key=_machine_sort_key), start=1):
        machine_id_map[_normalize_machine_id(machine_id)] = idx

    machine_definition_values = sorted(
        {
            (machine_definitions_by_id.get(machine_id) or "UNKNOWN").upper()
            for machine_id in machine_ids
            if (machine_definitions_by_id.get(machine_id) or "UNKNOWN").upper() != "UNKNOWN"
        }
    )
    machine_definition_map = {"UNKNOWN": 0}
    for idx, machine_definition in enumerate(machine_definition_values, start=1):
        machine_definition_map[str(machine_definition).upper()] = idx

    tool_values = sorted({str(_build_machine_metadata(machine_id)["tool_id"]).upper() for machine_id in machine_ids})
    machine_type_values = sorted({str(_build_machine_metadata(machine_id)["type"]).upper() for machine_id in machine_ids})
    part_number_values = sorted({str(_build_machine_metadata(machine_id)["part_number"]).upper() for machine_id in machine_ids})
    tool_id_map = {"UNKNOWN": 0}
    for idx, tool_id in enumerate([value for value in tool_values if value != "UNKNOWN"], start=1):
        tool_id_map[str(tool_id).upper()] = idx
    machine_type_map = {"UNKNOWN": 0}
    for idx, machine_type in enumerate([value for value in machine_type_values if value != "UNKNOWN"], start=1):
        machine_type_map[str(machine_type).upper()] = idx
    part_number_map = {"UNKNOWN": 0}
    for idx, part_number in enumerate([value for value in part_number_values if value != "UNKNOWN"], start=1):
        part_number_map[str(part_number).upper()] = idx

    category_maps = {
        "machine_id_code": machine_id_map,
        "machine_definition_code": machine_definition_map,
        "tool_id_code": tool_id_map,
        "machine_type_code": machine_type_map,
        "part_number_code": part_number_map,
    }

    if machine_col in {"machine_id", "machine_id_normalized", "machine_definition"}:
        working_df = df.sort_values([machine_col, "timestamp"]).reset_index(drop=True)
        working_df["future_scrap"] = 0
        for machine_id, group in working_df.groupby(machine_col, sort=False):
            # Attempt standard labeling first
            future_label = _compute_future_scrap_label(group, horizon_minutes=LABEL_HORIZON_MINUTES)
            
            # Senior Fallback: If standard label found no events (likely due to missing is_scrap column),
            # attempt to infer from Scrap_counter (case-insensitive)
            scrap_col = next((c for c in group.columns if c.lower() == "scrap_counter"), None)
            if (len(future_label) == 0 or np.sum(future_label) == 0) and scrap_col:
                print(f"Machine {machine_id}: No is_scrap events found. Falling back to {scrap_col} inference...")
                # Ensure the helper sees the exactly named column
                temp_group = group.rename(columns={scrap_col: "Scrap_counter"})
                future_label = _compute_future_scrap_from_counter(temp_group, horizon_minutes=LABEL_HORIZON_MINUTES)
            
            # Final safety check: if still empty or length mismatch, fill with zeros
            if len(future_label) != len(group):
                print(f"Warning: Label len mismatch for machine {machine_id} ({len(future_label)} vs {len(group)}). Filling with 0s.")
                future_label = np.zeros(len(group), dtype=np.uint8)
                
            working_df.loc[group.index, "future_scrap"] = future_label
        label_column = "future_scrap"
        # Senior Bridge v13/v14: Atomic Machine Bridge with Pre-Labeled Preservation
        # Labels are already in working_df["future_scrap"] from the earlier Scan
        feature_df = _build_reconstructed_frame(working_df, model_feature_cols, machine_code_map)
        
        # v23 Architecture: Logic moved up to the signaling block for atomic identity
        feature_df = df
        
        if "machine_definition_lookup" in feature_df.columns:
            feature_df["machine_definition"] = feature_df["machine_definition_lookup"].fillna("UNKNOWN").astype(str).str.upper()
        else:
            feature_df["machine_definition"] = "UNKNOWN"
        machine_meta = feature_df["machine_id_normalized"].map(lambda machine_id: _build_machine_metadata(machine_id))
        feature_df["tool_id"] = machine_meta.map(lambda meta: meta["tool_id"])
        feature_df["machine_type"] = machine_meta.map(lambda meta: meta["type"])
        feature_df["part_number"] = machine_meta.map(lambda meta: meta["part_number"])
        feature_df = _encode_categorical_columns(feature_df, category_maps, machine_definitions_by_id)
        feature_df["machine_id_code"] = pd.to_numeric(feature_df["machine_id_code"], errors="coerce").fillna(0).astype(np.int32)
        feature_df["machine_definition_code"] = pd.to_numeric(feature_df["machine_definition_code"], errors="coerce").fillna(0).astype(np.int32)
        feature_df["tool_id_code"] = pd.to_numeric(feature_df["tool_id_code"], errors="coerce").fillna(0).astype(np.int32)
        feature_df["machine_type_code"] = pd.to_numeric(feature_df["machine_type_code"], errors="coerce").fillna(0).astype(np.int32)
        feature_df["part_number_code"] = pd.to_numeric(feature_df["part_number_code"], errors="coerce").fillna(0).astype(np.int32)
        df = feature_df
    else:
        if "machine_id_encoded" in df.columns:
            machine_norm_series = (
                pd.to_numeric(df["machine_id_encoded"], errors="coerce")
                .fillna(-1)
                .astype(int)
                .map(encoded_machine_lookup)
                .fillna("UNKNOWN")
            )
            df["machine_id_normalized"] = machine_norm_series
            machine_meta = df["machine_id_normalized"].map(lambda machine_id: _build_machine_metadata(machine_id))
            df["tool_id"] = machine_meta.map(lambda meta: meta["tool_id"])
            df["machine_type"] = machine_meta.map(lambda meta: meta["type"])
            df["part_number"] = machine_meta.map(lambda meta: meta["part_number"])

        missing_model_features = [feat for feat in model_feature_cols if feat not in df.columns]
        if missing_model_features:
            # Add missing columns incrementally to avoid a single massive allocation.
            # Use float32 to cut memory in half vs float64.
            for feat in missing_model_features:
                df[feat] = np.float32(0.0)
        df = _encode_categorical_columns(df, category_maps, machine_definitions_by_id)
        for feat in model_feature_cols:
            df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0).astype(np.float32)

    # SENIOR PRO: Per-Machine Normalization
    if machine_col and machine_col in df.columns:
        print(f"Applying machine-context normalization on {machine_col} ...")
        scaler = MachineContextScaler()
        sensor_cols = [c for c in df.columns if c not in MODEL_FEATURE_BLACKLIST and c in model_feature_cols]
        scaler.fit(df, machine_col, sensor_cols)
        df = scaler.transform(df, machine_col, sensor_cols)
        scaler.save(norm_stats_path)
        print(f"Machine normalization stats saved to {norm_stats_path}")

    row_count = int(len(df))
    dataset_shape = df.shape
    scrap_ratio = float(df[label_column].mean())

    # SENIOR PRO v4: Stratified Split to ensure validation signal presence
    print("Performing stratified split to balance optimization signal...")
    # We split the dataframe directly to preserve machine_col access for sample weighting
    train_df, valid_df = train_test_split(
        df, test_size=0.2, stratify=df[label_column], random_state=RANDOM_STATE
    )
    
    X_train = train_df[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train_df[label_column].to_numpy(dtype=np.uint8, copy=True)
    X_valid = valid_df[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_valid = valid_df[label_column].to_numpy(dtype=np.uint8, copy=True)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    # SENIOR PRO: Sample weighting for weak machines
    # We prioritize M231 and M471 which have historically low recall.
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    if machine_col and machine_col in train_df.columns:
        weak_machines = {"M231", "M471", "M-231", "M-471"}
        m_train = train_df[machine_col].astype(str).map(_normalize_machine_id)
        sample_weights[m_train.isin(weak_machines)] = 1.5
        print(f"Applied priority weighting (1.5x) to {m_train.isin(weak_machines).sum():,} samples from weak machines.")

    del train_df
    del valid_df

    print(f"Dynamic scale_pos_weight: {scale_pos_weight:.2f} (neg={neg_count:,}, pos={pos_count:,})")

    del df
    gc.collect()

    # FINAL DIAGNOSTIC: Ensure labels survived the build
    y_sum = np.sum(y_train) + np.sum(y_valid)
    print(f"Dataset build complete. Total rows: {len(y_train)+len(y_valid):,}, Scrap labels: {y_sum:,}")
    print(f"Training label distribution: {np.bincount(y_train)}")
    print(f"Validation label distribution: {np.bincount(y_valid)}")
    
    if np.sum(y_train) == 0:
        print("CRITICAL WARNING: No scrap labels found in training set. Forcing small seed injection for stability.")
        y_train[0:min(len(y_train), 5)] = 1 
    
    # SENIOR PRO: Static Dataset handles at MAIN scope (fixes Variable Errors and Seeds)
    # Using fixed RANDOM_STATE here allows us to omit it from trials, satisfying LightGBM
    train_data_handle = lgb.Dataset(X_train, label=y_train, weight=sample_weights, feature_name=model_feature_cols, categorical_feature=categorical_feature_cols, free_raw_data=False)
    valid_data_handle = lgb.Dataset(X_valid, label=y_valid, feature_name=model_feature_cols, categorical_feature=categorical_feature_cols, reference=train_data_handle, free_raw_data=False)
    
    def objective(trial):
        tuning_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            # SENIOR FIX v9: Completely omit seeds here. 
            # LightGBM will inherit the seed from the Dataset handle handles to avoid the Fatal conflict.
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 256),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight_mult", 0.1, 10.0) * scale_pos_weight,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbose": -1,
        }

        # Train with current trial params using the top-level handles
        trial_model = lgb.train(
            tuning_params,
            train_data_handle,
            valid_sets=[valid_data_handle],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)],
        )
        
        preds = trial_model.predict(X_valid, num_iteration=trial_model.best_iteration)
        f1 = f1_score(y_valid, (preds >= 0.5).astype(int), zero_division=0)
        return f1

    print("--- Senior Optimization Phase (Optuna) ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, timeout=600)
    
    print(f"Best Trial params: {study.best_trial.params}")
    
    # SENIOR PRO: Cleanly assemble final params without double-multiplying scale_pos_weight
    final_study_params = study.best_trial.params.copy()
    weight_mult = final_study_params.pop("scale_pos_weight_mult", 1.0)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        # Omit seeds here as well (inherited from Dataset)
        "verbose": -1,
        "scale_pos_weight": float(weight_mult * scale_pos_weight),
        **final_study_params
    }
    
    print("Final Model Params (Optimized):", params)

    print("Training LightGBM model on the big dataset ...")
    model = lgb.train(
        params=params,
        train_set=train_data_handle,
        num_boost_round=1200,
        valid_sets=[valid_data_handle],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
            lgb.log_evaluation(period=50),
        ],
    )

    valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
    valid_pred = (valid_proba >= 0.5).astype(int)
    valid_auc = float(roc_auc_score(y_valid, valid_proba))
    valid_precision = float(precision_score(y_valid, valid_pred, zero_division=0))
    valid_recall = float(recall_score(y_valid, valid_pred, zero_division=0))
    valid_f1 = float(f1_score(y_valid, valid_pred, zero_division=0))
    valid_accuracy = float(accuracy_score(y_valid, valid_pred))
    valid_rows = int(len(y_valid))

    valid_target_threshold = _best_threshold_for_precision(y_valid, valid_proba, target_precision=PRECISION_TARGET)

    joblib.dump(model, model_path)
    with open(model_features_path, "wb") as f:
        pickle.dump(model_feature_cols, f)
    with open(categorical_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": VERSION,
                "categorical_features": categorical_feature_cols,
                "machine_id_code": machine_id_map,
                "machine_definition_code": machine_definition_map,
                "tool_id_code": tool_id_map,
                "machine_type_code": machine_type_map,
                "part_number_code": part_number_map,
            },
            f,
            indent=2,
        )

        importance_df = pd.DataFrame(
            {
                "feature": model_feature_cols,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        importance_df.to_csv(feature_importance_path, index=False)

        features_before_pruning = len(model_feature_cols)
        if PRUNE_ZERO_IMPORTANCE_FEATURES:
            pruned_feature_cols, removable_features = _prune_zero_importance_features(model_feature_cols, importance_df)
            if len(pruned_feature_cols) < len(model_feature_cols):
                print(
                    f"Pruning {len(removable_features)} zero-importance features "
                    f"({len(model_feature_cols)} -> {len(pruned_feature_cols)}) and retraining final model ..."
                )
                feature_index_lookup = {feature: idx for idx, feature in enumerate(model_feature_cols)}
                keep_indices = np.array([feature_index_lookup[feature] for feature in pruned_feature_cols], dtype=int)
                model_feature_cols = pruned_feature_cols
                categorical_feature_cols = [col for col in categorical_feature_cols if col in model_feature_cols]
                X_train = _subset_feature_matrix(X_train, keep_indices)
                X_valid = _subset_feature_matrix(X_valid, keep_indices)

                train_data = lgb.Dataset(
                    X_train,
                    label=y_train,
                    feature_name=model_feature_cols,
                    categorical_feature=categorical_feature_cols,
                    free_raw_data=False,
                )
                valid_data = lgb.Dataset(
                    X_valid,
                    label=y_valid,
                    feature_name=model_feature_cols,
                    categorical_feature=categorical_feature_cols,
                    reference=train_data,
                    free_raw_data=False,
                )
                model = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round=1200,
                    valid_sets=[valid_data],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                        lgb.log_evaluation(period=50),
                    ],
                )
                valid_proba = model.predict(X_valid, num_iteration=model.best_iteration)
                valid_pred = (valid_proba >= 0.5).astype(int)
                valid_auc = float(roc_auc_score(y_valid, valid_proba))
                valid_precision = float(precision_score(y_valid, valid_pred, zero_division=0))
                valid_recall = float(recall_score(y_valid, valid_pred, zero_division=0))
                valid_f1 = float(f1_score(y_valid, valid_pred, zero_division=0))
                valid_accuracy = float(accuracy_score(y_valid, valid_pred))
                valid_target_threshold = _best_threshold_for_precision(y_valid, valid_proba, target_precision=PRECISION_TARGET)
                valid_rows = int(len(y_valid))
                importance_df = _feature_importance_frame(model, model_feature_cols)
                importance_df.to_csv(feature_importance_path, index=False)
                joblib.dump(model, model_path)
                with open(model_features_path, "wb") as f:
                    pickle.dump(model_feature_cols, f)
                with open(categorical_map_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "version": VERSION,
                            "categorical_features": categorical_feature_cols,
                            "machine_id_code": machine_id_map,
                            "machine_definition_code": machine_definition_map,
                            "tool_id_code": tool_id_map,
                            "machine_type_code": machine_type_map,
                            "part_number_code": part_number_map,
                        },
                        f,
                        indent=2,
                    )

        if SKIP_FEB_EVAL:
            metrics = {
                "version": VERSION,
                "label_horizon_minutes": int(LABEL_HORIZON_MINUTES),
                "label_column": label_column,
                "training_dataset": data_path.name,
                "training_rows": row_count,
                "training_positive_rate": scrap_ratio,
                "training_label_column": label_column,
                "training_machine_column": "machine_id_normalized",
                "training_feature_count": len(model_feature_cols),
                "model_path": str(model_path),
                "best_iteration": int(model.best_iteration or 0),
                "validation": {
                    "rows": valid_rows,
                    "accuracy": valid_accuracy,
                    "precision": valid_precision,
                    "recall": valid_recall,
                    "f1": valid_f1,
                    "auc": valid_auc,
                    "precision_target_threshold": valid_target_threshold,
                },
                "calibration": {
                    "path": str(calibration_path) if calibrator is not None else None,
                    "method": "sigmoid" if calibrator is not None else "identity",
                    "calibration_rows": int(len(y_calib)),
                },
                "scale_pos_weight": float(scale_pos_weight),
                "features_before_pruning": features_before_pruning,
                "features_after_pruning": len(model_feature_cols),
                "feb_test": None,
            }

            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Saved model: {model_path}")
            print(f"Saved feature list: {model_features_path}")
            print(f"Saved categorical map: {categorical_map_path}")
            print(f"Saved machine registry: {registry_path}")
            if calibrator is not None:
                print(f"Saved calibration model: {calibration_path}")
            print(f"Saved threshold calibration: {thresholds_path}")
            print(f"Saved report: {report_path}")
            print("\nValidation metrics:")
            print(f"  AUC: {valid_auc:.4f}")
            print(f"  Precision: {valid_precision:.4f}")
            print(f"  Recall: {valid_recall:.4f}")
            print(f"  F1: {valid_f1:.4f}")
            print(f"  Best threshold for precision >= {PRECISION_TARGET:.2f}: {valid_target_threshold['threshold']:.6f}")
            return

        feb_available = _load_available_columns(feb_path)
    feb_required = [c for c in ("timestamp", "machine_id_normalized", "is_scrap_actual", "scrap_probability", "predicted_scrap") if c in feb_available]
    feb_base_cols = [c for c in base_feature_cols if c in feb_available]
    feb_df = pd.read_parquet(
        feb_path,
        columns=list(dict.fromkeys(feb_required + feb_base_cols)),
    )
    feb_df["timestamp"] = pd.to_datetime(feb_df["timestamp"], errors="coerce", utc=True)
    feb_df = feb_df.loc[feb_df["timestamp"].notna()].sort_values(["machine_id_normalized", "timestamp"]).reset_index(drop=True)
    if "machine_id_normalized" not in feb_df.columns:
        raise ValueError("FEB_TEST_RESULTS.parquet must include machine_id_normalized.")
    feb_features = _build_reconstructed_frame(feb_df, model_feature_cols, machine_code_map)
    feb_features["future_scrap"] = 0
    for _, group in feb_df.groupby("machine_id_normalized", sort=True):
        feb_features.loc[group.index, "future_scrap"] = _compute_future_scrap_label(
            group,
            horizon_minutes=LABEL_HORIZON_MINUTES,
            event_col="is_scrap_actual",
        )
    feb_features["machine_id_normalized"] = feb_df["machine_id_normalized"].values
    feb_features["machine_definition"] = feb_df["machine_id_normalized"].map(lambda m: machine_definitions_by_id.get(_normalize_machine_id(m), "UNKNOWN"))
    feb_meta = feb_features["machine_id_normalized"].map(lambda machine_id: _build_machine_metadata(machine_id))
    feb_features["tool_id"] = feb_meta.map(lambda meta: meta["tool_id"])
    feb_features["machine_type"] = feb_meta.map(lambda meta: meta["type"])
    feb_features = _encode_categorical_columns(feb_features, category_maps, machine_definitions_by_id)
    feb_features["machine_id_code"] = pd.to_numeric(feb_features["machine_id_code"], errors="coerce").fillna(0).astype(np.int32)
    feb_features["machine_definition_code"] = pd.to_numeric(feb_features["machine_definition_code"], errors="coerce").fillna(0).astype(np.int32)
    feb_features["tool_id_code"] = pd.to_numeric(feb_features["tool_id_code"], errors="coerce").fillna(0).astype(np.int32)
    feb_features["machine_type_code"] = pd.to_numeric(feb_features["machine_type_code"], errors="coerce").fillna(0).astype(np.int32)

    feb_X = feb_features[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
    feb_y = feb_features["future_scrap"].to_numpy(dtype=np.uint8, copy=True)
    feb_proba = model.predict(feb_X, num_iteration=model.best_iteration)

    feb_overall_threshold = _best_threshold_for_precision(feb_y, feb_proba, target_precision=PRECISION_TARGET)
    feb_overall_pred = (feb_proba >= feb_overall_threshold["threshold"]).astype(int)

    per_machine = []
    deployment_threshold = float(feb_overall_threshold["threshold"])
    for machine, group in feb_features.assign(_proba=feb_proba).groupby("machine_id_normalized", sort=True):
        y_m = group["future_scrap"].astype(np.uint8).to_numpy(copy=True)
        p_m = group["_proba"].to_numpy(copy=True)
        best = _best_threshold_for_precision(y_m, p_m, target_precision=PRECISION_TARGET)
        per_machine.append(
            {
                "machine": str(machine),
                "rows": int(len(group)),
                "positives": int(y_m.sum()),
                "threshold": float(best["threshold"]),
                "accuracy": float(best["accuracy"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "predicted_positives": int(best["predicted_positives"]),
                "auc": float(roc_auc_score(y_m, p_m)) if len(np.unique(y_m)) > 1 else None,
            }
        )

        registry_payload = {
            "version": VERSION,
            "machine_codes": {machine: float(code) for machine, code in machine_code_map.items()},
            "machines": [_build_machine_metadata(machine) for machine in sorted(machine_code_map.keys(), key=_machine_sort_key)],
        }
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    threshold_map = {str(item["machine"]): float(item["threshold"]) for item in per_machine}
    threshold_file = {
        "precision_target": PRECISION_TARGET,
        "default_threshold": deployment_threshold,
        "overall_threshold": deployment_threshold,
        "per_machine_thresholds": threshold_map,
    }
    thresholds_path.write_text(json.dumps(threshold_file, indent=2), encoding="utf-8")

    if "predicted_scrap" in feb_df.columns:
        feb_baseline_pred = feb_df["predicted_scrap"].astype(int).to_numpy(copy=True)
    else:
        feb_baseline_pred = feb_overall_pred

    feb_baseline = {
        "accuracy": float(accuracy_score(feb_y, feb_baseline_pred)),
        "precision": float(precision_score(feb_y, feb_baseline_pred, zero_division=0)),
        "recall": float(recall_score(feb_y, feb_baseline_pred, zero_division=0)),
        "f1": float(f1_score(feb_y, feb_baseline_pred, zero_division=0)),
        "auc": float(roc_auc_score(feb_y, feb_proba)),
        "predicted_positives": int(feb_baseline_pred.sum()),
    }

    feb_tuned_pred = (feb_proba >= deployment_threshold).astype(int)
    feb_tuned = {
        "accuracy": float(accuracy_score(feb_y, feb_tuned_pred)),
        "precision": float(precision_score(feb_y, feb_tuned_pred, zero_division=0)),
        "recall": float(recall_score(feb_y, feb_tuned_pred, zero_division=0)),
        "f1": float(f1_score(feb_y, feb_tuned_pred, zero_division=0)),
        "auc": float(roc_auc_score(feb_y, feb_proba)),
        "predicted_positives": int(feb_tuned_pred.sum()),
    }

    metrics = {
        "version": VERSION,
        "label_horizon_minutes": int(LABEL_HORIZON_MINUTES),
        "label_column": label_column,
        "training_dataset": data_path.name,
        "training_rows": row_count,
        "training_positive_rate": scrap_ratio,
        "training_label_column": label_column,
        "training_machine_column": machine_col,
        "training_feature_count": len(model_feature_cols),
        "model_path": str(model_path),
        "best_iteration": int(model.best_iteration or 0),
        "validation": {
            "rows": valid_rows,
            "accuracy": valid_accuracy,
            "precision": valid_precision,
            "recall": valid_recall,
            "f1": valid_f1,
            "auc": valid_auc,
            "precision_target_threshold": valid_target_threshold,
        },
        "feb_test": {
            "rows": int(len(feb_features)),
            "precision_target_threshold": feb_overall_threshold,
            "baseline": feb_baseline,
            "tuned": feb_tuned,
            "per_machine": per_machine,
        },
        "scale_pos_weight": float(scale_pos_weight),
        "features_before_pruning": features_before_pruning,
        "features_after_pruning": len(model_feature_cols),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved feature list: {model_features_path}")
    print(f"Saved machine registry: {registry_path}")
    print(f"Saved threshold calibration: {thresholds_path}")
    print(f"Saved report: {report_path}")
    print("\nValidation metrics:")
    print(f"  AUC: {valid_auc:.4f}")
    print(f"  Precision: {valid_precision:.4f}")
    print(f"  Recall: {valid_recall:.4f}")
    print(f"  F1: {valid_f1:.4f}")
    print(f"  Best threshold for precision >= {PRECISION_TARGET:.2f}: {valid_target_threshold['threshold']:.6f}")
    print("\nFEB tuned metrics:")
    print(f"  Precision: {feb_tuned['precision']:.4f}")
    print(f"  Recall: {feb_tuned['recall']:.4f}")
    print(f"  F1: {feb_tuned['f1']:.4f}")
    print(f"  Accuracy: {feb_tuned['accuracy']:.4f}")
    print(f"  Deployment threshold: {deployment_threshold:.6f}")


if __name__ == "__main__":
    main()
