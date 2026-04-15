from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

class MachineContextScaler:
    """
    Perform Z-score normalization for sensor data on a per-machine basis.
    This ensures that absolute baseline differences between different injection 
    molding machines are neutralized before being fed to the ML model.
    """
    def __init__(self, stats_path: Path | str | None = None):
        self.stats: dict[str, dict[str, dict[str, float]]] = {}
        if stats_path:
            self.load(stats_path)

    def fit(self, df: pd.DataFrame, machine_col: str, sensor_cols: list[str]):
        """
        Compute mean and std for each sensor per machine.
        """
        self.stats = {}
        for machine_id, group in df.groupby(machine_col):
            machine_stats = {}
            for sensor in sensor_cols:
                if sensor in group.columns:
                    vals = pd.to_numeric(group[sensor], errors='coerce').dropna()
                    if not vals.empty:
                        machine_stats[sensor] = {
                            "mean": float(vals.mean()),
                            "std": float(vals.std()) if vals.std() > 0 else 1.0
                        }
            self.stats[str(machine_id)] = machine_stats

    def transform(self, df: pd.DataFrame, machine_col: str, sensor_cols: list[str]) -> pd.DataFrame:
        """
        Apply (Value - Mean) / Std for each row based on its machine ID.
        """
        result = df.copy()
        for machine_id, group in df.groupby(machine_col):
            if str(machine_id) not in self.stats:
                continue
            
            m_stats = self.stats[str(machine_id)]
            for sensor in sensor_cols:
                if sensor in m_stats and sensor in result.columns:
                    mean = m_stats[sensor]["mean"]
                    std = m_stats[sensor]["std"]
                    result.loc[group.index, sensor] = (pd.to_numeric(result.loc[group.index, sensor], errors='coerce') - mean) / std
        return result

    def save(self, path: Path | str):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def load(self, path: Path | str):
        if Path(path).exists():
            with open(path, "r") as f:
                self.stats = json.load(f)
