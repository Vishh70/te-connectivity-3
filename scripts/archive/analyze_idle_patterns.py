from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENSORS = [
    "Cycle_time",
    "Injection_pressure",
    "Switch_pressure",
    "Peak_pressure_position",
]
PLOT_SENSORS = ["Cycle_time", "Injection_pressure", "Switch_pressure"]
ROLLING_WINDOW = 20
MIN_SEGMENT_LEN = 20
STD_EPS = 1e-9
RANGE_EPS = 1e-9
LOW_CYCLE = 0.5
LOW_PRESSURE = 5.0


@dataclass
class Segment:
    machine_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    window_length: int
    idle_type: str
    cycle_mean: float
    cycle_std: float
    cycle_min: float
    cycle_max: float
    cycle_nunique: int
    cycle_rate_of_change: float
    injection_pressure_mean: float
    injection_pressure_std: float
    injection_pressure_min: float
    injection_pressure_max: float
    injection_pressure_nunique: int
    injection_pressure_rate_of_change: float
    switch_pressure_mean: float
    switch_pressure_std: float
    switch_pressure_min: float
    switch_pressure_max: float
    switch_pressure_nunique: int
    switch_pressure_rate_of_change: float
    peak_pressure_position_mean: float
    peak_pressure_position_std: float
    peak_pressure_position_min: float
    peak_pressure_position_max: float
    peak_pressure_position_nunique: int
    peak_pressure_position_rate_of_change: float


def load_machine_wide(machine_file: Path) -> pd.DataFrame:
    chunks = []
    usecols = ["timestamp", "variable_name", "value"]
    for chunk in pd.read_csv(machine_file, usecols=usecols, chunksize=500_000):
        sub = chunk[chunk["variable_name"].isin(SENSORS)].copy()
        if sub.empty:
            continue
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["timestamp", "value"])
        chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=SENSORS)

    long_df = pd.concat(chunks, ignore_index=True)
    wide = (
        long_df.pivot_table(index="timestamp", columns="variable_name", values="value", aggfunc="mean")
        .sort_index()
        .dropna(how="all")
    )
    for sensor in SENSORS:
        if sensor not in wide.columns:
            wide[sensor] = np.nan
    wide = wide[SENSORS].dropna()
    return wide


def build_masks(wide: pd.DataFrame) -> Dict[str, pd.Series]:
    rolling_nunique = {s: wide[s].rolling(ROLLING_WINDOW).nunique() for s in SENSORS}
    rolling_std = {s: wide[s].rolling(ROLLING_WINDOW).std().fillna(0.0) for s in SENSORS}
    rolling_range = {
        s: wide[s].rolling(ROLLING_WINDOW).max() - wide[s].rolling(ROLLING_WINDOW).min()
        for s in SENSORS
    }
    rolling_mean = {s: wide[s].rolling(ROLLING_WINDOW).mean() for s in SENSORS}

    frozen = pd.Series(True, index=wide.index)
    constant = pd.Series(True, index=wide.index)
    for s in SENSORS:
        frozen = frozen & (rolling_nunique[s] == 1)
        constant = constant & (
            (rolling_std[s] <= STD_EPS) | (rolling_range[s].fillna(0.0) <= RANGE_EPS)
        )

    low_activity = (
        (rolling_mean["Cycle_time"] < LOW_CYCLE)
        & (rolling_mean["Injection_pressure"] < LOW_PRESSURE)
        & (rolling_mean["Switch_pressure"] < LOW_PRESSURE)
    )

    active = (
        (rolling_mean["Cycle_time"] > 1.0)
        & (rolling_mean["Injection_pressure"] > 50.0)
        & (rolling_mean["Switch_pressure"] > 50.0)
        & (~frozen.fillna(False))
    )

    idle_any = frozen.fillna(False) | constant.fillna(False) | low_activity.fillna(False)
    return {
        "frozen": frozen.fillna(False),
        "constant": constant.fillna(False),
        "low_activity": low_activity.fillna(False),
        "idle_any": idle_any,
        "active": active.fillna(False),
    }


def extract_segments(mask: pd.Series, min_len: int = MIN_SEGMENT_LEN) -> List[tuple]:
    if mask.empty:
        return []
    groups = mask.ne(mask.shift()).cumsum()
    out = []
    for _, idx in mask.groupby(groups).groups.items():
        idx = list(idx)
        if not mask.loc[idx[0]]:
            continue
        if len(idx) < min_len:
            continue
        out.append((idx[0], idx[-1], len(idx)))
    return out


def summarize_segment(machine_id: str, seg_df: pd.DataFrame, idle_type: str) -> Segment:
    stats: Dict[str, float] = {}
    for sensor in SENSORS:
        s = seg_df[sensor]
        roc = 0.0
        if len(s) > 1 and pd.notna(s.iloc[0]) and s.iloc[0] != 0:
            roc = float((s.iloc[-1] - s.iloc[0]) / max(abs(s.iloc[0]), 1e-9))
        stats[f"{sensor}_mean"] = float(s.mean())
        stats[f"{sensor}_std"] = float(s.std(ddof=0))
        stats[f"{sensor}_min"] = float(s.min())
        stats[f"{sensor}_max"] = float(s.max())
        stats[f"{sensor}_nunique"] = int(s.nunique())
        stats[f"{sensor}_rate_of_change"] = float(roc)

    return Segment(
        machine_id=machine_id,
        start_time=seg_df.index.min(),
        end_time=seg_df.index.max(),
        window_length=len(seg_df),
        idle_type=idle_type,
        cycle_mean=stats["Cycle_time_mean"],
        cycle_std=stats["Cycle_time_std"],
        cycle_min=stats["Cycle_time_min"],
        cycle_max=stats["Cycle_time_max"],
        cycle_nunique=stats["Cycle_time_nunique"],
        cycle_rate_of_change=stats["Cycle_time_rate_of_change"],
        injection_pressure_mean=stats["Injection_pressure_mean"],
        injection_pressure_std=stats["Injection_pressure_std"],
        injection_pressure_min=stats["Injection_pressure_min"],
        injection_pressure_max=stats["Injection_pressure_max"],
        injection_pressure_nunique=stats["Injection_pressure_nunique"],
        injection_pressure_rate_of_change=stats["Injection_pressure_rate_of_change"],
        switch_pressure_mean=stats["Switch_pressure_mean"],
        switch_pressure_std=stats["Switch_pressure_std"],
        switch_pressure_min=stats["Switch_pressure_min"],
        switch_pressure_max=stats["Switch_pressure_max"],
        switch_pressure_nunique=stats["Switch_pressure_nunique"],
        switch_pressure_rate_of_change=stats["Switch_pressure_rate_of_change"],
        peak_pressure_position_mean=stats["Peak_pressure_position_mean"],
        peak_pressure_position_std=stats["Peak_pressure_position_std"],
        peak_pressure_position_min=stats["Peak_pressure_position_min"],
        peak_pressure_position_max=stats["Peak_pressure_position_max"],
        peak_pressure_position_nunique=stats["Peak_pressure_position_nunique"],
        peak_pressure_position_rate_of_change=stats["Peak_pressure_position_rate_of_change"],
    )


def choose_idle_type(
    seg_idx: pd.Index, frozen_mask: pd.Series, constant_mask: pd.Series, low_mask: pd.Series
) -> str:
    frozen_share = frozen_mask.loc[seg_idx].mean()
    low_share = low_mask.loc[seg_idx].mean()
    const_share = constant_mask.loc[seg_idx].mean()
    if frozen_share >= max(low_share, const_share):
        return "frozen"
    if low_share >= max(frozen_share, const_share):
        return "low_activity"
    return "constant_low_variance"


def plot_segment(seg_df: pd.DataFrame, machine_id: str, out_file: Path):
    plt.figure(figsize=(12, 4))
    x = np.arange(len(seg_df))
    for sensor in PLOT_SENSORS:
        plt.plot(x, seg_df[sensor].values, label=sensor)
    plt.title(f"{machine_id} idle candidate: {seg_df.index.min()} to {seg_df.index.max()}")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def hydra_reference_info(hydra_file: Path) -> pd.DataFrame:
    if not hydra_file.exists():
        return pd.DataFrame()
    df = pd.read_excel(hydra_file, sheet_name="Data")
    out = pd.DataFrame()
    out["machine_id"] = df.get("machine_id")
    out["plant_shift_timestamp"] = pd.to_datetime(df.get("plant_shift_timestamp"), errors="coerce")
    out["machine_status_name"] = df.get("machine_status_name")
    out["scrap_quantity"] = pd.to_numeric(df.get("scrap_quantity"), errors="coerce")
    out["yield_quantity"] = pd.to_numeric(df.get("yield_quantity"), errors="coerce")
    return out.dropna(subset=["machine_id", "plant_shift_timestamp"])


def compare_hydra(idle_df: pd.DataFrame, hydra_df: pd.DataFrame) -> pd.DataFrame:
    if idle_df.empty:
        return pd.DataFrame()
    records = []
    for _, row in idle_df.iterrows():
        machine = str(row["machine_id"]).replace("-", "")
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        sub = hydra_df[
            hydra_df["machine_id"].astype(str).str.replace("-", "", regex=False).str.contains(machine[-3:])
        ]
        overlap = sub[(sub["plant_shift_timestamp"] >= start) & (sub["plant_shift_timestamp"] <= end)]
        records.append(
            {
                "machine_id": row["machine_id"],
                "start_time": start,
                "end_time": end,
                "hydra_rows_overlap": int(len(overlap)),
                "hydra_scrap_sum": float(overlap["scrap_quantity"].fillna(0).sum()) if len(overlap) else 0.0,
                "hydra_yield_sum": float(overlap["yield_quantity"].fillna(0).sum()) if len(overlap) else 0.0,
            }
        )
    return pd.DataFrame(records)


def main():
    raw_dir = Path(r"D:\new raw data")
    out_dir = Path("analysis") / "idle_patterns"
    out_dir.mkdir(parents=True, exist_ok=True)

    machine_files = sorted(raw_dir.glob("M*.csv"))
    all_idle: List[Segment] = []
    all_active: List[Segment] = []
    plotted = 0

    for machine_file in machine_files:
        machine_id = machine_file.stem.split("-")[0].replace("M", "M-")
        print(f"[INFO] Processing {machine_id} from {machine_file.name}")
        wide = load_machine_wide(machine_file)
        if wide.empty:
            print(f"[WARN] No usable sensor rows for {machine_id}")
            continue

        masks = build_masks(wide)
        idle_segments = extract_segments(masks["idle_any"], min_len=MIN_SEGMENT_LEN)
        active_segments = extract_segments(masks["active"], min_len=MIN_SEGMENT_LEN)

        for start, end, _ in idle_segments:
            seg_df = wide.loc[start:end]
            idle_type = choose_idle_type(seg_df.index, masks["frozen"], masks["constant"], masks["low_activity"])
            all_idle.append(summarize_segment(machine_id, seg_df, idle_type))
            if plotted < 5:
                plot_file = out_dir / f"idle_window_{plotted+1}_{machine_id}_{start:%Y%m%d_%H%M%S}.png"
                plot_segment(seg_df, machine_id, plot_file)
                plotted += 1

        for start, end, _ in active_segments[:20]:
            seg_df = wide.loc[start:end]
            all_active.append(summarize_segment(machine_id, seg_df, "active"))

    idle_df = pd.DataFrame([s.__dict__ for s in all_idle])
    active_df = pd.DataFrame([s.__dict__ for s in all_active])

    idle_csv = out_dir / "idle_windows.csv"
    active_csv = out_dir / "active_windows.csv"
    idle_df.to_csv(idle_csv, index=False)
    active_df.to_csv(active_csv, index=False)

    comp_rows = []
    for sensor in ["cycle", "injection_pressure", "switch_pressure", "peak_pressure_position"]:
        idle_std_col = f"{sensor}_std"
        active_std_col = f"{sensor}_std"
        if idle_std_col in idle_df.columns and active_std_col in active_df.columns:
            comp_rows.append(
                {
                    "sensor": sensor,
                    "idle_window_std_median": float(idle_df[idle_std_col].median()) if not idle_df.empty else np.nan,
                    "active_window_std_median": float(active_df[active_std_col].median()) if not active_df.empty else np.nan,
                    "idle_window_std_mean": float(idle_df[idle_std_col].mean()) if not idle_df.empty else np.nan,
                    "active_window_std_mean": float(active_df[active_std_col].mean()) if not active_df.empty else np.nan,
                }
            )
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "idle_vs_active_std_comparison.csv", index=False)

    hydra_df = hydra_reference_info(raw_dir / "MES_Manufacturing_M-231_M-356_M-471_M-607_M-612.xlsx")
    hydra_overlap_df = compare_hydra(idle_df, hydra_df)
    hydra_overlap_df.to_csv(out_dir / "idle_windows_hydra_overlap.csv", index=False)

    summary_file = out_dir / "IDLE_PATTERN_REPORT.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Idle Pattern Discovery Report\n\n")
        f.write(f"- Machines processed: {len(machine_files)}\n")
        f.write(f"- Idle windows detected: {len(idle_df)}\n")
        f.write(f"- Active windows sampled: {len(active_df)}\n")
        f.write(f"- Idle plots generated: {plotted}\n\n")
        if not idle_df.empty:
            f.write("## Idle Pattern Distribution\n\n")
            dist = idle_df["idle_type"].value_counts()
            for k, v in dist.items():
                f.write(f"- {k}: {int(v)} windows\n")
            f.write("\n## Idle Window Examples\n\n")
            cols = ["machine_id", "start_time", "end_time", "window_length", "idle_type"]
            f.write(idle_df[cols].head(20).to_markdown(index=False))
            f.write("\n\n")
        if not comp_df.empty:
            f.write("## Idle vs Active Variability Comparison\n\n")
            f.write(comp_df.to_markdown(index=False))
            f.write("\n\n")
        if not hydra_df.empty:
            f.write("## HYDRA Coverage\n\n")
            f.write(
                f"- HYDRA timestamp range: {hydra_df['plant_shift_timestamp'].min()} to {hydra_df['plant_shift_timestamp'].max()}\n"
            )
            if not hydra_overlap_df.empty:
                with_overlap = int((hydra_overlap_df["hydra_rows_overlap"] > 0).sum())
                f.write(f"- Idle windows with HYDRA overlap: {with_overlap}/{len(hydra_overlap_df)}\n")

    print(f"[DONE] Report: {summary_file}")
    print(f"[DONE] Idle windows CSV: {idle_csv}")
    print(f"[DONE] Active windows CSV: {active_csv}")
    print(f"[DONE] Comparison CSV: {out_dir / 'idle_vs_active_std_comparison.csv'}")
    print(f"[DONE] HYDRA overlap CSV: {out_dir / 'idle_windows_hydra_overlap.csv'}")


if __name__ == "__main__":
    main()
