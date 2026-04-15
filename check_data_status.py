import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent / "new_processed_data"

# Check FINAL_TRAINING_MASTER_V3
master = pd.read_parquet(base / "FINAL_TRAINING_MASTER_V3.parquet")
print("=== FINAL_TRAINING_MASTER_V3.parquet ===")
print(f"  Shape: {master.shape}")
ts = pd.to_datetime(master["timestamp"])
print(f"  Date range: {ts.min()} -> {ts.max()}")
print(f"  Machines: {sorted(master['machine_id'].unique())}")
scrap_col = None
for candidate in ("is_scrap", "is_scrap_actual", "future_scrap"):
    if candidate in master.columns:
        scrap_col = candidate
        break
if scrap_col is None:
    scrap_cols = [c for c in master.columns if c.startswith("scrap_")]
    scrap_col = scrap_cols[0] if scrap_cols else None

if scrap_col:
    print(f"  Total scrap rows ({scrap_col}): {int(master[scrap_col].sum())} ({master[scrap_col].mean()*100:.2f}%)")
else:
    print("  WARNING: No scrap label column found in FINAL_TRAINING_MASTER_V3.parquet")
print()

# Check OLD parquet sizes
print("=== OLD parquet row counts ===")
for f in sorted(base.glob("OLD_*_TRAIN.parquet")):
    df = pd.read_parquet(f)
    print(f"  {f.name}: {len(df):,} rows")

print()
print("=== NEW parquet row counts ===")
for f in sorted(base.glob("NEW_*_TRAIN.parquet")):
    df = pd.read_parquet(f)
    print(f"  {f.name}: {len(df):,} rows")

print()
# Check cleaned_dataset_v4.parquet
cleaned = pd.read_parquet(base / "cleaned_dataset_v4.parquet")
print("=== cleaned_dataset_v4.parquet ===")
print(f"  Shape: {cleaned.shape}")
col_sample = list(cleaned.columns[:5])
print(f"  First 5 cols: {col_sample}")
if "future_scrap" in cleaned.columns:
    print(f"  future_scrap ratio: {cleaned['future_scrap'].mean()*100:.3f}%")
else:
    print("  WARNING: 'future_scrap' column NOT found!")

print()
# Check cleaned_dataset_v4.5f.parquet
cleaned2 = pd.read_parquet(base / "cleaned_dataset_v4.5f.parquet")
print("=== cleaned_dataset_v4.5f.parquet ===")
print(f"  Shape: {cleaned2.shape}")
horizon_cols = [c for c in cleaned2.columns if c.startswith("scrap_")]
print(f"  Horizon label cols: {horizon_cols}")
