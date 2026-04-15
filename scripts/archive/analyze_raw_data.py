"""
analyze_raw_data.py
===================
Comprehensive analysis of all raw company data across:
  - D:/old data      (January data)
  - D:/new raw data  (newer raw data)
  - D:/new data      (newest data - larger files)
  - D:/teit04/te connectivity 3/new_processed_data (processed master)

Outputs a full analysis JSON + summary text file.
"""
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("D:/teit04/te connectivity 3/docs")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_SOURCES = {
    "old_data": {
        "base": "D:/old data",
        "files": {
            "M231": "M231Jan.csv",
            "M356": "M356Jan.csv",
            "M471": "M471Jan.csv",
            "M607": "M607Jan.csv",
            "M612": "M612Jan.csv",
        }
    },
    "new_raw_data": {
        "base": "D:/new raw data",
        "files": {
            "M231": "M231-11.csv",
            "M356": "M356-57.csv",
            "M471": "M471-23.csv",
            "M607": "M607-30.csv",
            "M612": "M612-33.csv",
        }
    },
    "new_data": {
        "base": "D:/new data",
        "files": {
            "M231": "M231-11.csv",
            "M356": "M356-57.csv",
            "M471": "M471-23.csv",
            "M607": "M607-30.csv",
            "M612": "M612-33.csv",
        }
    },
}

MES_FILES = {
    "old_data": "D:/old data/Hydra Dataset -- MES_Manufacturing_M-231_M-356_M-471_M-607_M-612.xlsx",
    "new_raw_data": "D:/new raw data/MES_Manufacturing_M-231_M-356_M-471_M-607_M-612.xlsx",
    "new_data": "D:/new data/MES_Manufacturing_M-231_M-356_M-471_M-607_M-612.xlsx",
}


def analyze_csv(path: str, nrows_sample=5000, machine_id=None) -> dict:
    """Load a CSV and compute statistics."""
    print(f"  Reading: {path} ...", flush=True)
    try:
        # Read full file for row count using chunking
        total_rows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
    except Exception:
        total_rows = -1

    try:
        df = pd.read_csv(path, nrows=nrows_sample, low_memory=False,
                         encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        return {"error": str(e)}

    # Basic structure
    result = {
        "file": str(path),
        "total_rows_approx": total_rows,
        "sample_rows": len(df),
        "columns": list(df.columns),
        "n_columns": len(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }

    # Date/time columns
    ts_cols = [c for c in df.columns if any(kw in c.lower() for kw in
               ["date", "time", "timestamp", "ts"])]
    result["timestamp_cols"] = ts_cols
    for tc in ts_cols[:2]:
        try:
            parsed = pd.to_datetime(df[tc], errors="coerce")
            result[f"date_range_{tc}"] = {
                "min": str(parsed.min()),
                "max": str(parsed.max()),
            }
        except Exception:
            pass

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result["n_numeric_cols"] = len(num_cols)
    result["numeric_cols"] = num_cols

    # Categorical / ID columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    result["n_categorical_cols"] = len(cat_cols)
    result["categorical_cols"] = cat_cols

    # Missing values
    missing = df.isnull().sum()
    result["missing_pct"] = {
        c: round(missing[c] / len(df) * 100, 2)
        for c in df.columns if missing[c] > 0
    }
    result["cols_with_missing"] = int((missing > 0).sum())

    # Scrap / target column detection
    scrap_related = [c for c in df.columns if any(kw in c.lower() for kw in
                     ["scrap", "reject", "ng", "defect", "is_bad", "label", "target"])]
    result["scrap_cols_detected"] = scrap_related
    for sc in scrap_related[:3]:
        try:
            vc = df[sc].value_counts(dropna=False)
            result[f"value_counts_{sc}"] = {str(k): int(v) for k, v in vc.items()}
            if df[sc].dtype in [np.int64, np.float64, np.int32, np.float32]:
                pos = (df[sc] == 1).sum()
                total = df[sc].notna().sum()
                result[f"scrap_rate_{sc}"] = round(pos / max(total, 1) * 100, 4)
        except Exception:
            pass

    # Part number / tool number detection
    identifier_cols = [c for c in df.columns if any(kw in c.lower() for kw in
                        ["part", "tool", "recipe", "mold", "setup", "program"])]
    result["identifier_cols"] = identifier_cols
    for ic in identifier_cols[:3]:
        try:
            vc = df[ic].value_counts(dropna=False)
            result[f"unique_{ic}"] = int(df[ic].nunique())
            result[f"top5_{ic}"] = {str(k): int(v) for k, v in vc.head(5).items()}
        except Exception:
            pass

    # Numeric sensor stats (sample only)
    sensor_stats = {}
    for col in num_cols[:30]:
        try:
            sensor_stats[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()), 4),
                "min":  round(float(df[col].min()), 4),
                "max":  round(float(df[col].max()), 4),
                "zero_pct": round((df[col] == 0).sum() / len(df) * 100, 2),
            }
        except Exception:
            pass
    result["sensor_stats_sample"] = sensor_stats

    # Check for obvious constant (dead) columns
    dead = [c for c in num_cols if df[c].std() == 0]
    result["dead_sensor_cols"] = dead

    return result


def analyze_mes(path: str) -> dict:
    """Analyze the MES Excel file for scrap labels."""
    print(f"  Reading MES: {path} ...", flush=True)
    try:
        xl = pd.ExcelFile(path)
        result = {"sheets": xl.sheet_names, "file": str(path)}
        for sheet in xl.sheet_names[:5]:
            df = xl.parse(sheet)
            result[f"sheet_{sheet}_shape"] = list(df.shape)
            result[f"sheet_{sheet}_cols"] = list(df.columns)

            scrap_related = [c for c in df.columns if any(kw in str(c).lower() for kw in
                             ["scrap", "reject", "ng", "defect", "label"])]
            result[f"sheet_{sheet}_scrap_cols"] = scrap_related
            for sc in scrap_related[:2]:
                try:
                    vc = df[sc].value_counts(dropna=False)
                    result[f"sheet_{sheet}_vc_{sc}"] = {str(k): int(v) for k, v in vc.head(10).items()}
                except Exception:
                    pass

            part_tool = [c for c in df.columns if any(kw in str(c).lower() for kw in
                         ["part", "tool", "recipe", "machine"])]
            result[f"sheet_{sheet}_id_cols"] = part_tool

        return result
    except Exception as e:
        return {"error": str(e), "file": str(path)}


def analyze_processed_master() -> dict:
    """Analyze the processed master parquet used for current training."""
    path = "D:/teit04/te connectivity 3/new_processed_data/FINAL_TRAINING_MASTER_V3.parquet"
    print(f"\n  Reading processed master: {path} ...", flush=True)
    try:
        df = pd.read_parquet(path)
        result = {
            "file": path,
            "total_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
        # Timestamp range
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            result["date_range"] = {"min": str(ts.min()), "max": str(ts.max())}

        # Per-machine row counts
        if "machine_id" in df.columns:
            mc = df["machine_id"].value_counts()
            result["rows_per_machine"] = {str(k): int(v) for k, v in mc.items()}

        # Label info
        if "is_scrap" in df.columns:
            result["global_scrap_rate_pct"] = round(df["is_scrap"].mean() * 100, 4)
            if "machine_id" in df.columns:
                result["scrap_rate_per_machine"] = {
                    str(m): round(g["is_scrap"].mean() * 100, 4)
                    for m, g in df.groupby("machine_id")
                }

        # Missing
        missing = df.isnull().sum()
        result["cols_with_missing"] = int((missing > 0).sum())
        result["missing_summary"] = {
            c: round(missing[c] / len(df) * 100, 2)
            for c in df.columns if missing[c] > 0
        }

        # Numeric variance check
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        dead = [c for c in num_cols if df[c].std() == 0]
        result["dead_cols"] = dead
        result["n_numeric_features"] = len(num_cols)

        # ROC feature presence
        roc_cols = [c for c in df.columns if "roc" in c.lower()]
        result["roc_feature_count"] = len(roc_cols)
        result["roc_feature_sample"] = roc_cols[:10]

        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    full_analysis = {"sources": {}, "mes": {}, "processed_master": {}}

    # Analyze processed master
    print("\n=== PROCESSED MASTER (current training data) ===")
    full_analysis["processed_master"] = analyze_processed_master()

    # Analyze one machine per source (M231 as representative; fast)
    for source_name, source_info in RAW_SOURCES.items():
        print(f"\n=== SOURCE: {source_name} ===")
        full_analysis["sources"][source_name] = {}
        for machine_id in ["M231", "M356"]:  # sample two machines
            file_path = Path(source_info["base"]) / source_info["files"][machine_id]
            print(f"  [{machine_id}] ...", flush=True)
            stats = analyze_csv(str(file_path), nrows_sample=3000, machine_id=machine_id)
            full_analysis["sources"][source_name][machine_id] = stats

    # Analyze MES files
    for source_name, mes_path in MES_FILES.items():
        print(f"\n=== MES: {source_name} ===")
        full_analysis["mes"][source_name] = analyze_mes(mes_path)

    # Write full JSON
    out_json = OUTPUT_DIR / "raw_data_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2, default=str)
    print(f"\nFull analysis saved to: {out_json}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pm = full_analysis["processed_master"]
    print(f"\nPROCESSED MASTER (training data):")
    print(f"  Rows: {pm.get('total_rows', 'N/A'):,}")
    print(f"  Columns: {pm.get('n_columns', 'N/A')}")
    print(f"  Date range: {pm.get('date_range', {})}")
    print(f"  Global scrap rate: {pm.get('global_scrap_rate_pct', 'N/A')}%")
    print(f"  Scrap per machine: {pm.get('scrap_rate_per_machine', {})}")
    print(f"  ROC features: {pm.get('roc_feature_count', 0)}")
    print(f"  Missing cols: {pm.get('cols_with_missing', 0)}")
    print(f"  Dead cols: {pm.get('dead_cols', [])}")

    for src in RAW_SOURCES:
        print(f"\nRAW SOURCE: {src}")
        for machine_id in ["M231", "M356"]:
            stats = full_analysis["sources"].get(src, {}).get(machine_id, {})
            if "error" in stats:
                print(f"  [{machine_id}] ERROR: {stats['error']}")
            else:
                print(f"  [{machine_id}] Rows ~{stats.get('total_rows_approx', 'N/A'):,}  "
                      f"Cols: {stats.get('n_columns', 'N/A')}  "
                      f"Scrap cols: {stats.get('scrap_cols_detected', [])}  "
                      f"Missing: {stats.get('cols_with_missing', 0)}  "
                      f"Dead: {len(stats.get('dead_sensor_cols', []))}")
                ts_col = stats.get('timestamp_cols', [])
                if ts_col:
                    tc = ts_col[0]
                    dr = stats.get(f"date_range_{tc}", {})
                    print(f"    Date range: {dr.get('min', 'N/A')} → {dr.get('max', 'N/A')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
