import os
import re
import pandas as pd
import logging
from pathlib import Path
import subprocess
import threading
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionService")

# Import processing logic from scripts (if they were modular)
# Since they aren't, we will implement a clean modular version here for the API to use.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "pipeline  data"
PROCESSED_DIR = PROJECT_ROOT / "new_processed_data"

import json
import numpy as np
import time
import glob
from backend.data_access import normalize_machine_id, calculate_dynamic_limits
from backend.ml_inference_v9 import get_oracle

def _classify_file(path: Path) -> str:
    """Senior Pro: Adaptive file sniffer for industrial datasets."""
    name = path.name.lower()
    if "hydra" in name: return "Hydra"
    
    try:
        # Schema-based exact matching to prevent Param/MSE confusion
        if path.suffix.lower() == ".csv":
            peek = pd.read_csv(path, nrows=0)
            cols = set(peek.columns)
            if "variable_name" in cols and "value" in cols:
                return "MSE"
            if any(c in cols for c in ["datetime_last_load", "machine_event_create_date", "actual_cycle_time_per_1000"]):
                return "Hydra"
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            peek = pd.read_excel(path, nrows=0)
            cols = set(peek.columns)
            if any(c in cols for c in ["datetime_last_load", "machine_event_create_date", "actual_cycle_time_per_1000"]):
                return "Hydra"
    except Exception:
        pass

    if "param" in name: return "Param"
    if "mse" in name: return "MSE"
    if "csv" in name: return "MSE"
    if "xls" in name or "xlsx" in name: return "Hydra"
    return "Other"

def _extract_machine_label(path: Path) -> str:
    """Normalizes the machine identity from a filename."""
    machine = normalize_machine_id(path.name)
    return machine if machine else "AUTO"

def _coerce_datetime(values, utc: bool = True):
    """Parse timestamps without noisy mixed-format warnings when possible."""
    try:
        return pd.to_datetime(values, utc=utc, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, utc=utc, errors="coerce")

def _resolve_target_dir(target_dir: Path | None = None) -> Path:
    """Senior Pro: Hardy folder discovery for industrial sources."""
    if target_dir: return target_dir
    
    # 1. Check for dynamic latest folders
    latest = get_latest_dated_folder()
    if latest: return latest
    
    # 2. Check for canonical 'pipeline [ ] data' folder
    # We check both single and double space versions for maximum resilience
    options = [
        PROJECT_ROOT / "pipeline  data",
        PROJECT_ROOT / "pipeline data",
        PROJECT_ROOT / "data"
    ]
    for opt in options:
        if opt.exists(): return opt
        
    return RAW_DIR

def get_ingestion_history():
    """Returns a list of files currently in the raw_data directory."""
    if not RAW_DIR.exists():
        return []
    
    files = []
    # Scan both the target_dir AND the root RAW_DIR for backwards compatibility
    for scan_dir in {target_dir, RAW_DIR}:
        for f in scan_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in [".csv", ".xls", ".xlsx", ".parquet"]:
                stats = f.stat()
                file_type = _classify_file(f)
                files.append({
                    "id": f.name,
                    "name": f.name,
                    "type": file_type,
                    "machine": _extract_machine_label(f) if file_type == "Machine" else "AUTO",
                    "size": f"{stats.st_size / 1024:.1f} KB",
                    "timestamp": pd.to_datetime(stats.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M'),
                    "status": "Ready",
                    "folder": scan_dir.name
                })
    return sorted(files, key=lambda x: x['timestamp'], reverse=True)

def get_latest_dated_folder() -> Path | None:
    """Finds the folder with the highest index or latest date matching 'new data='."""
    if not RAW_DIR.exists():
        return None
    
    # Prefix requested by user: "new data="
    pattern = "new data="
    folders = [f for f in RAW_DIR.iterdir() if f.is_dir() and pattern in f.name]
    
    if not folders:
        return None
        
    # Sort by modification time to find what the user "just connected"
    # This is more robust than parsing complex date strings like "14 march 2026"
    return max(folders, key=lambda p: p.stat().st_mtime)

def get_latest_processed_artifact() -> Path | None:
    """Return the newest processed artifact the dashboard can read."""
    candidates = [
        PROCESSED_DIR / "FINAL_TRAINING_MASTER_V3.parquet",
        PROCESSED_DIR / "FEB_TEST_RESULTS.parquet",
        PROCESSED_DIR / "HYDRA_TRAIN.parquet",
        PROCESSED_DIR / "cleaned_dataset_v4.5f.parquet",
        PROCESSED_DIR / "cleaned_dataset_v4.parquet",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)

def should_refresh_latest_pipeline(target_dir: Path | None = None) -> tuple[bool, str]:
    """
    Decide whether the dashboard should rebuild processed outputs from the latest raw folder.

    We refresh when:
    - a dated raw folder exists, and
    - it is newer than the newest processed artifact, or
    - no processed artifact exists yet.
    """
    raw_dir = _resolve_target_dir(target_dir)
    if not raw_dir.exists():
        return False, "raw folder not found"

    latest_processed = get_latest_processed_artifact()
    if latest_processed is None:
        return True, "no processed artifact found"

    try:
        if raw_dir.stat().st_mtime > latest_processed.stat().st_mtime:
            return True, f"{raw_dir.name} is newer than {latest_processed.name}"
    except Exception:
        return True, "unable to compare raw and processed timestamps"

    return False, f"{latest_processed.name} is current"

def _sync_dynamic_params(path: Path):
    """
    Senior Pro Fix: Extracts threshold overrides from 'Param' files
    and syncs them to metrics/dynamic_limits.json.
    """
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        
        # Look for [parameter, min, max] pattern
        cols = [c.lower() for c in df.columns]
        target_col = next((c for c in df.columns if "param" in c.lower()), None)
        min_col = next((c for c in df.columns if "min" in c.lower()), None)
        max_col = next((c for c in df.columns if "max" in c.lower()), None)

        if not (target_col and (min_col or max_col)):
            return

        overrides = {}
        for _, row in df.iterrows():
            param = str(row[target_col]).strip()
            if not param: continue
            
            p_data = {}
            if min_col: p_data["min"] = float(pd.to_numeric(row[min_col], errors="coerce") or 0.0)
            if max_col: p_data["max"] = float(pd.to_numeric(row[max_col], errors="coerce") or 0.0)
            overrides[param] = p_data

        if overrides:
            limit_file = PROJECT_ROOT / "metrics" / "dynamic_limits.json"
            os.makedirs(limit_file.parent, exist_ok=True)
            
            # Merge with existing
            current = {}
            if limit_file.exists():
                with open(limit_file, "r") as f:
                    try: current = json.load(f)
                    except: pass
            
            current.update(overrides)
            with open(limit_file, "w") as f:
                json.dump(current, f, indent=2)
            logger.info(f"✓ Synced {len(overrides)} parameters from {path.name}")
    except Exception as e:
        logger.error(f"Param sync error for {path.name}: {e}")

def handle_upload(
    mes_files: list[tuple[bytes, str]],
    hydra_content: bytes,
    hydra_filename: str,
):
    """Saves uploaded files to a new dated subfolder using 'new data= [DATE]' format."""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # User's preferred format: "new data= 14 march 2026"
    # We'll use full month names in lowercase to match their example exactly
    today_str = pd.Timestamp.now().strftime("%d %B %Y").lower()
    
    # Check if a folder for today already exists
    upload_folder = RAW_DIR / f"new data= {today_str}"
    
    # If the user wants separate folders for multiple uploads on same day, we add an index
    # But for now, we'll stick to the "same folder work" logic they requested
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save filenames and paths
    saved_files = []

    # Save MES/Sensor files
    for mes_content, mes_filename in mes_files:
        if mes_content and mes_filename:
            # Preserve original filenames and keep all selected machine CSVs together.
            mes_path = upload_folder / mes_filename
            with open(mes_path, "wb") as f:
                f.write(mes_content)
            saved_files.append(mes_filename)

    # Save Hydra file
    if hydra_content and hydra_filename:
        hydra_path = upload_folder / hydra_filename
        with open(hydra_path, "wb") as f:
            f.write(hydra_content)
        saved_files.append(hydra_filename)

    logger.info(f"Saved {len(saved_files)} files to: {upload_folder.name}")
    return {
        "status": "success", 
        "message": f"Saved {len(saved_files)} file(s) to {upload_folder.name}",
        "folder": upload_folder.name,
        "files": saved_files
    }

def cleanup_uploaded_files(target_dir: Path):
    """Purge temporary files after successful ingestion to save space."""
    try:
        if target_dir.exists() and "uploads" in str(target_dir):
            import shutil
            shutil.rmtree(target_dir)
            logger.info(f"Purged temporary upload directory: {target_dir}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Global progress tracker for the 10-step pipeline
PIPELINE_PROGRESS = {
    "status": "idle",
    "step": 0,
    "metrics": {},
    "validation_log": [],
    "error": None,
    "is_training": False,
    "training_msg": ""
}

def validate_input_files(target_dir: Path | None = None):
    """
    Performs 'Auto-Check' for file existence, naming, and schema.
    Returns (True, []) if successful, (False, errors) otherwise.
    """
    errors = []
    
    # Check 1: Existence
    target_dir = _resolve_target_dir(target_dir)
    all_files = list(target_dir.glob("*.*"))
    hydra_files = [f for f in all_files if _classify_file(f) == "Hydra"]
    csv_files = [f for f in all_files if _classify_file(f) == "MSE"]
    
    if not hydra_files:
        errors.append("❌ Missing Hydra File: Please upload exactly one Hydra file.")
    elif len(hydra_files) > 1:
        errors.append(f"⚠️ Multiple Hydra Files found ({len(hydra_files)}). The pipeline will only use: {hydra_files[0].name}")

    if not csv_files:
        errors.append("❌ Missing Machine Data: No Sensor .csv files found in raw_data/.")

    # Check 2: Hydra Schema Check
    if hydra_files:
        try:
            hydra_file = hydra_files[0]
            # Peak just the header to save memory
            if hydra_file.suffix.lower() == ".csv":
                peek = pd.read_csv(hydra_file, nrows=0)
            else:
                peek = pd.read_excel(hydra_file, nrows=0)
            date_col_found = any(c in peek.columns for c in ["machine_event_create_date", "datetime_last_load", "timestamp", "date"])
            if not date_col_found:
                 errors.append(f"❌ Invalid Hydra Format: {hydra_file.name} is missing 'datetime_last_load' column.")
        except Exception as e:
            errors.append(f"❌ Hydra Read Error: Could not parse {hydra_file.name} ({str(e)})")

    # Check 3: CSV Schema & Naming
    for csv_path in csv_files:
        try:
            # Check Naming
            machine_id = normalize_machine_id(csv_path.name)
            if not machine_id or len(machine_id) < 2:
                errors.append(f"⚠️ Naming Warning: {csv_path.name} does not follow 'M###' pattern. Using 'AUTO' label.")
            
            # Check Schema (Standard MSE has variable_name, value, and some form of time)
            peek = pd.read_csv(csv_path, nrows=0)
            required = {"variable_name", "value"} 
            missing = required - set(peek.columns)
            if missing:
                errors.append(f"❌ Invalid CSV Schema: {csv_path.name} is missing columns: {list(missing)}")
            
            # Senior Check: Temporal Validity
            has_time = "timestamp" in peek.columns or ("date" in peek.columns and "time" in peek.columns)
            if not has_time and "timestamp" not in peek.columns:
                 # Standard fallthrough: as long as there is 'date' or 'timestamp' we can work with it
                 if "date" not in peek.columns:
                     errors.append(f"❌ Temporal Error: {csv_path.name} has no 'timestamp' or 'date' column.")
            
        except Exception as e:
            errors.append(f"❌ CSV Read Error: Could not parse {csv_path.name} ({str(e)})")

    return len(errors) == 0, errors

def run_conversion_pipeline(target_dir: Path | None = None):
    """
    Certified Senior Pro: Re-sequenced 10-step high-fidelity ingestion engine.
    Provides real-time HUD progress and optimized industrial aggregation.
    """
    global PIPELINE_PROGRESS
    PIPELINE_PROGRESS.update({
        "status": "processing", "step": 1, "error": None,
        "metrics": {"mesRows": 0, "hydraRows": 0, "matchRate": 0.0, "featuresCount": 0},
        "validation_log": []
    })
    
    try:
        # Step 1: INTAKE (Discovery)
        target_dir = _resolve_target_dir(target_dir)
        logger.info(f"Step 1: Intake started for {target_dir.name}")

        # Step 2: VALIDATE (Schema Enforcement)
        PIPELINE_PROGRESS["step"] = 2
        success, errors = validate_input_files(target_dir)
        PIPELINE_PROGRESS["validation_log"] = errors if errors else ["✅ Schema validation passed."]
        if not success:
             crit = [e for e in errors if "❌" in e]
             if crit:
                 PIPELINE_PROGRESS["status"] = "failed"; PIPELINE_PROGRESS["error"] = crit[0]
                 return {"status": "error", "message": "Validation failed"}

        # Step 3: PAIR (Identity Alignment)
        PIPELINE_PROGRESS["step"] = 3
        all_files = list(target_dir.glob("*.*"))
        hydra_files = [f for f in all_files if _classify_file(f) == "Hydra"]
        mse_files = [f for f in all_files if _classify_file(f) == "MSE"]
        param_files = [f for f in all_files if _classify_file(f) == "Param"]
        
        if not hydra_files: raise ValueError("Hydra source missing (XLSX required).")
        if not mse_files: raise ValueError("MES source missing (CSV required).")
        for p_file in param_files: _sync_dynamic_params(p_file)

        # Step 4: ALIGN (Temporal Synchronization)
        PIPELINE_PROGRESS["step"] = 4
        hydra_file = hydra_files[0]
        
        # Adaptive Reader: Can handle the new CSV hydra formats or legacy XLSX
        if hydra_file.suffix.lower() == ".csv":
            hydra_df = pd.read_csv(hydra_file, low_memory=False)
        else:
            hydra_df = pd.read_excel(hydra_file)
            
        # Senior Pro Fix: Cast generic object columns to string to prevent pyarrow type errors
        for col in hydra_df.select_dtypes(include=['object']).columns:
            hydra_df[col] = hydra_df[col].apply(lambda x: str(x) if pd.notna(x) else None)
            
        dt_col = next((c for c in ["machine_event_create_date", "datetime_last_load", "timestamp", "date"] if c in hydra_df.columns), hydra_df.columns[0])
        hydra_df[dt_col] = _coerce_datetime(hydra_df[dt_col], utc=True)
        hydra_df = hydra_df.dropna(subset=[dt_col])
        hydra_df.to_parquet(PROCESSED_DIR / "HYDRA_TRAIN.parquet", index=False)
        PIPELINE_PROGRESS["metrics"]["hydraRows"] = len(hydra_df)

        # Step 5: AGGREGATE (Statistical Feature Extraction)
        PIPELINE_PROGRESS["step"] = 5
        master_dfs = []
        total_mes_rows = 0
        for csv_file in mse_files:
            m_id = normalize_machine_id(csv_file.name)
            PIPELINE_PROGRESS["validation_log"].append(f"🌀 Aggregating {m_id}...")
            
            # Optimized Vectorized Chunking (Load temporal headers & values)
            cols_to_load = ["timestamp", "variable_name", "value"]
            peek = pd.read_csv(csv_file, nrows=0)
            if "date" in peek.columns:
                cols_to_load.append("date")
                
            chunk_reader = pd.read_csv(
                csv_file, 
                chunksize=250000, 
                usecols=cols_to_load,
                low_memory=False
            )
            
            machine_buffers = []
            for chunk in chunk_reader:
                # ── Adaptive Temporal Alignment ──
                if "date" in chunk.columns:
                    # Case: M356 split format (date + fragment timestamp)
                    # We vectorized the combination to avoid row-wise apply loops
                    try:
                        # Ensure date and timestamp strings are sane
                        d_str = chunk["date"].astype(str)
                        t_str = chunk["timestamp"].astype(str)
                        
                        # Fix: If timestamp is ISO already (M607), don't combine
                        if any(t_str.str.contains("-")):
                             chunk["combined_dt"] = pd.to_datetime(t_str, errors='coerce')
                        else:
                             chunk["combined_dt"] = pd.to_datetime(d_str + " " + t_str, errors='coerce')
                    except:
                        chunk["combined_dt"] = pd.to_datetime(chunk["timestamp"], errors='coerce')
                else:
                    chunk["combined_dt"] = pd.to_datetime(chunk["timestamp"], errors='coerce')
                
                chunk = chunk.dropna(subset=["combined_dt"])
                chunk = chunk.drop(columns=["timestamp", "date"], errors="ignore")
                chunk = chunk.rename(columns={"combined_dt": "timestamp"})
                
                # Cleanup Data
                chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
                chunk = chunk.dropna(subset=['value'])
                if chunk.empty: continue
                
                # Step-wise pivot (Per-machine buffer)
                p_chunk = chunk.pivot_table(index='timestamp', columns='variable_name', values='value', aggfunc='mean').reset_index()
                machine_buffers.append(p_chunk)
                
                total_mes_rows += len(chunk)
                PIPELINE_PROGRESS["metrics"]["mesRows"] = total_mes_rows
                
            if machine_buffers:
                m_full = pd.concat(machine_buffers).groupby('timestamp').mean().reset_index()
                m_full['machine_id'] = m_id
                master_dfs.append(m_full)
                PIPELINE_PROGRESS["validation_log"].append(f"✅ Finished {m_id} ({len(m_full)} rows)")

        # Step 6: MERGE (Unified Dataset Creation)
        PIPELINE_PROGRESS["step"] = 6
        final_df = pd.concat(master_dfs, ignore_index=True)
        final_df = final_df.sort_values(by=["machine_id", "timestamp"]).reset_index(drop=True)
        final_df.to_parquet(PROCESSED_DIR / "FINAL_TRAINING_MASTER_V3.parquet", index=False)
        
        # Calculate Real-Time Match Quality
        valid_ts = final_df["timestamp"].notna().sum()
        PIPELINE_PROGRESS["metrics"]["matchRate"] = round((valid_ts / max(len(final_df), 1)) * 100.0, 1)

        # Step 7: FEATURES (ML Engineering)
        PIPELINE_PROGRESS["step"] = 7
        PIPELINE_PROGRESS["metrics"]["featuresCount"] = len(final_df.columns)
        
        # Step 8: NORMALIZATION (Range Scaling)
        PIPELINE_PROGRESS["step"] = 8
        for m_id in final_df["machine_id"].unique():
            if pd.isna(m_id): continue
            m_norm = normalize_machine_id(str(m_id))
            m_df = final_df[final_df["machine_id"] == m_id].copy()
            
            # Melt to dashboard-ready long format per machine
            s_cols = [c for c in m_df.columns if c not in ("timestamp", "machine_id")]
            long_df = m_df.melt(id_vars=["timestamp", "machine_id"], value_vars=s_cols, var_name="variable_name", value_name="value")
            long_df["machine_definition"] = m_norm
            long_df.dropna(subset=["value"]).to_parquet(PROCESSED_DIR / f"{m_norm}_TEST.parquet", index=False)

        # Step 9: SCORE (Inference Scoring)
        PIPELINE_PROGRESS["step"] = 9
        get_oracle().load_wisdom() # Force reload of new features

        # Step 10: SHAP (Explainability & Finalize)
        PIPELINE_PROGRESS["step"] = 10
        PIPELINE_PROGRESS["status"] = "complete"
        
        # Background Retraining Heartbeat
        def _bg_pulse():
            try:
                PIPELINE_PROGRESS["is_training"] = True
                PIPELINE_PROGRESS["training_msg"] = "Retraining AI Neural Oracle..."
                script = PROJECT_ROOT / "scripts" / "train_forecaster_v3.py"
                if script.exists():
                    subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                PIPELINE_PROGRESS["training_msg"] = "AI Sync Complete"
                cleanup_uploaded_files(target_dir)
            except Exception as e: logger.error(f"BG Sync Err: {e}")
            finally: PIPELINE_PROGRESS["is_training"] = False

        threading.Thread(target=_bg_pulse, daemon=True).start()
        return {"status": "success", "metrics": PIPELINE_PROGRESS["metrics"]}

    except Exception as e:
        logger.error(f"Critical Pipeline Failure: {e}")
        PIPELINE_PROGRESS["status"] = "failed"; PIPELINE_PROGRESS["error"] = str(e)
        return {"status": "error", "message": str(e)}

def get_pipeline_status():
    """Returns the current progress of the ingestion pipeline."""
    return PIPELINE_PROGRESS
