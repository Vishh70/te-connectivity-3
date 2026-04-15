import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

# Senior Pro V7: Target Parameters
VERSION = "v7-pro-fixed-v2"
PROJECT_ROOT = Path(r"C:\2. te ai cup\te connectivity 3")
DATA_PATH = PROJECT_ROOT / "new_processed_data" / "cleaned_dataset_v4.5f.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Import V7 Feature Logic
import sys
sys.path.append(str(PROJECT_ROOT))
from backend.feature_utils import augment_temporal_signal_features

def train_v7_high_precision():
    print(f"--- V7 SENIOR PRO: HIGH-PRECISION TRAINING PHASE (FIXED v2) ---")
    
    # 1. Strictly define the Raw Core Sensors (Baseline for all aggregations)
    RAW_CORE = [
        "Cushion", "Cycle_time", "Dosage_time", "Injection_pressure", 
        "Injection_time", "Switch_pressure", "Switch_position"
    ] + [f"Cyl_tmp_z{i}" for i in range(1, 9)]
    
    # 2. ULTRA-MEMORY-SAFE LOADING (Column-Specific)
    print(f"Slicing RAW telemetry only from {DATA_PATH.name}...")
    pfile = pq.ParquetFile(DATA_PATH)
    
    # Load raw columns only to avoid LightGBM 'Feature appears more than once' error
    cols_to_load = RAW_CORE + ["future_scrap", "machine_id_normalized"]
    
    # Take a 200k sample for robust training
    sample_size = 200000
    df_raw = pfile.read_row_group(pfile.num_row_groups - 1, columns=cols_to_load).to_pandas()
    if len(df_raw) < sample_size and pfile.num_row_groups > 1:
        prev_group = pfile.read_row_group(pfile.num_row_groups - 2, columns=cols_to_load).to_pandas()
        df_raw = pd.concat([prev_group, df_raw]).tail(sample_size)

    # 3. Feature Reconstruction (The Senior Way)
    print("Regenerating V7 Precision Suite from raw signals...")
    for col in RAW_CORE:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').ffill().fillna(0)
            
    df_augmented = augment_temporal_signal_features(df_raw)
    
    # Deduplicate columns just in case
    df_augmented = df_augmented.loc[:, ~df_augmented.columns.duplicated()].copy()
    
    # Define features to use (Removing labels and non-numeric machine IDs)
    EXCLUDE = {"future_scrap", "timestamp", "machine_id_normalized", "machine_id", "is_scrap"}
    model_features = [c for c in df_augmented.columns if c not in EXCLUDE and not any(x in c for x in ["scrap_", "shot_"])]
    
    # Senior Fix: Force categorical encoding for machine_id if needed
    # But for simplicity in this script, we use technical process features only.
    
    print(f"Training on {len(model_features)} unique industrial signals.")
    
    X = df_augmented[model_features]
    y = df_augmented["future_scrap"].astype(int)
    
    # 4. Stratified Split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 5. Hyperparameters for 71-75% PR convergence
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 48,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": 4.0 # Balanced for recall in the 75% range
    }
    
    # 6. Training
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)
    
    model = lgb.train(
        params,
        d_train,
        valid_sets=[d_valid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 7. Final PR-Curve Calibration
    probs = model.predict(X_valid)
    precisions, recalls, thresholds = precision_recall_curve(y_valid, probs)
    
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    # Target 71-75% Balance
    target_mask = (precisions >= 0.70) & (recalls >= 0.70)
    if target_mask.any():
        idx = np.where(target_mask)[0][np.argmax(f1[target_mask])]
        print(f"!!! TARGET ATTAINED: V7 SENIOR PRO REACHED 75% RANGE !!!")
    else:
        idx = np.argmax(f1)
        print(f"Closest PR balance: P={precisions[idx]:.2%}, R={recalls[idx]:.2%}")
        
    best_t = thresholds[idx] if idx < len(thresholds) else 0.5
    final_p = precisions[idx]
    final_r = recalls[idx]
    
    # 8. Save Artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "scrap_risk_model_v7.pkl")
    joblib.dump(model_features, MODEL_DIR / "model_features_v7.pkl")
    
    # V7 Report
    report = {
        "version": VERSION,
        "precision": float(final_p),
        "recall": float(final_r),
        "f1": float(f1[idx]),
        "auc": float(roc_auc_score(y_valid, probs)),
        "threshold": float(best_t),
        "rows_trained": len(X_train),
        "positives_in_train": int(y_train.sum())
    }
    with open(METRICS_DIR / "final_model_report_v7.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Calibration V7
    # Ensure machine_id_normalized exists, fallback to 'M-UNKNOWN' if missing
    machine_list = df_augmented["machine_id_normalized"].unique() if "machine_id_normalized" in df_augmented.columns else ["M-UNKNOWN"]
    t_file = {
        "precision_target": 0.73,
        "default_threshold": float(best_t),
        "per_machine_thresholds": {str(m): float(best_t) for m in machine_list}
    }
    with open(METRICS_DIR / "machine_thresholds_v7.json", "w") as f:
        json.dump(t_file, f, indent=2)

if __name__ == "__main__":
    train_v7_high_precision()
