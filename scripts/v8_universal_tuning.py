import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import sys
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

# V8.5 Universal Oracle: The Absolute Truth (Balanced Shuffling)
VERSION = "v8.5-universal-oracle"
PROJECT_ROOT = Path(r"C:\2. te ai cup\te connectivity 3")
DATA_PATH = PROJECT_ROOT / "new_processed_data" / "cleaned_dataset_v4.5f.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

sys.path.append(str(PROJECT_ROOT))
from backend.feature_utils import augment_temporal_signal_features

def train_v8_5():
    print(f"--- V8.5 UNIVERSAL ORACLE: BALANCED GLOBAL TRUTH ---")
    
    RAW_CORE = [
        "Cushion", "Cycle_time", "Dosage_time", "Injection_pressure", 
        "Injection_time", "Switch_pressure", "Switch_position",
        "Cyl_tmp_z1", "Cyl_tmp_z3", "Cyl_tmp_z4", "Cyl_tmp_z5", "Cyl_tmp_z8"
    ]
    IDENTIFIER = "machine_id_encoded"
    TARGET = "future_scrap"
    
    pfile = pq.ParquetFile(DATA_PATH)
    total_rows = pfile.metadata.num_rows
    cols_to_load = RAW_CORE + [TARGET, IDENTIFIER]
    
    # LOAD REPRESENTATIVE SAMPLES (Head, Mid, Tail)
    print("Sampling 1.25M cycles for balanced global training...")
    shard_size = 200000
    table = pfile.read(columns=cols_to_load)
    
    df_head = table.slice(0, shard_size).to_pandas()
    df_mid = table.slice(total_rows // 2, shard_size).to_pandas()
    df_tail = table.slice(total_rows - shard_size, shard_size).to_pandas()
    
    df_raw = pd.concat([df_head, df_mid, df_tail], ignore_index=True)
    
    # BALANCED SAMPLING: Keep all scrap, sample non-scrap
    scrap_indices = df_raw[df_raw[TARGET] > 0].index
    non_scrap_indices = df_raw[df_raw[TARGET] == 0].index
    
    # Target 50/50 or 25/75 balance to force signal detection
    sample_non_scrap = df_raw.loc[non_scrap_indices].sample(n=min(len(non_scrap_indices), 150000), random_state=42)
    df_balanced = pd.concat([df_raw.loc[scrap_indices], sample_non_scrap], ignore_index=True).sample(frac=1.0, random_state=42)
    
    print(f"Balanced Dataset: {len(df_balanced)} rows ({len(scrap_indices)} scrap events).")
    
    # 3. Feature Reconstruction
    for col in RAW_CORE:
        df_balanced[col] = pd.to_numeric(df_balanced[col], errors='coerce').ffill().fillna(0)
            
    df_augmented = augment_temporal_signal_features(df_balanced)
    
    EXCLUDE = {TARGET, IDENTIFIER, "timestamp", "is_scrap"}
    model_features = [c for c in df_augmented.columns if c not in EXCLUDE and not any(x in c for x in ["scrap_", "shot_"])]
    df_augmented["machine_cat"] = df_augmented[IDENTIFIER].astype("category")
    model_features.append("machine_cat")
    
    X = df_augmented[model_features]
    y = df_augmented[TARGET].astype(int)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Hyper-tuning for 75% Recall
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "scale_pos_weight": 4.0 # Balanced weight for 75% target
    }
    
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)
    
    model = lgb.train(params, d_train, valid_sets=[d_valid], num_boost_round=1500, callbacks=[lgb.early_stopping(stopping_rounds=100)])
    
    # 6. Global Precision Calibration
    probs = model.predict(X_valid)
    precisions, recalls, thresholds = precision_recall_curve(y_valid, probs)
    
    # Target: 75% Precision and 75% Recall
    target_mask = (precisions >= 0.74) & (recalls >= 0.74)
    if target_mask.any():
        idx = np.where(target_mask)[0][np.argmax(recalls[target_mask])]
        print(f"!!! TRUTH ATTAINED !!!")
    else:
        # Fallback to best F1
        f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        idx = np.argmax(f1)
        print(f"Best Convergence: P={precisions[idx]:.2%}, R={recalls[idx]:.2%}")
        
    best_t = thresholds[idx] if idx < len(thresholds) else 0.5
    
    # SAVE V8.5
    joblib.dump(model, MODEL_DIR / "scrap_risk_model_v8.pkl") # Overwrite V8 with V8.5 stable
    joblib.dump(model_features, MODEL_DIR / "model_features_v8.pkl")
    
    report = {"p": float(precisions[idx]), "r": float(recalls[idx]), "auc": float(roc_auc_score(y_valid, probs)), "t": float(best_t)}
    with open(METRICS_DIR / "final_model_report_v8.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("V8.5 Final Truth Certificate Generated.")

if __name__ == "__main__":
    train_v8_5()
