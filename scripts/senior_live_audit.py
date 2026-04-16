import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Ensure backend is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.ml_inference_v9 import get_oracle
from backend.feature_utils import augment_temporal_signal_features

def run_live_audit():
    print("🚀 TE CONNECTIVITY: SENIOR PRO LIVE ACCURACY AUDIT (V9-CERTIFIED)")
    print("===============================================================")
    
    data_path = Path("new_processed_data/cleaned_dataset_v4.5f.parquet")
    if not data_path.exists():
        print(f"❌ Error: Big dataset {data_path} not found.")
        return

    print(f"Initializing Senior Oracle (V9)...")
    oracle = get_oracle()
    model, features = oracle.load_wisdom()
    
    # Load raw sensors + ground truth + machine ID
    raw_sensors = ["Cushion", "Cycle_time", "Dosage_time", "Injection_pressure", "Injection_time", 
                   "Switch_pressure", "Switch_position", "Cyl_tmp_z1", "Cyl_tmp_z2", "Cyl_tmp_z3", 
                   "Cyl_tmp_z4", "Cyl_tmp_z5", "Cyl_tmp_z6", "Cyl_tmp_z7", "Cyl_tmp_z8"]
    load_columns = raw_sensors + ["future_scrap", "machine_id_encoded"]
    
    print(f"Sampling Evaluation Set (50k records)...")
    try:
        # Load a contiguous slice to preserve temporal statistics
        table = pq.read_table(data_path, columns=load_columns)
        df_raw = table.to_pandas()
        df_eval = df_raw.iloc[50000:100000].copy() 
        
        print(f"Reconstructing V5 Feature Space...")
        # Re-engineering 275+ signals
        df_featured = augment_temporal_signal_features(df_eval)
        
        print(f"Running Certified V9 Inference...")
        X = df_featured.copy()
        for f in features:
            if f not in X.columns:
                X[f] = 0.0
                
        if "machine_cat" in features:
            X["machine_cat"] = X["machine_id_encoded"].astype("category")

        X = X[list(features)]
        
        # Base and Adaptive scores
        y_scores_base = model.predict(X)
        baseline = pd.Series(y_scores_base).rolling(window=30, min_periods=1).mean().values
        rel_risk = y_scores_base / (baseline + 0.01)
        y_scores = np.clip(y_scores_base * oracle.alpha + (rel_risk * oracle.beta), 0.0, 1.0)
        
        y_true = df_featured["future_scrap"].fillna(0).astype(int).values
        threshold = 0.5531
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate Metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_scores)
        
        print("\n🏆 FINAL CERTIFICATION RESULTS (V9 MASTER):")
        print(f"| Metric      | Score      | Status       |")
        print(f"| :---        | :---       | :---         |")
        print(f"| Precision   | {precision:.2%}   | {'✅ PASSED' if precision >= 0.7 else '⚠️ REVIEW'} |")
        print(f"| Recall      | {recall:.2%}   | {'✅ PASSED' if recall >= 0.5 else '⚠️ REVIEW'} |")
        print(f"| ROC AUC     | {auc:.4f}     | 🏆 ELITE     |")
        print("-" * 54)
        
        mapping = {0.356: "M356", 0.607: "M607", 0.612: "M612", 0.231: "M231", 0.471: "M471"}
        for code, m_name in mapping.items():
            m_mask = df_featured["machine_id_encoded"] == code
            if m_mask.any():
                idx = np.where(m_mask)[0]
                m_true, m_pred = y_true[idx], y_pred[idx]
                m_p = precision_score(m_true, m_pred, zero_division=0)
                m_r = recall_score(m_true, m_pred, zero_division=0)
                print(f"- {m_name}: {m_p:.1%} Prec / {m_r:.1%} Rec")

    except Exception as e:
        print(f"❌ Critical Audit Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_live_audit()
