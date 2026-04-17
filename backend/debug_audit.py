import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.data_access import _build_machine_feb_history, normalize_machine_id, get_machine_code
from backend.feature_utils import augment_temporal_signal_features

def debug_predictive_mismatch():
    machine_id = "M-356"
    machine_norm = normalize_machine_id(machine_id)
    m_code = get_machine_code(machine_norm)
    
    # 1. Load history for Case 1 (16:31)
    history, _ = _build_machine_feb_history(machine_norm, time_window_minutes=120)
    history_aug = augment_temporal_signal_features(history)
    
    # 2. Pick a row 15 minutes before the scrap (approx)
    scrap_time = pd.to_datetime("2024-02-15 16:31:00").tz_localize("UTC")
    lead_point = scrap_time - pd.Timedelta(minutes=15)
    
    # Find closest row
    history_aug["timestamp"] = pd.to_datetime(history_aug["timestamp"])
    closest_idx = (history_aug["timestamp"] - lead_point).abs().idxmin()
    row = history_aug.loc[closest_idx]
    
    print(f"DEBUG: Using row at {row['timestamp']} for scrap at {scrap_time}")
    
    # 3. Build the feature dictionary like the audit hub does
    feat_dict = row.dropna().to_dict()
    feat_dict["machine_id_encoded"] = m_code
    feat_dict["machine_cat"] = m_code
    feat_dict[f"Machine_{machine_id.replace('-', '')}"] = 1.0
    
    # 4. Load the 15m model and check features
    model_path = Path("models/future_models/model_scrap_15m.pkl")
    model = joblib.load(model_path)
    m_features = model.feature_name_
    
    print(f"DEBUG: Model expects {len(m_features)} features.")
    
    # Check for missing features
    missing = [f for f in m_features if f not in feat_dict]
    print(f"DEBUG: Missing features in feat_dict ({len(missing)}): {missing[:20]}...")
    
    # 5. Run prediction
    X = pd.DataFrame([{f: feat_dict.get(f, 0.0) for f in m_features}])[list(m_features)].fillna(0)
    
    # Senior Pro: Print top 5 values
    print("DEBUG: Top 5 feature values in X:")
    for f in m_features[:5]:
        print(f"  {f}: {X[f].values[0]}")
    
    prob = model.predict_proba(X)[0, 1]
    print(f"DEBUG: 15m Prediction Probability: {prob}")

if __name__ == "__main__":
    debug_predictive_mismatch()
