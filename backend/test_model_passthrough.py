import pandas as pd
import numpy as np
from data_access import build_control_room_payload, _load_control_model_and_features, _build_machine_feb_history, build_realtime_model_vector, normalize_machine_id
from future_predictor import predict_future_risk
import json

def run_verification_test(machine_id="M-607", anchor_time="2026-03-14T11:25:00Z"):
    print(f"=== Starting Verification Test for {machine_id} at {anchor_time} ===")
    
    # Step 1: Run the full backend API payload generation
    print("\n1. Generating Dashboard Payload (Simulating API Call)...")
    payload = build_control_room_payload(
        machine_id=machine_id,
        time_window=120,
        future_window=30,
        anchor_time=anchor_time
    )
    
    # Extract future points from the timeline
    timeline = payload.get("timeline", [])
    future_points = [p for p in timeline if p.get("is_future") and p.get("type") == "future" and p.get("horizon_minutes")]
    
    if not future_points:
        print("ERROR: No future points found in payload. Are the models loaded?")
        return False
        
    dashboard_predictions = {}
    for p in future_points:
        h = p["horizon_minutes"]
        score = p["risk_score"]
        dashboard_predictions[f"{h}m"] = score
        print(f"   Dashboard Payload [{h}m]: {score}")

    # Step 2: Manually load the raw data and run the raw models natively
    print("\n2. Reconstructing Raw Model Pipeline...")
    machine_norm = normalize_machine_id(machine_id)
    history, _ = _build_machine_feb_history(machine_norm, time_window_minutes=120, anchor_time=anchor_time)
    anchor = pd.to_datetime(anchor_time, utc=True)
    cutoff = anchor - pd.Timedelta(minutes=120)
    past_window = history[
        (history["timestamp"] >= cutoff) & (history["timestamp"] <= anchor)
    ].copy()
    
    _, feature_columns = _load_control_model_and_features()
    feature_row = build_realtime_model_vector(past_window, machine_norm=machine_id, strict=True)
    
    print("\n3. Executing Raw Models Natively...")
    raw_future_preds = predict_future_risk(feature_row, feature_columns)
    
    # Step 3: Compare Side-by-Side
    print("\n=== VERIFICATION RESULTS ===")
    all_match = True
    for h in [5, 10, 15, 20, 25, 30]:
        key = f"{h}m"
        if key not in raw_future_preds:
            continue
            
        raw_val = float(raw_future_preds[key])
        raw_clamped = round(max(0.0, min(1.0, raw_val)), 4)
        
        dash_val = dashboard_predictions.get(key)
        
        match = (dash_val == raw_clamped)
        status = "PASS" if match else "FAIL"
        print(f"Horizon [{key}]: Raw Model -> {raw_clamped:.4f} | Dashboard -> {dash_val:.4f} | Status: {status}")
        
        if not match:
            all_match = False

    print("\n=== FINAL VERIFICATION CONCLUSION ===")
    if all_match:
        print("SUCCESS! ALl values from the backend to the frontend exactly match the native un-smoothed model predictions. No phantom data detected.")
    else:
        print("FAILURE! The dashboard values differ from the raw model outputs. Smoothing or leakage may still exist.")
        
    return all_match

if __name__ == "__main__":
    # Test for M-607 as seen in the screenshot
    run_verification_test("M-607", "2026-03-14T11:25:00Z")
    
    # Optional test for M-356 from the other screenshot
    print("\n\n")
    run_verification_test("M-356", "2026-03-30T17:20:00Z")
