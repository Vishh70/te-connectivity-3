import pandas as pd
import numpy as np
from datetime import timedelta
from backend.data_access import (
    build_control_room_payload, 
    get_available_machines, 
    _load_sensor_forecaster, 
    prepare_tcn_input_window,
    build_realtime_model_vector,
    normalize_machine_id,
    _build_machine_feb_history
)
from backend.forecasting import SensorForecasterWrapper
from backend.ml_inference_v9 import predict_scrap_v9 as predict_scrap_probability
from backend.config_limits import ML_THRESHOLDS

def run_4h_lookahead():
    """
    Performs a 4-hour (240m) future scrap prediction for all machines
    by projecting sensor states autoregressively and classifying each step.
    """
    
    # 1. Load the multi-sensor forecaster
    model, sensor_columns, input_features, num_lags, _ = _load_sensor_forecaster()
    if model is None:
        return {"error": "Sensor forecaster model not found."}
    
    forecaster = SensorForecasterWrapper(
        model=model,
        sensor_columns=sensor_columns,
        num_lags=num_lags,
        num_points=24 # 4 hours with 10-min resolution
    )
    
    machines = get_available_machines()
    results = []
    
    # Medium threshold for "predicted scrap" classification
    THRESHOLD = float(ML_THRESHOLDS.get("MEDIUM", 0.19))
    
    for m in machines:
        mid = m['id']
        machine_norm = normalize_machine_id(mid)
        
        try:
            # Get last 24h history for past count and future starting point
            history, _ = _build_machine_feb_history(machine_norm)
            if history.empty:
                results.append(create_empty_result(m))
                continue
            
            # A. Past 24h count
            past_scrap = int(history['is_scrap_actual'].sum())
            
            # B. Future 4h projection
            # 1. Prepare input tensor from the latest window
            window_data = prepare_tcn_input_window(
                history, 
                sensor_columns=sensor_columns, 
                num_lags=num_lags
            )
            input_tensor = window_data['tensor']
            
            # 2. Project sensors 4 hours ahead
            # Shape: (24 points, num_sensors)
            projected_sensors = forecaster.predict(input_tensor)
            
            # 3. Classify each projected point
            future_scrap_count = 0
            
            # To calculate derived features (lags/rolling) accurately for the future, 
            # we'll build a "synthetic" future dataframe.
            synthetic_future = history.iloc[-num_lags:].copy()
            
            for i in range(len(projected_sensors)):
                # Create next row
                next_row = {s: projected_sensors[i, idx] for idx, s in enumerate(sensor_columns)}
                # Add timestamp (approx 10 min steps)
                next_row['timestamp'] = synthetic_future['timestamp'].iloc[-1] + timedelta(minutes=10)
                
                # Append to synthetic history
                synthetic_future = pd.concat([synthetic_future, pd.DataFrame([next_row])], ignore_index=True)
                
                # Calculate full 55-feature vector for this projected point
                # build_realtime_model_vector uses the current window state
                feature_row = build_realtime_model_vector(synthetic_future, machine_norm=machine_norm)
                
                # Predict risk
                risk = predict_scrap_probability(feature_row)
                if risk >= THRESHOLD:
                    future_scrap_count += 1
            
            results.append({
                "Machine ID": mid,
                "Type": m['type'],
                "Past Scrap (24h)": past_scrap,
                "Future Scrap (4h)": future_scrap_count
            })
            
        except Exception as e:
            print(f"Error processing {mid}: {e}")
            results.append(create_empty_result(m))
            
    return results

def create_empty_result(m):
    return {
        "Machine ID": m['id'],
        "Type": m['type'],
        "Past Scrap (24h)": "N/A",
        "Future Scrap (4h)": "N/A"
    }

if __name__ == "__main__":
    data = run_4h_lookahead()
    import json
    print(json.dumps(data, indent=2))
