import re
import numpy as np

def _get_base_sensor_name(feature_name: str) -> str:
    """Removes all engineered statistical suffixes to return the clean original sensor name."""
    # Pattern designed to catch both v3 and v4 pipelined suffixes
    pattern = r"_(rolling_mean_5|rolling_std_5|rolling_min_5|rolling_max_5|lag_1|lag_3|lag_5|rate_of_change_5|rate_of_change_30|roc_5|roc_30)"
    
    # Strip compound suffixes until no more remain (e.g. "_roc_30_rate_of_change_30")
    cleaned = feature_name
    while True:
        new_cleaned = re.sub(pattern, "", cleaned)
        if new_cleaned == cleaned:
            break
        cleaned = new_cleaned
        
    return cleaned

def _map_feature_to_cause(feature_name: str) -> str:
    if "Cycle_time" in feature_name:
        return "Cycle Time Instability"
    if "Injection_pressure" in feature_name:
        return "Injection Pressure Instability"
    if "Switch_pressure" in feature_name:
        return "Switch Pressure Variation"
    if "Cyl_tmp" in feature_name:
        return "Cylinder Temperature Instability"
    if "Peak_pressure" in feature_name:
        return "Peak Pressure Position Drift"
    return "Other Sensor Variation"


def compute_root_causes(model, feature_row, feature_names):
    """
    Compute root causes from LightGBM feature contributions (SHAP-style).
    Returns grouped causes with detailed parameter-level breakdown using cleanly mapped Base Sensors.
    """

    contrib = model.predict(feature_row, pred_contrib=True)
    contrib = np.asarray(contrib)

    if contrib.ndim == 2:
        contrib = contrib[0]
    elif contrib.ndim > 2:
        contrib = contrib.reshape(-1)

    if contrib.size == 0:
        return []

    # Last contribution is model bias term for LightGBM.
    feature_contrib = contrib[:-1]

    grouped = {}
    for name, value in zip(feature_names, feature_contrib):
        value = float(value)
        if abs(value) < 1e-4:
            continue
        
        category = _map_feature_to_cause(name)
        if category not in grouped:
            grouped[category] = {
                "total_impact": 0.0,
                "risk_increasing": 0.0,
                "risk_decreasing": 0.0,
                "sensor_agg": {}
            }
            
        base_sensor = _get_base_sensor_name(name)
        if base_sensor not in grouped[category]["sensor_agg"]:
            grouped[category]["sensor_agg"][base_sensor] = 0.0
            
        # Aggregate the impact of all derivatives back into the single base sensor!
        grouped[category]["sensor_agg"][base_sensor] += value
        
        # Prevent cancellation by tracking positive and negative separately
        if value > 0:
            grouped[category]["risk_increasing"] += value
        else:
            grouped[category]["risk_decreasing"] += value
            
        # Overall grouped magnitude
        grouped[category]["total_impact"] += value

    result = []
    for category, data in grouped.items():
        # Build the final parameters list from the aggregated base sensors
        top_params = [
            {"parameter": sensor, "impact": imp}
            for sensor, imp in data["sensor_agg"].items()
            if abs(imp) >= 1e-4 # ignore sensors perfectly canceled out
        ]
        
        # Sort parameters by absolute impact to find the key drivers within this category
        top_params.sort(key=lambda x: abs(x["impact"]), reverse=True)
        # Keep only the top 5 highest responsible physical sensors
        top_params = top_params[:5]
        
        result.append({
            "cause": category,                       # For backward compatibility
            "impact": data["total_impact"],          # For backward compatibility
            "category": category,
            "total_impact": data["total_impact"],
            "risk_increasing": data["risk_increasing"],
            "risk_decreasing": data["risk_decreasing"],
            "top_parameters": top_params
        })

    # Sort categories by absolute total impact
    result.sort(key=lambda item: abs(item["total_impact"]), reverse=True)
    return result[:5]
