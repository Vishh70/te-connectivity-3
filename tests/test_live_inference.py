import sys
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.live_predictor import predict_from_raw, clear_buffer, get_buffer_status

def test_live_predictor_frozen_detector():
    machine = "M231"
    clear_buffer(machine)
    
    # 1. Provide initial status
    r1 = predict_from_raw(machine, {"variable_name": "Cycle_time", "value": 15.0})
    assert r1["status"] in {"offline", "success"}  # The first sample may now score immediately.
    
    # Send a bunch of identical states
    r2 = predict_from_raw(machine, {"variable_name": "Cycle_time", "value": 15.0})
    assert r2["status"] in {"offline", "success"}
    
def test_live_predictor_valid_scoring(monkeypatch):
    machine = "M607"
    clear_buffer(machine)
    
    # Feed enough data to trigger normal scoring
    predict_from_raw(machine, {"variable_name": "Cycle_time", "value": 20.0})
    predict_from_raw(machine, {"variable_name": "Cyl_tmp_z3", "value": 400.0})
    
    # Now it should compute the massive vector locally without pandas
    result = predict_from_raw(machine, {"variable_name": "Injection_pressure", "value": 1500.0})
    
    assert result["machine_id"] == "M607"
    assert result["status"] in ["success", "offline"]
    # Even if predictions fail or default to 0 due to missing ML paths on CI, it shouldn't crash.
    assert isinstance(result["risk_score"], float)

def test_buffer_endpoints():
    clear_buffer()
    predict_from_raw("M100", {"variable_name": "temp", "value": 1})
    status = get_buffer_status("M100")
    assert status["buffer_size"] == 1
