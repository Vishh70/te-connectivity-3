import requests
import json
import time

BASE_URL = "http://127.0.0.1:8080"

def test_connectivity():
    print("--- 1. Testing Machines Endpoint ---")
    try:
        r = requests.get(f"{BASE_URL}/api/machines")
        machines = r.json()
        print(f"Detected {len(machines)} machines.")
        for m in machines:
            print(f" - {m['id']} ({m['type']})")
    except Exception as e:
        print(f"FAIL: {e}")
        return

    print("\n--- 2. Testing Control Room Endpoint (Live Data Match) ---")
    target = "M-231"
    try:
        r = requests.get(f"{BASE_URL}/api/control-room/{target}")
        data = r.json()
        
        # Check if we have real telemetry sensors
        sensors = data.get("telemetry_grid", [])
        if len(sensors) > 0:
            print(f"SUCCESS: Received {len(sensors)} sensor metrics for {target}")
            print(f"Status: {data['current_health']['status']}, Risk: {data['current_health']['risk_score']}")
        else:
            print("WARNING: Control room returned empty telemetry grid.")
            
        # Check for SHAP root causes
        causes = data.get("root_causes", [])
        if len(causes) > 0:
            print(f"SUCCESS: Distributed AI Root Cause Analysis detected {len(causes)} impact factors.")
        else:
            print("INFO: No root causes detected (likely low risk).")
            
        # Check timeline continuity
        timeline = data.get("timeline", [])
        if len(timeline) > 0:
            types = [p.get('type') for p in timeline]
            print(f"SUCCESS: Timeline has {len(timeline)} points. Includes: {set(types)}")
        else:
            print("FAIL: Empty timeline.")
            
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n--- 3. Testing Ingestion Status ---")
    try:
        r = requests.get(f"{BASE_URL}/api/ingest/status")
        status = r.json()
        print(f"Ingestion Pilot Status: {status.get('status')}, Step: {status.get('step')}/10")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    test_connectivity()
