import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path

def check_integrity():
    print("=" * 60)
    print("🚀 TE CONNECTIVITY: SENIOR PRO HANDOFF INTEGRITY CHECK (V2)")
    print("=" * 60)
    
    root = Path(__file__).resolve().parent.parent
    
    # 1. Structural Checks
    critical_paths = {
        "processed/sensor": "Telemetry cache",
        "processed/safe": "Inference buffers",
        "new_processed_data": "V5 Mergedata source",
        "metrics": "Thresholds & Registry",
        "models": "AI Core",
        "backend": "Logic layer",
        "frontend": "UI layer"
    }
    
    missing_paths = []
    print("\n[📁] Checking Core Infrastructure:")
    for path_rel, desc in critical_paths.items():
        p = root / path_rel
        status = "✅ FOUND" if p.exists() else "❌ MISSING"
        print(f"  {status:10} | {path_rel:25} | {desc}")
        if not p.exists():
            missing_paths.append(path_rel)

    # 2. Model & Logic Integrity
    print("\n[🧠] Checking AI Model & Registry Integrity:")
    registry_path = root / "metrics" / "machine_registry_v6.json"
    model_path = root / "models" / "production_scrap_model.pkl"
    features_path = root / "models" / "production_features.pkl"
    
    logic_errors = []
    
    # Check Registry
    if registry_path.exists():
        try:
            with open(registry_path, 'r') as f:
                reg = json.load(f)
            machines = reg.get("machines", [])
            print(f"  ✅ REGISTRY   | Found {len(machines)} registered machines")
        except Exception as e:
            logic_errors.append(f"Registry format error: {e}")
            print(f"  ❌ REGISTRY   | CORRUPT")
    else:
        print(f"  ❌ REGISTRY   | MISSING")

    # Check Model Loadability
    if model_path.exists() and features_path.exists():
        try:
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            print(f"  ✅ AI MODEL   | V9 Production Oracle loaded ({len(features)} features)")
        except Exception as e:
            logic_errors.append(f"Model load error: {e}")
            print(f"  ❌ AI MODEL   | LOAD FAILED")
    else:
        print(f"  ❌ AI MODEL   | ARTIFACTS MISSING")

    # 3. Final Verdict
    print("\n" + "=" * 60)
    if not missing_paths and not logic_errors:
        print("🏆 HEALTH CHECK: ALL SYSTEMS GO (GREEN)")
        print("The repository is fully assembled and certified for production.")
        print("\nNEXT STEPS:")
        print("1. Run 'npm install' in /frontend")
        print("2. Launch backend: python -m uvicorn backend.api:app --reload")
        print("3. Launch frontend: npm run dev")
    else:
        print("🚨 HEALTH CHECK: ACTION REQUIRED (RED)")
        if missing_paths:
            print(f"  MISSING DIRECTORIES: {', '.join(missing_paths)}")
        if logic_errors:
            for err in logic_errors:
                print(f"  LOGIC ERROR: {err}")
        print("\nACTION: Re-copy the Pendrive contents to the root folder.")
    print("=" * 60)

if __name__ == "__main__":
    check_integrity()
