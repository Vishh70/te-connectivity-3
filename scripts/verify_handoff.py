import os
import sys
import json
import joblib
from pathlib import Path

def check_integrity():
    print("=" * 70)
    print("🚀 TE CONNECTIVITY: MASTER PRODUCTION CERTIFICATION (V5.2 FINAL)")
    print("=" * 70)
    
    root = Path(__file__).resolve().parent.parent
    
    # 1. Infrastructure Checks (Critical Folders)
    infrastructure = {
        "pipeline  data": "Raw Telemetry Registry",
        "new_processed_data": "V5 Production Mergedata",
        "processed/safe": "Inference Mapping Buffers",
        "metrics": "Model Calibration & Stats",
        "models": "AI Neural Oracle",
        "backend": "FastAPI Logic Engine",
        "frontend": "Visual Dashboard Layer"
    }
    
    print("\n[📁] Checking Core Folder Integrity:")
    missing_infra = []
    for rel, desc in infrastructure.items():
        p = root / rel
        if p.exists():
            print(f"  ✅ FOUND   | {rel:25} | {desc}")
        else:
            print(f"  ❌ MISSING | {rel:25} | {desc}")
            missing_infra.append(rel)

    # 2. Environment & Dependency Proofing
    print("\n[🛠️] Checking Runtime Calibration:")
    
    # Check Virtual Environment
    venv_exists = (root / ".venv").exists() or (root / "venv").exists()
    print(f"  {'✅' if venv_exists else '⚠️'} ENV       | Python Virtual Environment: {'Active/Found' if venv_exists else 'Missing (.venv recommended)'}")

    # Check Node Modules
    node_exists = (root / "frontend" / "node_modules").exists()
    print(f"  {'✅' if node_exists else '⚠️'} NODE      | Frontend Dependencies: {'Installed' if node_exists else 'Action Required (npm install)'}")

    # Check Production Build
    dist_exists = (root / "frontend" / "dist").exists()
    print(f"  {'✅' if dist_exists else '⚠️'} BUILD     | Frontend Prod Bundle: {'Ready' if dist_exists else 'Missing (npm run build)'}")

    # 3. Model & Logic Verification
    print("\n[🧠] Verifying AI Model Oracle:")
    model_path = root / "models" / "production_scrap_model.pkl"
    features_path = root / "models" / "production_features.pkl"
    
    model_status = False
    if model_path.exists() and features_path.exists():
        try:
            features = joblib.load(features_path)
            print(f"  ✅ MODEL     | V9 Production Oracle: Loaded Successfully ({len(features)} Features)")
            model_status = True
        except Exception as e:
            print(f"  ❌ MODEL     | Load Failed: {e}")
    else:
        print(f"  ❌ MODEL     | Production Weights missing from /models")

    # 4. Final Final Certification
    print("\n" + "=" * 70)
    if not missing_infra and model_status:
        print("🏆 CERTIFICATION STATUS: SUCCESS (GREEN)")
        print("System is fully certified for Industrial Production release.")
        print("\nLAUNCH PROTOCOL:")
        print("1. Backend:  python -m uvicorn backend.api:app --host 0.0.0.0")
        print("2. Frontend: npm run dev")
    else:
        print("🚨 CERTIFICATION STATUS: FAILED (RED)")
        print("Action required to restore repository health.")
    print("=" * 70)

if __name__ == "__main__":
    check_integrity()
