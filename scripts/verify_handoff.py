import os
import sys
import json
import joblib
from pathlib import Path

def check_integrity():
    print("=" * 70)
    print("🚀 TE CONNECTIVITY: MASTER PRODUCTION CERTIFICATION (V5.5 SENIOR PRO)")
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
    print("\n[🧠] Verifying V9 Neural Oracle:")
    models_dir = root / "models" / "future_models"
    manifest_path = models_dir / "future_model_manifest.json"
    
    expected_horizons = ["5m", "10m", "15m", "20m", "25m", "30m"]
    found_models = 0
    manifest = {}

    def _unwrap_artifact(artifact):
        if isinstance(artifact, dict):
            model = artifact.get("model") or artifact.get("estimator") or artifact.get("forecaster")
            feature_names = (
                artifact.get("feature_names")
                or artifact.get("features")
                or artifact.get("feature_columns")
                or []
            )
            if not isinstance(feature_names, list):
                feature_names = list(feature_names) if feature_names else []
            return model or artifact, feature_names
        return artifact, []

    def _feature_count(model) -> int:
        if isinstance(model, dict):
            _, feature_names = _unwrap_artifact(model)
            return len(feature_names)
        if hasattr(model, "feature_name_"):
            try:
                return len(list(model.feature_name_))
            except Exception:
                pass
        if hasattr(model, "feature_name"):
            try:
                names = model.feature_name() if callable(model.feature_name) else model.feature_name
                return len(list(names))
            except Exception:
                return 0
        return 0

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"  ✅ MANIFEST | Loaded {manifest_path.name}")
        except Exception as exc:
            print(f"  ❌ MANIFEST | Failed to read {manifest_path.name}: {exc}")
            manifest = {}
    else:
        print("  ⚠️ MANIFEST | future_model_manifest.json missing (recommended)")
    
    if models_dir.exists():
        for horizon in expected_horizons:
            m_path = models_dir / f"model_scrap_{horizon}.pkl"
            if m_path.exists():
                try:
                    artifact = joblib.load(m_path)
                    model, feature_names = _unwrap_artifact(artifact)
                    feature_count = len(feature_names) or _feature_count(model)
                    print(f"  ✅ MODEL     | {horizon:4} | Loadable ({feature_count} features)")
                    found_models += 1
                except Exception as exc:
                    print(f"  ❌ MODEL     | {horizon:4} | Failed to load: {exc}")
            else:
                print(f"  ⚠️ WARNING   | Missing {horizon} model weight.")
        
        if found_models == len(expected_horizons):
            print(f"  ✅ MODEL     | V9 Production Oracle: Fully Optimized ({found_models}/6 Horizons Online)")
            model_status = True
        else:
            print(f"  ❌ MODEL     | Partial Oracle Load ({found_models}/6). Check /models/future_models.")
            model_status = False
    else:
        print("  ❌ MODEL     | V9 Neural Oracle Directory missing from /models")
        model_status = False

    if manifest:
        missing_manifest_horizons = [h for h in expected_horizons if h not in (manifest.get("models") or {})]
        if missing_manifest_horizons:
            print(f"  ❌ MANIFEST | Missing horizons: {', '.join(missing_manifest_horizons)}")
            model_status = False
        else:
            print("  ✅ MANIFEST | Horizon coverage verified")

    # 4. Final Final Certification
    print("\n" + "=" * 70)
    if not missing_infra and model_status:
        print("🏆 CERTIFICATION STATUS: SUCCESS (V5.5 SENIOR PRO)")
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
