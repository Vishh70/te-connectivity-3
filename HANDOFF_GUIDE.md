# 🏭 TE Connectivity: Production V5 Assembly Guide

Greetings! This guide will help you assemble the full Predictive Maintenance system. Since large datasets and calibration files are excluded from GitHub, you will need to manually place them into the project structure.

## 📦 Pendrive Manifest
Ensure you have copied the following folders from the Pendrive to your project root:

| Folder Name | Description | Placement |
| :--- | :--- | :--- |
| `processed/` | Intermediate data & Parameter info | Project Root |
| `pipeline data/` | Raw telemetry datasets (Parquet) | Project Root |
| `new_processed_data/` | High-fidelity "Mergedata" for V5 predictions | Project Root |
| `metrics/` | Fleet thresholds & Z-Score calibration JSONs | Project Root |
| `test_raw_data/` | Ground-truth cases for the Validation Hub | Project Root |

---

## 🛠️ Assembly Steps

### 1. Extract GitHub Code
Extract the ZIP or `git clone` the repository into a folder on your computer (e.g., `C:\TE_Connectivity`).

### 2. Manual Data Injection
Copy the folders from the **Pendrive Manifest** (above) and paste them directly into `C:\TE_Connectivity`.

### 3. Initialize Environment (Python)
Open a terminal in the project root:
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Initialize Dashboard (Frontend)
Open a separate terminal in the `frontend/` directory:
```bash
npm install
npm run dev
```

---

## 🔍 Validation
To ensure everything is correctly placed, run our high-fidelity integrity check:
```powershell
python scripts/verify_handoff.py
```
If you see **🏆 CERTIFICATION STATUS: SUCCESS (V5.5 SENIOR PRO)**, you are ready.

## 🧠 AI Neural Oracle (V9)
The V5.5 system utilizes a Multi-Horizon Neural Oracle. Unlike previous versions that used mathematical smoothing, this engine performs discrete inference for 5, 10, 15, 20, 25, and 30-minute horizons simultaneously.
- **Model Storage**: `models/future_models/`
- **Fidelity**: 100% predictive loyalty to April 2026 ground-truth cases.

## 🌀 10-Step Ingestion HUB
The new **Flow Synchronizer** provides a synchronized, professional data pipeline:
1. **Intake** -> 2. **Validate** -> 3. **Pair** -> 4. **Align** -> 5. **Aggregate** -> 6. **Merge** -> 7. **Features** -> 8. **Normalize** -> 9. **Score** -> 10. **SHAP**

## 🚀 Launching (One-Click)
We have provided a professional launch script to start both the backend and frontend in separate windows:
```powershell
./Launch_All.ps1
```

## 🕰️ Using the "Time Machine"
To verify historical scrap events (e.g., March 30th):
1. Locate the **Date/Time Picker** in the Dashboard Header.
2. Select the desired historical window.
3. Click **Jump**. The dashboard will enter **Replay Mode** (Static), allowing you to audit the scrap zone without live data interference.

Enjoy the "V5.5" Fleet Monitoring Experience! — *Atharva & Vishnu*
