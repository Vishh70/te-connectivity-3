# 🏭 TE Connectivity: AI-Powered Predictive Maintenance
### *The "Senior Pro (V5)" Fleet Monitoring Solution*

[![Version](https://img.shields.io/badge/Release-v5.1.0--Senior--Pro-7916ff?style=for-the-badge&logo=rocket)](https://github.com/Vishh70/te-connectivity-3)
[![Status](https://img.shields.io/badge/Status-Deployment--Ready-2ea44f?style=for-the-badge&logo=checkmarx)](https://github.com/Vishh70/te-connectivity-3)
[![FastAPI](https://img.shields.io/badge/FastAPI-005863?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org)

The TE Connectivity Predictive Maintenance system is a state-of-the-art solution designed to monitor, predict, and explain scrap risks across mechanical production fleets. By leveraging high-precision machine learning and fleet-wide normalization, we provide operators with a **30-minute intervention window** before hardware failures occur.

---

## 🏛️ System Architecture

Our decoupled architecture ensures that 10GB+ of telemetry can be processed in real-time without dashboard lag.

```mermaid
graph TD
    classDef machine fill:#f9f,stroke:#333,stroke-width:2px;
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px;
    classDef frontend fill:#bfb,stroke:#333,stroke-width:2px;

    subgraph "Production Fleet (200+ Assets)"
        M1["🏭 Machine 231"]:::machine
        M2["🏭 Machine 607"]:::machine
        Mn["🏭 Machine N"]:::machine
    end

    subgraph "Intelligent Backend (FastAPI)"
        DA["💾 Data Access Layer"]:::backend
        INF["🧠 Senior Pro V5 Engine"]:::backend
        VAL["⚖️ Audit Hub"]:::backend
        API["📡 REST API"]:::backend
    end

    subgraph "Digital Twin Dashboard"
        DR["🕹️ Control Room"]:::frontend
        AH["📊 Analytics"]:::frontend
        XAI["🔍 Explainability Tool"]:::frontend
    end

    M1 & M2 & Mn --> DA
    DA --> INF
    INF --> API
    VAL --> API
    API --> DR
    API --> AH
    API --> XAI
```

---

## 🌟 Core Innovations

### 🧠 Senior Pro (V5) Inference
Built on **LightGBM**, our engine is tuned specifically for the extreme conditions of manufacturing.
*   **55%+ Precision**: Unmatched signal-to-noise ratio in high-vibration environments.
*   **30m Lead Time**: Proactive maintenance alerts for the entire fleet.

### ⚖️ Fleet-Wide Normalization
Each machine is a unique hardware ecosystem. We use **Machine-Contextual Z-Scores** to eliminate bias.
*   `Normalized_Value = (Sensor_Value - Machine_Mean) / Machine_Std`
*   Instant scalability to new machines without retraining.

### 🔍 Explainable AI (SHAP)
The system tells you **WHY** a machine is at risk.
*   **Root Cause**: Directly identifies at-risk components like injection valves or cycle timers.
*   **Confidence Scores**: Provides transparency into the model's decision-making process.

---

## 🛠️ Tech Stack & Requirements

| Layer | Technologies |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=flat-square&logo=javascript&logoColor=%23F7DF1E) |
| **Backend** | **FastAPI**, Uvicorn, Pandas, LightGBM, SHAP |
| **Frontend** | **Vite**, React 18, Framer Motion, Recharts, TailwindCSS |
| **OS Support** | Windows / Linux / MacOS |

---

## 🔧 Installation & Deployment

> [!IMPORTANT]
> Ensure you have **Node.js 18+** and **Python 3.12+** installed before proceeding.

### 1️⃣ Environment Setup
```powershell
# Create & Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install Production Dependencies
pip install -r requirements.txt
```

### 2️⃣ Frontend Initialization
```powershell
cd frontend
npm install
```

---

## 🚀 Execution Guide

> [!TIP]
> Use the included PowerShell script for the fastest one-click deployment.

### 🖥️ Option A: Automated (Fastest)
```powershell
./run-dev.ps1
```

### ⌨️ Option B: Manual
*   **Backend**: `uvicorn api:app --reload` (from `/backend`)
*   **Frontend**: `npm run dev` (from `/frontend`)

Access the dashboard at **`http://localhost:5173`**.

---

## 📂 Project Structure

*   `backend/` - API, Logic, and Inference Pipeline.
*   `frontend/` - Interactive Digital Twin Control Room.
*   `models/` - Senior Pro V5 production weights.
*   `scripts/` - R&D, EDA, and Model Sweeping tools.
*   `tests/` - System validation and smoke tests.

---
**Status:** ✅ 100% Finalized | **Designed For:** TE Connectivity AI Cup 🏆
