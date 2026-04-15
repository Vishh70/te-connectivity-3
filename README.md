# 🏭 TE Connectivity: AI-Powered Predictive Maintenance
### *The "Senior Pro (V5)" Fleet Monitoring Solution*

[![Version](https://img.shields.io/badge/Release-v5.1.0--Senior--Pro-7916ff?style=for-the-badge&logo=rocket)](https://github.com/Vishh70/te-connectivity-3)
[![Status](https://img.shields.io/badge/Status-Deployment--Ready-2ea44f?style=for-the-badge&logo=checkmarx)](https://github.com/Vishh70/te-connectivity-3)
[![FastAPI](https://img.shields.io/badge/FastAPI-005863?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

The TE Connectivity Predictive Maintenance system is a state-of-the-art solution designed to monitor, predict, and explain scrap risks across mechanical production fleets. By leveraging high-precision machine learning and fleet-wide normalization, we provide operators with a **30-minute intervention window** before hardware failures occur.

---

## 🏛️ Advanced Architecture

Our architecture is optimized for high-performance telemetry processing and explainable AI insights.

```mermaid
graph LR
    classDef machine fill:#ff99cc,stroke:#333,stroke-width:2px,color:#000;
    classDef backend fill:#99ccff,stroke:#333,stroke-width:2px,color:#000;
    classDef frontend fill:#99ffcc,stroke:#333,stroke-width:2px,color:#000;
    classDef data fill:#ffffff,stroke:#333,stroke-width:2px,color:#000,stroke-dasharray: 5 5;

    subgraph "Production Fleet"
        M1["🏭 Machine 231"]:::machine
        M2["🏭 Machine 607"]:::machine
        Mn["🏭 Machine N"]:::machine
    end

    subgraph "Data Storage"
        PQT[(".parquet Telemetry")]:::data
        PKL[(".pkl Model Weights")]:::data
        JSON[("Config & Limits")]:::data
    end

    subgraph "Senior Pro Backend"
        DA["💾 Data Access Layer"]:::backend
        NORM["⚖️ Z-Score Normalization"]:::backend
        INF["🧠 Senior Pro V5 Engine"]:::backend
        SHAP["🔍 SHAP Processor"]:::backend
        API["📡 REST / WebSocket API"]:::backend
    end

    subgraph "Digital Twin Dashboard"
        DR["🕹️ Control Room"]:::frontend
        AH["📊 Analytics"]:::frontend
        XAI["🔍 Explainability Tool"]:::frontend
    end

    M1 & M2 & Mn --> DA
    PQT --> DA
    DA --> NORM
    NORM --> INF
    INF --> SHAP
    SHAP --> API
    PKL & JSON -.-> INF
    API --> DR
    API --> AH
    API --> XAI
```

---

## 📡 API Specification (V5)

| Endpoint | Method | Description | Payload / Query |
| :--- | :--- | :--- | :--- |
| `/api/login` | `POST` | Authenticates user and issues JWT. | `{username, password}` |
| `/api/machines` | `GET` | Lists all available 200+ fleet assets. | - |
| `/api/status/{id}`| `GET` | Fetches real-time alert level and probability. | `machine_id` |
| `/api/control-room/{id}`| `GET/WS` | High-fidelity digital twin payload. | `time_window`, `future_window` |
| `/api/trend/{id}/{p}`| `GET` | Historic sensor trend + safety limits. | `machine_id`, `parameter` |
| `/api/audit/validation`| `GET` | Ground-truth verification results. | - |

---

## 🐋 Production Deployment (Docker)

> [!TIP]
> This project is containerized for professional, one-click deployment.

### ⛴️ Unified Deployment (Quickstart)
To launch the entire Senior Pro suite (Backend + Frontend + Nginx):
```bash
docker-compose up --build
```

### 📦 Individual Containers
- **Backend Only**: `docker build -t te-backend .`
- **Frontend Only**: `docker build -t te-frontend ./frontend`

---

## 🔧 Local Development & Execution

> [!IMPORTANT]
> Ensure you have **Node.js 18+** and **Python 3.12+** installed for local execution.

### 1️⃣ Automated Startup (Windows)
```powershell
./run-dev.ps1
```

### 2️⃣ Manual Startup
*   **Backend**: `uvicorn backend.api:app --reload`
*   **Frontend**: `cd frontend && npm run dev`

---

## 📂 Project Structure

- `backend/` - Core API, Normalization Engine, and Inference.
- `frontend/` - React Source, Visual Analytics, and XAI Dashboard.
- `models/` - Production Weights for Senior Pro model.
- `metrics/` - Fleet-wide thresholds and normalization calibration.

---
**Status:** ✅ 100% Finalized & Perfected | **Designed For:** TE Connectivity AI Cup 🏆
