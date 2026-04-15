# 🤝 Project Collaboration Handbook: Atharva & Vishnu

Welcome to the **Senior Pro** development workflow. To ensure both Atharva and Vishnu can collaborate without conflicts and maintain a professional production environment, follow these protocols.

---

## 🔄 The "Sync Loop" Protocol

Always follow this ritual before starting work and after pushing changes to ensure both sides are updated.

### 1. Daily Pull & Sync
Before adding new code, ensure your local environment matches the remote state:
```bash
git pull origin main
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Side-by-Side Commits
If both **Atharva** and **Vishnu** are working on different features, use branches:
- `feature/atharva-ui-updates`
- `feature/vishnu-ml-calibration`

Merge them back to `main` via a **Pull Request** to allow the Automated CI to verify the merge.

---

## 💾 Data Synchronization (Critical)

> [!IMPORTANT]
> Large `.parquet` and `.csv` files are NOT tracked by Git. If one user generates a new "Mergedata" set, they must share the file path or upload it via the **Management Dashboard Ingestion Tool**.

### Handoff Steps:
1. Generate new processed data in `new_processed_data/`.
2. Notify the other user.
3. The other user runs the `ingest` command or puts the file in their local folder.

---

## ⚖️ Quality Standards

As **Senior Developers**, you must ensure:
1. **Tests Pass**: Run `pytest` locally before pushing.
2. **API Parity**: Any changes to `backend/api.py` must be reflected in the Frontend `apiClient`.
3. **No Bloat**: Never commit `node_modules/`, `.venv/`, or large raw datasets.

---

## 🚀 Deployment Ritual

When code is ready for "Vishnu Check" or "Atharva Validation":
1. Push to `main`.
2. GitHub Actions will automatically Run Smoke Tests.
3. If the ✅ badge appears in the repo, the build is stable.

---
**Status:** 📊 Operational | **Role**: Senior Technical Lead Guidance
