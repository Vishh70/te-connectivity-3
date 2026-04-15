# EDA Report

## Scope
- This report summarizes the current training master, the FEB evaluation snapshot, and the control-room model contract.
- The historical raw batches began with M231 and M356, while the current merged master and dashboard flow cover all five machines: M231, M356, M471, M607, and M612.

## Processed Master Summary
- Rows: `1,250,000`
- Columns: `66`
- Date range: `2025-10-01 02:01:18+00:00 to 2026-01-12 01:00:15+00:00`
- ROC features: `42`
- Dead columns: `Cyl_tmp_z2`, `Cyl_tmp_z6`, `Cyl_tmp_z7`, `Cyl_tmp_z2_roc_5`, `Cyl_tmp_z2_roc_30`, `Cyl_tmp_z6_roc_5`, `Cyl_tmp_z6_roc_30`, `Cyl_tmp_z7_roc_5`, `Cyl_tmp_z7_roc_30`
- Rows per machine: `250,000` each for M231, M356, M471, M607, and M612
- Scrap rate per machine:
  - M231: `2.46%`
  - M356: `1.07%`
  - M471: `0.99%`
  - M607: `1.66%`
  - M612: `1.51%`

## Evaluation Snapshot
- Evaluation rows: `777006`
- Baseline precision / recall / AUC: `27.24% / 17.85% / 0.65997`
- Tuned precision / recall / F1 / AUC: `51.74% / 28.79% / 36.99% / 0.65997`
- Precision-target operating point: `55.01%` precision, `36.54%` recall
- Best per-machine tuned result: `M356` at `52.08%` precision and `39.85%` recall

## Model Reference
- Training feature count: `201`
- Label horizon: `15` minutes

## Notes
- The dashboard consumes root-cause payloads from `backend/data_access.py` through the control-room contract in `backend/api.py`.
- The repo is wired correctly; the remaining blocker is model ceiling and further feature-quality iteration, not frontend/backend connectivity.
