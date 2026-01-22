# Reports — Outputs & Results (Patient Journey Analytics: Unscheduled Care)

This folder contains the **machine-readable outputs** produced by the pipeline (tables + metrics + interactive HTML). These artefacts support reproducible analysis and make it easy to cite results in a CV, statement, or write-up.

---

## Key outputs

### `metrics.json`
**Primary model results and trustworthiness metrics**, including:
- AUROC (Logistic Regression, Gradient Boosting)
- Brier score / calibration summaries (ECE-style)
Use this file to quote headline performance numbers consistently.

### `cluster_summary.csv`
**Cluster-level journey typologies**, summarising each cluster’s:
- size (n)
- average confidence / stability measure (if applicable)
- dominant pathway characteristics (derived from engineered features)

### `cluster_assignments.csv` (if present)
Row-level mapping of each patient journey to a **cluster label**, used for:
- cohort comparisons
- downstream modelling
- operational interpretation of journey types

### `sankey_pathways.html`
Interactive **pathway flow visualisation** showing the most frequent transitions and major patient flows through the unscheduled-care process.

---

## How to regenerate
Run the pipeline end-to-end from the repo root:

```bash
python scripts/01_generate_synthetic_ehr.py
python scripts/02_build_sequences.py
python scripts/03_visualize_pathways.py
python scripts/04_cluster_journeys.py
python scripts/05_predict_outcomes.py

