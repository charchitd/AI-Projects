
# Scripts — Reproducible Pipeline (Patient Journey Analytics: Unscheduled Care)

These scripts implement an end-to-end, **reproducible** workflow: generate a shareable event log → build journey sequences → visualise pathways → discover journey types → predict admission with calibrated probabilities.

Run scripts in numeric order from the repo root.

---

## Pipeline overview (what each script produces)

### `01_generate_synthetic_ehr.py`
Creates a **synthetic unscheduled-care event log** (shareable, no patient data).  
**Output:** structured event dataset saved under `data/` for downstream steps.

### `02_build_sequences.py`
Transforms event logs into **journey sequences** and derives early-pathway features (timing, event counts, transitions).  
**Output:** modelling table used for clustering and admission prediction.

### `03_visualize_pathways.py`
Builds pathway visualisations for interpretability:
- interactive Sankey flow (`reports/sankey_pathways.html`)
- transition heatmap (`figures/transition_heatmap.png`)  
**Outcome:** rapid understanding of dominant flows and bottlenecks.

### `04_cluster_journeys.py`
Groups journeys into **typologies** (clusters) using engineered pathway features.  
**Output:** `reports/cluster_summary.csv`, cluster plots in `figures/`.

### `05_predict_outcomes.py`
Trains baseline models to predict **admission risk** from early signals and evaluates:
- discrimination (ROC/AUROC)
- calibration (probability trustworthiness)  
**Output:** `figures/roc_admission.png`, `figures/calibration_admission.png`, and `reports/metrics.json`.

---

## Run all steps
```bash
python scripts/01_generate_synthetic_ehr.py
python scripts/02_build_sequences.py
python scripts/03_visualize_pathways.py
python scripts/04_cluster_journeys.py
python scripts/05_predict_outcomes.py

