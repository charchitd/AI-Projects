# Patient Journey Pathway Analytics (Synthetic FHIR EHR)

## Overview
This mini-project demonstrates an end-to-end workflow for analysing **longitudinal patient journeys** using a synthetic, FHIR-style EHR dataset. The goal is to reproduce the core methodological pattern used in unscheduled-care research: **pathway visualisation**, **journey clustering**, and **predictive modelling** for downstream decision support.

## Why this project
Healthcare systems often need to understand how patients move across services, where bottlenecks occur, and what drives adverse outcomes. This repository provides a reproducible template for:
- mapping end-to-end pathways,
- identifying common journey “types” via clustering,
- predicting outcomes (e.g., admission risk / long waits) from early pathway signals.


## Repo structure

1. `data/` (synthetic raw + processed)
2. `scripts/` (pipeline steps)
3. `figures/` (saved plots)
4. `reports/` (metrics and summaries)

## Data
- Synthetic EHR generated with **Synthea** (FHIR-like JSON bundles).
- Encounters are converted into an event log (patient_id, timestamp, service_type, outcome).

> Synthetic data is used to ensure the repo is shareable and fully reproducible.

## Methods (high-level)
1. **Pre-processing**: parse FHIR bundles → ordered event sequences per patient.
2. **Pathway visualisation**: Sankey/network plots of service transitions.
3. **Clustering**: sequence feature extraction + k-medoids / hierarchical clustering.
4. **Prediction**: baseline logistic regression + tree model to predict an outcome (AUROC, calibration).
5. **Reporting**: saved figures + metrics JSON for review.

## Results (Outputs you will generate)
- `figures/sankey_pathways.png` – pathway flow visualisation
- `figures/cluster_profiles.png` – top journey clusters and characteristics
- `reports/metrics.json` – AUROC + calibration + cluster statistics

## How to run
```bash
pip install -r requirements.txt
python scripts/01_generate_or_load_data.py
python scripts/02_build_event_log.py
python scripts/03_visualise_pathways.py
python scripts/04_cluster_journeys.py
python scripts/05_predict_outcomes.py
```
## Future scope

1. Add fairness/equity slices (performance across demographic groups).
2. Replace synthetic records with an approved de-identified dataset (if available).
3. Extend to time-to-event modelling and causal pathway comparisons.

