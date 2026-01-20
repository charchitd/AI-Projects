
# Interpretable Navigational Biomarkers from Trajectory Data (Self-Supervised + XAI)

## Overview
This mini-project builds an interpretable modelling pipeline for **navigation trajectories**. It is inspired by VR-based navigation tasks used in dementia research, where the aim is to derive **navigational biomarkers** of cognitive change and link them to clinically meaningful outcomes.

## Data
- Public GPS/trajectory dataset (e.g., Microsoft Geolife) or a lightweight trajectory dataset.
- Each trajectory is segmented into fixed-length sequences for modelling.

## Approach
1. **Trajectory biomarkers** (interpretable features)
   - speed variability, stop frequency, turning-angle entropy, path efficiency, revisit rate.
2. **Self-supervised learning**
   - contrastive learning on trajectory segments to learn robust embeddings.
3. **Outcome modelling**
   - predict a proxy “navigation difficulty / progression score” (or supervised label if available).
4. **Explainability**
   - feature attribution for the predictor + sensitivity analysis for biomarker stability.

## Outputs (Results you will generate)
- `figures/biomarker_distributions.png`
- `figures/embedding_clusters.png`
- `figures/explainability_top_features.png`
- `reports/metrics.json` (RMSE/AUROC depending on task)

## Run
```bash
pip install -r requirements.txt
python scripts/01_download_or_prepare_data.py
python scripts/02_extract_biomarkers.py
python scripts/03_self_supervised_train.py
python scripts/04_train_predictor.py
python scripts/05_explainability.py
```
## Future Scope

1. Replace proxy outcomes with labelled cognitive/affect targets (where accessible).
2. Extend to multimodal inputs (trajectory + gaze/interaction logs).
3. Evaluate generalisation across sessions (longitudinal consistency).

