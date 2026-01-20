# Microscopy Segmentation + Probabilistic Multi-view Clustering (BARseq-inspired Demo)

## Overview
This repository demonstrates two core building blocks relevant to barcoded-anatomy pipelines:
1) **cell segmentation** from microscopy images, and
2) **probabilistic multi-view clustering** to integrate complementary feature spaces.

The aim is to provide a compact, reproducible portfolio project that mirrors the technical foundations used in BARseq-style analysis.

## Data
- Public microscopy segmentation dataset (e.g., Data Science Bowl 2018 nuclei, or BBBC).
- Optional: synthetic “barcode/channel” features generated per segmented cell to form a second view.

## Methods
1. **Segmentation**
   - baseline Cellpose/U-Net
   - evaluation: IoU/Dice, qualitative overlays
2. **Feature extraction (View 1)**
   - morphology: area, eccentricity, intensity stats, texture
3. **Second view (View 2)**
   - simulated channel/barcode-like signals (or real multi-channel features if dataset provides them)
4. **Probabilistic multi-view clustering**
   - mixture models / variational clustering to group cell populations across views

## Outputs (Results you will generate)
- `figures/segmentation_overlays.png`
- `figures/segmentation_metrics.png`
- `figures/clusters_by_view.png`
- `reports/cluster_summary.json`

## Run
```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py
python scripts/02_segment_cells.py
python scripts/03_extract_features.py
python scripts/04_multiview_clustering.py
python scripts/05_visualise_results.py
```
## Future scope

1. Replace simulated second view with real multi-omic/multi-channel measurements (when available).
2. Compare clustering stability and uncertainty across cohorts.
3. Extend to joint embedding + clustering with uncertainty quantification.

