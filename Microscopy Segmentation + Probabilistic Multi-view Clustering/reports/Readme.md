# Reports — Outputs & Results (Microscopy Multi-view Clustering)

This folder contains the **final datasets**, clustering outputs, and **metrics** produced by the pipeline. These artefacts are intended to support reproducibility and make it easy to quote results in applications or a short write-up.

---

## Key outputs

### `metrics.json`
**Headline performance and uncertainty results**, including:
- segmentation overlap metrics (binary Dice / IoU vs ground truth union)
- clustering quality (ARI vs synthetic true type)
- posterior uncertainty summary (mean max posterior + quantiles)

Use this file to cite the key numbers in a consistent and reproducible way.

---

### `cell_table.csv`
The final **multi-view dataset** (one row per segmented region), containing:
- morphology + intensity features (view 1)
- barcode / projection features (view 2)
- the synthetic “true type” label (for evaluation only)

This is the main table used for clustering.

---

### `cluster_assignments.csv`
Row-level assignments produced by probabilistic clustering:
- predicted cluster per cell
- confidence score (posterior-derived)

Useful for cluster-level comparisons and downstream analysis.

---

### `cluster_summary.csv`
Cluster-wise statistics:
- cluster sizes
- average confidence
- dominant true type (for evaluation/interpretation)

---

## Reproducibility notes
- The pipeline is **deterministic by default** via fixed random seeds.
- Running scripts overwrites outputs in `data/`, `figures/`, and `reports/`.
- To preserve multiple runs, copy `reports/` to a timestamped folder before re-running.

