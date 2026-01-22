# Scripts — Reproducible Pipeline (Microscopy Segmentation + Multi-view Probabilistic Clustering)

These scripts implement an end-to-end workflow:
generate synthetic microscopy → segment cells → extract features → fuse multi-view data → cluster with uncertainty → produce figures and metrics.

Run scripts in numeric order from the repo root.

---

## Pipeline overview (what each script produces)

### `01_generate_synthetic_microscopy.py`
Generates a synthetic microscopy image + ground-truth instance mask and a second “omics-like” barcode/projection view.  
**Outputs:** `data/raw/*.npy`, `data/raw/*.csv`, and `figures/synthetic_image_preview.png`

---

### `02_segment_cells.py`
Performs baseline segmentation using Otsu thresholding + watershed.  
**Outputs:** `data/processed/seg_mask.npy` and `figures/segmentation_overlay.png`

---

### `03_extract_features.py`
Extracts morphology/intensity features from segmented regions and maps regions to ground-truth cell IDs for evaluation and barcode joining.  
**Outputs:** `data/processed/morph_features.csv` and `data/processed/mapped_cells.csv`

---

### `04_build_multiview.py`
Assembles the final multi-view dataset (view 1 + view 2).  
**Output:** `reports/cell_table.csv`

---

### `05_multiview_clustering.py`
Runs probabilistic clustering (GMM) on fused features and reports uncertainty via posterior confidence (smoothed for demo stability).  
**Outputs:** `reports/cluster_assignments.csv`, `reports/cluster_summary.csv`, and `reports/metrics.json`

---

### `06_visualise_results.py`
Creates the key plots (PCA, confusion matrix) and updates metrics with segmentation overlap results.  
**Outputs:** `figures/pca_clusters.png`, `figures/confusion_matrix.png`, and updated `reports/metrics.json`

---

## Run all steps
```bash
pip install -r requirements.txt

python scripts/01_generate_synthetic_microscopy.py
python scripts/02_segment_cells.py
python scripts/03_extract_features.py
python scripts/04_build_multiview.py
python scripts/05_multiview_clustering.py
python scripts/06_visualise_results.py

