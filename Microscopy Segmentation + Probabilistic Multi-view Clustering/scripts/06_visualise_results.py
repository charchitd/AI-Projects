import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from utils import seed_everything, normalize01

IN_IMG = os.path.join("data", "raw", "microscopy_image.npy")
IN_GT  = os.path.join("data", "raw", "gt_mask.npy")
IN_SEG = os.path.join("data", "processed", "seg_mask.npy")

IN_CELL = os.path.join("reports", "cell_table.csv")
IN_ASSIGN = os.path.join("reports", "cluster_assignments.csv")
METRICS_PATH = os.path.join("reports", "metrics.json")

FIG_PCA = os.path.join("figures", "pca_clusters.png")
FIG_CM  = os.path.join("figures", "confusion_matrix.png")
FIG_TYPE = os.path.join("figures", "true_type_pca.png")

def dice_iou_binary(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = (a & b).sum()
    da = a.sum()
    db = b.sum()
    dice = (2*inter) / (da + db + 1e-12)
    iou = inter / ((da + db - inter) + 1e-12)
    return float(dice), float(iou)

def plot_confusion(cm, out_path, title):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted cluster")
    plt.ylabel("True type")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    seed_everything(42)
    os.makedirs("figures", exist_ok=True)

    img = np.load(IN_IMG)
    gt = np.load(IN_GT)
    seg = np.load(IN_SEG)

    # segmentation quality (binary dice/iou against GT union)
    dice, iou = dice_iou_binary(seg, gt)

    # load tables
    cell = pd.read_csv(IN_CELL)
    assign = pd.read_csv(IN_ASSIGN)

    merged = cell.merge(assign[["seg_id","cluster","cluster_confidence"]], on="seg_id", how="left")
    y_true = merged["true_type"].astype(int).values
    y_pred = merged["cluster"].astype(int).values

    # PCA plot (features)
    feature_cols = [c for c in merged.columns if c.startswith(("area","eccentricity","perimeter","solidity","mean_intensity","max_intensity","bar_","prj_"))]
    X = merged[feature_cols].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    # plot predicted clusters
    plt.figure(figsize=(6,5))
    for c in sorted(np.unique(y_pred)):
        m = y_pred == c
        plt.scatter(Z[m,0], Z[m,1], s=18, alpha=0.8, label=f"C{c}")
    plt.title("PCA of multi-view features (coloured by predicted cluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_PCA, dpi=200)
    plt.close()

    # plot true types
    plt.figure(figsize=(6,5))
    for t in sorted(np.unique(y_true)):
        m = y_true == t
        plt.scatter(Z[m,0], Z[m,1], s=18, alpha=0.8, label=f"T{t}")
    plt.title("PCA of multi-view features (coloured by true type)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_TYPE, dpi=200)
    plt.close()

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, FIG_CM, "Confusion matrix (true type vs predicted cluster)")

    # update metrics.json
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    metrics.update({
        "segmentation_binary_dice": dice,
        "segmentation_binary_iou": iou,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    })

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {FIG_PCA}, {FIG_TYPE}, {FIG_CM}")
    print(f"Updated metrics: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
