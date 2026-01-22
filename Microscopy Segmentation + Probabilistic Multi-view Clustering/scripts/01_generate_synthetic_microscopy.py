import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.util import random_noise
from utils import seed_everything, normalize01

OUT_IMG = os.path.join("data", "raw", "microscopy_image.npy")
OUT_GT  = os.path.join("data", "raw", "gt_mask.npy")
OUT_META = os.path.join("data", "raw", "cell_metadata.csv")

# "BARseq-like" second view (synthetic)
OUT_BARCODES = os.path.join("data", "raw", "barcode_view.csv")

def main():
    seed_everything(42)
    rng = np.random.default_rng(42)

    H, W = 256, 256
    n_cells = 180
    n_types = 4  # synthetic "cell types"
    barcode_dim = 12
    proj_dim = 8

    # base background with gentle gradient
    yy, xx = np.mgrid[0:H, 0:W]
    background = 0.15 + 0.15*(xx/W) + 0.10*(yy/H)

    img = background.copy()
    gt = np.zeros((H, W), dtype=np.int32)

    rows = []
    barcode_rows = []

    # cell-type prototypes for barcode/projection
    type_proto_bar = 1.0 * rng.normal(0, 1, size=(n_types, barcode_dim))
    type_proto_prj = 1.0 * rng.normal(0, 1, size=(n_types, proj_dim))

    # place cells (avoid borders)
    centers = []
    for cid in range(1, n_cells+1):
        for _ in range(200):
            cy = int(rng.integers(12, H-12))
            cx = int(rng.integers(12, W-12))
            if all((cy-y)**2 + (cx-x)**2 > 10**2 for y,x in centers):
                centers.append((cy,cx))
                break

        ctype = int(rng.integers(0, n_types))
        radius = int(rng.integers(5, 12))
        intensity = float(rng.uniform(0.45, 0.95)) + 0.08*ctype

        rr, cc = disk((cy, cx), radius=radius, shape=img.shape)

        # add a bright nucleus-like blob (soft)
        img[rr, cc] += intensity

        # write GT label
        gt[rr, cc] = cid

        rows.append({
            "cell_id": cid,
            "true_type": ctype,
            "cy": cy,
            "cx": cx,
            "radius": radius,
            "intensity": intensity
        })

        # barcode/projection view per cell (type + noise)
        bar = type_proto_bar[ctype] + rng.normal(0, 0.8, size=barcode_dim)
        prj = type_proto_prj[ctype] + rng.normal(0, 0.8, size=proj_dim)
        br = {"cell_id": cid, "true_type": ctype}
        for j in range(barcode_dim):
            br[f"bar_{j+1}"] = float(bar[j])
        for j in range(proj_dim):
            br[f"prj_{j+1}"] = float(prj[j])
        barcode_rows.append(br)

    # add noise + normalize to [0,1]
    img = random_noise(img, mode="gaussian", var=0.01)
    img = normalize01(img)

    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    np.save(OUT_IMG, img.astype(np.float32))
    np.save(OUT_GT, gt)

    pd.DataFrame(rows).to_csv(OUT_META, index=False)
    pd.DataFrame(barcode_rows).to_csv(OUT_BARCODES, index=False)

    # save a quick preview
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title("Synthetic microscopy image (preview)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("figures/synthetic_image_preview.png", dpi=200)
    plt.close()

    print(f"Saved: {OUT_IMG}, {OUT_GT}, {OUT_META}, {OUT_BARCODES}")
    print("Preview: figures/synthetic_image_preview.png")

if __name__ == "__main__":
    main()
