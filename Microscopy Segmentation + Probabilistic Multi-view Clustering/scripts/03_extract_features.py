import os
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from utils import seed_everything

IN_IMG = os.path.join("data", "raw", "microscopy_image.npy")
IN_GT  = os.path.join("data", "raw", "gt_mask.npy")
IN_SEG = os.path.join("data", "processed", "seg_mask.npy")
IN_BV  = os.path.join("data", "raw", "barcode_view.csv")

OUT_MORPH = os.path.join("data", "processed", "morph_features.csv")
OUT_MAPPED = os.path.join("data", "processed", "mapped_cells.csv")

def majority_gt_label(seg_region_pixels, gt):
    vals = gt[seg_region_pixels]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0
    return int(pd.Series(vals).value_counts().index[0])

def main():
    seed_everything(42)
    img = np.load(IN_IMG)
    gt = np.load(IN_GT)
    seg = np.load(IN_SEG)

    props = regionprops_table(
        seg,
        intensity_image=img,
        properties=[
            "label",
            "area",
            "eccentricity",
            "perimeter",
            "solidity",
            "mean_intensity",
            "max_intensity",
        ],
    )
    morph = pd.DataFrame(props).rename(columns={"label":"seg_id"})
    morph.to_csv(OUT_MORPH, index=False)

    # Map each seg region to GT cell_id via majority overlap
    # Build a pixel-index list per region by iterating over seg_id masks
    mapped = []
    for seg_id in morph["seg_id"].tolist():
        mask = seg == seg_id
        gt_id = int(pd.Series(gt[mask][gt[mask] > 0]).value_counts().index[0]) if np.any(gt[mask] > 0) else 0
        mapped.append({"seg_id": seg_id, "gt_cell_id": gt_id})

    map_df = pd.DataFrame(mapped)

    bv = pd.read_csv(IN_BV)
    out = morph.merge(map_df, on="seg_id", how="left").merge(bv, left_on="gt_cell_id", right_on="cell_id", how="left")

    # drop empty mappings
    out = out[out["gt_cell_id"] > 0].copy()
    out.to_csv(OUT_MAPPED, index=False)

    print(f"Saved morphology: {OUT_MORPH}")
    print(f"Saved mapped table: {OUT_MAPPED} ({len(out):,} segmented regions mapped to GT cells)")

if __name__ == "__main__":
    main()
