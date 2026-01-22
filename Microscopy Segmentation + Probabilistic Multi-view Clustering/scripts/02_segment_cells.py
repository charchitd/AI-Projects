import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import label
from utils import seed_everything

IN_IMG = os.path.join("data", "raw", "microscopy_image.npy")
IN_GT  = os.path.join("data", "raw", "gt_mask.npy")

OUT_MASK = os.path.join("data", "processed", "seg_mask.npy")
OUT_OVERLAY = os.path.join("figures", "segmentation_overlay.png")

def main():
    seed_everything(42)
    img = np.load(IN_IMG)
    gt = np.load(IN_GT)

    # smooth and threshold
    smooth = gaussian(img, sigma=1.0, preserve_range=True)
    thr = threshold_otsu(smooth)
    bw = smooth > thr

    # cleanup
    bw = binary_opening(bw, footprint=disk(2))
    bw = remove_small_objects(bw, min_size=40)

    # distance + watershed
    dist = ndi.distance_transform_edt(bw)
    # local maxima as markers (simple)
    markers = label(dist > np.percentile(dist[bw], 75))
    seg = watershed(-dist, markers, mask=bw)

    os.makedirs(os.path.dirname(OUT_MASK), exist_ok=True)
    np.save(OUT_MASK, seg.astype(np.int32))

    # overlay plot
    os.makedirs("figures", exist_ok=True)
    boundaries = find_boundaries(seg, mode="outer")
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.imshow(np.ma.masked_where(~boundaries, boundaries), cmap="autumn", alpha=0.9)
    plt.title("Segmentation overlay (baseline)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_OVERLAY, dpi=200)
    plt.close()

    print(f"Saved segmentation: {OUT_MASK}")
    print(f"Saved overlay: {OUT_OVERLAY}")

if __name__ == "__main__":
    main()
