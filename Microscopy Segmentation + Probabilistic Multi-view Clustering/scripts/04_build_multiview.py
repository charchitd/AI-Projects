import os
import pandas as pd
from utils import seed_everything

IN_MAPPED = os.path.join("data", "processed", "mapped_cells.csv")
OUT_CELL_TABLE = os.path.join("reports", "cell_table.csv")

def main():
    seed_everything(42)
    df = pd.read_csv(IN_MAPPED)

    # View 1: morphology/intensity features
    morph_cols = ["area","eccentricity","perimeter","solidity","mean_intensity","max_intensity"]
    # View 2: barcodes/projections
    view2_cols = [c for c in df.columns if c.startswith("bar_") or c.startswith("prj_")]

    keep = ["seg_id","gt_cell_id","true_type"] + morph_cols + view2_cols
    out = df[keep].copy()

    os.makedirs(os.path.dirname(OUT_CELL_TABLE), exist_ok=True)
    out.to_csv(OUT_CELL_TABLE, index=False)

    print(f"Saved multi-view dataset: {OUT_CELL_TABLE} ({out.shape[0]:,} x {out.shape[1]:,})")
    print(f"View1 cols: {len(morph_cols)} | View2 cols: {len(view2_cols)}")

if __name__ == "__main__":
    main()
