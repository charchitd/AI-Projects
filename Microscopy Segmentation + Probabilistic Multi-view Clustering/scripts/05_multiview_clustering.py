import os
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from utils import seed_everything

IN_CELL = os.path.join("reports", "cell_table.csv")
OUT_ASSIGN = os.path.join("reports", "cluster_assignments.csv")
OUT_SUM = os.path.join("reports", "cluster_summary.csv")
OUT_METRICS = os.path.join("reports", "metrics.json")

def temper_probs(probs: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    # Temperature-smooth responsibilities to avoid overconfident posteriors.
    # This is a practical calibration-style trick for demo settings.
    p = np.clip(probs, 1e-12, 1.0)
    p = p ** (1.0 / temperature)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    return p

def main():
    seed_everything(42)
    df = pd.read_csv(IN_CELL)

    y_true = df["true_type"].astype(int).values
    feature_cols = [c for c in df.columns if c not in ["seg_id","gt_cell_id","true_type"]]
    X = df[feature_cols].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(n_components=4, covariance_type="full", reg_covar=5e-2, random_state=42))
    ])
    model.fit(X)

    Xs = model.named_steps["scaler"].transform(X)
    probs_raw = model.named_steps["gmm"].predict_proba(Xs)

    temperature = 10.0
    probs = temper_probs(probs_raw, temperature=temperature)

    y_pred = probs.argmax(axis=1)
    maxp = probs.max(axis=1)

    ari = adjusted_rand_score(y_true, y_pred)

    out = df[["seg_id","gt_cell_id","true_type"]].copy()
    out["cluster"] = y_pred
    out["cluster_confidence"] = maxp
    out.to_csv(OUT_ASSIGN, index=False)

    summ = out.groupby("cluster").agg(
        n=("seg_id","count"),
        avg_conf=("cluster_confidence","mean"),
        dominant_true_type=("true_type", lambda x: int(pd.Series(x).value_counts().index[0])),
    ).reset_index()
    summ.to_csv(OUT_SUM, index=False)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "n_cells": int(len(df)),
        "ari_vs_true_type": float(ari),
        "posterior_temperature": float(temperature),
        "mean_max_posterior": float(maxp.mean()),
        "max_posterior_quantiles": {
            "p10": float(np.quantile(maxp, 0.10)),
            "p50": float(np.quantile(maxp, 0.50)),
            "p90": float(np.quantile(maxp, 0.90)),
        },
        "confusion_matrix": cm.tolist(),
    }

    if os.path.exists(OUT_METRICS):
        with open(OUT_METRICS, "r") as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing

    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved assignments: {OUT_ASSIGN}")
    print(f"Saved summary: {OUT_SUM}")
    print(f"Updated metrics: {OUT_METRICS}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
