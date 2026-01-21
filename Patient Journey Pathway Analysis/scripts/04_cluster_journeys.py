import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import seed_everything

FEAT_PATH = os.path.join("reports","journey_features.csv")
OUT_ASSIGN = os.path.join("reports","journey_clusters.csv")
OUT_SUM = os.path.join("reports","cluster_summary.csv")
FIG_OUT = os.path.join("figures","cluster_profiles.png")

def main():
    seed_everything(42)
    df = pd.read_csv(FEAT_PATH)

    cols = [c for c in df.columns if c.startswith("trans_") or (c.startswith("early") and not c.startswith("early_event_"))]
    cols += ["age","deprivation","n_events","wait_time_min"]
    X = df[cols].values

    scaler = StandardScaler(with_mean=False) if np.any(X==0) else StandardScaler()
    Xs = scaler.fit_transform(X)

    k = 5
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)

    df_out = df[["patient_id","admitted","age","deprivation","n_events","wait_time_min","los_min"]].copy()
    df_out["cluster"] = labels
    df_out.to_csv(OUT_ASSIGN, index=False)

    summ = df_out.groupby("cluster").agg(
        n=("patient_id","count"),
        admit_rate=("admitted","mean"),
        avg_events=("n_events","mean"),
        avg_wait=("wait_time_min","mean"),
        avg_los=("los_min","mean"),
        avg_age=("age","mean"),
        avg_deprivation=("deprivation","mean"),
    ).reset_index()
    summ.to_csv(OUT_SUM, index=False)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.figure(figsize=(10,6))
    x = np.arange(k)
    plt.bar(x-0.2, summ["admit_rate"], width=0.4, label="admission rate")
    plt.bar(x+0.2, (summ["avg_wait"]/summ["avg_wait"].max()).fillna(0), width=0.4, label="avg wait (norm)")
    plt.xticks(x, [f"C{c}" for c in summ["cluster"]])
    plt.ylim(0,1.05)
    plt.title("Journey typologies (clusters): admission vs wait time (normalised)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=200)
    plt.close()

    print(f"Saved assignments: {OUT_ASSIGN}")
    print(f"Saved summary: {OUT_SUM}")
    print(f"Saved figure: {FIG_OUT}")

if __name__ == "__main__":
    main()
