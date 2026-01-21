import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import seed_everything

FEAT_PATH = os.path.join("reports","journey_features.csv")
METRICS_OUT = os.path.join("reports","metrics.json")
ROC_OUT = os.path.join("figures","roc_admission.png")
CAL_OUT = os.path.join("figures","calibration_admission.png")

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum()/len(y_true)) * abs(acc - conf)
    return float(ece)

def main():
    seed_everything(42)
    df = pd.read_csv(FEAT_PATH)
    y = df["admitted"].values

    cols = [c for c in df.columns if c.startswith("early") and not c.startswith("early_event_")] + ["age","deprivation","wait_time_min","n_events"]
    X = df[cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False) if np.any(X==0) else StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    gb = GradientBoostingClassifier(random_state=42)

    lr.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    p_lr = lr.predict_proba(X_test)[:,1]
    p_gb = gb.predict_proba(X_test)[:,1]

    auc_lr = roc_auc_score(y_test, p_lr)
    auc_gb = roc_auc_score(y_test, p_gb)

    brier_lr = brier_score_loss(y_test, p_lr)
    brier_gb = brier_score_loss(y_test, p_gb)

    ece_lr = ece_score(y_test, p_lr)
    ece_gb = ece_score(y_test, p_gb)

    os.makedirs(os.path.dirname(ROC_OUT), exist_ok=True)
    plt.figure(figsize=(7,5))
    fpr, tpr, _ = roc_curve(y_test, p_lr)
    plt.plot(fpr, tpr, label=f"LogReg (AUROC={auc_lr:.3f})")
    fpr, tpr, _ = roc_curve(y_test, p_gb)
    plt.plot(fpr, tpr, label=f"GradBoost (AUROC={auc_gb:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Admission prediction from early pathway signals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_OUT, dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    for name, p in [("LogReg", p_lr), ("GradBoost", p_gb)]:
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker='o', label=name)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (admission)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CAL_OUT, dpi=200)
    plt.close()

    metrics = {
        "n_patients": int(df.shape[0]),
        "test_size": int(len(y_test)),
        "models": {
            "logistic_regression": {"auroc": float(auc_lr), "brier": float(brier_lr), "ece": ece_lr},
            "gradient_boosting": {"auroc": float(auc_gb), "brier": float(brier_gb), "ece": ece_gb},
        }
    }

    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved ROC: {ROC_OUT}")
    print(f"Saved calibration: {CAL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
