import os
import numpy as np
import pandas as pd
from collections import Counter
from utils import seed_everything

IN_PATH = os.path.join("data", "raw", "event_log.csv")
OUT_SEQ = os.path.join("data", "processed", "journeys.csv")
OUT_FEAT = os.path.join("reports", "journey_features.csv")

EVENT_VOCAB = ["ED_ARRIVAL","TRIAGE","VITALS","LABS","IMAGING","TREATMENT","OBSERVATION","SPECIALIST_REVIEW","DISCHARGE","ADMIT"]

def transition_counts(events):
    c = Counter()
    for a,b in zip(events[:-1], events[1:]):
        c[f"{a}->{b}"] += 1
    return c

def main():
    seed_everything(42)
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df.sort_values(["patient_id","timestamp"], inplace=True)

    journeys = []
    features = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("timestamp")
        ev = g["event"].tolist()
        admitted = int(g["admitted"].iloc[0])
        age = int(g["age"].iloc[0])
        sex = g["sex"].iloc[0]
        deprivation = int(g["deprivation"].iloc[0])

        t0 = g["timestamp"].iloc[0]
        t_end = g["timestamp"].iloc[-1]
        los = int((t_end - t0).total_seconds() / 60)
        wait = int(g["wait_time_min"].iloc[0])

        journeys.append({
            "patient_id": pid,
            "events": " | ".join(ev),
            "n_events": len(ev),
            "admitted": admitted,
            "age": age,
            "sex": sex,
            "deprivation": deprivation,
            "wait_time_min": wait,
            "los_min": los
        })

        trans = transition_counts(ev)
        feat = {
            "patient_id": pid,
            "admitted": admitted,
            "age": age,
            "sex_M": 1 if sex == "M" else 0,
            "deprivation": deprivation,
            "n_events": len(ev),
            "wait_time_min": wait,
            "los_min": los,
        }

        first_k = 5
        early = ev[:min(first_k, len(ev))]
        for i in range(first_k):
            feat[f"early_event_{i+1}"] = early[i] if i < len(early) else "NONE"

        for i in range(first_k):
            for token in (EVENT_VOCAB + ["NONE"]):
                feat[f"early{i+1}_{token}"] = 1 if feat[f"early_event_{i+1}"] == token else 0

        for k,v in trans.items():
            feat[f"trans_{k}"] = v

        features.append(feat)

    jdf = pd.DataFrame(journeys)
    fdf = pd.DataFrame(features).fillna(0)

    os.makedirs(os.path.dirname(OUT_SEQ), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_FEAT), exist_ok=True)
    jdf.to_csv(OUT_SEQ, index=False)
    fdf.to_csv(OUT_FEAT, index=False)

    print(f"Saved journeys: {OUT_SEQ} ({len(jdf):,})")
    print(f"Saved features: {OUT_FEAT} ({fdf.shape[0]:,} x {fdf.shape[1]:,})")

if __name__ == "__main__":
    main()
