import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils import seed_everything

OUT_PATH = os.path.join("data", "raw", "event_log.csv")

def sample_demographics(n, rng):
    age = rng.integers(18, 92, size=n)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    deprivation = rng.integers(1, 6, size=n)  # synthetic proxy
    return age, sex, deprivation

def generate_journey(rng, patient_id, base_time, age, deprivation):
    # Latent severity drives journey complexity and admission probability
    severity = np.clip(rng.normal(loc=0.0, scale=1.0), -2.5, 2.5)
    severity += 0.015*(age-50) + 0.12*(deprivation-3)

    events = ["ED_ARRIVAL", "TRIAGE", "VITALS"]

    p_labs = 1/(1+np.exp(-(severity-0.1)))
    p_img  = 1/(1+np.exp(-(severity-0.3)))
    p_obs  = 1/(1+np.exp(-(severity-0.2)))
    p_spec = 1/(1+np.exp(-(severity-0.6)))

    if rng.random() < p_labs: events.append("LABS")
    if rng.random() < p_img: events.append("IMAGING")
    events.append("TREATMENT")
    if rng.random() < p_obs: events.append("OBSERVATION")
    if rng.random() < p_spec: events.append("SPECIALIST_REVIEW")

    p_admit = 1/(1+np.exp(-(severity-0.4)))
    admitted = int(rng.random() < p_admit)
    events.append("ADMIT" if admitted else "DISCHARGE")

    t = base_time
    rows = []
    for i, e in enumerate(events):
        if i == 0:
            gap = 0
        else:
            base_gap = rng.integers(8, 35)
            complexity = 1 + 0.25*max(0, severity)
            gap = int(base_gap * complexity)
        t = t + timedelta(minutes=gap)
        rows.append({
            "patient_id": patient_id,
            "timestamp": t.isoformat(),
            "event": e,
            "age": int(age),
            "sex": None,
            "deprivation": int(deprivation),
            "latent_severity": float(severity),
        })

    wait_time = int((pd.to_datetime(rows[1]["timestamp"]) - pd.to_datetime(rows[0]["timestamp"])).total_seconds() / 60)
    los = int((pd.to_datetime(rows[-1]["timestamp"]) - pd.to_datetime(rows[0]["timestamp"])).total_seconds() / 60)

    for r in rows:
        r["admitted"] = admitted
        r["wait_time_min"] = wait_time
        r["los_min"] = los

    return rows

def main():
    seed_everything(42)
    rng = np.random.default_rng(42)

    n_patients = 1200
    age, sex, deprivation = sample_demographics(n_patients, rng)

    start = datetime(2024, 1, 1, 8, 0, 0)
    all_rows = []
    for i in range(n_patients):
        base_time = start + timedelta(days=int(rng.integers(0, 180)), minutes=int(rng.integers(0, 24*60)))
        rows = generate_journey(rng, f"P{i:05d}", base_time, age[i], deprivation[i])
        for r in rows:
            r["sex"] = sex[i]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(["patient_id", "timestamp"], inplace=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved synthetic event log: {OUT_PATH} ({len(df):,} rows, {df['patient_id'].nunique():,} patients)")

if __name__ == "__main__":
    main()
