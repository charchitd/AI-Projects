import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils import seed_everything, EVENTS, SERVICE_ORDER

IN_PATH = os.path.join("data", "raw", "event_log.csv")
HEATMAP_OUT = os.path.join("figures", "transition_heatmap.png")
SANKEY_OUT = os.path.join("reports", "sankey_pathways.html")

def build_transition_matrix(df):
    counts = np.zeros((len(EVENTS), len(EVENTS)), dtype=int)
    for pid, g in df.groupby("patient_id"):
        ev = g.sort_values("timestamp")["event"].tolist()
        for a,b in zip(ev[:-1], ev[1:]):
            counts[SERVICE_ORDER[a], SERVICE_ORDER[b]] += 1
    return counts

def save_heatmap(mat):
    os.makedirs(os.path.dirname(HEATMAP_OUT), exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect='auto')
    plt.xticks(range(len(EVENTS)), EVENTS, rotation=45, ha='right')
    plt.yticks(range(len(EVENTS)), EVENTS)
    plt.title("Transition heatmap (synthetic unscheduled-care journeys)")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(HEATMAP_OUT, dpi=200)
    plt.close()

def save_sankey(mat):
    links = []
    for i,a in enumerate(EVENTS):
        for j,b in enumerate(EVENTS):
            if mat[i,j] > 0:
                links.append((a,b,int(mat[i,j])))
    links.sort(key=lambda x: x[2], reverse=True)
    top = links[:30]

    label = EVENTS
    src = [SERVICE_ORDER[a] for a,b,v in top]
    tgt = [SERVICE_ORDER[b] for a,b,v in top]
    val = [v for a,b,v in top]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(width=0.5), label=label),
        link=dict(source=src, target=tgt, value=val)
    )])
    fig.update_layout(title_text="Top unscheduled-care pathway flows (synthetic)", font_size=11)
    os.makedirs(os.path.dirname(SANKEY_OUT), exist_ok=True)
    fig.write_html(SANKEY_OUT)

def main():
    seed_everything(42)
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    mat = build_transition_matrix(df)
    save_heatmap(mat)
    save_sankey(mat)
    print(f"Saved: {HEATMAP_OUT}")
    print(f"Saved: {SANKEY_OUT}")

if __name__ == "__main__":
    main()
