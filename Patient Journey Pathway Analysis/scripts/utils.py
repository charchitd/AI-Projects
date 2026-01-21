import numpy as np
import pandas as pd

EVENTS = [
    "ED_ARRIVAL",
    "TRIAGE",
    "VITALS",
    "LABS",
    "IMAGING",
    "TREATMENT",
    "OBSERVATION",
    "SPECIALIST_REVIEW",
    "DISCHARGE",
    "ADMIT"
]

SERVICE_ORDER = {e:i for i,e in enumerate(EVENTS)}

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)
