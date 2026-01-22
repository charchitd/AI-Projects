import numpy as np

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    x = x - np.min(x)
    d = np.max(x) - np.min(x)
    return x / (d + 1e-12)
