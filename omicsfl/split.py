"""Train/test/val splitting utilities."""
from __future__ import annotations

import numpy as np


def train_test_val_split(x, y, train_ratio=0.8, val_ratio=0.0, rng=None):
    """Split into train/test (and optionally val). Ensures >= 1 test sample."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(1, int(n * train_ratio))
    n_val = int(n * val_ratio) if val_ratio > 0 else 0

    # Ensure at least 1 test sample
    if n_train + n_val >= n:
        n_val = 0
        n_train = n - 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val] if n_val > 0 else np.array([], dtype=int)
    test_idx = indices[n_train + n_val:]

    if len(test_idx) == 0:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    splits = {
        "train": {
            "x": x[train_idx].astype(np.float32),
            "y": y[train_idx].astype(np.int64),
        },
        "test": {
            "x": x[test_idx].astype(np.float32),
            "y": y[test_idx].astype(np.int64),
        },
    }

    if n_val > 0 and len(val_idx) > 0:
        splits["val"] = {
            "x": x[val_idx].astype(np.float32),
            "y": y[val_idx].astype(np.int64),
        }

    return splits
