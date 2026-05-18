"""Dirichlet-based non-IID partitioning."""
from __future__ import annotations

import numpy as np
import pandas as pd


def partition_dirichlet(expression, labels, num_clients=10, alpha=0.5,
                        min_samples=10, seed=42):
    """Split samples across num_clients using Dirichlet(alpha) label skew."""
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    x_all = expression.values.astype(np.float32)
    sample_ids = expression.index.tolist()

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(class_idx)).astype(int)
        remainder = len(class_idx) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1

        start = 0
        for k in range(num_clients):
            end = start + counts[k]
            client_indices[k].extend(class_idx[start:end].tolist())
            start = end

    clients = []
    for k in range(num_clients):
        idx = np.array(client_indices[k])
        if len(idx) < min_samples:
            continue

        clients.append({
            "center_id": f"client_{k:03d}",
            "tss_codes": [],
            "x": x_all[idx],
            "y": labels[idx].astype(np.int64),
            "sample_ids": [sample_ids[i] for i in idx],
        })

    clients.sort(key=lambda c: len(c["y"]), reverse=True)
    print(f"  {len(clients)} Dirichlet clients (alpha={alpha})")
    return clients
