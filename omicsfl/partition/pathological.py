"""Pathological non-IID partitioning (McMahan et al. 2017)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def partition_pathological(expression, labels, num_clients=10,
                          classes_per_client=2, seed=42):
    """Sort by label, shard, and distribute so each client gets ~K classes."""
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    x_all = expression.values.astype(np.float32)
    sample_ids = expression.index.tolist()
    n = len(labels)

    num_shards = num_clients * classes_per_client
    sorted_idx = np.argsort(labels)
    shard_size = n // num_shards

    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size if i < num_shards - 1 else n
        shards.append(sorted_idx[start:end])

    shard_order = np.arange(num_shards)
    rng.shuffle(shard_order)

    clients = []
    for k in range(num_clients):
        client_shards = shard_order[
            k * classes_per_client:(k + 1) * classes_per_client
        ]
        idx = np.concatenate([shards[s] for s in client_shards])

        clients.append({
            "center_id": f"client_{k:03d}",
            "tss_codes": [],
            "x": x_all[idx],
            "y": labels[idx].astype(np.int64),
            "sample_ids": [sample_ids[i] for i in idx],
        })

    clients.sort(key=lambda c: len(c["y"]), reverse=True)
    print(f"  {len(clients)} pathological clients "
          f"({classes_per_client} classes each)")
    return clients
