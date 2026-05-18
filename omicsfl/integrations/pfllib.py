"""PFLlib integration adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from omicsfl.core import Partition, load_partition


def to_pfllib_format(partition_path):
    """Convert partition to PFLlib-compatible data dict."""
    partition = load_partition(partition_path)

    train_data = {}
    test_data = {}

    for i in range(partition.num_clients):
        x_train, y_train = partition.get_train(i)
        x_test, y_test = partition.get_test(i)

        train_data[i] = {"x": x_train, "y": y_train}
        test_data[i] = {"x": x_test, "y": y_test}

    return {
        "num_clients": partition.num_clients,
        "num_classes": partition.num_classes,
        "num_features": partition.num_features,
        "train_data": train_data,
        "test_data": test_data,
        "client_ids": partition.client_ids,
    }


def get_dataloaders(partition_path, batch_size=32, num_workers=0):
    """Return (train_loaders, test_loaders) — one DataLoader per client."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Run: pip install omicsfl[torch]"
        )

    partition = load_partition(partition_path)
    train_loaders = []
    test_loaders = []

    for i in range(partition.num_clients):
        x_train, y_train = partition.get_train(i)
        x_test, y_test = partition.get_test(i)

        train_ds = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        )
        test_ds = TensorDataset(
            torch.from_numpy(x_test),
            torch.from_numpy(y_test),
        )

        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
        ))
        test_loaders.append(DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        ))

    return train_loaders, test_loaders
