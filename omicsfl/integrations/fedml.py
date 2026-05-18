"""FedML integration adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from omicsfl.core import Partition, load_partition


def load_data(partition_path, batch_size=32):
    """Load partition in FedML's 8-tuple format.

    Returns (client_num, train_data_num, test_data_num,
             train_global, test_global, train_local_dict,
             test_local_dict, class_num).
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Run: pip install omicsfl[torch]"
        )

    partition = load_partition(partition_path)

    train_data_local_dict = {}
    test_data_local_dict = {}
    all_train_datasets = []
    all_test_datasets = []
    train_data_num = 0
    test_data_num = 0

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

        train_data_local_dict[i] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
        )
        test_data_local_dict[i] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
        )

        all_train_datasets.append(train_ds)
        all_test_datasets.append(test_ds)
        train_data_num += len(x_train)
        test_data_num += len(x_test)

    train_data_global = DataLoader(
        ConcatDataset(all_train_datasets),
        batch_size=batch_size, shuffle=True,
    )
    test_data_global = DataLoader(
        ConcatDataset(all_test_datasets),
        batch_size=batch_size, shuffle=False,
    )

    return (
        partition.num_clients,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_dict,
        test_data_local_dict,
        partition.num_classes,
    )
