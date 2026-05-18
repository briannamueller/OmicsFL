"""Flower (flwr) integration — requires pip install omicsfl[flower]."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from omicsfl.core import Partition, load_partition


def get_client_fn(partition_path, batch_size=32):
    """Create a Flower client_fn for fl.simulation.start_simulation."""
    try:
        import flwr as fl
    except ImportError:
        raise ImportError(
            "Flower not installed. Run: pip install omicsfl[flower]"
        )

    partition = load_partition(partition_path)

    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_idx = int(cid)
        x_train, y_train = partition.get_train(client_idx)
        x_test, y_test = partition.get_test(client_idx)

        class OmicsFLClient(fl.client.NumPyClient):
            def get_properties(self, config):
                return {
                    "client_id": partition.client_ids[client_idx],
                    "num_samples": len(y_train),
                }

            def fit(self, parameters, config):
                # Users override this with their model training logic
                raise NotImplementedError(
                    "Subclass OmicsFLClient or use get_numpy_data() instead. "
                    "This base client provides data access only."
                )

            def evaluate(self, parameters, config):
                raise NotImplementedError(
                    "Subclass OmicsFLClient or use get_numpy_data() instead."
                )

        return OmicsFLClient()

    return client_fn


def get_numpy_data(partition_path):
    """Load all clients as dict[client_idx -> {"train": (x, y), "test": (x, y)}]."""
    partition = load_partition(partition_path)
    result = {}
    for i in range(partition.num_clients):
        result[i] = {
            "train": partition.get_train(i),
            "test": partition.get_test(i),
        }
        if partition.has_val():
            result[i]["val"] = partition.get_val(i)
    return result
