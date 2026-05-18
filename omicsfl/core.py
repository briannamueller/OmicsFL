"""Core data structures for loading and accessing OmicsFL partitions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class Partition:
    """A loaded OmicsFL federated partition."""

    def __init__(self, path: Path):
        self.path = Path(path)
        config_path = self.path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.json found at {self.path}. "
                f"Run `omicsfl generate` first."
            )
        with open(config_path) as f:
            self.config = json.load(f)

        self._cache: dict[tuple[int, str], dict] = {}

    @property
    def num_clients(self) -> int:
        return self.config["num_clients"]

    @property
    def num_classes(self) -> int:
        return self.config["num_classes"]

    @property
    def num_features(self) -> int:
        return self.config["num_features"]

    @property
    def client_ids(self) -> list[str]:
        return self.config["client_ids"]

    @property
    def partition_id(self) -> str:
        return self.config["partition_id"]

    @property
    def task(self) -> str:
        return self.config["task"]

    @property
    def cohorts(self) -> list[str]:
        return self.config["cohorts"]

    def _load_split(self, client: int, split: str) -> dict:
        key = (client, split)
        if key not in self._cache:
            split_dir = self.path / split
            npz_path = split_dir / f"{client}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"No data for client {client} split '{split}' "
                    f"at {npz_path}"
                )
            data = np.load(npz_path, allow_pickle=True)
            self._cache[key] = {
                "x": data["x"],
                "y": data["y"],
            }
        return self._cache[key]

    def get_train(self, client: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) training arrays for the given client."""
        d = self._load_split(client, "train")
        return d["x"], d["y"]

    def get_test(self, client: int) -> tuple[np.ndarray, np.ndarray]:
        d = self._load_split(client, "test")
        return d["x"], d["y"]

    def get_val(self, client: int) -> tuple[np.ndarray, np.ndarray]:
        d = self._load_split(client, "val")
        return d["x"], d["y"]

    def has_val(self) -> bool:
        val_dir = self.path / "val"
        return val_dir.exists() and any(val_dir.glob("*.npz"))

    def client_summary(self, client: int) -> dict:
        x_train, y_train = self.get_train(client)
        x_test, y_test = self.get_test(client)
        summary = {
            "client_id": self.client_ids[client],
            "n_train": len(y_train),
            "n_test": len(y_test),
            "label_distribution_train": _label_dist(y_train, self.num_classes),
            "label_distribution_test": _label_dist(y_test, self.num_classes),
        }
        if self.has_val():
            x_val, y_val = self.get_val(client)
            summary["n_val"] = len(y_val)
            summary["label_distribution_val"] = _label_dist(y_val, self.num_classes)
        return summary

    def summary(self) -> dict:
        return {
            "partition_id": self.partition_id,
            "task": self.task,
            "cohorts": self.cohorts,
            "num_clients": self.num_clients,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "has_val": self.has_val(),
            "client_ids": self.client_ids,
        }

    def __repr__(self) -> str:
        return (
            f"Partition(task={self.task!r}, clients={self.num_clients}, "
            f"classes={self.num_classes}, features={self.num_features})"
        )


def _label_dist(y, num_classes):
    counts = {}
    for c in range(num_classes):
        n = int((y == c).sum())
        if n > 0:
            counts[c] = n
    return counts


def load_partition(path: str | Path) -> Partition:
    """Load a saved partition directory (must contain config.json)."""
    return Partition(Path(path))
