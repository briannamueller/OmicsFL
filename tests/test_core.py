"""Tests for core utilities: split, fingerprint, partition strategies, Partition class."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicsfl.core import Partition, load_partition
from omicsfl.fingerprint import build_partition_id
from omicsfl.split import train_test_val_split
from omicsfl.preprocess.gene_filter import (
    filter_by_variance, filter_by_mean_expression, filter_by_nonzero_fraction,
)
from omicsfl.preprocess.normalize import (
    log2_transform, size_factor_normalize, total_count_normalize,
)
from omicsfl.preprocess.labels import encode_cancer_type, encode_stage, encode_gender
from omicsfl.partition.natural import partition_by_center, extract_tss
from omicsfl.partition.dirichlet import partition_dirichlet
from omicsfl.partition.pathological import partition_pathological
from omicsfl.partition.by_cohort import partition_by_cohort
from omicsfl.download.registry import validate_cohorts


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Create synthetic expression data with TCGA-like barcodes."""
    rng = np.random.default_rng(0)
    n_samples = 200
    n_genes = 500

    # Generate barcodes from a few TSS codes
    tss_codes = ["A7", "B6", "3C", "D8", "E9", "F2"]
    barcodes = []
    for i in range(n_samples):
        tss = tss_codes[i % len(tss_codes)]
        barcodes.append(f"TCGA-{tss}-{i:04d}-01A")

    expression = pd.DataFrame(
        rng.random((n_samples, n_genes)).astype(np.float32),
        index=barcodes,
        columns=[f"GENE_{i}" for i in range(n_genes)],
    )
    labels = rng.integers(0, 4, size=n_samples).astype(np.int64)
    return expression, labels


@pytest.fixture
def partition_dir():
    """Create a temporary partition directory with fake data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "train").mkdir()
        (path / "test").mkdir()

        rng = np.random.default_rng(42)
        n_clients = 3
        n_features = 100

        for i in range(n_clients):
            x_train = rng.random((20, n_features)).astype(np.float32)
            y_train = rng.integers(0, 3, size=20).astype(np.int64)
            x_test = rng.random((5, n_features)).astype(np.float32)
            y_test = rng.integers(0, 3, size=5).astype(np.int64)

            np.savez_compressed(path / "train" / f"{i}.npz", x=x_train, y=y_train)
            np.savez_compressed(path / "test" / f"{i}.npz", x=x_test, y=y_test)

        config = {
            "partition_id": "test123",
            "task": "cancer_type",
            "cohorts": ["LUAD", "BRCA", "COAD"],
            "num_clients": n_clients,
            "num_classes": 3,
            "num_features": n_features,
            "client_ids": ["Hospital_A", "Hospital_B", "Hospital_C"],
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        yield path


# ── Test: Partition class ────────────────────────────────────────────────────

class TestPartition:
    def test_load(self, partition_dir):
        p = load_partition(partition_dir)
        assert p.num_clients == 3
        assert p.num_classes == 3
        assert p.num_features == 100
        assert len(p.client_ids) == 3

    def test_get_train(self, partition_dir):
        p = load_partition(partition_dir)
        x, y = p.get_train(0)
        assert x.shape == (20, 100)
        assert y.shape == (20,)
        assert x.dtype == np.float32
        assert y.dtype == np.int64

    def test_get_test(self, partition_dir):
        p = load_partition(partition_dir)
        x, y = p.get_test(1)
        assert x.shape == (5, 100)
        assert y.shape == (5,)

    def test_client_summary(self, partition_dir):
        p = load_partition(partition_dir)
        s = p.client_summary(0)
        assert s["client_id"] == "Hospital_A"
        assert s["n_train"] == 20
        assert s["n_test"] == 5

    def test_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_partition(tmp_path)

    def test_repr(self, partition_dir):
        p = load_partition(partition_dir)
        r = repr(p)
        assert "clients=3" in r
        assert "cancer_type" in r


# ── Test: Fingerprinting ─────────────────────────────────────────────────────

class TestFingerprint:
    def test_deterministic(self):
        params = dict(
            source="xena", cohorts=["LUAD", "BRCA"], task="cancer_type",
            partition_strategy="natural", top_genes=2000, min_samples=20,
            train_ratio=0.8, val_ratio=0.0, seed=42,
        )
        id1 = build_partition_id(**params)
        id2 = build_partition_id(**params)
        assert id1 == id2
        assert len(id1) == 12

    def test_different_params_different_id(self):
        base = dict(
            source="xena", cohorts=["LUAD", "BRCA"], task="cancer_type",
            partition_strategy="natural", top_genes=2000, min_samples=20,
            train_ratio=0.8, val_ratio=0.0, seed=42,
        )
        id1 = build_partition_id(**base)
        id2 = build_partition_id(**{**base, "seed": 99})
        assert id1 != id2

    def test_cohort_order_irrelevant(self):
        params = dict(
            source="xena", cohorts=["BRCA", "LUAD"], task="cancer_type",
            partition_strategy="natural", top_genes=2000, min_samples=20,
            train_ratio=0.8, val_ratio=0.0, seed=42,
        )
        id1 = build_partition_id(**params)
        params["cohorts"] = ["LUAD", "BRCA"]
        id2 = build_partition_id(**params)
        assert id1 == id2


# ── Test: Splitting ──────────────────────────────────────────────────────────

class TestSplit:
    def test_basic_split(self):
        rng = np.random.default_rng(0)
        x = rng.random((100, 50)).astype(np.float32)
        y = rng.integers(0, 3, size=100).astype(np.int64)

        splits = train_test_val_split(x, y, train_ratio=0.8, rng=rng)
        assert "train" in splits
        assert "test" in splits
        assert "val" not in splits
        assert len(splits["train"]["y"]) == 80
        assert len(splits["test"]["y"]) == 20

    def test_with_val(self):
        rng = np.random.default_rng(0)
        x = rng.random((100, 50)).astype(np.float32)
        y = rng.integers(0, 3, size=100).astype(np.int64)

        splits = train_test_val_split(x, y, train_ratio=0.7, val_ratio=0.1, rng=rng)
        assert "val" in splits
        assert len(splits["train"]["y"]) == 70
        assert len(splits["val"]["y"]) == 10
        assert len(splits["test"]["y"]) == 20

    def test_small_dataset(self):
        rng = np.random.default_rng(0)
        x = rng.random((3, 10)).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)

        splits = train_test_val_split(x, y, train_ratio=0.8, rng=rng)
        total = len(splits["train"]["y"]) + len(splits["test"]["y"])
        assert total == 3
        assert len(splits["test"]["y"]) >= 1

    def test_dtypes(self):
        rng = np.random.default_rng(0)
        x = rng.random((50, 20))  # float64 input
        y = np.arange(50)  # int input

        splits = train_test_val_split(x, y, rng=rng)
        assert splits["train"]["x"].dtype == np.float32
        assert splits["train"]["y"].dtype == np.int64


# ── Test: Gene filtering ─────────────────────────────────────────────────────

class TestGeneFilter:
    def test_variance_filter(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.random((50, 200)),
            columns=[f"G{i}" for i in range(200)],
        )
        filtered = filter_by_variance(df, top_k=50)
        assert filtered.shape == (50, 50)

    def test_variance_filter_zero_keeps_all(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.random((10, 100)))
        filtered = filter_by_variance(df, top_k=0)
        assert filtered.shape == (10, 100)

    def test_mean_filter(self):
        df = pd.DataFrame({
            "high": [10.0, 20.0, 30.0],
            "low": [0.01, 0.02, 0.03],
            "mid": [1.0, 2.0, 3.0],
        })
        filtered = filter_by_mean_expression(df, min_mean=1.0)
        assert "high" in filtered.columns
        assert "mid" in filtered.columns
        assert "low" not in filtered.columns

    def test_nonzero_filter(self):
        df = pd.DataFrame({
            "expressed": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sparse": [0.0, 0.0, 0.0, 0.0, 1.0],
            "dead": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        filtered = filter_by_nonzero_fraction(df, min_fraction=0.5)
        assert "expressed" in filtered.columns
        assert "sparse" not in filtered.columns
        assert "dead" not in filtered.columns


# ── Test: Normalization ──────────────────────────────────────────────────────

class TestNormalization:
    def test_log2(self):
        df = pd.DataFrame({"A": [0, 1, 3], "B": [7, 15, 31]})
        result = log2_transform(df, pseudocount=1.0)
        assert np.isclose(result.iloc[0, 0], 0.0)  # log2(0+1) = 0
        assert np.isclose(result.iloc[1, 0], 1.0)  # log2(1+1) = 1
        assert np.isclose(result.iloc[2, 0], 2.0)  # log2(3+1) = 2

    def test_cpm(self):
        df = pd.DataFrame({"A": [10, 20], "B": [90, 80]})
        result = total_count_normalize(df, target_sum=1e6)
        # Row sums should be 1e6
        assert np.allclose(result.sum(axis=1), 1e6)

    def test_size_factor(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.integers(1, 1000, size=(20, 50)).astype(float),
        )
        result = size_factor_normalize(df)
        assert result.shape == df.shape
        assert not result.isna().any().any()


# ── Test: Label encoding ─────────────────────────────────────────────────────

class TestLabels:
    def test_cancer_type(self):
        samples = ["S1", "S2", "S3", "S4"]
        assignments = pd.Series(["LUAD", "BRCA", "LUAD", "BRCA"], index=samples)
        labels, cmap = encode_cancer_type(samples, assignments, ["BRCA", "LUAD"])
        assert set(labels) == {0, 1}
        assert cmap[0] == "BRCA"
        assert cmap[1] == "LUAD"

    def test_stage(self):
        stages = pd.Series(["Stage I", "Stage IV", "Stage IVA", "", "Stage II"])
        labels, cmap = encode_stage(stages)
        assert labels[0] == 0.0  # Non-advanced
        assert labels[1] == 1.0  # Advanced
        assert labels[2] == 1.0  # Advanced
        assert np.isnan(labels[3])  # Missing
        assert labels[4] == 0.0

    def test_gender(self):
        genders = pd.Series(["male", "female", "MALE", "unknown"])
        labels, cmap = encode_gender(genders)
        assert labels[0] == 0.0
        assert labels[1] == 1.0
        assert labels[2] == 0.0
        assert np.isnan(labels[3])


# ── Test: Partition strategies ───────────────────────────────────────────────

class TestPartitionStrategies:
    def test_natural_partition(self, synthetic_data):
        expression, labels = synthetic_data
        # We need a centers map for our fake TSS codes
        centers_map = {
            "A7": "Hospital Alpha",
            "B6": "Hospital Beta",
            "3C": "Hospital Gamma",
            "D8": "Hospital Delta",
            "E9": "Hospital Epsilon",
            "F2": "Hospital Zeta",
        }
        clients = partition_by_center(
            expression, labels, min_samples=10, centers_map=centers_map,
        )
        assert len(clients) > 0
        # Each client should have x and y arrays
        for c in clients:
            assert c["x"].shape[0] == len(c["y"])
            assert c["x"].shape[1] == 500
            assert len(c["y"]) >= 10

    def test_natural_min_samples_filter(self, synthetic_data):
        expression, labels = synthetic_data
        centers_map = {"A7": "H1", "B6": "H2", "3C": "H3",
                       "D8": "H4", "E9": "H5", "F2": "H6"}
        # With 200 samples / 6 centers ≈ 33 per center
        # Setting min_samples=50 should filter most out
        clients = partition_by_center(
            expression, labels, min_samples=50, centers_map=centers_map,
        )
        assert len(clients) < 6

    def test_dirichlet_partition(self, synthetic_data):
        expression, labels = synthetic_data
        clients = partition_dirichlet(
            expression, labels,
            num_clients=5, alpha=0.5, min_samples=5, seed=42,
        )
        assert len(clients) <= 5
        total_samples = sum(len(c["y"]) for c in clients)
        assert total_samples == 200

    def test_dirichlet_alpha_affects_heterogeneity(self, synthetic_data):
        expression, labels = synthetic_data

        # Very low alpha → very non-IID
        clients_low = partition_dirichlet(
            expression, labels, num_clients=5, alpha=0.01, seed=42, min_samples=1,
        )
        # High alpha → near-IID
        clients_high = partition_dirichlet(
            expression, labels, num_clients=5, alpha=100.0, seed=42, min_samples=1,
        )

        # Low alpha clients should have more skewed label distributions
        def max_class_fraction(client):
            _, counts = np.unique(client["y"], return_counts=True)
            return counts.max() / counts.sum()

        avg_skew_low = np.mean([max_class_fraction(c) for c in clients_low])
        avg_skew_high = np.mean([max_class_fraction(c) for c in clients_high])
        assert avg_skew_low > avg_skew_high

    def test_pathological_partition(self, synthetic_data):
        expression, labels = synthetic_data
        clients = partition_pathological(
            expression, labels,
            num_clients=5, classes_per_client=2, seed=42,
        )
        assert len(clients) == 5
        # Each client should be dominated by few classes (shard boundaries
        # may not align perfectly with class boundaries)
        for c in clients:
            unique_labels, counts = np.unique(c["y"], return_counts=True)
            # Top 2 classes should hold the vast majority of samples
            top2 = sorted(counts, reverse=True)[:2]
            assert sum(top2) / len(c["y"]) >= 0.8

    def test_by_cohort(self, synthetic_data):
        expression, labels = synthetic_data
        cohort_assignments = pd.Series(
            ["LUAD" if i < 100 else "BRCA" for i in range(200)],
            index=expression.index,
        )
        clients = partition_by_cohort(expression, labels, cohort_assignments)
        assert len(clients) == 2
        assert clients[0]["center_id"] in ["BRCA", "LUAD"]


# ── Test: Registry ───────────────────────────────────────────────────────────

class TestRegistry:
    def test_validate_cohorts(self):
        result = validate_cohorts(["luad", "BRCA", "TCGA-COAD"])
        assert result == ["BRCA", "COAD", "LUAD"]

    def test_invalid_cohort(self):
        with pytest.raises(ValueError, match="Unknown TCGA cohort"):
            validate_cohorts(["FAKE"])

    def test_extract_tss(self):
        assert extract_tss("TCGA-A7-A0CG-01A") == "A7"
        assert extract_tss("TCGA-B6-1234-01") == "B6"
        assert extract_tss("NOT-A-BARCODE") == ""
