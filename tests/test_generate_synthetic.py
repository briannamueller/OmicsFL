"""End-to-end test of the generate pipeline using mocked download."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from omicsfl.generate import generate
from omicsfl.core import load_partition


def _mock_download(source, cohorts, cache_dir):
    """Return synthetic expression + phenotype data instead of downloading."""
    rng = np.random.default_rng(123)
    n_samples = 300
    n_genes = 1000

    # Simulate samples from multiple TSS/hospitals across cohorts
    tss_pool = ["A7", "B6", "3C", "D8", "E9", "F2", "G1", "H4"]
    barcodes = []
    cohort_list = []
    for i in range(n_samples):
        tss = tss_pool[i % len(tss_pool)]
        cohort = cohorts[i % len(cohorts)]
        barcodes.append(f"TCGA-{tss}-{i:04d}-01A")
        cohort_list.append(cohort)

    expression = pd.DataFrame(
        rng.random((n_samples, n_genes)).astype(np.float32),
        index=barcodes,
        columns=[f"ENSG{i:010d}" for i in range(n_genes)],
    )

    phenotype = pd.DataFrame({
        "cohort": cohort_list,
        "gender": rng.choice(["male", "female"], size=n_samples),
        "stage": rng.choice(
            ["Stage I", "Stage II", "Stage III", "Stage IV", ""],
            size=n_samples,
        ),
    }, index=barcodes)

    return expression, phenotype


@pytest.fixture
def output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestGenerateEndToEnd:
    def test_cancer_type_natural(self, output_dir):
        cohorts = ["LUAD", "BRCA", "COAD"]
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path = generate(
                cohorts=cohorts,
                task="cancer_type",
                partition="natural",
                top_genes=200,
                min_samples=10,
                seed=42,
                output_dir=str(output_dir / "test1"),
            )

        partition = load_partition(path)
        assert partition.num_clients > 0
        assert partition.num_classes == 3
        assert partition.num_features == 200
        assert partition.task == "cancer_type"

        # Verify data loads correctly
        x, y = partition.get_train(0)
        assert x.shape[1] == 200
        assert set(np.unique(y)).issubset({0, 1, 2})

    def test_stage_natural(self, output_dir):
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path = generate(
                cohorts=["LUAD"],
                task="stage",
                partition="natural",
                top_genes=100,
                min_samples=5,
                seed=42,
                output_dir=str(output_dir / "test2"),
            )

        partition = load_partition(path)
        assert partition.num_classes == 2
        x, y = partition.get_train(0)
        assert set(np.unique(y)).issubset({0, 1})

    def test_dirichlet(self, output_dir):
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path = generate(
                cohorts=["LUAD", "BRCA"],
                task="cancer_type",
                partition="dirichlet",
                num_clients=8,
                alpha=0.3,
                top_genes=50,
                min_samples=5,
                seed=42,
                output_dir=str(output_dir / "test3"),
            )

        partition = load_partition(path)
        assert partition.num_clients <= 8
        assert partition.num_classes == 2

    def test_with_val_split(self, output_dir):
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path = generate(
                cohorts=["LUAD", "BRCA", "COAD"],
                task="cancer_type",
                partition="natural",
                top_genes=50,
                min_samples=10,
                train_ratio=0.7,
                val_ratio=0.1,
                seed=42,
                output_dir=str(output_dir / "test4"),
            )

        partition = load_partition(path)
        assert partition.has_val()
        x_val, y_val = partition.get_val(0)
        assert len(y_val) > 0

    def test_idempotent(self, output_dir):
        """Running generate twice with same params should skip second time."""
        kwargs = dict(
            cohorts=["LUAD", "BRCA"],
            task="cancer_type",
            partition="natural",
            top_genes=50,
            min_samples=10,
            seed=42,
            output_dir=str(output_dir / "test5"),
        )
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path1 = generate(**kwargs)
            # Second call should detect existing partition
            path2 = generate(**kwargs)

        assert path1 == path2

    def test_config_json_complete(self, output_dir):
        with patch("omicsfl.generate._download", side_effect=lambda *a: _mock_download(*a)):
            path = generate(
                cohorts=["LUAD", "BRCA"],
                task="cancer_type",
                partition="natural",
                top_genes=100,
                min_samples=10,
                seed=42,
                output_dir=str(output_dir / "test6"),
            )

        config = json.loads((path / "config.json").read_text())
        required_keys = [
            "partition_id", "source", "cohorts", "task",
            "partition_strategy", "num_clients", "num_classes",
            "num_features", "client_ids", "client_sizes", "class_map",
        ]
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
