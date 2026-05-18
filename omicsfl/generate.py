"""Main generate pipeline — ties download, preprocess, partition, and split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from omicsfl.download.registry import validate_cohorts
from omicsfl.fingerprint import build_partition_id
from omicsfl.split import train_test_val_split


def generate(
    cohorts: list[str] | str = "all",
    task: str = "cancer_type",
    source: str = "xena",
    partition: str = "natural",
    top_genes: int = 2000,
    min_samples: int = 20,
    train_ratio: float = 0.8,
    val_ratio: float = 0.0,
    seed: int = 42,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    # Dirichlet/pathological params
    num_clients: int = 10,
    alpha: float = 0.5,
    classes_per_client: int = 2,
) -> Path:
    """Generate a federated learning partition from TCGA data.

    Orchestrates download -> preprocess -> partition -> split -> save.
    Returns the path to the output directory.
    """
    from omicsfl.download.registry import TCGA_COHORTS

    # Resolve cohorts
    if cohorts == "all" or cohorts == ["all"]:
        cohorts = list(TCGA_COHORTS)
    elif isinstance(cohorts, str):
        cohorts = [c.strip() for c in cohorts.split(",")]
    cohorts = validate_cohorts(cohorts)

    # Validate task
    valid_tasks = ["cancer_type", "stage", "gender"]
    if task not in valid_tasks:
        raise ValueError(f"Unknown task: '{task}'. Choose from {valid_tasks}")

    if task == "cancer_type" and len(cohorts) < 2:
        raise ValueError(
            "cancer_type task requires at least 2 cohorts. "
            "Use --task stage/gender for single-cohort experiments."
        )

    # Resolve paths
    if cache_dir is None:
        cache_dir = Path(f"./omicsfl_cache/{source}")
    cache_dir = Path(cache_dir)

    partition_id = build_partition_id(
        source=source,
        cohorts=cohorts,
        task=task,
        partition_strategy=partition,
        top_genes=top_genes,
        min_samples=min_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        num_clients=num_clients if partition in ("dirichlet", "pathological") else 0,
        alpha=alpha if partition == "dirichlet" else 0,
        classes_per_client=classes_per_client if partition == "pathological" else 0,
    )

    if output_dir is None:
        output_dir = Path(f"./omicsfl_data/{partition_id}")
    output_dir = Path(output_dir)

    config_path = output_dir / "config.json"
    if config_path.exists():
        print(f"[omicsfl] Partition already exists at {output_dir}")
        print(f"  Partition ID: {partition_id}")
        return output_dir

    print(f"\n{'='*60}")
    print(f" OmicsFL Generate")
    print(f"  Source:    {source}")
    print(f"  Cohorts:   {', '.join(cohorts[:5])}{'...' if len(cohorts) > 5 else ''}")
    print(f"  Task:      {task}")
    print(f"  Partition: {partition}")
    print(f"  Genes:     top {top_genes}" if top_genes > 0 else "  Genes:     all")
    print(f"{'='*60}\n")

    print("[1/5] Downloading data ...")
    expression, phenotype = _download(source, cohorts, cache_dir)

    print(f"\n[2/5] Encoding labels (task={task}) ...")
    labels, class_map, valid_mask = _encode_labels(
        expression, phenotype, cohorts, task,
    )
    expression = expression.loc[valid_mask]
    labels = labels[valid_mask.values]
    print(f"  {len(labels)} samples with valid labels")
    print(f"  Classes: {class_map}")

    print(f"\n[3/5] Gene filtering ...")
    from omicsfl.preprocess.gene_filter import filter_by_variance, filter_by_nonzero_fraction
    n_genes_raw = expression.shape[1]
    expression = filter_by_nonzero_fraction(expression, min_fraction=0.1)
    if top_genes > 0:
        expression = filter_by_variance(expression, top_k=top_genes)
    n_genes = expression.shape[1]
    print(f"  {n_genes_raw} -> {n_genes} genes")

    print(f"\n[4/5] Partitioning ({partition}) ...")
    clients = _partition(
        expression, labels, partition, cohorts, phenotype,
        min_samples=min_samples,
        num_clients=num_clients,
        alpha=alpha,
        classes_per_client=classes_per_client,
        seed=seed,
    )

    if not clients:
        raise RuntimeError("No clients remaining after partitioning.")

    print(f"\n[5/5] Splitting and saving ({len(clients)} clients) ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    if val_ratio > 0:
        (output_dir / "val").mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    num_classes = len(class_map)
    client_ids = []
    client_sizes = []

    for idx, client in enumerate(clients):
        splits = train_test_val_split(
            client["x"], client["y"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            rng=rng,
        )

        for split_name, split_data in splits.items():
            np.savez_compressed(
                output_dir / split_name / f"{idx}.npz",
                x=split_data["x"],
                y=split_data["y"],
            )

        client_ids.append(client["center_id"])
        client_sizes.append(len(client["y"]))

    config = {
        "partition_id": partition_id,
        "source": source,
        "cohorts": sorted(cohorts),
        "task": task,
        "partition_strategy": partition,
        "num_clients": len(clients),
        "num_classes": num_classes,
        "num_features": n_genes,
        "top_genes": top_genes,
        "min_samples": min_samples,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "client_ids": client_ids,
        "client_sizes": client_sizes,
        "class_map": {str(k): v for k, v in class_map.items()},
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    total = sum(client_sizes)
    print(f"\n[omicsfl] Done!")
    print(f"  Output:   {output_dir}")
    print(f"  ID:       {partition_id}")
    print(f"  Clients:  {len(clients)}")
    print(f"  Classes:  {num_classes}")
    print(f"  Features: {n_genes}")
    print(f"  Samples:  {total}")

    return output_dir


def _download(
    source: str,
    cohorts: list[str],
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and return (expression, phenotype) DataFrames."""
    if source == "xena":
        from omicsfl.download.xena import (
            download, load_expression, load_phenotype,
        )
        paths = download(cohorts, output_dir=cache_dir)
        expression = load_expression(
            paths["expression_path"], cohorts, paths["phenotype_path"],
        )
        phenotype = load_phenotype(
            paths["phenotype_path"], samples=expression.index.tolist(),
        )
    elif source == "gdc":
        from omicsfl.download.gdc import download as gdc_download
        results = gdc_download(cohorts, output_dir=cache_dir)
        all_expr = []
        all_clin = []
        for cohort, paths in results.items():
            expr = pd.read_parquet(paths["counts"])
            clin = pd.read_csv(paths["clinical"])
            clin["cohort"] = cohort
            all_expr.append(expr)
            all_clin.append(clin)

        if len(all_expr) > 1:
            shared = set(all_expr[0].columns)
            for e in all_expr[1:]:
                shared &= set(e.columns)
            shared = sorted(shared)
            all_expr = [e[shared] for e in all_expr]

        expression = pd.concat(all_expr, axis=0)
        phenotype = pd.concat(all_clin, axis=0).set_index("case_id")
    else:
        raise ValueError(f"Unknown source: '{source}'. Use 'xena' or 'gdc'.")

    return expression, phenotype


def _encode_labels(expression, phenotype, cohorts, task):
    from omicsfl.preprocess.labels import (
        encode_cancer_type, encode_stage, encode_gender, encode_survival,
    )

    sample_ids = expression.index.tolist()

    if task == "cancer_type":
        if "cohort" in phenotype.columns:
            cohort_assignments = phenotype["cohort"].reindex(expression.index)
        else:
            cohort_assignments = pd.Series(
                [_barcode_to_cohort(s, cohorts) for s in sample_ids],
                index=expression.index,
            )
        labels, class_map = encode_cancer_type(
            sample_ids, cohort_assignments, cohorts,
        )
        valid_mask = pd.Series(True, index=expression.index)

    elif task == "stage":
        if "stage" not in phenotype.columns:
            raise ValueError("Phenotype data missing 'stage' column for stage task.")
        stages = phenotype["stage"].reindex(expression.index)
        labels, class_map = encode_stage(stages)
        valid_mask = pd.Series(~np.isnan(labels), index=expression.index)

    elif task == "gender":
        if "gender" not in phenotype.columns:
            raise ValueError("Phenotype data missing 'gender' column.")
        genders = phenotype["gender"].reindex(expression.index)
        labels, class_map = encode_gender(genders)
        valid_mask = pd.Series(~np.isnan(labels), index=expression.index)

    else:
        raise ValueError(f"Unknown task: {task}")

    return labels, class_map, valid_mask


def _barcode_to_cohort(barcode, cohorts):
    """Fallback cohort assignment from TCGA barcode."""
    return cohorts[0] if len(cohorts) == 1 else "UNKNOWN"


def _partition(expression, labels, strategy, cohorts, phenotype, **kwargs):
    if strategy == "natural":
        from omicsfl.partition.natural import partition_by_center
        return partition_by_center(
            expression, labels,
            min_samples=kwargs.get("min_samples", 20),
        )
    elif strategy == "by_cohort":
        from omicsfl.partition.by_cohort import partition_by_cohort
        if "cohort" in phenotype.columns:
            cohort_assignments = phenotype["cohort"].reindex(expression.index)
        else:
            cohort_assignments = pd.Series("UNKNOWN", index=expression.index)
        return partition_by_cohort(expression, labels, cohort_assignments)
    elif strategy == "dirichlet":
        from omicsfl.partition.dirichlet import partition_dirichlet
        return partition_dirichlet(
            expression, labels,
            num_clients=kwargs.get("num_clients", 10),
            alpha=kwargs.get("alpha", 0.5),
            min_samples=kwargs.get("min_samples", 10),
            seed=kwargs.get("seed", 42),
        )
    elif strategy == "pathological":
        from omicsfl.partition.pathological import partition_pathological
        return partition_pathological(
            expression, labels,
            num_clients=kwargs.get("num_clients", 10),
            classes_per_client=kwargs.get("classes_per_client", 2),
            seed=kwargs.get("seed", 42),
        )
    else:
        raise ValueError(
            f"Unknown partition strategy: '{strategy}'. "
            f"Choose from: natural, by_cohort, dirichlet, pathological"
        )
