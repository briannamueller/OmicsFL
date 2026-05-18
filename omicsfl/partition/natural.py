"""Partition by contributing hospital using TCGA TSS codes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_centers_map(assets_dir=None):
    """Load TSS code -> hospital name mapping from bundled centers.csv."""
    if assets_dir is None:
        assets_dir = Path(__file__).parent.parent / "assets"

    centers_file = assets_dir / "centers.csv"
    if not centers_file.exists():
        raise FileNotFoundError(
            f"centers.csv not found at {centers_file}. "
            f"Ensure the OmicsFL package is installed correctly."
        )

    df = pd.read_csv(centers_file)
    return dict(zip(df["TSS Code"].astype(str), df["Source Site"].astype(str)))


def extract_tss(barcode: str) -> str:
    """Extract TSS code from TCGA barcode (e.g. TCGA-A7-A0CG-01 -> A7)."""
    parts = barcode.split("-")
    if len(parts) >= 2 and parts[0] == "TCGA":
        return parts[1]
    return ""


def partition_by_center(expression, labels, min_samples=20, centers_map=None):
    """Group samples by hospital (merging TSS codes), filter by min_samples."""
    if centers_map is None:
        centers_map = load_centers_map()

    sample_ids = expression.index.tolist()
    tss_codes = pd.Series(
        [extract_tss(s) for s in sample_ids],
        index=expression.index,
    )
    hospital_names = tss_codes.map(
        lambda t: centers_map.get(str(t), f"TSS_{t}")
    )

    clients = []
    skipped = 0

    for hospital in sorted(hospital_names.unique()):
        mask = hospital_names == hospital
        n = mask.sum()

        if n < min_samples:
            skipped += 1
            continue

        group_tss = sorted(tss_codes[mask].unique())
        group_ids = expression.index[mask].tolist()

        clients.append({
            "center_id": hospital,
            "tss_codes": group_tss,
            "x": expression.loc[mask].values.astype(np.float32),
            "y": labels[mask.values].astype(np.int64),
            "sample_ids": group_ids,
        })

    clients.sort(key=lambda c: len(c["y"]), reverse=True)

    if skipped > 0:
        print(f"  Skipped {skipped} centers with < {min_samples} samples")
    print(f"  {len(clients)} centers qualify")

    return clients
