"""Partition by cancer cohort — each TCGA cohort becomes one FL client."""
from __future__ import annotations

import numpy as np
import pandas as pd


def partition_by_cohort(expression, labels, cohort_assignments):
    """One client per cohort."""
    clients = []

    for cohort in sorted(cohort_assignments.unique()):
        mask = cohort_assignments == cohort
        sample_ids = expression.index[mask].tolist()

        clients.append({
            "center_id": cohort,
            "tss_codes": [],
            "x": expression.loc[mask].values.astype(np.float32),
            "y": labels[mask.values].astype(np.int64),
            "sample_ids": sample_ids,
        })

    clients.sort(key=lambda c: len(c["y"]), reverse=True)
    print(f"  {len(clients)} cohort-clients created")
    return clients
