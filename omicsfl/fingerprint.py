"""Deterministic partition ID generation from parameters."""
from __future__ import annotations

import hashlib
import json


def build_partition_id(source, cohorts, task, partition_strategy, top_genes,
                       min_samples, train_ratio, val_ratio, seed, **extra):
    """Deterministic 12-char hex hash of partition parameters."""
    params = {
        "source": source,
        "cohorts": sorted(cohorts),
        "task": task,
        "partition_strategy": partition_strategy,
        "top_genes": top_genes,
        "min_samples": min_samples,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
    }
    params.update(extra)
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
