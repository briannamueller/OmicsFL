"""Normalization methods for gene expression data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log2_transform(expression, pseudocount=1.0):
    """Apply log2(x + pseudocount). Use on raw counts (GDC); Xena is already log-scaled."""
    return np.log2(expression + pseudocount)


def size_factor_normalize(expression):
    """DESeq2-style median-of-ratios normalization (count space, not logged)."""
    log_expr = np.log(expression.replace(0, np.nan))
    geo_means = log_expr.mean(axis=0)

    finite_mask = np.isfinite(geo_means)
    if finite_mask.sum() == 0:
        return total_count_normalize(expression)

    log_ratios = log_expr.loc[:, finite_mask].subtract(
        geo_means[finite_mask], axis=1
    )
    size_factors = np.exp(log_ratios.median(axis=1))
    size_factors = size_factors.replace(0, 1.0)

    return expression.div(size_factors, axis=0)


def total_count_normalize(expression, target_sum=1e6):
    """CPM (counts per million) normalization."""
    total_per_sample = expression.sum(axis=1)
    total_per_sample = total_per_sample.replace(0, 1.0)
    return expression.div(total_per_sample, axis=0) * target_sum
