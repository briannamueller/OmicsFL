"""Gene filtering strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_by_variance(expression, top_k=2000):
    """Keep the top-K genes by variance. 0 or >= total genes keeps all."""
    if top_k <= 0 or top_k >= expression.shape[1]:
        return expression

    gene_var = expression.var(axis=0)
    top_genes = gene_var.nlargest(top_k).index
    return expression[top_genes]


def filter_by_mean_expression(expression, min_mean=1.0):
    """Remove genes with mean expression below min_mean."""
    gene_means = expression.mean(axis=0)
    keep = gene_means >= min_mean
    return expression.loc[:, keep]


def filter_by_nonzero_fraction(expression, min_fraction=0.1):
    """Remove genes expressed in fewer than min_fraction of samples."""
    nonzero_frac = (expression > 0).mean(axis=0)
    keep = nonzero_frac >= min_fraction
    return expression.loc[:, keep]
