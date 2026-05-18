"""OmicsFL — TCGA transcriptomics data for federated learning."""

__version__ = "0.1.0"

from omicsfl.core import load_partition, Partition

def generate(*args, **kwargs):
    """Convenience re-export of omicsfl.generate.generate."""
    from omicsfl.generate import generate as _generate
    return _generate(*args, **kwargs)
