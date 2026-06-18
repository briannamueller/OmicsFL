# OmicsFL

**Download, preprocess, and partition TCGA transcriptomics data for federated learning experiments.**

OmicsFL bridges the gap between cancer genomics data and federated learning research. It provides a single command to go from raw TCGA data to FL client partitions with real-world institutional heterogeneity from TCGA's contributing hospitals.

## Why OmicsFL?

Federated learning benchmarks typically use vision/NLP datasets with artificial non-IID splits. Real genomics data offers *natural* heterogeneity: each hospital contributed different cancer types, patient populations, and sample sizes. OmicsFL makes this data accessible without the bioinformatics plumbing.

| | fedpydeseq2-datasets | OncoLearn | FLamby | **OmicsFL** |
|--|--|--|--|--|
| Multi-cohort (33 types) | 8 cohorts | ✗ | ✗ | ✓ |
| Real genomics (RNA-seq) | ✓ | ✓ | ✗ | ✓ |
| Multiple classification tasks | ✗ | ✗ | ✗ | ✓ |
| Natural + artificial partitions | Natural only | Unclear | Artificial | Both |
| Framework-agnostic output | ✗ | ✗ | Partial | ✓ |
| One-command UX | ✗ | ✗ | ✓ | ✓ |

## Quick Start

```bash
pip install git+https://github.com/briannamueller/OmicsFL.git
```

### One command

```bash
# Multi-cohort cancer type classification, partitioned by hospital
omicsfl generate --cohorts LUAD,LUSC,BRCA,COAD --task cancer_type --partition natural
```

### Python API

```python
from omicsfl import generate, load_partition

path = generate(
    cohorts=["LUAD", "LUSC", "BRCA", "COAD"],
    task="cancer_type",
    partition="natural",
    top_genes=2000,
)

partition = load_partition(path)
print(partition)  # Partition(task='cancer_type', clients=29, classes=4, features=2000)

x_train, y_train = partition.get_train(client=0)
```

## Features

### Data Sources

| Source | What you get | Best for |
|--------|-------------|----------|
| **Xena** (default) | Pre-normalized log2(TPM+1) matrices, fast download | Quick experiments, ML classification |
| **GDC** | Raw STAR counts, latest data release | Custom normalization, maximum sample count |

```bash
omicsfl generate --source xena ...   # Default: fast, ML-ready
omicsfl generate --source gdc ...    # Raw counts, more samples
```

### Classification Tasks

| Task | Type | Description |
|------|------|-------------|
| `cancer_type` | Multi-class | Cancer type classification (requires 2+ cohorts) |
| `stage` | Binary | Advanced (Stage IV) vs non-advanced |
| `survival` | Binary | Survived past threshold vs deceased (default: 3 years, Xena only) |

### Partition Strategies

| Strategy | Description | Use case |
|----------|-------------|----------|
| `natural` | Each contributing hospital = one client | Real-world FL heterogeneity |
| `by_cohort` | Each cancer type = one client | Cross-cancer feature shift |
| `dirichlet` | Synthetic non-IID via Dirichlet(α) | Controlled label skew benchmarks |
| `pathological` | K classes per client | Extreme non-IID (McMahan et al. 2017) |

### All 33 TCGA Cohorts

ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM

## Output Format

```
output_dir/
├── config.json       # Full metadata for reproducibility
├── train/
│   ├── 0.npz        # {"x": float32[n, genes], "y": int64[n]}
│   ├── 1.npz
│   └── ...
└── test/
    ├── 0.npz
    └── ...
```

Load with any framework:

```python
import numpy as np
data = np.load("train/0.npz")
x, y = data["x"], data["y"]
```

## Framework Integrations

OmicsFL outputs are framework-agnostic (NumPy arrays), but optional adapters make integration seamless:

### Flower

```bash
pip install "omicsfl[flower] @ git+https://github.com/briannamueller/OmicsFL.git"
```

```python
from omicsfl.integrations.flower import get_numpy_data

data = get_numpy_data("./omicsfl_data/f3f4811ed1a6/")
# data[client_idx]["train"] -> (x, y)
```

### PFLlib

```bash
pip install "omicsfl[torch] @ git+https://github.com/briannamueller/OmicsFL.git"
```

```python
from omicsfl.integrations.pfllib import get_dataloaders

train_loaders, test_loaders = get_dataloaders("./omicsfl_data/f3f4811ed1a6/")
# One DataLoader per client, ready for PFLlib training loops
```

### FedML

```python
from omicsfl.integrations.fedml import load_data

(client_num, train_data_num, test_data_num,
 train_global, test_global,
 train_local, test_local, class_num) = load_data("./omicsfl_data/f3f4811ed1a6/")
```

## Advanced Usage

### Full CLI options

```bash
omicsfl generate \
    --source xena \
    --cohorts LUAD,LUSC,BRCA,PAAD \
    --task cancer_type \
    --partition natural \
    --min-samples 20 \
    --top-genes 2000 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --seed 42 \
    --output-dir ./my_experiment
```

### Dirichlet non-IID with controllable heterogeneity

```bash
omicsfl generate \
    --cohorts all \
    --task cancer_type \
    --partition dirichlet \
    --num-clients 20 \
    --alpha 0.1    # Very non-IID (each client dominated by 1-2 classes)
```

### List available cohorts and tasks

```bash
omicsfl info
```

## Installation

### Recommended (install directly from GitHub)

```bash
pip install git+https://github.com/briannamueller/OmicsFL.git
```

### Optional extras

Extras require quoted `"pkg @ url"` syntax — without quotes, the shell expands
`[...]` as a glob and the space in `@ url` splits the argument:

```bash
pip install "omicsfl[xena] @ git+https://github.com/briannamueller/OmicsFL.git"    # Xena download backend (xenaPython)
pip install "omicsfl[torch] @ git+https://github.com/briannamueller/OmicsFL.git"   # PyTorch DataLoaders for integrations
pip install "omicsfl[flower] @ git+https://github.com/briannamueller/OmicsFL.git"  # Flower FL framework adapter
pip install "omicsfl[all] @ git+https://github.com/briannamueller/OmicsFL.git"     # Everything
```

### From source (development)

```bash
git clone https://github.com/briannamueller/OmicsFL.git
cd OmicsFL
pip install -e .
```

### Core dependencies (minimal)

- numpy >= 1.22
- pandas >= 1.4
- pyarrow >= 8.0

## How It Works

1. **Download** — Fetches TCGA Pan-Cancer expression data from UCSC Xena (pre-processed) or GDC API (raw counts)
2. **Preprocess** — Filters low-expression genes, selects top-K by variance, encodes classification labels from clinical metadata
3. **Partition** — Groups samples by contributing hospital (natural federation) or applies synthetic non-IID schemes
4. **Split** — Creates deterministic per-client train/test/val splits
5. **Save** — Writes per-client `.npz` files + `config.json` with full reproducibility metadata

Each run produces a deterministic partition ID (hash of all parameters), preventing accidental overwrites and ensuring reproducibility.

## Reproducibility

Same parameters always produce the same partition ID and identical output:

```python
from omicsfl.fingerprint import build_partition_id

# This hash is deterministic — same params = same ID
pid = build_partition_id(
    source="xena", cohorts=["LUAD", "BRCA"], task="cancer_type",
    partition_strategy="natural", top_genes=2000, min_samples=20,
    train_ratio=0.8, val_ratio=0.0, seed=42,
)
```

## License

MIT
