#!/bin/bash
#$ -S /bin/bash
#$ -N omicsfl_gdc33
#$ -q UI,MANSCI
#$ -pe smp 8
#$ -l mf=64G
#$ -o /Users/brimueller/OmicsFL/logs/test_gdc_all33.out
#$ -e /Users/brimueller/OmicsFL/logs/test_gdc_all33.err
#$ -cwd

# OmicsFL: Full 33-cohort test via GDC API
# Submit with: qsub scripts/test_gdc_all33.sh

set -e

PYTHON="/old_Users/brimueller/.local/share/mamba/envs/pygtest-cu117/bin/python"
REPO="/Users/brimueller/OmicsFL"
CACHE_DIR="/Users/brimueller/OmicsFL/cache/gdc"
OUTPUT_DIR="/Users/brimueller/OmicsFL/data"

export PYTHONPATH="${REPO}"

echo "=================================================="
echo " OmicsFL: All 33 TCGA Cohorts via GDC (min=100)"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo " Slots: ${NSLOTS}"
echo "=================================================="
echo ""

mkdir -p "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"

${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO}')

from pathlib import Path
from omicsfl.download.gdc import download as gdc_download
from omicsfl.download.registry import TCGA_COHORTS
from omicsfl.preprocess.gene_filter import filter_by_variance, filter_by_nonzero_fraction
from omicsfl.preprocess.normalize import log2_transform
from omicsfl.preprocess.labels import encode_cancer_type
from omicsfl.partition.natural import partition_by_center
from omicsfl.split import train_test_val_split
from omicsfl.fingerprint import build_partition_id
from omicsfl.core import load_partition
import numpy as np
import pandas as pd
import json

cache_dir = Path('${CACHE_DIR}')
output_dir = Path('${OUTPUT_DIR}')

cohorts = list(TCGA_COHORTS)  # All 33

print('=' * 60)
print(f' OmicsFL GDC: ALL {len(cohorts)} TCGA cohorts')
print(f' min_samples=100, top_genes=2000')
print('=' * 60)
print()

# Step 1: Download all cohorts from GDC
print('[1/7] Downloading from GDC API ...')
print(f'  This will download {len(cohorts)} cohorts. May take a while.')
print()
results = gdc_download(cohorts, output_dir=cache_dir)
print()

# Step 2: Load and merge
print('[2/7] Loading and merging count matrices ...')
all_expr = []
all_cohort_labels = []

for cohort in sorted(results.keys()):
    paths = results[cohort]
    expr = pd.read_parquet(paths['counts'])
    # Filter to tumor samples
    tumor_mask = []
    for idx in expr.index:
        parts = idx.split('-')
        if len(parts) >= 4:
            try:
                is_tumor = int(parts[3][:2]) < 10
            except ValueError:
                is_tumor = True
        else:
            is_tumor = True
        tumor_mask.append(is_tumor)
    expr = expr.loc[tumor_mask]
    all_expr.append(expr)
    all_cohort_labels.extend([cohort] * len(expr))
    print(f'  {cohort}: {len(expr)} tumor samples')

# Align on shared genes
print(f'  Aligning on shared genes ...')
shared_genes = set(all_expr[0].columns)
for e in all_expr[1:]:
    shared_genes &= set(e.columns)
shared_genes = sorted(shared_genes)
print(f'  Shared genes: {len(shared_genes)}')

all_expr = [e[shared_genes] for e in all_expr]
expression = pd.concat(all_expr, axis=0)
cohort_assignments = pd.Series(all_cohort_labels, index=expression.index)
print(f'  Combined: {expression.shape[0]} samples x {expression.shape[1]} genes')

# Step 3: Normalize (raw counts -> log2)
print()
print('[3/7] Normalizing (log2(counts + 1)) ...')
expression = log2_transform(expression, pseudocount=1.0)
print(f'  Range: [{expression.values.min():.2f}, {expression.values.max():.2f}]')

# Step 4: Encode labels
print()
print('[4/7] Encoding cancer type labels ...')
actual_cohorts = sorted(cohort_assignments.unique().tolist())
labels, class_map = encode_cancer_type(
    expression.index.tolist(), cohort_assignments, actual_cohorts,
)
print(f'  Classes: {len(class_map)}')
print(f'  Samples: {len(labels)}')
cohort_dist = cohort_assignments.value_counts()
print(f'  Top 5 cohorts: {cohort_dist.head(5).to_dict()}')
print(f'  Bottom 5 cohorts: {cohort_dist.tail(5).to_dict()}')

# Step 5: Gene filtering
print()
print('[5/7] Filtering genes ...')
expression = filter_by_nonzero_fraction(expression, min_fraction=0.1)
print(f'  After nonzero filter: {expression.shape[1]} genes')
expression = filter_by_variance(expression, top_k=2000)
print(f'  After variance filter: {expression.shape[1]} genes')

# Step 6: Partition by center (min 100)
print()
print('[6/7] Partitioning by hospital (min 100 samples) ...')
clients = partition_by_center(expression, labels, min_samples=100)
print(f'  {len(clients)} hospital clients')
print(f'  Top 10:')
for c in clients[:10]:
    unique_labels = np.unique(c['y'])
    print(f'    {c[\"center_id\"]}: {len(c[\"y\"])} samples, {len(unique_labels)} cancer types')
if len(clients) > 10:
    print(f'  ... and {len(clients) - 10} more')

# Step 7: Save
print()
print('[7/7] Splitting and saving ...')
partition_id = build_partition_id(
    source='gdc', cohorts=actual_cohorts, task='cancer_type',
    partition_strategy='natural', top_genes=2000,
    min_samples=100, train_ratio=0.8, val_ratio=0.0, seed=42,
)
out_path = output_dir / f'gdc_all33_{partition_id}'
out_path.mkdir(parents=True, exist_ok=True)
(out_path / 'train').mkdir(exist_ok=True)
(out_path / 'test').mkdir(exist_ok=True)

rng = np.random.default_rng(42)
client_ids = []
client_sizes = []

for idx, client in enumerate(clients):
    splits = train_test_val_split(client['x'], client['y'], train_ratio=0.8, rng=rng)
    np.savez_compressed(out_path / 'train' / f'{idx}.npz',
                        x=splits['train']['x'], y=splits['train']['y'])
    np.savez_compressed(out_path / 'test' / f'{idx}.npz',
                        x=splits['test']['x'], y=splits['test']['y'])
    client_ids.append(client['center_id'])
    client_sizes.append(len(client['y']))

config = {
    'partition_id': partition_id,
    'source': 'gdc',
    'cohorts': sorted(actual_cohorts),
    'task': 'cancer_type',
    'partition_strategy': 'natural',
    'num_clients': len(clients),
    'num_classes': len(class_map),
    'num_features': 2000,
    'top_genes': 2000,
    'min_samples': 100,
    'train_ratio': 0.8,
    'val_ratio': 0.0,
    'seed': 42,
    'client_ids': client_ids,
    'client_sizes': client_sizes,
    'class_map': {str(k): v for k, v in class_map.items()},
}
with open(out_path / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Verify
p = load_partition(out_path)
x, y = p.get_train(0)

print(f'  Saved to: {out_path}')
print(f'  {p}')
print(f'  Client 0 train: {x.shape}')

# Summary
print()
print('=' * 60)
print(' RESULTS SUMMARY')
print('=' * 60)
print(f'  Source:          GDC (Data Release latest)')
print(f'  Partition ID:    {partition_id}')
print(f'  Cancer types:    {len(class_map)}')
print(f'  Hospital clients:{len(clients)}')
print(f'  Total samples:   {sum(client_sizes)}')
print(f'  Features:        2000 genes')
print(f'  Avg samples/client: {sum(client_sizes) // len(clients)}')
print(f'  Min client size: {min(client_sizes)}')
print(f'  Max client size: {max(client_sizes)}')
print()
print('  Label heterogeneity:')
n_types_per_client = [len(np.unique(c['y'])) for c in clients]
print(f'    Avg cancer types per client: {np.mean(n_types_per_client):.1f}')
print(f'    Min: {min(n_types_per_client)}, Max: {max(n_types_per_client)}')
print()
print('=' * 60)
print(' SUCCESS')
print('=' * 60)
"

echo ""
echo "Finished: $(date)"
