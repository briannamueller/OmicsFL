#!/bin/bash
#$ -S /bin/bash
#$ -N omicsfl_all33
#$ -q UI,MANSCI
#$ -pe smp 8
#$ -l mf=32G
#$ -o /Users/brimueller/OmicsFL/logs/test_all33.out
#$ -e /Users/brimueller/OmicsFL/logs/test_all33.err
#$ -cwd

# OmicsFL: Full 33-cohort cancer type classification test
# Submit with: qsub scripts/test_all_cohorts.sh

set -e

PYTHON="/old_Users/brimueller/.local/share/mamba/envs/pygtest-cu117/bin/python"
REPO="/Users/brimueller/OmicsFL"
CACHE_DIR="/Users/brimueller/OmicsFL/cache/xena"
OUTPUT_DIR="/Users/brimueller/OmicsFL/data"

export PYTHONPATH="${REPO}"

echo "=================================================="
echo " OmicsFL: All 33 TCGA Cohorts — Cancer Type"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo " Slots: ${NSLOTS}"
echo "=================================================="
echo ""

# Expression matrix should already be cached from prior job
if [ ! -f "${CACHE_DIR}/tcga_RSEM_gene_tpm.gz" ]; then
    echo "ERROR: Expression matrix not cached. Run download_xena.sh first."
    exit 1
fi

echo "Using cached data from: ${CACHE_DIR}"
ls -lh "${CACHE_DIR}/"
echo ""

${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO}')

from pathlib import Path
from omicsfl.download.xena import load_expression, load_phenotype
from omicsfl.download.registry import TCGA_COHORTS
from omicsfl.preprocess.gene_filter import filter_by_variance, filter_by_nonzero_fraction
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
print(f' OmicsFL: ALL {len(cohorts)} TCGA cohorts, cancer_type, natural')
print('=' * 60)
print(f' Cohorts: {cohorts}')
print()

# Load expression
print('[1/6] Loading expression matrix ...')
expr = load_expression(
    cache_dir / 'tcga_RSEM_gene_tpm.gz',
    cohorts=cohorts,
    phenotype_path=cache_dir / 'TcgaTargetGTEX_phenotype.txt.gz',
)
print(f'  Expression: {expr.shape[0]} samples x {expr.shape[1]} genes')

# Load phenotype
print()
print('[2/6] Loading phenotype ...')
pheno = load_phenotype(cache_dir / 'TcgaTargetGTEX_phenotype.txt.gz',
                       samples=expr.index.tolist())
print(f'  Phenotype: {pheno.shape}')
cohort_dist = pheno['cohort'].value_counts()
print(f'  Cohorts represented: {len(cohort_dist)}')
print(f'  Top 10 by size:')
for c, n in cohort_dist.head(10).items():
    print(f'    {c}: {n}')
print(f'  Bottom 5 by size:')
for c, n in cohort_dist.tail(5).items():
    print(f'    {c}: {n}')

# Encode labels
print()
print('[3/6] Encoding cancer type labels ...')
cohort_assignments = pheno['cohort'].reindex(expr.index)
# Only use cohorts that actually appear in the data
actual_cohorts = sorted(cohort_assignments.dropna().unique().tolist())
print(f'  Cohorts with data: {len(actual_cohorts)}')
labels, class_map = encode_cancer_type(
    expr.index.tolist(), cohort_assignments, actual_cohorts,
)
print(f'  Classes: {len(class_map)}')
print(f'  Total labeled samples: {len(labels)}')

# Gene filtering
print()
print('[4/6] Filtering genes ...')
expr = filter_by_nonzero_fraction(expr, min_fraction=0.1)
print(f'  After nonzero filter: {expr.shape[1]} genes')
expr = filter_by_variance(expr, top_k=2000)
print(f'  After variance filter: {expr.shape[1]} genes')

# Partition by center
print()
print('[5/6] Partitioning by hospital (min 20 samples) ...')
clients = partition_by_center(expr, labels, min_samples=20)
print(f'  {len(clients)} hospital clients')
print(f'  Top 10 by size:')
for c in clients[:10]:
    unique_labels = np.unique(c['y'])
    print(f'    {c[\"center_id\"]}: {len(c[\"y\"])} samples, {len(unique_labels)} cancer types')
if len(clients) > 10:
    print(f'  ... and {len(clients) - 10} more')

# Save
print()
print('[6/6] Splitting and saving ...')
partition_id = build_partition_id(
    source='xena', cohorts=actual_cohorts, task='cancer_type',
    partition_strategy='natural', top_genes=2000,
    min_samples=20, train_ratio=0.8, val_ratio=0.0, seed=42,
)
out_path = output_dir / f'all33_{partition_id}'
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
    'source': 'xena',
    'cohorts': sorted(actual_cohorts),
    'task': 'cancer_type',
    'partition_strategy': 'natural',
    'num_clients': len(clients),
    'num_classes': len(class_map),
    'num_features': 2000,
    'top_genes': 2000,
    'min_samples': 20,
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

# Summary stats
print()
print('=' * 60)
print(' RESULTS SUMMARY')
print('=' * 60)
print(f'  Partition ID:    {partition_id}')
print(f'  Cancer types:    {len(class_map)}')
print(f'  Hospital clients:{len(clients)}')
print(f'  Total samples:   {sum(client_sizes)}')
print(f'  Features:        2000 genes')
print(f'  Avg samples/client: {sum(client_sizes) // len(clients)}')
print(f'  Min client size: {min(client_sizes)}')
print(f'  Max client size: {max(client_sizes)}')
print()

# Label heterogeneity stats
print('  Label heterogeneity across clients:')
n_types_per_client = []
for c in clients:
    n_types_per_client.append(len(np.unique(c['y'])))
print(f'    Avg cancer types per client: {np.mean(n_types_per_client):.1f}')
print(f'    Min: {min(n_types_per_client)}, Max: {max(n_types_per_client)}')
print()
print('=' * 60)
print(' SUCCESS')
print('=' * 60)
"

echo ""
echo "Finished: $(date)"
