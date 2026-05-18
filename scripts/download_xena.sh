#!/bin/bash
#$ -S /bin/bash
#$ -N omicsfl_xena_download
#$ -q UI,MANSCI
#$ -pe smp 4
#$ -l mf=16G
#$ -o /Users/brimueller/OmicsFL/logs/download_xena.out
#$ -e /Users/brimueller/OmicsFL/logs/download_xena.err
#$ -cwd

# OmicsFL: Download TCGA data from Xena and run end-to-end test
# Submit with: qsub scripts/download_xena.sh

set -e

PYTHON="/old_Users/brimueller/.local/share/mamba/envs/pygtest-cu117/bin/python"
REPO="/Users/brimueller/OmicsFL"
CACHE_DIR="/Users/brimueller/OmicsFL/cache/xena"
OUTPUT_DIR="/Users/brimueller/OmicsFL/data"

export PYTHONPATH="${REPO}"

echo "=================================================="
echo " OmicsFL Xena Download + End-to-End Test"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo " Slots: ${NSLOTS}"
echo "=================================================="
echo ""

# Create directories
mkdir -p "${CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO}/logs"

# Step 1: Download phenotype (small, ~136 KB)
echo "[1/4] Downloading phenotype data ..."
PHENO_FILE="${CACHE_DIR}/TcgaTargetGTEX_phenotype.txt.gz"
if [ ! -f "${PHENO_FILE}" ]; then
    curl -L -o "${PHENO_FILE}" \
        "https://toil.xenahubs.net/download/TcgaTargetGTEX_phenotype.txt.gz"
    echo "  Done: $(ls -lh ${PHENO_FILE} | awk '{print $5}')"
else
    echo "  Already cached"
fi

# Step 2: Download survival data (~407 KB)
echo "[2/4] Downloading survival data ..."
SURV_FILE="${CACHE_DIR}/TCGA_survival_data.tsv"
if [ ! -f "${SURV_FILE}" ]; then
    curl -L -o "${SURV_FILE}" \
        "https://toil.xenahubs.net/download/TCGA_survival_data"
    echo "  Done: $(ls -lh ${SURV_FILE} | awk '{print $5}')"
else
    echo "  Already cached"
fi

# Step 3: Download expression matrix (~741 MB)
echo "[3/4] Downloading expression matrix (~741 MB, may take several minutes) ..."
EXPR_FILE="${CACHE_DIR}/tcga_RSEM_gene_tpm.gz"
if [ ! -f "${EXPR_FILE}" ]; then
    curl -L -o "${EXPR_FILE}" \
        "https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz"
    echo "  Done: $(ls -lh ${EXPR_FILE} | awk '{print $5}')"
else
    echo "  Already cached"
fi

echo ""
echo "All downloads complete:"
ls -lh "${CACHE_DIR}/"
echo ""

# Step 4: Run end-to-end generate pipeline
echo "[4/4] Running end-to-end generate test ..."
echo ""

${PYTHON} -c "
import sys
sys.path.insert(0, '${REPO}')

from pathlib import Path
from omicsfl.download.xena import load_expression, load_phenotype
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

cohorts = ['LUAD', 'LUSC', 'BRCA', 'COAD']

print('=' * 60)
print(' OmicsFL End-to-End Test: 4 cohorts, cancer_type, natural')
print('=' * 60)

# Load and filter expression
print('\n[1/6] Loading expression matrix ...')
expr = load_expression(
    cache_dir / 'tcga_RSEM_gene_tpm.gz',
    cohorts=cohorts,
    phenotype_path=cache_dir / 'TcgaTargetGTEX_phenotype.txt.gz',
)
print(f'  Expression: {expr.shape[0]} samples x {expr.shape[1]} genes')

# Load phenotype
print('\n[2/6] Loading phenotype ...')
pheno = load_phenotype(cache_dir / 'TcgaTargetGTEX_phenotype.txt.gz',
                       samples=expr.index.tolist())
print(f'  Phenotype: {pheno.shape}')
print(f'  Cohort distribution: {pheno[\"cohort\"].value_counts().to_dict()}')

# Encode labels
print('\n[3/6] Encoding labels ...')
cohort_assignments = pheno['cohort'].reindex(expr.index)
from omicsfl.preprocess.labels import encode_cancer_type
labels, class_map = encode_cancer_type(
    expr.index.tolist(), cohort_assignments, cohorts,
)
print(f'  Classes: {class_map}')
print(f'  Label dist: {dict(zip(*[a.tolist() for a in np.unique(labels, return_counts=True)]))}')

# Gene filtering
print('\n[4/6] Filtering genes ...')
expr = filter_by_nonzero_fraction(expr, min_fraction=0.1)
print(f'  After nonzero filter: {expr.shape[1]} genes')
expr = filter_by_variance(expr, top_k=2000)
print(f'  After variance filter: {expr.shape[1]} genes')

# Partition by center
print('\n[5/6] Partitioning by hospital ...')
clients = partition_by_center(expr, labels, min_samples=20)
print(f'  {len(clients)} clients')
for c in clients[:5]:
    print(f'    {c[\"center_id\"]}: {len(c[\"y\"])} samples')
if len(clients) > 5:
    print(f'    ... and {len(clients) - 5} more')

# Save
print('\n[6/6] Splitting and saving ...')
partition_id = build_partition_id(
    source='xena', cohorts=cohorts, task='cancer_type',
    partition_strategy='natural', top_genes=2000,
    min_samples=20, train_ratio=0.8, val_ratio=0.0, seed=42,
)
out_path = output_dir / partition_id
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
    'cohorts': sorted(cohorts),
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
print(f'\n  Saved to: {out_path}')
p = load_partition(out_path)
print(f'  {p}')
x, y = p.get_train(0)
print(f'  Client 0 train: {x.shape}')

print('\n' + '=' * 60)
print(' SUCCESS — OmicsFL end-to-end test complete')
print(f' Partition ID: {partition_id}')
print(f' Clients: {len(clients)}')
print(f' Total samples: {sum(client_sizes)}')
print('=' * 60)
"

echo ""
echo "Finished: $(date)"
