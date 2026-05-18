"""Example: custom partitioning with different strategies and parameters."""

from omicsfl import generate, load_partition

# Natural federation (default)
# Each hospital that contributed samples becomes a federated client.
path = generate(
    cohorts=["LUAD", "BRCA", "COAD", "SKCM"],
    task="cancer_type",
    partition="natural",
    min_samples=30,  # Only include hospitals with 30+ samples
    top_genes=5000,  # Keep more genes for richer features
    train_ratio=0.7,
    val_ratio=0.1,   # 70/10/20 train/val/test split
    seed=123,
)
partition = load_partition(path)
print(f"Natural: {partition.num_clients} hospitals as clients")

# Dirichlet non-IID — synthetic clients with controllable label heterogeneity.
path = generate(
    cohorts=["LUAD", "BRCA", "COAD", "SKCM"],
    task="cancer_type",
    partition="dirichlet",
    num_clients=20,
    alpha=0.1,  # Very non-IID (each client dominated by 1-2 classes)
    seed=42,
)
partition = load_partition(path)
print(f"Dirichlet(0.1): {partition.num_clients} clients")
for i in range(min(3, partition.num_clients)):
    info = partition.client_summary(i)
    print(f"  Client {i}: {info['label_distribution_train']}")

# Single cohort, binary task — stage prediction within one cancer type.
path = generate(
    cohorts=["LUAD"],
    task="stage",
    partition="natural",
    top_genes=1000,
)
partition = load_partition(path)
print(f"\nLUAD stage: {partition.num_clients} clients, "
      f"{partition.num_classes} classes")
