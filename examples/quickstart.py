"""OmicsFL Quickstart — generate a TCGA federated partition in 5 lines."""

from omicsfl import generate, load_partition

# Generate a multi-cohort cancer type classification partition
# (downloads data on first run, cached for subsequent calls)
path = generate(
    cohorts=["LUAD", "LUSC", "BRCA", "PAAD"],
    task="cancer_type",
    partition="natural",
    top_genes=2000,
)

# Load and inspect
partition = load_partition(path)
print(partition)
print(f"Clients: {partition.client_ids[:5]} ...")

# Access data for client 0
x_train, y_train = partition.get_train(client=0)
print(f"Client 0: {x_train.shape[0]} train samples, {x_train.shape[1]} features")
