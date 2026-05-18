"""Example: Using OmicsFL with Flower for federated simulation.

Requires: pip install omicsfl[flower,torch]
"""

from omicsfl import generate, load_partition
from omicsfl.integrations.flower import get_numpy_data

# Step 1: Generate partition
path = generate(
    cohorts=["LUAD", "LUSC", "BRCA"],
    task="cancer_type",
    partition="natural",
    top_genes=2000,
)

# Step 2: Load data in Flower-friendly format
data = get_numpy_data(path)
partition = load_partition(path)

print(f"Ready for Flower simulation:")
print(f"  {partition.num_clients} clients")
print(f"  {partition.num_features} features")
print(f"  {partition.num_classes} classes")

# Step 3: Use with Flower (example structure)
# import flwr as fl
# import torch
# import torch.nn as nn
#
# class SimpleModel(nn.Module):
#     def __init__(self, n_features, n_classes):
#         super().__init__()
#         self.fc = nn.Linear(n_features, n_classes)
#
#     def forward(self, x):
#         return self.fc(x)
#
# def client_fn(cid: str):
#     x_train, y_train = data[int(cid)]["train"]
#     x_test, y_test = data[int(cid)]["test"]
#     # ... define Flower NumPyClient with your training logic
#
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=partition.num_clients,
#     config=fl.server.ServerConfig(num_rounds=10),
# )
