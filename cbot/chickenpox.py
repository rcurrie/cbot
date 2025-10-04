"""Train a temporal GNN on the chickenpox dataset."""

# %%
import torch
from tqdm import tqdm

torch.manual_seed(1)

import torch.nn.functional as F
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.nn.recurrent import DCRNN

# %%
loader = ChickenpoxDatasetLoader(index=True)

train_dataLoader, val_dataLoader, test_dataLoader, edges, edge_weights = (
    loader.get_index_dataset()
)

print("Loaded chickenpox dataset:")
print(f"Number of nodes: {torch.unique(edges).size(0)}")
print(f"Number of edges: {edges.size(1)}")


# %%
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features: int):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(in_channels=node_features, out_channels=32, K=1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(
        self,
        x: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
    ) -> torch.Tensor:
        h: torch.Tensor = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


# %%
model = RecurrentGCN(node_features=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(100)):
    cost = torch.Tensor([0.0])
    for batch_idx, (x, y) in enumerate(train_dataLoader):
        # x shape: [batch, time_steps, nodes, features]
        # Process each time step in the batch sequentially
        batch_size, time_steps, num_nodes, num_features = x.shape

        for t in range(time_steps):
            # Get snapshot at time t: [batch, nodes, features]
            x_t = x[:, t, :, :].float()  # Shape: [batch, nodes, features]
            y_t = y[:, t, :, :].squeeze(-1).float()  # Shape: [batch, nodes]

            # Process each item in batch
            for b in range(batch_size):
                # x_t[b] shape: [nodes, features]
                y_hat = model(x_t[b], edges, edge_weights)
                cost = cost + torch.mean((y_hat.squeeze() - y_t[b]) ** 2)

    total_samples = (batch_idx + 1) * batch_size * time_steps
    cost = cost / total_samples
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()


# %%
