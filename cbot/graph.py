# %%
import torch
from torch_geometric.data import Data

# %%
device = torch.device("mps")

# %%
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long, device=device)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float, device=device)

data = Data(x=x, edge_index=edge_index)
# %%
