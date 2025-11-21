"""Token movement prediction using TGCN on DEX swap networks.

Architecture:
- Nodes: Tokens
- Edges: Swaps between tokens with temporal and flow information
- Node features: Static (materialized state) from swap bars
- Node labels: Triple-barrier method outputs (up/down/stay predictions)
- Edge features: Flow, volume, time delta, tick count

The model predicts if a token will move up/down/stay the same within a future window.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from tgm import DGraph
from tgm.data import DGData, DGDataLoader
from tgm.nn import TGCN, NodePredictor
from torch import nn
from tqdm import tqdm

# %%
# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CHECKPOINT_DIR / "tgcn_model.pt"
ENCODER_PATH = CHECKPOINT_DIR / "label_encoder.pkl"
DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")

# Hyperparameters
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 3  # up (0), stay (1), down (2)
LABEL_UP = 2  # Class label for price increase
LABEL_DOWN = 0  # Class label for price decrease
BATCH_TIME_GRAN = "h"  # Hourly batches for temporal iteration (lowercase h for hours)
LEARNING_RATE = 1e-3
EPOCHS = 1
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logger.info("Using device: %s", DEVICE)

# %%
# Load labeled swap bars
logger.info("Loading labeled bars from %s", DATA_PATH)
df = pl.read_parquet(DATA_PATH).sort("bar_close_timestamp")
logger.info("Bars loaded: %s", f"{df.shape[0]:,}")

# %%
# Encode token IDs (0-indexed for TGM)
logger.info("Encoding token IDs")
le = LabelEncoder()
all_tokens = np.concatenate(
    [
        df["src_token_id"].to_numpy(),
        df["dest_token_id"].to_numpy(),
    ],
)
le.fit(all_tokens)
num_tokens = len(le.classes_)
logger.info("Unique tokens: %s", f"{num_tokens:,}")

src = le.transform(df["src_token_id"].to_numpy()).astype(np.int32)
dst = le.transform(df["dest_token_id"].to_numpy()).astype(np.int32)

# %%
# Prepare DGData: Separate node features from labels
# Strategy:
# - Node features: src_fracdiff, rolling_volatility (materialized from swap bars)
# - Node labels: Triple-barrier outputs {-1,0,1} → {0,1,2}
# - Edge features: src_flow_usdc, dest_flow_usdc, tick_count, bar_time_delta_sec

logger.info("Preparing DGData structure")

# Convert timestamps to seconds since start (TGM requirement)
timestamps_sec = (
    (df["bar_close_timestamp"] - df["bar_close_timestamp"].min())
    .dt.total_seconds()
    .cast(pl.Int32)
    .to_numpy()
    .astype(np.int32)
)

# Edge features: pack financial metrics for temporal memory updates
# Per Prado: src_fracdiff and dest_fracdiff are IID stationary streams
# key to the model's learning signal
edge_feats = (
    df.select(
        [
            "src_fracdiff",
            # "dest_fracdiff", may have nulls...
            "src_flow_usdc",
            "dest_flow_usdc",
            "tick_count",
            "rolling_volatility",
            "bar_time_delta_sec",
        ],
    )
    .fill_null(0.0)
    .to_numpy()
    .astype(np.float32)
)

# Node events: one per swap with the SOURCE token's label
# This represents the outcome for the token being swapped out
node_timestamps = timestamps_sec  # Same as bar close times
node_ids = src  # Predict on source token (the one being sold)
# Labels: shift from {-1,0,1} to {0,1,2} for CrossEntropyLoss
dynamic_node_feats = (
    (df["label"].fill_null(0).to_numpy().astype(np.int64) + 1)
    .reshape(-1, 1)
    .astype(np.float32)
)

logger.info(
    "Node events: %s | Edge events: %s | Node feature dim: %s",
    len(node_ids),
    len(src),
    dynamic_node_feats.shape[1],
)

# %%
# Construct DGData object
data = DGData.from_raw(
    edge_timestamps=torch.from_numpy(timestamps_sec),
    edge_index=torch.from_numpy(np.column_stack((src, dst))),
    edge_feats=torch.from_numpy(edge_feats),
    node_timestamps=torch.from_numpy(node_timestamps),
    node_ids=torch.from_numpy(node_ids),
    dynamic_node_feats=torch.from_numpy(dynamic_node_feats),
    time_delta="s",  # seconds
)

logger.info("DGData created: %s nodes, %s edges", num_tokens, len(src))

# %%
# Create static node features (random initialization)
# In practice, you could add token metadata here (market cap, liquidity, etc.)
static_node_feats = torch.randn(
    (num_tokens, EMBED_DIM),
    device=DEVICE,
).float()


# %%
# Build model architecture
class TokenPredictorModel(nn.Module):
    """Temporal GCN encoder + linear decoder for node-level token prediction."""

    def __init__(
        self,
        num_nodes: int,
        static_feat_dim: int,
        embed_dim: int,
        output_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize token predictor model.

        Args:
            num_nodes: Total number of nodes (tokens) in the graph.
            static_feat_dim: Dimension of static node features.
            embed_dim: Embedding dimension for TGCN hidden state.
            output_dim: Output dimension (number of classes).
            device: Device to place tensors on.

        """
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # Temporal GCN backbone for learning node embeddings
        self.tgcn = TGCN(
            in_channels=static_feat_dim,
            out_channels=embed_dim,
        )

        # Classification head
        self.decoder = NodePredictor(
            in_dim=embed_dim,
            out_dim=output_dim,
            hidden_dim=embed_dim,
        )

    def forward(
        self,
        batch: Any,
        static_node_feats: torch.Tensor,
    ) -> torch.Tensor | None:
        """Forward pass through encoder and decoder.

        Args:
            batch: DGBatch with src, dst, time, edge_feats, etc.
            static_node_feats: [num_nodes, static_feat_dim].

        Returns:
            logits: [num_nodes_in_batch, output_dim] or None if no nodes.

        """
        # Get node embeddings from TGCN
        # TGCN signature: forward(X, edge_index, edge_weight=None, H=None)
        # X is node features, edge_index is [2, num_edges]
        z = self.tgcn(static_node_feats, batch.edge_index)

        # Only predict on nodes with labels in this batch
        if batch.node_ids is not None:
            z_node = z[batch.node_ids]
            logits = self.decoder(z_node)
            return logits
        return None


# %%
# Initialize model
logger.info("Initializing model")
model = TokenPredictorModel(
    num_nodes=num_tokens,
    static_feat_dim=EMBED_DIM,
    embed_dim=HIDDEN_DIM,
    output_dim=NUM_CLASSES,
    device=DEVICE,
).to(DEVICE)

# %%
# Training setup
dg = DGraph(data, device=DEVICE)
logger.info("DGraph created: %s edges total", len(dg.edges[0]))

# Create temporal dataloader (batches by hourly windows)
loader = DGDataLoader(dg, batch_unit=BATCH_TIME_GRAN)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction="mean")


# %%
def train_epoch(
    loader: DGDataLoader,
    model: TokenPredictorModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: str,
) -> float:
    """Train for one epoch.

    Args:
        loader: DataLoader for batches.
        model: The model to train.
        optimizer: Optimizer for gradient updates.
        criterion: Loss function.
        device: Device to train on.

    Returns:
        Average loss for the epoch.

    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        # Skip empty batches
        if batch.src.shape[0] == 0 or batch.node_ids is None:
            continue

        # Move batch to device
        batch.src = batch.src.to(device)
        batch.dst = batch.dst.to(device)
        batch.time = batch.time.to(device)

        # Construct edge index for TGCN
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        batch.edge_index = edge_index

        # Move node features to device
        batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
        batch.node_ids = batch.node_ids.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch, static_node_feats)

        if logits is None:
            continue

        # Target labels (extract from dynamic_node_feats and convert to LongTensor)
        y_true = batch.dynamic_node_feats.long().squeeze(-1)

        # Compute loss and backprop
        loss = criterion(logits, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info("Train Loss: %.4f", avg_loss)
    return avg_loss


# %%
@torch.no_grad()
def eval_epoch(
    loader: DGDataLoader,
    model: TokenPredictorModel,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the model and extract bullish recommendations.

    Args:
        loader: DataLoader for batches.
        model: The model to evaluate.
        device: Device to evaluate on.

    Returns:
        Tuple of (predictions, node_ids).

    """
    model.eval()
    predictions: list[np.ndarray] = []
    node_ids_pred: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Evaluating"):
        if batch.src.shape[0] == 0 or batch.node_ids is None:
            continue

        # Move to device
        batch.src = batch.src.to(device)
        batch.dst = batch.dst.to(device)
        batch.time = batch.time.to(device)
        edge_index = torch.stack([batch.src, batch.dst], dim=0)
        batch.edge_index = edge_index

        batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
        batch.node_ids = batch.node_ids.to(device)

        # Forward pass
        logits = model(batch, static_node_feats)
        if logits is None:
            continue

        # Get class predictions
        preds = torch.argmax(logits, dim=1)  # [0, 1, 2] → {down, stay, up}
        predictions.append(preds.cpu().numpy())
        node_ids_pred.append(batch.node_ids.cpu().numpy())

    if len(predictions) == 0:
        logger.warning("No predictions generated")
        return np.array([]), np.array([])

    all_preds = np.concatenate(predictions)
    all_node_ids = np.concatenate(node_ids_pred)

    return all_preds, all_node_ids


# %%
# Training loop
logger.info("Starting training for %d epochs", EPOCHS)
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(loader, model, optimizer, criterion, DEVICE)
    logger.info("Epoch %d/%d - Loss: %.4f", epoch, EPOCHS, loss)

# %%
# Evaluation and recommendations
logger.info("Evaluating model and extracting recommendations")
preds, node_ids_list = eval_epoch(loader, model, DEVICE)

# Aggregate predictions by token
token_predictions: dict[int, list[int]] = {}
for node_id, pred in zip(node_ids_list, preds, strict=True):
    if node_id not in token_predictions:
        token_predictions[node_id] = []
    token_predictions[node_id].append(pred)

# Calculate bullish score per token (% of "up" predictions)
recommendations: list[dict[str, Any]] = []
for node_id, pred_list in token_predictions.items():
    up_count = sum(1 for p in pred_list if p == LABEL_UP)
    down_count = sum(1 for p in pred_list if p == LABEL_DOWN)
    bullish_score = up_count / len(pred_list)

    # Decode token ID back to original token address
    token_addr = le.inverse_transform([node_id])[0]
    recommendations.append(
        {
            "token": token_addr,
            "bullish_score": bullish_score,
            "up_count": up_count,
            "down_count": down_count,
            "total_samples": len(pred_list),
        },
    )

# Sort by bullish score (highest first)
recommendations.sort(key=lambda x: x["bullish_score"], reverse=True)

# %%
# Display recommendations
logger.info("=" * 80)
logger.info("TOKEN PRICE MOVEMENT RECOMMENDATIONS")
logger.info("=" * 80)
logger.info("Tokens ranked by probability of price increase:")
logger.info("-" * 80)

for i, rec in enumerate(recommendations[:20], 1):  # Top 20
    logger.info(
        "%2d. %s | Bullish: %.1f%% (Up: %d, Down: %d, Samples: %d)",
        i,
        str(rec["token"])[:8],
        rec["bullish_score"] * 100,
        rec["up_count"],
        rec["down_count"],
        rec["total_samples"],
    )

logger.info("-" * 80)
logger.info("Analysis complete. %d unique tokens evaluated.", len(recommendations))

# %%
# Save model checkpoint
logger.info("Saving model to %s", MODEL_PATH)
torch.save(model.state_dict(), MODEL_PATH)
logger.info("Model saved successfully")
