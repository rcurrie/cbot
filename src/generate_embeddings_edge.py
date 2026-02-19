"""Generate GNN node embeddings via self-supervised edge regression.

WHY: Link prediction failed (AUROC 0.4954 = random) because pools (links) always
exist in Uniswap. The signal isn't "which tokens swap" but "how much flows through
existing pools". Edge regression predicts flow magnitude, capturing trading intensity
rather than link existence.

WHAT: Train Temporal GNN (TGCN) using heterogeneous graph where pools are edges:
- Nodes = Tokens (59 unique)
- Edges = Pools (145 unique, static adjacency)
- Node Features = fracdiff, volatility (token properties)
- Edge Features = liquidity_close, tick_delta (pool state)
- Regression Target = src_flow_usdc at time t+1

HOW:
1. Build static pool-token adjacency (145 pools connecting 59 tokens)
2. Create temporal edge updates (pool features + flow targets shifted by +1)
3. Train via Huber loss (robust to flow outliers)
4. Extract token embeddings for downstream supervised learning

INPUT: data/labeled_log_fracdiff_price.parquet (505K bars, 138 pools, 59 tokens)
OUTPUT: data/embeddings/edge_regression_{timestamp}.npz

References:
- Uniswap V3 Whitepaper: Concentrated liquidity AMM design
- Heterogeneous GNN: https://arxiv.org/abs/1903.07293
- Huber Loss for Robustness: Statistical robustness to outliers

"""

import json
import logging
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from torch import nn

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console = Console()

# Paths
DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")
TOKENS_PATH = Path("data/tokens.json")
OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters for Edge Regression
NODE_FEAT_DIM = 2  # fracdiff, volatility (token properties)
EDGE_FEAT_DIM = 2  # liquidity_close, tick_delta (pool state)
EMBED_DIM = 64  # Token embedding dimension
BATCH_SIZE = 200  # seconds
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0  # No L2 regularization - signal is too fragile
PATIENCE = 5  # Early stopping patience (stop at peak performance)
DEVICE = "cpu"  # CPU for compatibility

# CLI App
app = typer.Typer(help="Generate token embeddings via edge regression on heterogeneous graph.")


# ============================================================================
# Phase 1: Heterogeneous Graph Construction
# ============================================================================


def build_heterogeneous_graph(df: pl.DataFrame) -> dict[str, Any]:
    """Build static pool-token adjacency matrix.

    Args:
        df: DataFrame with columns: pool_id, src_token_id, dest_token_id.

    Returns:
        Dictionary with:
            - pool_ids: List[str] - 145 unique pool addresses
            - pool_to_idx: Dict[str, int] - pool address → pool index
            - edge_index: np.ndarray - [2, num_pools] static adjacency
            - token_to_idx: Dict[str, int] - token address → node index
            - idx_to_token: Dict[int, str] - node index → token address
            - num_pools: int
            - num_tokens: int

    """
    logger.info("Building heterogeneous graph (pools as edges, tokens as nodes)")

    # Step 1: Extract unique pools and their token pairs
    # Each pool consistently connects the same 2 tokens
    pool_structure = (
        df.group_by("pool_id")
        .agg([
            pl.col("src_token_id").first().alias("token_a"),
            pl.col("dest_token_id").first().alias("token_b"),
            pl.len().alias("num_bars"),
        ])
        .sort("num_bars", descending=True)
    )

    pool_ids = pool_structure["pool_id"].to_list()
    token_a_list = pool_structure["token_a"].to_list()
    token_b_list = pool_structure["token_b"].to_list()

    logger.info("  Found %d unique pools", len(pool_ids))

    # Step 2: Encode tokens as node IDs
    all_tokens = sorted(set(token_a_list + token_b_list))
    token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}

    logger.info("  Found %d unique tokens", len(all_tokens))

    # Step 3: Build static edge_index [2, num_pools]
    # edge_index[0] = source token IDs (token_a)
    # edge_index[1] = dest token IDs (token_b)
    token_a_ids = np.array([token_to_idx[t] for t in token_a_list], dtype=np.int64)
    token_b_ids = np.array([token_to_idx[t] for t in token_b_list], dtype=np.int64)

    edge_index = np.stack([token_a_ids, token_b_ids], axis=0)  # [2, num_pools]

    logger.info("  Edge index shape: %s", edge_index.shape)

    # Step 4: Create pool index mapping
    pool_to_idx = {pool_id: idx for idx, pool_id in enumerate(pool_ids)}

    return {
        "pool_ids": pool_ids,
        "pool_to_idx": pool_to_idx,
        "edge_index": edge_index,
        "token_to_idx": token_to_idx,
        "idx_to_token": idx_to_token,
        "num_pools": len(pool_ids),
        "num_tokens": len(all_tokens),
    }


def prepare_temporal_splits(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data chronologically for train/val/test.

    Args:
        df: DataFrame sorted by timestamp.
        train_ratio: Fraction for training (default 70%).
        val_ratio: Fraction for validation (default 15%).

    Returns:
        Tuple of (train_df, val_df, test_df) with no temporal leakage.

    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    logger.info("Temporal split:")
    logger.info("  Train: %d bars (%.1f%%)", len(train_df), 100 * train_ratio)
    logger.info("  Val:   %d bars (%.1f%%)", len(val_df), 100 * val_ratio)
    logger.info("  Test:  %d bars (%.1f%%)", len(test_df), 100 * (1 - train_ratio - val_ratio))

    return train_df, val_df, test_df


def build_temporal_edge_data(
    df: pl.DataFrame,
    graph_structure: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Convert bars to temporal edge + node updates with flow targets.

    CRITICAL: Shift targets by +1 timestep to avoid temporal leakage.
    For bar at time t: features from t, target = flow at t+1 (next bar for same pool).

    Args:
        df: Bar-level data sorted chronologically by timestamp.
        graph_structure: Static pool-token adjacency from build_heterogeneous_graph().

    Returns:
        Dictionary with numpy arrays:
            - timestamps: [num_bars] - bar close timestamps (seconds since epoch)
            - pool_indices: [num_bars] - pool index for each bar
            - src_token_indices: [num_bars] - source token for each bar
            - dest_token_indices: [num_bars] - dest token for each bar
            - node_features: [num_bars, 2, NODE_FEAT_DIM] - [src_feats, dst_feats]
            - edge_features: [num_bars, EDGE_FEAT_DIM] - pool state features
            - flow_targets: [num_bars] - log-transformed flow at t+1 (shifted)
            - valid_mask: [num_bars] - bool mask (False for last bar per pool)

    """
    logger.info("Building temporal edge data with shifted flow targets")

    pool_to_idx = graph_structure["pool_to_idx"]
    token_to_idx = graph_structure["token_to_idx"]

    # Extract columns
    timestamps = df["bar_close_timestamp"].cast(pl.Int64) // 1_000_000  # μs → ms
    timestamps = (timestamps // 1000).to_numpy()  # ms → seconds

    pool_ids = df["pool_id"].to_list()
    src_tokens = df["src_token_id"].to_list()
    dest_tokens = df["dest_token_id"].to_list()

    # Node features (2-dim): fracdiff, volatility
    src_fracdiff = df["src_fracdiff"].to_numpy()
    dest_fracdiff = df["dest_fracdiff"].to_numpy()
    volatility = df["rolling_volatility"].to_numpy()

    # Edge features (2-dim): liquidity, tick_delta
    src_liquidity = df["src_liquidity_close"].to_numpy()
    src_tick_delta = df["src_tick_delta"].to_numpy()

    # Flow target (regression): src_flow_usdc
    src_flow = df["src_flow_usdc"].to_numpy()

    num_bars = len(df)

    # Convert to indices
    pool_indices = np.array([pool_to_idx[pid] for pid in pool_ids], dtype=np.int64)
    src_token_indices = np.array([token_to_idx[t] for t in src_tokens], dtype=np.int64)
    dest_token_indices = np.array([token_to_idx[t] for t in dest_tokens], dtype=np.int64)

    # Node features: [num_bars, 2, NODE_FEAT_DIM]
    # Axis 1: [src_features, dest_features]
    node_features = np.zeros((num_bars, 2, NODE_FEAT_DIM), dtype=np.float32)
    node_features[:, 0, 0] = src_fracdiff  # src fracdiff
    node_features[:, 0, 1] = volatility  # src volatility
    node_features[:, 1, 0] = dest_fracdiff  # dest fracdiff
    node_features[:, 1, 1] = volatility  # dest volatility (same for both)

    # Edge features: [num_bars, EDGE_FEAT_DIM]
    # CRITICAL: Normalize edge features to match scale of node features (mean≈0, std≈1)
    edge_features = np.zeros((num_bars, EDGE_FEAT_DIM), dtype=np.float32)

    # Normalize liquidity (currently mean~41, std~7.5)
    liquidity_norm = (src_liquidity - src_liquidity.mean()) / src_liquidity.std()
    edge_features[:, 0] = liquidity_norm

    # tick_delta already normalized (mean≈0, std≈0.37)
    edge_features[:, 1] = src_tick_delta

    # Flow targets: log-transform to handle wide range (10K-10M USDC)
    # Use log(abs(flow) + 1) to handle negative flows (sells)
    flow_log = np.log(np.abs(src_flow) + 1.0).astype(np.float32)

    # CRITICAL: Shift targets by +1 timestep (predict t+1 from t)
    # Group by pool and shift within each pool
    flow_targets = np.zeros(num_bars, dtype=np.float32)
    valid_mask = np.ones(num_bars, dtype=bool)

    # Sort by pool and timestamp to shift correctly
    pool_df = df.select(["pool_id"]).with_row_index()
    for pool_id in set(pool_ids):
        pool_mask = np.array([p == pool_id for p in pool_ids])
        pool_bar_indices = np.where(pool_mask)[0]

        # Shift: target[i] = flow[i+1]
        if len(pool_bar_indices) > 1:
            flow_targets[pool_bar_indices[:-1]] = flow_log[pool_bar_indices[1:]]
            valid_mask[pool_bar_indices[-1]] = False  # Last bar has no future target
        else:
            valid_mask[pool_bar_indices[0]] = False  # Single bar pool

    logger.info("  Total bars: %d", num_bars)
    logger.info("  Valid bars (with future targets): %d", valid_mask.sum())
    logger.info("  Removed bars (last per pool): %d", (~valid_mask).sum())

    # Replace NaNs with 0
    node_features = np.nan_to_num(node_features, nan=0.0)
    edge_features = np.nan_to_num(edge_features, nan=0.0)

    return {
        "timestamps": timestamps,
        "pool_indices": pool_indices,
        "src_token_indices": src_token_indices,
        "dest_token_indices": dest_token_indices,
        "node_features": node_features,
        "edge_features": edge_features,
        "flow_targets": flow_targets,
        "valid_mask": valid_mask,
    }


# ============================================================================
# Phase 2: Edge Regression Model
# ============================================================================


class EdgeRegressionModel(nn.Module):
    """GNN model for edge regression (predict flow magnitude on pool edges).

    Architecture:
        Token Node Features [fracdiff, volatility] → Node Embeddings [num_tokens, 64]
                                                       ↓
                              Simple MLP Message Passing over Pool Edges
                                                       ↓
                             Updated Node Embeddings [num_tokens, 64]
                                                       ↓
                  Edge Features [liquidity, tick] + Node Embeddings (concat)
                                                       ↓
                                Edge Regression Head → flow_prediction

    Note: We use a simplified MLP-based message passing instead of TGCN to avoid
    TGM library complexities. The model still captures pool-level interactions.

    """

    def __init__(
        self,
        num_tokens: int,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        embed_dim: int = EMBED_DIM,
    ):
        """Initialize edge regression model.

        Args:
            num_tokens: Number of unique tokens (nodes).
            node_feat_dim: Dimension of node features (2: fracdiff, volatility).
            edge_feat_dim: Dimension of edge features (2: liquidity, tick_delta).
            embed_dim: Token embedding dimension (64).

        """
        super().__init__()

        # 1. Node Feature Projector
        self.node_projector = nn.Linear(node_feat_dim, embed_dim)

        # 2. Learnable Token Embeddings (static personality)
        # Captures static traits not in time-series features
        self.node_embeddings = nn.Parameter(torch.randn(num_tokens, embed_dim) * 0.1)

        # 3. Message Passing Layer (simplified - just MLP aggregation)
        # Takes concatenated [src_embed, dst_embed] and updates src
        self.message_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 4. Edge Regression Head
        # Input: concat(src_embed, dst_embed, edge_features)
        # Output: single flow prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2 + edge_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Minimal dropout - signal is fragile
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        src_node_feats: torch.Tensor,
        dst_node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for edge regression.

        Args:
            src_node_feats: [batch_size, node_feat_dim] - source token features.
            dst_node_feats: [batch_size, node_feat_dim] - dest token features.
            edge_feats: [batch_size, edge_feat_dim] - pool state features.
            src_indices: [batch_size] - source token indices.
            dst_indices: [batch_size] - dest token indices.

        Returns:
            flow_predictions: [batch_size] - predicted log-flow values.

        """
        batch_size = src_node_feats.shape[0]

        # Project node features to embedding space
        src_proj = self.node_projector(src_node_feats)  # [batch_size, embed_dim]
        dst_proj = self.node_projector(dst_node_feats)  # [batch_size, embed_dim]

        # Add learnable embeddings
        src_embeds = src_proj + self.node_embeddings[src_indices]  # [batch_size, embed_dim]
        dst_embeds = dst_proj + self.node_embeddings[dst_indices]  # [batch_size, embed_dim]

        # Message passing: update src embeddings based on dst
        # Simplified aggregation - concat and transform
        messages = torch.cat([src_embeds, dst_embeds], dim=1)  # [batch_size, 2*embed_dim]
        updated_src = self.message_mlp(messages)  # [batch_size, embed_dim]

        # Edge prediction: combine updated embeddings + edge features
        edge_input = torch.cat(
            [updated_src, dst_embeds, edge_feats], dim=1
        )  # [batch_size, 2*embed_dim + edge_feat_dim]

        flow_predictions = self.edge_predictor(edge_input).squeeze(-1)  # [batch_size]

        return flow_predictions


# ============================================================================
# Phase 3: Training Loop
# ============================================================================


def train_epoch(
    model: EdgeRegressionModel,
    data: dict[str, np.ndarray],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    """Train for one epoch on edge regression task.

    Args:
        model: EdgeRegressionModel instance.
        data: Dictionary with temporal edge data (from build_temporal_edge_data).
        optimizer: PyTorch optimizer.
        loss_fn: Loss function (Huber loss).
        device: Device to run on.

    Returns:
        Average training loss for the epoch.

    """
    model.train()

    # Filter to valid bars only (those with future targets)
    valid_mask = data["valid_mask"]
    valid_indices = np.where(valid_mask)[0]

    # Shuffle valid indices for training
    np.random.shuffle(valid_indices)

    # Extract data
    node_features = data["node_features"][valid_indices]  # [N, 2, node_feat_dim]
    edge_features = data["edge_features"][valid_indices]  # [N, edge_feat_dim]
    src_indices = data["src_token_indices"][valid_indices]  # [N]
    dst_indices = data["dest_token_indices"][valid_indices]  # [N]
    targets = data["flow_targets"][valid_indices]  # [N]

    # Convert to tensors
    src_node_feats = torch.from_numpy(node_features[:, 0, :]).to(device)  # [N, node_feat_dim]
    dst_node_feats = torch.from_numpy(node_features[:, 1, :]).to(device)  # [N, node_feat_dim]
    edge_feats = torch.from_numpy(edge_features).to(device)  # [N, edge_feat_dim]
    src_idx_tensor = torch.from_numpy(src_indices).to(device)  # [N]
    dst_idx_tensor = torch.from_numpy(dst_indices).to(device)  # [N]
    targets_tensor = torch.from_numpy(targets).to(device)  # [N]

    # Forward pass
    optimizer.zero_grad()
    predictions = model(src_node_feats, dst_node_feats, edge_feats, src_idx_tensor, dst_idx_tensor)

    # Compute loss
    loss = loss_fn(predictions, targets_tensor)

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def validate(
    model: EdgeRegressionModel,
    data: dict[str, np.ndarray],
    device: str,
) -> dict[str, float]:
    """Validate model on regression metrics.

    Args:
        model: EdgeRegressionModel instance.
        data: Dictionary with temporal edge data.
        device: Device to run on.

    Returns:
        Dictionary with metrics: mse, mae, r2.

    """
    model.eval()

    # Filter to valid bars
    valid_mask = data["valid_mask"]
    valid_indices = np.where(valid_mask)[0]

    # Extract data
    node_features = data["node_features"][valid_indices]
    edge_features = data["edge_features"][valid_indices]
    src_indices = data["src_token_indices"][valid_indices]
    dst_indices = data["dest_token_indices"][valid_indices]
    targets = data["flow_targets"][valid_indices]

    # Convert to tensors
    src_node_feats = torch.from_numpy(node_features[:, 0, :]).to(device)
    dst_node_feats = torch.from_numpy(node_features[:, 1, :]).to(device)
    edge_feats = torch.from_numpy(edge_features).to(device)
    src_idx_tensor = torch.from_numpy(src_indices).to(device)
    dst_idx_tensor = torch.from_numpy(dst_indices).to(device)

    with torch.no_grad():
        predictions = model(src_node_feats, dst_node_feats, edge_feats, src_idx_tensor, dst_idx_tensor)

    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets

    # Compute metrics
    mse = float(np.mean((preds_np - targets_np) ** 2))
    mae = float(np.mean(np.abs(preds_np - targets_np)))

    # R² score
    ss_res = np.sum((targets_np - preds_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {"mse": mse, "mae": mae, "r2": r2}


# ============================================================================
# Phase 4: Main Training Loop
# ============================================================================


@app.command()
def main(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    architecture: str = typer.Option("EdgeRegression", help="Model architecture name"),
) -> None:
    """Train edge regression model on heterogeneous graph."""
    logger.info("=" * 70)
    logger.info("GNN EDGE REGRESSION TRAINING")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading data from %s", DATA_PATH)
    df = pl.read_parquet(DATA_PATH)
    logger.info("  Loaded %d bars", len(df))

    # Sort by timestamp
    df = df.sort("bar_close_timestamp")

    # Build heterogeneous graph
    graph_structure = build_heterogeneous_graph(df)

    logger.info("\nGraph structure:")
    logger.info("  Tokens (nodes): %d", graph_structure["num_tokens"])
    logger.info("  Pools (edges): %d", graph_structure["num_pools"])
    logger.info("  Static edge_index shape: %s", graph_structure["edge_index"].shape)

    # Split data
    train_df, val_df, test_df = prepare_temporal_splits(df)

    # Build temporal edge data for each split
    logger.info("\nBuilding temporal edge data...")
    logger.info("  Train split:")
    train_data = build_temporal_edge_data(train_df, graph_structure)

    logger.info("  Val split:")
    val_data = build_temporal_edge_data(val_df, graph_structure)

    logger.info("  Test split:")
    test_data = build_temporal_edge_data(test_df, graph_structure)

    # Initialize model
    logger.info("\nInitializing EdgeRegressionModel")
    model = EdgeRegressionModel(
        num_tokens=graph_structure["num_tokens"],
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    logger.info("  Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )  # L2 regularization
    loss_fn = nn.HuberLoss(delta=1.0)  # Robust to outliers

    # Training loop
    logger.info("\nStarting training for %d epochs", epochs)
    logger.info("  Device: %s", DEVICE)
    logger.info("  Learning rate: %.4f", LEARNING_RATE)
    logger.info("  Weight decay (L2): %.0e (disabled)", WEIGHT_DECAY)
    logger.info("  Dropout: 0.1")
    logger.info("  Early stopping patience: %d epochs", PATIENCE)
    logger.info("  Loss function: Huber (delta=1.0)")

    best_val_r2 = -np.inf
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_mse": [], "val_mae": [], "val_r2": []}

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_data, optimizer, loss_fn, DEVICE)
        history["train_loss"].append(train_loss)

        # Validate
        val_metrics = validate(model, val_data, DEVICE)
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %d/%d | Loss: %.4f | Val R²: %.4f | Val MAE: %.4f | Time: %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_metrics["r2"],
            val_metrics["mae"],
            epoch_time,
        )

        # Early stopping
        if val_metrics["r2"] > best_val_r2:
            best_val_r2 = val_metrics["r2"]
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered (patience=%d)", PATIENCE)
            break

    training_time = time.time() - start_time

    logger.info("\nTraining complete!")
    logger.info("  Best Val R²: %.4f (epoch %d)", best_val_r2, best_epoch)
    logger.info("  Training time: %.1f seconds", training_time)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = validate(model, test_data, DEVICE)
    logger.info("  Test R²:  %.4f", test_metrics["r2"])
    logger.info("  Test MAE: %.4f", test_metrics["mae"])
    logger.info("  Test MSE: %.4f", test_metrics["mse"])

    # Extract token embeddings
    logger.info("\nExtracting token embeddings...")
    embeddings = model.node_embeddings.detach().cpu().numpy()  # [num_tokens, embed_dim]
    logger.info("  Embeddings shape: %s", embeddings.shape)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"edge_regression_{timestamp}.npz"

    metadata = {
        "architecture": architecture,
        "epochs_trained": best_epoch,
        "best_val_r2": float(best_val_r2),
        "best_val_mae": float(history["val_mae"][best_epoch - 1]),
        "test_r2": float(test_metrics["r2"]),
        "test_mae": float(test_metrics["mae"]),
        "test_mse": float(test_metrics["mse"]),
        "node_feat_dim": NODE_FEAT_DIM,
        "edge_feat_dim": EDGE_FEAT_DIM,
        "embed_dim": EMBED_DIM,
        "num_tokens": graph_structure["num_tokens"],
        "num_pools": graph_structure["num_pools"],
        "training_time_sec": training_time,
    }

    # Save embeddings + metadata + token mapping
    np.savez(
        output_path,
        embeddings=embeddings,
        token_addresses=np.array(
            [graph_structure["idx_to_token"][i] for i in range(graph_structure["num_tokens"])]
        ),
        metadata=metadata,
        history=history,
    )

    logger.info("\nSaved embeddings to: %s", output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    app()