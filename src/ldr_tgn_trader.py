"""LDR-based Temporal Graph Network (TGN) trader with SupCon + CE loss.

WHY: Supervised Contrastive Learning (SupCon) structures the embedding space by
pulling same-class embeddings together and pushing different-class embeddings apart.
Combined with cross-entropy, this produces discriminative embeddings where:
1. Embeddings are interpretable (linear probes work)
2. Classes are well-separated (robust to noise)
3. Within-class structure is preserved (similar tokens cluster)

Unlike MCR², SupCon operates on pairs of normalized embeddings directly — no
covariance matrices needed. This avoids the representation collapse failure mode
where MCR² gradients vanish through rank-deficient covariance (Cholesky logdet).

WHAT: Heterogeneous bipartite graph trading system:
1. Nodes: Tokens (price dynamics) <-> Pools (liquidity structure)
2. Edges: Token trades through Pool (bidirectional for message passing)
3. Memory: TGN maintains temporal state for both node types
4. Embedding: GATv2Conv with dynamic attention for selective message passing
5. Loss: SupCon pulls same-class together + CE for decision boundaries

HOW: Walk-forward validation with memory persistence:
1. Train on N days with SupCon + CE loss (learn class-clustered embeddings)
2. Freeze weights, continue memory updates on trade day
3. Generate signals from embedding distances to class centroids
4. Retrain at end of day when true labels revealed

INPUT: data/labeled_log_fracdiff_price.parquet (swap events with labels)
       data/tokens.json (token metadata for display)
OUTPUT: Console backtest report, optional embeddings to data/ldr_embeddings/

References:
- Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
- Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
- Brody et al. "How Attentive are Graph Attention Networks?" (GATv2)
- Prado AFML Ch. 3-5: Triple-Barrier, Sample Weights, Fractional Differentiation

"""

# ruff: noqa: N801, N812, PLR0915, ARG001

import json
import logging
import math
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import typer
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DIVIDER_LENGTH = 70

app = typer.Typer(
    help="LDR-TGN trading system with SupCon + CE loss for discriminative embeddings.",
)

# Backtest parameters
TRAIN_WINDOW_DAYS = 30
TRADE_WINDOW_HOURS = (9, 17)  # 9am to 5pm EST
TOP_N_TOKENS = 5
SLIDE_STEP_DAYS = 1

# Risk management
STOP_LOSS_PCT = 0.20
TAKE_PROFIT_PCT = 2.0

# Model hyperparameters
TOKEN_FEAT_DIM = 4  # fracdiff, volatility, flow_magnitude, buying_pressure
POOL_FEAT_DIM = 4  # fee_tier, tick_spacing, volume, liquidity_util
EDGE_FEAT_DIM = 4  # amount_in, amount_out, tick_delta, time_delta
EMBED_DIM = 16  # Embedding dimension — kept small for sample sizes
HIDDEN_DIM = 64
NUM_CLASSES = 3  # down (0), neutral (1), up (2)
LABEL_DOWN = 0
LABEL_NEUTRAL = 1
LABEL_UP = 2

# SupCon hyperparameters (disabled by default — see USE_SUPCON)
SUPCON_TEMPERATURE = 0.5  # Temperature for contrastive loss
# Disabled: batch aggregates per-token, destroying per-event labels
USE_SUPCON = False

# GATv2 hyperparameters
GAT_HEADS = 4
GAT_DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
MEMORY_DIM = 32
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001  # Minimum relative improvement to reset patience

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logger.info(
    "Device: %s | Train window: %d days | Trade hours: %d-%d EST",
    DEVICE,
    TRAIN_WINDOW_DAYS,
    TRADE_WINDOW_HOURS[0],
    TRADE_WINDOW_HOURS[1],
)


# ==============================================================================
# SUPERVISED CONTRASTIVE LOSS
# ==============================================================================


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = SUPCON_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Supervised Contrastive loss (Khosla et al. 2020).

    For each anchor, pulls same-class embeddings closer and pushes
    different-class embeddings apart. Operates on L2-normalized embeddings.

    Unlike MCR², this works on pairs — no covariance matrices, no collapse mode.
    Gradients are strong even when embeddings start identical (the temperature
    creates non-zero gradients from any similarity difference).

    Args:
        z: Embeddings tensor of shape [n, d].
        labels: Class labels of shape [n].
        temperature: Temperature scaling (lower = sharper contrast).

    Returns:
        Tuple of (loss, metrics_dict).

    """
    z_norm = F.normalize(z, p=2, dim=1)
    n = z_norm.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=z.device), {"supcon": 0.0}

    # Pairwise cosine similarity [n, n]
    sim_matrix = z_norm @ z_norm.T / temperature

    # Mask: same class = 1, different class = 0
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [n, n]
    # Remove self-similarity from positives
    self_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    positive_mask = label_eq & self_mask

    # For numerical stability, subtract max from each row
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()

    # Log-sum-exp of all non-self similarities (denominator)
    exp_sim = torch.exp(sim_matrix) * self_mask.float()
    log_sum_exp = torch.log(exp_sim.sum(dim=1) + 1e-8)

    # For each anchor, mean of log(exp(sim_pos) / sum(exp(sim_all)))
    # = mean of (sim_pos - log_sum_exp)
    n_positives = positive_mask.sum(dim=1).float()
    has_positives = n_positives > 0

    if not has_positives.any():
        return torch.tensor(0.0, device=z.device), {"supcon": 0.0}

    # Sum of positive similarities per anchor
    positive_sim_sum = (sim_matrix * positive_mask.float()).sum(dim=1)

    # Loss per anchor: -mean(sim_pos - log_sum_exp)
    loss_per_anchor = -(positive_sim_sum / n_positives.clamp(min=1)) + log_sum_exp

    # Average only over anchors that have positives
    loss = loss_per_anchor[has_positives].mean()

    # Diagnostic metrics
    avg_positives = n_positives[has_positives].mean().item()
    frac_with_positives = has_positives.float().mean().item()
    metrics = {
        "supcon": loss.item(),
        "avg_positives": avg_positives,
        "frac_with_positives": frac_with_positives,
        "n_samples": n,
    }
    return loss, metrics


# ==============================================================================
# MEMORY MODULE (TGN-style)
# ==============================================================================


class TemporalMemory(nn.Module):
    """GRU-based memory module for temporal node state.

    Maintains a memory vector for each node that evolves with events.
    This captures the "history" of each token and pool.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        message_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize temporal memory.

        Args:
            num_nodes: Total number of nodes (tokens + pools).
            memory_dim: Dimension of memory vectors.
            message_dim: Dimension of incoming messages.
            device: Device for tensors.

        """
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.device = device

        # Memory storage (not a parameter - updated externally)
        self.register_buffer(
            "memory",
            torch.zeros(num_nodes, memory_dim, device=device),
        )
        self.register_buffer(
            "last_update",
            torch.zeros(num_nodes, device=device),
        )

        # GRU cell for memory updates
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve memory for specified nodes.

        Args:
            node_ids: Node indices to retrieve.

        Returns:
            Memory vectors of shape [len(node_ids), memory_dim].

        """
        return self.memory[node_ids]

    def update_memory(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Update memory for nodes with new messages.

        Args:
            node_ids: Node indices to update.
            messages: Message vectors of shape [len(node_ids), message_dim].
            timestamps: Event timestamps for each update.

        """
        if len(node_ids) == 0:
            return

        # Get current memory
        current_memory = self.memory[node_ids]

        # GRU update
        new_memory = self.gru(messages, current_memory)

        # Store updated memory
        self.memory[node_ids] = new_memory.detach()
        self.last_update[node_ids] = timestamps.float()

    def reset(self) -> None:
        """Reset all memory to zeros."""
        self.memory.zero_()
        self.last_update.zero_()

    def detach_memory(self) -> None:
        """Detach memory from computation graph."""
        self.memory = self.memory.detach()


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================


class HeteroGATEncoder(nn.Module):
    """Heterogeneous GATv2 encoder for token-pool bipartite graph.

    Uses GATv2Conv which computes dynamic attention - the attention scores
    depend on both source and destination node features. This allows the
    model to "cut" dependencies between tokens sharing a pool when their
    signals conflict.
    """

    def __init__(
        self,
        token_in_dim: int,
        pool_in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = GAT_HEADS,
        dropout: float = GAT_DROPOUT,
    ) -> None:
        """Initialize heterogeneous GAT encoder.

        Args:
            token_in_dim: Input dimension for token nodes.
            pool_in_dim: Input dimension for pool nodes.
            hidden_dim: Hidden layer dimension.
            out_dim: Output embedding dimension.
            heads: Number of attention heads.
            dropout: Dropout rate.

        """
        super().__init__()

        # Project inputs to common dimension with normalization
        self.token_proj = nn.Linear(token_in_dim, hidden_dim)
        self.pool_proj = nn.Linear(pool_in_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # Post-conv normalization
        self.hidden_token_norm = nn.LayerNorm(hidden_dim)
        self.hidden_pool_norm = nn.LayerNorm(hidden_dim)

        # First heterogeneous convolution layer
        self.conv1 = HeteroConv(
            {
                ("token", "trades_in", "pool"): GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
                ("pool", "rev_trades_in", "token"): GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                ),
            },
            aggr="sum",
        )

        # Second heterogeneous convolution layer
        # With concat=False, GATv2Conv averages over heads, so output = out_channels
        self.conv2 = HeteroConv(
            {
                ("token", "trades_in", "pool"): GATv2Conv(
                    hidden_dim,
                    out_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,  # Mean aggregation for final layer
                ),
                ("pool", "rev_trades_in", "token"): GATv2Conv(
                    hidden_dim,
                    out_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=False,
                ),
            },
            aggr="sum",
        )

        self.dropout = nn.Dropout(dropout)

        # Fallback projections for batches with no edges
        self.no_edge_proj_token = nn.Linear(hidden_dim, out_dim)
        self.no_edge_proj_pool = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass through heterogeneous GAT layers.

        Args:
            x_dict: Dictionary mapping node type to feature tensors.
            edge_index_dict: Dictionary mapping edge type to edge indices.

        Returns:
            Dictionary mapping node type to output embeddings.

        """
        # Project inputs with LayerNorm for stable feature scales
        x_proj = {
            "token": F.elu(self.token_norm(self.token_proj(x_dict["token"]))),
            "pool": F.elu(self.pool_norm(self.pool_proj(x_dict["pool"]))),
        }

        if not edge_index_dict:
            # No edges: project to output dimension directly
            return {
                "token": self.no_edge_proj_token(x_proj["token"]),
                "pool": self.no_edge_proj_pool(x_proj["pool"]),
            }

        # First conv layer — preserve projected features for isolated nodes
        conv1_out = self.conv1(x_proj, edge_index_dict)
        x_hidden = {
            "token": F.elu(self.dropout(self.hidden_token_norm(
                conv1_out.get("token", x_proj["token"]),
            ))),
            "pool": F.elu(self.dropout(self.hidden_pool_norm(
                conv1_out.get("pool", x_proj["pool"]),
            ))),
        }

        # Second conv layer
        conv2_out = self.conv2(x_hidden, edge_index_dict)
        return {k: conv2_out.get(k, x_hidden[k]) for k in x_proj}


class LDR_TGN(nn.Module):
    """LDR-based Temporal Graph Network for trading signals.

    Combines:
    - Temporal memory (TGN): Maintains evolving node state
    - Heterogeneous GATv2: Dynamic attention on token-pool graph
    - SupCon + CE objective: Learns class-clustered embeddings

    """

    def __init__(
        self,
        num_tokens: int,
        num_pools: int,
        token_feat_dim: int,
        pool_feat_dim: int,
        embed_dim: int,
        memory_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize LDR-TGN model.

        Args:
            num_tokens: Number of unique tokens.
            num_pools: Number of unique pools.
            token_feat_dim: Dimension of token features.
            pool_feat_dim: Dimension of pool features.
            embed_dim: Output embedding dimension.
            memory_dim: Temporal memory dimension.
            device: Device for tensors.

        """
        super().__init__()
        self.num_tokens = num_tokens
        self.num_pools = num_pools
        self.embed_dim = embed_dim
        self.device = device

        # Temporal memory for tokens and pools
        self.token_memory = TemporalMemory(
            num_tokens,
            memory_dim,
            message_dim=token_feat_dim + embed_dim,
            device=device,
        )
        self.pool_memory = TemporalMemory(
            num_pools,
            memory_dim,
            message_dim=pool_feat_dim + embed_dim,
            device=device,
        )

        # Input dimension includes memory
        token_in_dim = token_feat_dim + memory_dim
        pool_in_dim = pool_feat_dim + memory_dim

        # Heterogeneous GAT encoder
        self.encoder = HeteroGATEncoder(
            token_in_dim=token_in_dim,
            pool_in_dim=pool_in_dim,
            hidden_dim=HIDDEN_DIM,
            out_dim=embed_dim,
        )

        # Message encoders for memory updates
        self.token_message_encoder = nn.Linear(
            token_feat_dim + embed_dim,
            token_feat_dim + embed_dim,
        )
        self.pool_message_encoder = nn.Linear(
            pool_feat_dim + embed_dim,
            pool_feat_dim + embed_dim,
        )

        # Event feature encoder: project 4-dim features to embed_dim
        # This gives event features equal weight vs the graph embedding
        self.event_feat_encoder = nn.Sequential(
            nn.Linear(TOKEN_FEAT_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

        # Event-level MLP: combines graph embedding + encoded event features
        # Input: embed_dim (graph) + embed_dim (event features) = 2 * embed_dim
        event_input_dim = embed_dim * 2
        self.event_mlp = nn.Sequential(
            nn.Linear(event_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Classification head for joint cross-entropy training
        self.classifier = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(
        self,
        data: HeteroData,
        update_memory: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional memory update.

        Args:
            data: HeteroData with token/pool features and edge indices.
            update_memory: Whether to update temporal memory.

        Returns:
            Dictionary with token and pool embeddings.

        """
        # Get node features
        token_feats = data["token"].x
        pool_feats = data["pool"].x
        token_ids = data["token"].node_ids
        pool_ids = data["pool"].node_ids

        # Retrieve memory
        token_memory = self.token_memory.get_memory(token_ids)
        pool_memory = self.pool_memory.get_memory(pool_ids)

        # Concatenate features with memory
        x_dict = {
            "token": torch.cat([token_feats, token_memory], dim=-1),
            "pool": torch.cat([pool_feats, pool_memory], dim=-1),
        }

        # Build edge index dict
        edge_index_dict = {}
        if ("token", "trades_in", "pool") in data.edge_types:
            edge_index_dict[("token", "trades_in", "pool")] = data[
                "token", "trades_in", "pool",
            ].edge_index
            # Add reverse edges
            edge_index_dict[("pool", "rev_trades_in", "token")] = data[
                "token", "trades_in", "pool",
            ].edge_index.flip(0)

        # Encode with GATv2
        raw_embeddings = self.encoder(x_dict, edge_index_dict)

        # Update memory using raw (unnormalized) embeddings
        if update_memory and "timestamps" in data["token"]:
            token_messages = self.token_message_encoder(
                torch.cat([token_feats, raw_embeddings["token"]], dim=-1),
            )
            pool_messages = self.pool_message_encoder(
                torch.cat([pool_feats, raw_embeddings["pool"]], dim=-1),
            )

            self.token_memory.update_memory(
                token_ids,
                token_messages,
                data["token"].timestamps,
            )
            self.pool_memory.update_memory(
                pool_ids,
                pool_messages,
                data["pool"].timestamps,
            )

        return raw_embeddings

    def reset_memory(self) -> None:
        """Reset all temporal memory."""
        self.token_memory.reset()
        self.pool_memory.reset()

    def detach_memory(self) -> None:
        """Detach memory from computation graph."""
        self.token_memory.detach_memory()
        self.pool_memory.detach_memory()


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================


def load_token_metadata(tokens_path: Path) -> dict[str, dict[str, Any]]:
    """Load token metadata from tokens.json.

    Args:
        tokens_path: Path to tokens.json file.

    Returns:
        Dictionary mapping token address to metadata.

    """
    with tokens_path.open() as f:
        metadata: dict[str, dict[str, Any]] = json.load(f)
    return metadata


def validate_input_data(data_path: Path) -> None:
    """Validate input data exists and meets basic requirements.

    Args:
        data_path: Path to the parquet file with labeled bars.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("INPUT DATA VALIDATION")
    logger.info("=" * DIVIDER_LENGTH)

    if not data_path.exists():
        logger.error("  ERROR: Input file not found: %s", data_path)
        msg = f"Input file not found: {data_path}"
        raise FileNotFoundError(msg)

    logger.info("  OK: Input file exists: %s", data_path)

    df = pl.read_parquet(data_path)
    required_cols = [
        "bar_close_timestamp",
        "pool_id",
        "src_token_id",
        "dest_token_id",
        "src_fracdiff",
        "dest_fracdiff",
        "label",
        "sample_weight",
        "rolling_volatility",
        "src_flow_usdc",
        "dest_flow_usdc",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("  ERROR: Missing required columns: %s", missing_cols)
        msg = f"Missing required columns: {missing_cols}"
        raise ValueError(msg)

    logger.info("  OK: All required columns present")
    logger.info(
        "  Dataset shape: %s rows x %d columns",
        f"{df.shape[0]:,}",
        df.shape[1],
    )
    logger.info("=" * DIVIDER_LENGTH)


def load_and_filter_bars(data_path: Path) -> pl.DataFrame:
    """Load and filter swap bar data.

    Args:
        data_path: Path to the parquet file with labeled bars.

    Returns:
        Cleaned and sorted DataFrame with swap bars.

    """
    logger.info("Loading labeled bars from %s", data_path)
    df = pl.read_parquet(data_path)
    logger.info("Bars loaded: %s", f"{df.shape[0]:,}")

    # Sort by time
    df = df.sort(["bar_close_timestamp", "src_token_id", "dest_token_id"])

    # Filter out rows with NaN labels
    df = df.filter(pl.col("label").is_not_null() & pl.col("label").is_finite())
    logger.info("After filtering null labels: %s", f"{df.shape[0]:,}")

    return df


def prepare_data(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, LabelEncoder, LabelEncoder, int, int, pl.DataFrame]:
    """Encode token and pool IDs and create date index.

    Args:
        df: DataFrame with swap bars.

    Returns:
        Tuple of (df, token_encoder, pool_encoder, num_tokens, num_pools,
        unique_dates).

    """
    logger.info("Encoding token and pool IDs")

    # Encode tokens (both src and dest)
    token_le = LabelEncoder()
    all_tokens = np.concatenate(
        [
            df["src_token_id"].to_numpy(),
            df["dest_token_id"].to_numpy(),
        ],
    )
    token_le.fit(all_tokens)
    num_tokens = len(token_le.classes_)
    logger.info("Unique tokens: %s", f"{num_tokens:,}")

    # Encode pools
    pool_le = LabelEncoder()
    pool_le.fit(df["pool_id"].to_numpy())
    num_pools = len(pool_le.classes_)
    logger.info("Unique pools: %s", f"{num_pools:,}")

    # Add trade date column
    df = df.with_columns(
        pl.col("bar_close_timestamp").dt.date().alias("trade_date"),
    )

    # Add encoded IDs
    src_encoded = token_le.transform(df["src_token_id"].to_numpy())
    dest_encoded = token_le.transform(df["dest_token_id"].to_numpy())
    pool_encoded = pool_le.transform(df["pool_id"].to_numpy())

    df = df.with_columns(
        [
            pl.Series("src_token_encoded", src_encoded, dtype=pl.Int32),
            pl.Series("dest_token_encoded", dest_encoded, dtype=pl.Int32),
            pl.Series("pool_encoded", pool_encoded, dtype=pl.Int32),
        ],
    )

    # Map labels from {-1, 0, 1} to {0, 1, 2}
    df = df.with_columns(
        (pl.col("label").cast(pl.Int32) + 1).alias("label_encoded"),
    )

    unique_dates = df.select("trade_date").unique().sort("trade_date")
    logger.info(
        "Date range: %s to %s | Total days: %d",
        unique_dates["trade_date"][0],
        unique_dates["trade_date"][-1],
        len(unique_dates),
    )

    return df, token_le, pool_le, num_tokens, num_pools, unique_dates


def build_hetero_data(
    df: pl.DataFrame,
    num_tokens: int,
    num_pools: int,
    device: str = "cpu",
) -> HeteroData:
    """Build HeteroData object from DataFrame.

    Creates a heterogeneous bipartite graph:
    - Token nodes with price/flow features
    - Pool nodes with liquidity features
    - Edges connecting tokens to pools they trade through

    Args:
        df: DataFrame with swap events.
        num_tokens: Total number of tokens (unused, for API consistency).
        num_pools: Total number of pools (unused, for API consistency).
        device: Device for tensors.

    Returns:
        HeteroData object for PyG.

    """
    if len(df) == 0:
        # Return empty HeteroData
        data = HeteroData()
        data["token"].x = torch.zeros((0, TOKEN_FEAT_DIM), device=device)
        data["token"].node_ids = torch.zeros(0, dtype=torch.long, device=device)
        data["pool"].x = torch.zeros((0, POOL_FEAT_DIM), device=device)
        data["pool"].node_ids = torch.zeros(0, dtype=torch.long, device=device)
        return data

    # Get unique tokens and pools in this batch
    src_tokens = df["src_token_encoded"].to_numpy()
    dest_tokens = df["dest_token_encoded"].to_numpy()
    pools = df["pool_encoded"].to_numpy()

    unique_tokens = np.unique(np.concatenate([src_tokens, dest_tokens]))
    unique_pools = np.unique(pools)

    # Create local index mappings
    token_to_local = {t: i for i, t in enumerate(unique_tokens)}
    pool_to_local = {p: i for i, p in enumerate(unique_pools)}

    # Aggregate token features (mean over all events for each token)
    token_features = np.zeros(
        (len(unique_tokens), TOKEN_FEAT_DIM), dtype=np.float32,
    )
    token_labels = np.full(len(unique_tokens), -1, dtype=np.int64)
    token_weights = np.zeros(len(unique_tokens), dtype=np.float32)
    token_counts = np.zeros(len(unique_tokens), dtype=np.int32)

    # Process src tokens
    for i, row in enumerate(df.iter_rows(named=True)):
        src_local = token_to_local[src_tokens[i]]
        # Features: fracdiff, volatility, flow_magnitude, buying_pressure
        flow_mag = abs(row["src_flow_usdc"]) if row["src_flow_usdc"] else 0.0
        dest_flow = abs(row["dest_flow_usdc"]) if row["dest_flow_usdc"] else 0.0
        buying_pressure = flow_mag / (flow_mag + dest_flow + 1e-8)

        fracdiff_val = row["src_fracdiff"] if row["src_fracdiff"] else 0
        vol_val = row["rolling_volatility"] if row["rolling_volatility"] else 0
        token_features[src_local, 0] += fracdiff_val
        token_features[src_local, 1] += vol_val
        token_features[src_local, 2] += math.log1p(flow_mag)
        token_features[src_local, 3] += buying_pressure
        token_counts[src_local] += 1

        # Take the last label and weight
        if row["label"] is not None and not np.isnan(row["label"]):
            token_labels[src_local] = int(row["label"]) + 1  # Map to 0,1,2
            weight_val = row["sample_weight"] if row["sample_weight"] else 1.0
            token_weights[src_local] = weight_val

    # Average features
    token_counts = np.maximum(token_counts, 1)
    token_features /= token_counts[:, np.newaxis]

    # Aggregate pool features
    pool_features = np.zeros((len(unique_pools), POOL_FEAT_DIM), dtype=np.float32)
    pool_counts = np.zeros(len(unique_pools), dtype=np.int32)

    for i, row in enumerate(df.iter_rows(named=True)):
        pool_local = pool_to_local[pools[i]]
        # Features: log(tick_count), tick_delta, log(volume), log(liquidity_util)
        volume = abs(row["src_flow_usdc"] or 0) + abs(row["dest_flow_usdc"] or 0)
        liquidity = row.get("src_liquidity_close") or 1e12
        # Use log1p to tame the enormous range of liquidity_util
        liquidity_util = math.log1p(volume / (liquidity + 1e-8))

        tick_count_val = row.get("tick_count", 1) or 1
        tick_delta_val = row.get("src_tick_delta", 0) or 0
        pool_features[pool_local, 0] += math.log1p(tick_count_val)
        pool_features[pool_local, 1] += tick_delta_val
        pool_features[pool_local, 2] += math.log1p(volume)
        pool_features[pool_local, 3] += liquidity_util
        pool_counts[pool_local] += 1

    pool_counts = np.maximum(pool_counts, 1)
    pool_features /= pool_counts[:, np.newaxis]

    # Build edges (token -> pool)
    # Each swap creates edge: src_token -> pool and dest_token -> pool
    edge_src = []
    edge_dst = []

    for i in range(len(df)):
        src_local = token_to_local[src_tokens[i]]
        dest_local = token_to_local[dest_tokens[i]]
        pool_local = pool_to_local[pools[i]]

        edge_src.extend([src_local, dest_local])
        edge_dst.extend([pool_local, pool_local])

    # De-duplicate edges
    edges = set(zip(edge_src, edge_dst, strict=False))
    edge_src = [e[0] for e in edges]
    edge_dst = [e[1] for e in edges]

    # Create HeteroData
    data = HeteroData()

    # Token nodes
    data["token"].x = torch.tensor(
        token_features, dtype=torch.float32, device=device,
    )
    data["token"].node_ids = torch.tensor(
        unique_tokens, dtype=torch.long, device=device,
    )
    data["token"].y = torch.tensor(token_labels, dtype=torch.long, device=device)
    data["token"].weight = torch.tensor(
        token_weights, dtype=torch.float32, device=device,
    )

    # Timestamps (use last timestamp for simplicity)
    timestamps = df["bar_close_timestamp"].to_numpy()
    last_ts = timestamps[-1].astype("datetime64[s]").astype(np.int64)
    data["token"].timestamps = torch.full(
        (len(unique_tokens),),
        last_ts,
        dtype=torch.float32,
        device=device,
    )

    # Pool nodes
    data["pool"].x = torch.tensor(pool_features, dtype=torch.float32, device=device)
    data["pool"].node_ids = torch.tensor(
        unique_pools, dtype=torch.long, device=device,
    )
    data["pool"].timestamps = torch.full(
        (len(unique_pools),),
        last_ts,
        dtype=torch.float32,
        device=device,
    )

    # Edges
    if edge_src:
        edge_index = torch.tensor(
            [edge_src, edge_dst],
            dtype=torch.long,
            device=device,
        )
        data["token", "trades_in", "pool"].edge_index = edge_index

    # Handle NaN values
    data["token"].x = torch.nan_to_num(data["token"].x, nan=0.0)
    data["pool"].x = torch.nan_to_num(data["pool"].x, nan=0.0)

    # Event-level data for SupCon loss (expand per-token -> per-event)
    # Each event maps to its src token's local index
    event_token_local = np.array(
        [token_to_local[t] for t in src_tokens], dtype=np.int64,
    )
    event_labels_raw = df["label"].to_numpy()
    event_labels = np.where(
        np.isfinite(event_labels_raw),
        event_labels_raw.astype(np.int64) + 1,  # Map {-1,0,1} -> {0,1,2}
        -1,
    )
    event_weights_raw = df["sample_weight"].to_numpy()
    event_weights = np.where(
        np.isfinite(event_weights_raw), event_weights_raw, 1.0,
    ).astype(np.float32)

    data["event_token_indices"] = torch.tensor(
        event_token_local, dtype=torch.long, device=device,
    )
    data["event_labels"] = torch.tensor(
        event_labels, dtype=torch.long, device=device,
    )
    data["event_weights"] = torch.tensor(
        event_weights, dtype=torch.float32, device=device,
    )

    # Per-event features for the event MLP (unique per event, not aggregated)
    src_fracdiff = df["src_fracdiff"].fill_null(0).to_numpy().astype(np.float32)
    volatility = df["rolling_volatility"].fill_null(0).to_numpy().astype(np.float32)
    src_flow = df["src_flow_usdc"].fill_null(0).to_numpy().astype(np.float32)
    dest_flow = df["dest_flow_usdc"].fill_null(0).to_numpy().astype(np.float32)
    flow_mag = np.abs(src_flow)
    buying_pres = flow_mag / (flow_mag + np.abs(dest_flow) + 1e-8)
    event_features = np.stack(
        [src_fracdiff, volatility, np.log1p(flow_mag), buying_pres],
        axis=1,
    )
    data["event_features"] = torch.tensor(
        np.nan_to_num(event_features, nan=0.0),
        dtype=torch.float32,
        device=device,
    )

    return data


# ==============================================================================
# TRAINING
# ==============================================================================


def train_model(  # noqa: C901, PLR0912
    model: LDR_TGN,
    df: pl.DataFrame,
    num_tokens: int,
    num_pools: int,
    device: str,
    epochs: int = 10,
    batch_size: int = BATCH_SIZE,
) -> dict[str, list[float]]:
    """Train model with SupCon + CE loss.

    Args:
        model: LDR_TGN model instance.
        df: Training DataFrame.
        num_tokens: Total number of tokens.
        num_pools: Total number of pools.
        device: Device for training.
        epochs: Number of training epochs.
        batch_size: Batch size for training.

    Returns:
        Dictionary with training metrics history.

    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Sort by timestamp
    df = df.sort("bar_close_timestamp")

    # Calculate number of batches
    num_events = len(df)
    num_batches = (num_events + batch_size - 1) // batch_size

    history: dict[str, list[float]] = {
        "loss": [],
        "supcon": [],
        "ce": [],
    }

    # Early stopping state
    best_loss = float("inf")
    patience_counter = 0

    overall_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_supcon = 0.0
        epoch_ce = 0.0
        epoch_avg_positives = 0.0
        epoch_frac_with_positives = 0.0
        epoch_n_samples = 0.0
        n_batches = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_events)

            df_batch = df.slice(start_idx, end_idx - start_idx)
            if len(df_batch) == 0:
                continue

            # Build HeteroData for batch
            data = build_hetero_data(df_batch, num_tokens, num_pools, device)

            if data["token"].x.shape[0] == 0:
                continue

            # Log batch diagnostics on first epoch
            if epoch == 0 and batch_idx < 3:
                n_tok = data["token"].x.shape[0]
                n_pool = data["pool"].x.shape[0]
                n_edges = (
                    data["token", "trades_in", "pool"].edge_index.shape[1]
                    if ("token", "trades_in", "pool") in data.edge_types
                    else 0
                )
                tok_feat_norms = data["token"].x.norm(dim=1)
                pool_feat_norms = data["pool"].x.norm(dim=1)
                ev_labels = data["event_labels"]
                label_counts = [
                    (ev_labels == c).sum().item() for c in range(NUM_CLASSES)
                ]
                logger.info(
                    "  [Batch %d] events=%d, tokens=%d, pools=%d,"
                    " edges=%d, labels=[D=%d,N=%d,U=%d]",
                    batch_idx,
                    len(df_batch),
                    n_tok,
                    n_pool,
                    n_edges,
                    label_counts[0],
                    label_counts[1],
                    label_counts[2],
                )
                logger.info(
                    "    token feat norms: min=%.2f, max=%.2f,"
                    " mean=%.2f | pool feat norms: min=%.2f,"
                    " max=%.2f, mean=%.2f",
                    tok_feat_norms.min().item(),
                    tok_feat_norms.max().item(),
                    tok_feat_norms.mean().item(),
                    pool_feat_norms.min().item(),
                    pool_feat_norms.max().item(),
                    pool_feat_norms.mean().item(),
                )

            # Forward pass
            optimizer.zero_grad()
            embeddings = model(data, update_memory=True)

            # Create per-event embeddings via event MLP
            # Each event = graph_embedding(src_token) + event_features -> MLP
            token_emb = embeddings["token"]
            event_indices = data["event_token_indices"]
            event_labels = data["event_labels"]
            event_features = data["event_features"]

            # Gather token embeddings and encode event features
            token_per_event = token_emb[event_indices]
            encoded_feats = model.event_feat_encoder(event_features)
            event_input = torch.cat([token_per_event, encoded_feats], dim=-1)
            event_emb = model.event_mlp(event_input)

            # Filter for labeled events
            labeled_mask = event_labels >= 0
            if not labeled_mask.any():
                continue

            z = event_emb[labeled_mask]
            y = event_labels[labeled_mask]

            # Compute loss: weighted CE, optionally with SupCon
            logits = model.classifier(z)
            # Use Prado sample weights in CE loss
            event_w = data["event_weights"][labeled_mask]
            ce_loss = F.cross_entropy(logits, y, reduction="none")
            ce_loss = (ce_loss * event_w).sum() / event_w.sum()

            if USE_SUPCON:
                sc_loss, metrics = supcon_loss(z, y)
                loss = sc_loss + ce_loss
            else:
                sc_loss = torch.tensor(0.0, device=device)
                metrics = {
                    "supcon": 0.0,
                    "avg_positives": 0.0,
                    "frac_with_positives": 0.0,
                    "n_samples": len(z),
                }
                loss = ce_loss

            metrics["ce"] = ce_loss.item()
            metrics["loss"] = loss.item()

            # Log embedding diagnostics on first epoch
            if epoch == 0 and batch_idx < 3:
                z_norms = z.detach().norm(dim=1)
                logger.info(
                    "    emb norms: min=%.3f, max=%.3f,"
                    " mean=%.3f | pos_pairs=%.1f,"
                    " frac_with_pos=%.2f",
                    z_norms.min().item(),
                    z_norms.max().item(),
                    z_norms.mean().item(),
                    metrics.get("avg_positives", 0),
                    metrics.get("frac_with_positives", 0),
                )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Detach memory to prevent backprop through time
            model.detach_memory()

            epoch_loss += metrics["loss"]
            epoch_supcon += metrics["supcon"]
            epoch_ce += metrics["ce"]
            epoch_avg_positives += metrics.get("avg_positives", 0)
            epoch_frac_with_positives += metrics.get(
                "frac_with_positives", 0,
            )
            epoch_n_samples += metrics.get("n_samples", 0)
            n_batches += 1

        # Log epoch metrics
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_supcon = epoch_supcon / n_batches
            avg_ce = epoch_ce / n_batches
            avg_pos = epoch_avg_positives / n_batches
            avg_frac_pos = epoch_frac_with_positives / n_batches
            avg_samples = epoch_n_samples / n_batches

            history["loss"].append(avg_loss)
            history["supcon"].append(avg_supcon)
            history["ce"].append(avg_ce)

            logger.info(
                "  Epoch %d/%d: loss=%.4f, SupCon=%.4f,"
                " CE=%.4f | pos_pairs=%.1f,"
                " frac_pos=%.2f, samples=%.0f",
                epoch + 1,
                epochs,
                avg_loss,
                avg_supcon,
                avg_ce,
                avg_pos,
                avg_frac_pos,
                avg_samples,
            )

            # Early stopping: check for relative improvement
            relative_improvement = (
                (best_loss - avg_loss) / abs(best_loss)
                if best_loss != float("inf")
                else 1.0
            )
            if relative_improvement > EARLY_STOP_MIN_DELTA:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "  Early stopping at epoch %d "
                        "(no improvement for %d epochs)",
                        epoch + 1,
                        EARLY_STOP_PATIENCE,
                    )
                    break

    train_time = time.time() - overall_start
    logger.info("Training completed in %.2fs", train_time)

    # Compute class centroids from final epoch embeddings for prediction
    model.eval()
    centroids: dict[int, torch.Tensor] = {}
    all_z: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_events)
            df_batch = df.slice(start_idx, end_idx - start_idx)
            if len(df_batch) == 0:
                continue
            data = build_hetero_data(df_batch, num_tokens, num_pools, device)
            if data["token"].x.shape[0] == 0:
                continue
            emb = model(data, update_memory=False)
            token_emb = emb["token"]
            token_per_event = token_emb[data["event_token_indices"]]
            encoded_feats = model.event_feat_encoder(data["event_features"])
            event_input = torch.cat(
                [token_per_event, encoded_feats], dim=-1,
            )
            event_emb = model.event_mlp(event_input)
            # Normalize for centroid computation (consistent with SupCon loss)
            event_emb = F.normalize(event_emb, p=2, dim=-1)
            event_labels = data["event_labels"]
            mask = event_labels >= 0
            if mask.any():
                all_z.append(event_emb[mask])
                all_y.append(event_labels[mask])

    if all_z:
        z_all = torch.cat(all_z)
        y_all = torch.cat(all_y)
        for label in torch.unique(y_all):
            centroids[label.item()] = z_all[y_all == label].mean(dim=0)

    history["centroids"] = centroids
    return history


# ==============================================================================
# PREDICTION AND TRADING
# ==============================================================================


def predict_token_signals(
    model: LDR_TGN,
    df: pl.DataFrame,
    token_le: LabelEncoder,
    num_tokens: int,
    num_pools: int,
    centroids: dict[int, torch.Tensor],
    device: str,
    top_n: int = TOP_N_TOKENS,
) -> list[tuple[str, float, int]]:
    """Generate trading signals from embedding distances to class centroids.

    For each token, compute cosine similarity to each class centroid.
    Signal strength = sim(UP) - sim(DOWN). Positive = long signal.

    Args:
        model: Trained LDR_TGN model.
        df: DataFrame with events for prediction.
        token_le: Token label encoder.
        num_tokens: Total number of tokens.
        num_pools: Total number of pools.
        centroids: Class centroids from training {label: tensor}.
        device: Device for inference.
        top_n: Number of top signals to return.

    Returns:
        List of (token_address, signal_strength, predicted_class) tuples.
        signal_strength > 0 for long, < 0 for short.

    """
    model.eval()

    if len(df) == 0 or not centroids:
        return []

    # Stack centroids for vectorized computation
    centroid_labels = sorted(centroids.keys())
    centroid_matrix = torch.stack(
        [centroids[lbl] for lbl in centroid_labels],
    )  # [num_classes, embed_dim]
    centroid_matrix = F.normalize(centroid_matrix, p=2, dim=1)

    with torch.no_grad():
        data = build_hetero_data(df, num_tokens, num_pools, device)

        if data["token"].x.shape[0] == 0:
            return []

        embeddings = model(data, update_memory=False)
        token_emb = embeddings["token"]  # Already L2-normalized
        token_ids = data["token"].node_ids

        # Cosine similarity to each centroid [num_tokens, num_classes]
        similarities = token_emb @ centroid_matrix.T

        signals = []
        for i, token_id in enumerate(token_ids):
            sims = {
                lbl: similarities[i, j].item()
                for j, lbl in enumerate(centroid_labels)
            }

            # Predicted class = highest similarity centroid
            pred_class = max(sims, key=sims.get)  # type: ignore[arg-type]

            # Signal strength: sim(UP) - sim(DOWN)
            sim_up = sims.get(LABEL_UP, 0.0)
            sim_down = sims.get(LABEL_DOWN, 0.0)
            signal_strength = sim_up - sim_down

            token_addr = token_le.inverse_transform(
                [token_id.cpu().item()],
            )[0]
            signals.append((token_addr, signal_strength, pred_class))

        # Sort by signal strength descending (strongest long signal first)
        signals.sort(key=lambda x: x[1], reverse=True)

        # Filter for long signals only (positive signal = closer to UP)
        long_signals = [
            (addr, strength, cls)
            for addr, strength, cls in signals
            if strength > 0
        ]

        return long_signals[:top_n]


def calculate_daily_returns(
    df_trade: pl.DataFrame,
    top_tokens: list[tuple[str, float, int]],
) -> dict[str, tuple[float, float]]:
    """Calculate daily returns for selected tokens.

    Args:
        df_trade: DataFrame with swap events for the trade date.
        top_tokens: List of (token_address, signal_strength, predicted_class).

    Returns:
        Dict mapping token_address to (daily_return, position_size).

    """
    returns: dict[str, tuple[float, float]] = {}

    for token_addr, signal_strength, _ in top_tokens:
        # Position size based on signal strength (capped Kelly-like sizing)
        position_size = min(abs(signal_strength), 0.5)

        token_events = df_trade.filter(
            pl.col("src_token_id") == token_addr,
        ).sort("bar_close_timestamp")

        if len(token_events) == 0:
            returns[token_addr] = (0.0, position_size)
            continue

        prices = token_events.select("src_price_usdc").to_numpy().flatten()
        prices = prices[~np.isnan(prices)]

        if len(prices) < 2:
            returns[token_addr] = (0.0, position_size)
            continue

        start_price = prices[0]
        stop_loss_price = start_price * (1 - STOP_LOSS_PCT)
        take_profit_price = start_price * (1 + TAKE_PROFIT_PCT)

        position_closed = False
        final_return = 0.0

        for price in prices[1:]:
            if price <= stop_loss_price:
                final_return = -STOP_LOSS_PCT
                position_closed = True
                break

            if price >= take_profit_price:
                final_return = TAKE_PROFIT_PCT
                position_closed = True
                break

        if not position_closed:
            end_price = prices[-1]
            if start_price > 0:
                final_return = (end_price - start_price) / start_price
            else:
                final_return = 0.0

        returns[token_addr] = (final_return, position_size)

    return returns


def save_embeddings(
    model: LDR_TGN,
    df: pl.DataFrame,
    token_le: LabelEncoder,
    num_tokens: int,
    num_pools: int,
    trade_date: date,
    output_dir: Path,
    device: str,
) -> Path:
    """Save token embeddings for evaluation.

    Args:
        model: Trained LDR_TGN model.
        df: DataFrame with events.
        token_le: Token label encoder.
        num_tokens: Total number of tokens.
        num_pools: Total number of pools.
        trade_date: Trading date.
        output_dir: Directory to save embeddings.
        device: Device for inference.

    Returns:
        Path to saved embeddings file.

    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        data = build_hetero_data(df, num_tokens, num_pools, device)

        if data["token"].x.shape[0] == 0:
            logger.warning("No tokens to save embeddings for")
            return output_dir / "empty.npz"

        embeddings_dict = model(data, update_memory=False)
        token_emb = embeddings_dict["token"]

        # Compute event-level embeddings via MLP, then normalize
        token_per_event = token_emb[data["event_token_indices"]]
        encoded_feats = model.event_feat_encoder(data["event_features"])
        event_input = torch.cat(
            [token_per_event, encoded_feats], dim=-1,
        )
        event_emb = F.normalize(
            model.event_mlp(event_input), p=2, dim=-1,
        ).cpu().numpy()

        event_labels = data["event_labels"].cpu().numpy()

        # Map event indices back to token addresses
        src_tokens = df["src_token_encoded"].to_numpy()
        event_token_addrs = token_le.inverse_transform(src_tokens)

        date_str = trade_date.strftime("%Y%m%d")
        output_path = output_dir / f"ldr_embeddings_{date_str}.npz"

        np.savez(
            output_path,
            embeddings=event_emb,
            token_addresses=event_token_addrs,
            labels=event_labels,
            trade_date=str(trade_date),
            embed_dim=model.embed_dim,
        )

        logger.info("Saved embeddings to: %s", output_path)
        logger.info("  Shape: %s", event_emb.shape)

    return output_path


# ==============================================================================
# BACKTEST
# ==============================================================================


def compute_backtest_summary(results: list[dict[str, Any]]) -> None:
    """Compute and log comprehensive backtest performance summary."""
    if not results:
        logger.warning("No results to summarize")
        return

    portfolio_returns = [r["portfolio_return"] for r in results]

    total_return = np.prod([1 + r for r in portfolio_returns]) - 1
    cumulative_return = np.sum(portfolio_returns)
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

    winning_days = sum(1 for r in portfolio_returns if r > 0)
    losing_days = sum(1 for r in portfolio_returns if r < 0)
    win_rate = winning_days / len(portfolio_returns) if portfolio_returns else 0.0

    cumulative_wealth = np.cumprod([1 + r for r in portfolio_returns])
    running_max_wealth = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max_wealth) / running_max_wealth
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    best_day_idx = np.argmax(portfolio_returns) if portfolio_returns else 0
    worst_day_idx = np.argmin(portfolio_returns) if portfolio_returns else 0
    best_day_return = portfolio_returns[best_day_idx]
    worst_day_return = portfolio_returns[worst_day_idx]
    best_day_date = results[best_day_idx]["trade_date"]
    worst_day_date = results[worst_day_idx]["trade_date"]

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("BACKTEST SUMMARY (LDR-TGN with SupCon)")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("  Total trading days: %d", len(results))
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("RETURNS SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)

    ret_icon = "OK" if total_return > 0 else "WARN" if total_return == 0 else "FAIL"
    logger.info("  %s Total Return: %+.2f%%", ret_icon, total_return * 100)
    logger.info("  Cumulative Return: %+.2f%%", cumulative_return * 100)
    logger.info("  Mean Daily Return: %+.2f%%", mean_return * 100)
    logger.info("  Std Dev: %.2f%%", std_return * 100)

    sharpe_icon = "OK" if sharpe > 1.0 else "WARN" if sharpe > 0 else "FAIL"
    logger.info("  %s Sharpe Ratio (annualized): %.2f", sharpe_icon, sharpe)

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("WIN/LOSS STATISTICS")
    logger.info("=" * DIVIDER_LENGTH)

    win_icon = "OK" if win_rate > 0.5 else "WARN" if win_rate > 0.4 else "FAIL"
    logger.info(
        "  %s Winning Days: %d / %d (%.1f%%)",
        win_icon,
        winning_days,
        len(results),
        win_rate * 100,
    )
    logger.info("  Losing Days: %d", losing_days)
    logger.info("  Break-even Days: %d", len(results) - winning_days - losing_days)

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("DRAWDOWN ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)

    dd_icon = "OK" if max_drawdown > -0.1 else "WARN" if max_drawdown > -0.2 else "FAIL"
    logger.info("  %s Max Drawdown: %.2f%%", dd_icon, max_drawdown * 100)

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EXTREME DAYS")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("  Best Day: %s (%+.2f%%)", best_day_date, best_day_return * 100)
    logger.info("  Worst Day: %s (%+.2f%%)", worst_day_date, worst_day_return * 100)
    logger.info("=" * DIVIDER_LENGTH)

    logger.info("")
    if total_return > 0 and sharpe > 1.0 and win_rate > 0.5:
        logger.info("Strategy shows strong performance!")
    elif total_return > 0:
        logger.info("Strategy is profitable but could be improved")
    else:
        logger.info("Strategy underperformed - needs adjustment")
    logger.info("=" * DIVIDER_LENGTH)


def backtest_slide(
    df: pl.DataFrame,
    token_le: LabelEncoder,
    pool_le: LabelEncoder,
    num_tokens: int,
    num_pools: int,
    unique_dates: pl.DataFrame,
    tokens_metadata: dict[str, dict[str, Any]],
    epochs: int,
    trading_days: int | None = None,
    save_embeddings_flag: bool = False,
    train_window_days: int = TRAIN_WINDOW_DAYS,
) -> None:
    """Run sliding backtest with walk-forward validation.

    Args:
        df: DataFrame with swap bars.
        token_le: Token label encoder.
        pool_le: Pool label encoder (unused, for API consistency).
        num_tokens: Total number of tokens.
        num_pools: Total number of pools.
        unique_dates: DataFrame with unique trade dates.
        tokens_metadata: Dictionary of token metadata.
        epochs: Number of training epochs per day.
        trading_days: Number of days to trade (None = all available).
        save_embeddings_flag: Whether to save embeddings.
        train_window_days: Number of days in training window.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("LDR-TGN SLIDING BACKTEST - WALK-FORWARD VALIDATION")
    logger.info("=" * DIVIDER_LENGTH)

    dates_array = unique_dates["trade_date"].to_list()

    max_trading_days = len(dates_array) - train_window_days
    if trading_days is None:
        trading_days = max_trading_days
    else:
        trading_days = min(trading_days, max_trading_days)

    logger.info("Total available days: %d", len(dates_array))
    logger.info("Training window: %d days", train_window_days)
    logger.info("Trading days: %d", trading_days)

    results: list[dict[str, Any]] = []

    # Initialize model once and persist across trading days
    # The model accumulates knowledge; each day we fine-tune on the sliding window
    model = LDR_TGN(
        num_tokens=num_tokens,
        num_pools=num_pools,
        token_feat_dim=TOKEN_FEAT_DIM,
        pool_feat_dim=POOL_FEAT_DIM,
        embed_dim=EMBED_DIM,
        memory_dim=MEMORY_DIM,
        device=DEVICE,
    ).to(DEVICE)

    for day_idx in range(train_window_days, train_window_days + trading_days):
        trade_date = dates_array[day_idx]
        train_start_date = dates_array[day_idx - train_window_days]
        train_end_date = dates_array[day_idx - 1]

        logger.info("")
        logger.info("=" * DIVIDER_LENGTH)
        logger.info("Trade Day: %s", trade_date)
        logger.info("=" * DIVIDER_LENGTH)
        logger.info("Training on: %s to %s", train_start_date, train_end_date)

        # Get training data
        df_train = df.filter(
            (pl.col("trade_date") >= train_start_date)
            & (pl.col("trade_date") <= train_end_date),
        )

        logger.info("Training events: %s", f"{len(df_train):,}")

        # Train model and get class centroids
        history = train_model(
            model,
            df_train,
            num_tokens,
            num_pools,
            DEVICE,
            epochs=epochs,
        )
        centroids = history.get("centroids", {})

        # Optional: save embeddings
        if save_embeddings_flag:
            save_embeddings(
                model,
                df_train,
                token_le,
                num_tokens,
                num_pools,
                trade_date,
                Path("data/ldr_embeddings"),
                DEVICE,
            )

        # Get morning data for prediction context
        df_morning = df.filter(
            (pl.col("trade_date") == trade_date)
            & (pl.col("bar_close_timestamp").dt.hour() < TRADE_WINDOW_HOURS[0]),
        )

        # Predict signals using centroid distances
        top_tokens = predict_token_signals(
            model,
            df_morning if len(df_morning) > 0 else df_train,
            token_le,
            num_tokens,
            num_pools,
            centroids,
            DEVICE,
            top_n=TOP_N_TOKENS,
        )

        logger.info("Top tokens with positive signals: %d", len(top_tokens))

        if len(top_tokens) == 0:
            logger.info("No positive signals, skipping trading day")
            results.append(
                {
                    "train_start": train_start_date,
                    "train_end": train_end_date,
                    "trade_date": trade_date,
                    "top_tokens": [],
                    "token_returns": {},
                    "portfolio_return": 0.0,
                },
            )
            continue

        # Get trading window data
        df_trading = df.filter(
            (pl.col("trade_date") == trade_date)
            & (pl.col("bar_close_timestamp").dt.hour() >= TRADE_WINDOW_HOURS[0])
            & (pl.col("bar_close_timestamp").dt.hour() < TRADE_WINDOW_HOURS[1]),
        )

        logger.info("Trading window events: %s", f"{len(df_trading):,}")

        if len(df_trading) == 0:
            logger.warning("No trading data for %s, skipping", trade_date)
            continue

        # Calculate returns
        token_returns = calculate_daily_returns(df_trading, top_tokens)

        for rank, (token_addr, signal_strength, pred_class) in enumerate(
            top_tokens, 1,
        ):
            token_addr_lower = token_addr.lower()
            token_meta = tokens_metadata.get(token_addr_lower, {})
            symbol = token_meta.get("symbol", token_addr[:10])
            daily_return, position_size = token_returns.get(
                token_addr,
                (0.0, abs(signal_strength)),
            )
            class_name = ["DOWN", "NEUTRAL", "UP"][pred_class]
            logger.info(
                "  %d. %s | Signal: %.3f (%s) | Size: %.1f%% | Return: %+.2f%%",
                rank,
                symbol,
                signal_strength,
                class_name,
                position_size * 100,
                daily_return * 100,
            )

        # Calculate portfolio return
        total_position_size = sum(pos for _, pos in token_returns.values())
        weighted_return = sum(ret * pos for ret, pos in token_returns.values())
        portfolio_return = (
            weighted_return / total_position_size if total_position_size > 0 else 0.0
        )

        logger.info(
            "Portfolio return: %+.2f%% (%.1f%% capital deployed)",
            portfolio_return * 100,
            total_position_size * 100,
        )

        results.append(
            {
                "train_start": train_start_date,
                "train_end": train_end_date,
                "trade_date": trade_date,
                "top_tokens": top_tokens,
                "token_returns": token_returns,
                "portfolio_return": portfolio_return,
            },
        )

    compute_backtest_summary(results)


# ==============================================================================
# CLI
# ==============================================================================


@app.command()
def main(
    data_path: Path = typer.Option(
        Path("data/labeled_log_fracdiff_price.parquet"),
        "--data",
        "-d",
        help="Path to labeled training data parquet file",
    ),
    tokens_path: Path = typer.Option(
        Path("data/tokens.json"),
        "--tokens",
        "-t",
        help="Path to tokens metadata JSON file",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Number of training epochs per day",
    ),
    trading_days: int | None = typer.Option(
        None,
        "--trading-days",
        help="Number of days to trade (default: all available)",
    ),
    save_embeddings: bool = typer.Option(
        False,
        "--save-embeddings",
        help="Save token embeddings to data/ldr_embeddings/",
    ),
    train_days: int = typer.Option(
        TRAIN_WINDOW_DAYS,
        "--train-days",
        help="Number of days in training window before trading begins",
    ),
) -> None:
    """Run LDR-TGN trading backtest with SupCon + CE loss.

    Uses heterogeneous graph with token-pool bipartite structure.
    SupCon loss learns class-clustered representations.
    Walk-forward validation with memory persistence.
    """
    validate_input_data(data_path)

    logger.info("Loading token metadata from %s", tokens_path)
    tokens_metadata = load_token_metadata(tokens_path)
    logger.info("Loaded metadata for %d tokens", len(tokens_metadata))

    df = load_and_filter_bars(data_path)
    df, token_le, pool_le, num_tokens, num_pools, unique_dates = prepare_data(df)

    backtest_slide(
        df,
        token_le,
        pool_le,
        num_tokens,
        num_pools,
        unique_dates,
        tokens_metadata,
        epochs,
        trading_days,
        save_embeddings,
        train_days,
    )


if __name__ == "__main__":
    app()
