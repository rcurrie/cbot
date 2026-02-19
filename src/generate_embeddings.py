"""Generate LDR token embeddings via Bipartite GNN with MCR² loss.

WHY: Learn Linear Discriminative Representations (LDR) where tokens cluster into
orthogonal subspaces based on trading behavior. This enables better downstream
prediction by capturing structural relationships between tokens and pools.

WHAT: Train a heterogeneous temporal GNN on a bipartite Token-Pool graph using:
1. GATv2Conv for attention-based message passing (enables subspace orthogonality)
2. MCR² (Maximal Coding Rate Reduction) loss for LDR structure
3. TGM library for temporal batching and graph construction

HOW:
1. Build bipartite graph: tokens (52) ↔ pools (136) connected by swap events
2. Use GATv2-GRU cells for temporal message passing with dynamic attention
3. Train with MCR² loss: maximize between-cluster separation, minimize within-cluster variance
4. Extract final embeddings for both tokens and pools

INPUT: data/labeled_log_fracdiff_price.parquet (503K events, 52 tokens, 136 pools)
       data/pools.json (pool metadata: fee tier, tick spacing)
OUTPUT: data/embeddings/mcr2_bipartite_{timestamp}.npz

References:
- MCR² Loss: https://arxiv.org/abs/2006.08558
- GATv2: https://arxiv.org/abs/2105.14491
- TGM Library: https://github.com/tgm-team/tgm

"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from tgm import DGBatch, DGraph
from tgm.data import DGData, DGDataLoader
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")
POOLS_PATH = Path("data/pools.json")
TOKENS_PATH = Path("data/tokens.json")
OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
NUM_CLUSTERS = 8
NODE_FEAT_DIM = 5  # fracdiff, volatility, flow, liquidity, tick_delta
EDGE_FEAT_DIM = 4  # amount, tick_count, time_delta, direction
BATCH_SIZE = 200  # seconds
BATCH_UNIT = "s"
LEARNING_RATE = 1e-3
MCR2_EPS = 0.5
MCR2_GAMMA = 1.0
CLUSTER_BALANCE_WEIGHT = 0.1
PATIENCE = 10

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("Device: %s", DEVICE)


# ============================================================================
# Data Loading & Bipartite Graph Construction
# ============================================================================


def load_data() -> tuple[pl.DataFrame, dict[str, dict[str, Any]]]:
    """Load parquet data and pool metadata.

    Returns:
        Tuple of (DataFrame, pool_metadata dict).

    """
    logger.info("Loading data from %s", DATA_PATH)
    df = pl.read_parquet(DATA_PATH)
    logger.info("Loaded %s bars", f"{df.shape[0]:,}")

    # Sort by time for chronological processing
    df = df.sort(["bar_close_timestamp", "pool_id", "src_token_id"])

    # Load pool metadata
    logger.info("Loading pool metadata from %s", POOLS_PATH)
    with POOLS_PATH.open() as f:
        pools_raw = json.load(f)

    # Build pool address -> metadata mapping
    pool_meta: dict[str, dict[str, Any]] = {}
    for pool in pools_raw.get("data", []):
        addr = pool.get("address", "").lower()
        if addr:
            pool_meta[addr] = {
                "fee": float(pool.get("fee", 0.003)),
                "tick_spacing": int(pool.get("tickSpacing", 60))
                if pool.get("tickSpacing")
                else 60,
                "protocol": pool.get("protocol", "unknown"),
                "name": pool.get("name", ""),
            }

    logger.info("Loaded metadata for %s pools", len(pool_meta))
    return df, pool_meta


def build_bipartite_topology(
    df: pl.DataFrame,
) -> dict[str, Any]:
    """Build token-pool bipartite graph topology.

    In the bipartite structure:
    - Tokens are nodes 0 to num_tokens-1
    - Pools are nodes num_tokens to num_tokens+num_pools-1

    Each swap creates edges:
    - src_token → pool (token sold into pool)
    - pool → dest_token (pool gave out token)
    - Reverse edges for bidirectional message passing

    Args:
        df: DataFrame with swap events.

    Returns:
        Dictionary with encoders and topology info.

    """
    logger.info("Building bipartite topology...")

    # Get unique tokens (union of src and dest)
    src_tokens = set(df["src_token_id"].unique().to_list())
    dest_tokens = set(df["dest_token_id"].unique().to_list())
    all_tokens = sorted(src_tokens | dest_tokens)

    # Get unique pools
    all_pools = sorted(df["pool_id"].unique().to_list())

    # Create encoders
    token_encoder = LabelEncoder()
    token_encoder.fit(all_tokens)
    num_tokens = len(all_tokens)

    pool_encoder = LabelEncoder()
    pool_encoder.fit(all_pools)
    num_pools = len(all_pools)

    # Total nodes = tokens + pools
    num_nodes = num_tokens + num_pools

    # Node types: 0 = token, 1 = pool
    node_type = np.zeros(num_nodes, dtype=np.int32)
    node_type[num_tokens:] = 1

    logger.info("  Tokens: %s", num_tokens)
    logger.info("  Pools: %s", num_pools)
    logger.info("  Total nodes: %s", num_nodes)

    return {
        "token_encoder": token_encoder,
        "pool_encoder": pool_encoder,
        "num_tokens": num_tokens,
        "num_pools": num_pools,
        "num_nodes": num_nodes,
        "node_type": node_type,
    }


def build_streaming_data(
    df: pl.DataFrame,
    topology: dict[str, Any],
    pool_meta: dict[str, dict[str, Any]],
) -> DGData:
    """Convert DataFrame to TGM DGData with bipartite edges.

    Creates a temporal graph where:
    - Each swap creates 4 edges (bidirectional token↔pool)
    - Tokens get 5-dim features: fracdiff, volatility, flow, liquidity, tick_delta
    - Pools get 5-dim features: fee, tick_spacing, combined_liq, volume_proxy, tick_mvmt

    Args:
        df: DataFrame with swap events.
        topology: Topology dict from build_bipartite_topology.
        pool_meta: Pool metadata dict.

    Returns:
        DGData object for TGM.

    """
    logger.info("Building streaming DGData for bipartite graph...")

    token_encoder = topology["token_encoder"]
    pool_encoder = topology["pool_encoder"]
    num_tokens = topology["num_tokens"]

    num_events = len(df)

    # Encode node IDs
    src_token_ids = token_encoder.transform(df["src_token_id"].to_numpy())
    dest_token_ids = token_encoder.transform(df["dest_token_id"].to_numpy())
    pool_ids = pool_encoder.transform(df["pool_id"].to_numpy()) + num_tokens

    # Timestamps (milliseconds from start)
    timestamps_raw = df["bar_close_timestamp"].to_numpy()
    first_ts = timestamps_raw[0]
    timestamps = ((timestamps_raw - first_ts) / np.timedelta64(1, "ms")).astype(
        np.int64
    )

    # =========================================================================
    # Build edges: each swap creates 4 edges
    # 1. src_token → pool
    # 2. pool → dest_token
    # 3. dest_token → pool (reverse)
    # 4. pool → src_token (reverse)
    # =========================================================================

    # Forward edges (src_token → pool, pool → dest_token)
    edge_src_to_pool = np.column_stack((src_token_ids, pool_ids))
    edge_pool_to_dest = np.column_stack((pool_ids, dest_token_ids))

    # Reverse edges (bidirectional for message passing)
    edge_pool_to_src = np.column_stack((pool_ids, src_token_ids))
    edge_dest_to_pool = np.column_stack((dest_token_ids, pool_ids))

    # Combine all edges
    edge_index = np.vstack(
        [edge_src_to_pool, edge_pool_to_dest, edge_pool_to_src, edge_dest_to_pool]
    )

    # Edge timestamps (repeat 4x for the 4 edge types)
    edge_timestamps = np.tile(timestamps, 4)

    # =========================================================================
    # Edge features: 4 dimensions per edge
    # [amount, tick_count, time_delta, direction]
    # =========================================================================
    src_flow = df["src_flow_usdc"].to_numpy()
    dest_flow = df["dest_flow_usdc"].to_numpy()
    tick_count = df["tick_count"].fill_null(0).to_numpy()
    time_delta = df["bar_time_delta_sec"].fill_null(0).to_numpy()

    # Normalize features
    src_flow_norm = np.clip(src_flow / 1e6, -10, 10)  # Scale to millions
    dest_flow_norm = np.clip(dest_flow / 1e6, -10, 10)
    tick_count_norm = np.clip(tick_count / 100, 0, 10)
    time_delta_norm = np.clip(time_delta / 3600, 0, 24)  # Hours

    # Create edge features for each edge type
    # Direction: 0=token→pool (sell), 1=pool→token (buy)
    feat_src_to_pool = np.column_stack(
        [
            np.abs(src_flow_norm),
            tick_count_norm,
            time_delta_norm,
            np.zeros(num_events),  # direction=0
        ]
    )
    feat_pool_to_dest = np.column_stack(
        [
            np.abs(dest_flow_norm),
            tick_count_norm,
            time_delta_norm,
            np.ones(num_events),  # direction=1
        ]
    )
    feat_pool_to_src = np.column_stack(
        [
            np.abs(src_flow_norm),
            tick_count_norm,
            time_delta_norm,
            np.ones(num_events),  # direction=1
        ]
    )
    feat_dest_to_pool = np.column_stack(
        [
            np.abs(dest_flow_norm),
            tick_count_norm,
            time_delta_norm,
            np.zeros(num_events),  # direction=0
        ]
    )

    edge_feats = np.vstack(
        [feat_src_to_pool, feat_pool_to_dest, feat_pool_to_src, feat_dest_to_pool]
    ).astype(np.float32)

    # =========================================================================
    # Node features: each event updates 3 nodes (src_token, dest_token, pool)
    # Token features: [fracdiff, volatility, flow, liquidity, tick_delta]
    # Pool features: [fee, tick_spacing, combined_liq, volume_proxy, tick_mvmt]
    # =========================================================================

    # Node timestamps (3 updates per event)
    node_timestamps = np.repeat(timestamps, 3)

    # Node IDs
    node_ids = np.empty(num_events * 3, dtype=np.int32)
    node_ids[0::3] = src_token_ids  # src token
    node_ids[1::3] = dest_token_ids  # dest token
    node_ids[2::3] = pool_ids  # pool

    # Extract features
    src_fracdiff = df["src_fracdiff"].fill_null(0).to_numpy()
    dest_fracdiff = df["dest_fracdiff"].fill_null(0).to_numpy()
    volatility = df["rolling_volatility"].fill_null(0).to_numpy()
    src_liq = df["src_liquidity_close"].fill_null(0).to_numpy()
    dest_liq = df["dest_liquidity_close"].fill_null(0).to_numpy()
    src_tick_delta = df["src_tick_delta"].fill_null(0).to_numpy()
    dest_tick_delta = df["dest_tick_delta"].fill_null(0).to_numpy()

    # Get pool static features
    pool_addresses = df["pool_id"].to_numpy()
    pool_fees = np.array(
        [pool_meta.get(p.lower(), {}).get("fee", 0.003) for p in pool_addresses]
    )
    pool_tick_spacing = np.array(
        [
            pool_meta.get(p.lower(), {}).get("tick_spacing", 60) / 200.0
            for p in pool_addresses
        ]
    )  # Normalize

    # Normalize liquidity (log scale)
    src_liq_norm = np.log1p(np.abs(src_liq)) / 20.0
    dest_liq_norm = np.log1p(np.abs(dest_liq)) / 20.0
    combined_liq_norm = (src_liq_norm + dest_liq_norm) / 2.0

    # Dynamic node features [num_events * 3, 5]
    dynamic_node_feats = np.zeros((num_events * 3, NODE_FEAT_DIM), dtype=np.float32)

    # Src token features
    dynamic_node_feats[0::3, 0] = src_fracdiff
    dynamic_node_feats[0::3, 1] = volatility
    dynamic_node_feats[0::3, 2] = src_flow_norm
    dynamic_node_feats[0::3, 3] = src_liq_norm
    dynamic_node_feats[0::3, 4] = np.clip(src_tick_delta / 1000, -10, 10)

    # Dest token features
    dynamic_node_feats[1::3, 0] = dest_fracdiff
    dynamic_node_feats[1::3, 1] = volatility
    dynamic_node_feats[1::3, 2] = dest_flow_norm
    dynamic_node_feats[1::3, 3] = dest_liq_norm
    dynamic_node_feats[1::3, 4] = np.clip(dest_tick_delta / 1000, -10, 10)

    # Pool features
    dynamic_node_feats[2::3, 0] = pool_fees * 100  # Scale fee to ~0.3-1 range
    dynamic_node_feats[2::3, 1] = pool_tick_spacing
    dynamic_node_feats[2::3, 2] = combined_liq_norm
    dynamic_node_feats[2::3, 3] = (np.abs(src_flow_norm) + np.abs(dest_flow_norm)) / 2
    dynamic_node_feats[2::3, 4] = (
        np.clip(src_tick_delta / 1000, -10, 10)
        + np.clip(dest_tick_delta / 1000, -10, 10)
    ) / 2

    # Handle NaN
    dynamic_node_feats = np.nan_to_num(dynamic_node_feats, nan=0.0)
    edge_feats = np.nan_to_num(edge_feats, nan=0.0)

    logger.info("  Events: %s", f"{num_events:,}")
    logger.info("  Edges: %s", f"{len(edge_index):,}")
    logger.info("  Node updates: %s", f"{len(node_ids):,}")

    return DGData.from_raw(
        edge_timestamps=torch.from_numpy(edge_timestamps),
        edge_index=torch.from_numpy(edge_index),
        edge_feats=torch.from_numpy(edge_feats),
        node_timestamps=torch.from_numpy(node_timestamps),
        node_ids=torch.from_numpy(node_ids),
        dynamic_node_feats=torch.from_numpy(dynamic_node_feats),
        time_delta="ms",
    )


# ============================================================================
# Model Architecture
# ============================================================================


class GATv2GRUCell(nn.Module):
    """GRU cell with GATv2Conv for message passing.

    Replaces TGCN's GCNConv with GATv2Conv for LDR-compatible attention.
    GATv2 provides "dynamic attention" where attention weights depend on both
    source and target features, enabling nodes sharing neighbors to learn
    different aggregation patterns (required for subspace orthogonality).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 4,
    ) -> None:
        """Initialize GATv2-GRU cell.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            heads: Number of attention heads.
            edge_dim: Edge feature dimension.

        """
        super().__init__()
        self.out_channels = out_channels

        # GATv2 convolutions for GRU gates
        self.conv_c = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,  # Average heads
            edge_dim=edge_dim,
            add_self_loops=False,
        )
        self.conv_u = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=False,
        )
        self.conv_r = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        # Linear layers for GRU gates
        self.linear_c = nn.Linear(2 * out_channels, out_channels)
        self.linear_u = nn.Linear(2 * out_channels, out_channels)
        self.linear_r = nn.Linear(2 * out_channels, out_channels)

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        H: Tensor | None = None,
    ) -> Tensor:
        """GRU update with GATv2 message passing.

        Args:
            X: Node features [num_nodes, in_channels].
            edge_index: Edge connectivity [2, num_edges].
            edge_attr: Edge features [num_edges, edge_dim].
            H: Previous hidden state [num_nodes, out_channels].

        Returns:
            New hidden state [num_nodes, out_channels].

        """
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels, device=X.device)

        # Handle empty edge_index
        if edge_index.shape[1] == 0:
            # No edges - just return identity with projection
            return H

        # Update gate
        conv_u_out = self.conv_u(X, edge_index, edge_attr)
        u = torch.sigmoid(self.linear_u(torch.cat([conv_u_out, H], dim=-1)))

        # Reset gate
        conv_r_out = self.conv_r(X, edge_index, edge_attr)
        r = torch.sigmoid(self.linear_r(torch.cat([conv_r_out, H], dim=-1)))

        # Candidate state
        conv_c_out = self.conv_c(X, edge_index, edge_attr)
        c = torch.tanh(self.linear_c(torch.cat([conv_c_out, r * H], dim=-1)))

        # New hidden state
        return u * H + (1 - u) * c


class BipartiteMCR2Model(nn.Module):
    """TGM-compatible model with GATv2 encoder and MCR² loss heads.

    Architecture:
        Input → Feature Projector → Learnable Embeddings → GATv2-GRU Layers
              → Embed Head (Z) + Cluster Head (Π)

    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        edge_feat_dim: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_clusters: int = 8,
    ) -> None:
        """Initialize bipartite MCR² model.

        Args:
            num_nodes: Total nodes (tokens + pools).
            node_feat_dim: Node feature dimension.
            edge_feat_dim: Edge feature dimension.
            embed_dim: Embedding dimension.
            num_layers: Number of GATv2-GRU layers.
            num_heads: Attention heads per layer.
            num_clusters: Number of clusters for MCR².

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        # Feature projector
        self.projector = nn.Sequential(
            nn.Linear(node_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Learnable node embeddings (captures static "personality")
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        # GATv2-GRU layers
        self.layers = nn.ModuleList(
            [
                GATv2GRUCell(embed_dim, embed_dim, num_heads, edge_feat_dim)
                for _ in range(num_layers)
            ]
        )

        # Embed head: final Z embeddings for MCR²
        self.embed_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cluster head: soft cluster assignments Π
        self.cluster_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_clusters),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        node_feats: Tensor,
        node_ids: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning embeddings and cluster assignments.

        Args:
            node_feats: Dynamic node features [batch_nodes, feat_dim].
            node_ids: Global node IDs [batch_nodes].
            edge_index: Edge connectivity [2, num_edges] (batch-local).
            edge_attr: Edge features [num_edges, edge_dim].

        Returns:
            Tuple of (Z embeddings, Π cluster assignments).

        """
        # Project features
        x = self.projector(node_feats)

        # Add learnable embeddings (clamp indices to valid range)
        valid_ids = torch.clamp(node_ids, 0, self.num_nodes - 1)
        x = x + self.node_embeddings[valid_ids]

        # Message passing through GATv2-GRU layers
        H = None
        for layer in self.layers:
            H = layer(x, edge_index, edge_attr, H)

        # If no edges were processed, use projected features
        if H is None:
            H = x

        # Output heads
        Z = self.embed_head(H)
        Pi = self.cluster_head(H)

        return Z, Pi


# ============================================================================
# MCR² Loss
# ============================================================================


def stable_logdet(A: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute numerically stable log determinant via Cholesky.

    Args:
        A: Positive definite matrix.
        eps: Small constant for numerical stability.

    Returns:
        Log determinant of A.

    """
    A = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    try:
        L = torch.linalg.cholesky(A)
        return 2.0 * torch.sum(torch.log(torch.diag(L)))
    except RuntimeError:
        # Fallback to eigenvalue decomposition
        eigenvalues = torch.linalg.eigvalsh(A)
        eigenvalues = torch.clamp(eigenvalues, min=eps)
        return torch.sum(torch.log(eigenvalues))


class MCR2Loss(nn.Module):
    """Maximal Coding Rate Reduction loss for LDR embeddings.

    L = -R_expansion + γ * R_compression

    Expansion: Encourages embeddings to spread across the full space
    Compression: Encourages within-cluster compactness

    References:
        - https://arxiv.org/abs/2006.08558

    """

    def __init__(
        self,
        embed_dim: int,
        eps: float = 0.5,
        gamma: float = 1.0,
    ) -> None:
        """Initialize MCR² loss.

        Args:
            embed_dim: Embedding dimension.
            eps: Precision scalar (controls tightness).
            gamma: Weight for compression term.

        """
        super().__init__()
        self.d = embed_dim
        self.eps = eps
        self.gamma = gamma

    def forward(self, Z: Tensor, Pi: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Compute MCR² loss.

        Args:
            Z: Embeddings [N, d].
            Pi: Soft cluster assignments [N, k].

        Returns:
            Tuple of (loss, metrics dict).

        """
        N = Z.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=Z.device), {
                "expansion": 0.0,
                "compression": 0.0,
            }

        I = torch.eye(self.d, device=Z.device, dtype=Z.dtype)

        # Normalize embeddings
        Z_norm = F.normalize(Z, dim=1)

        # Expansion: log det(I + scalar * Z^T Z)
        scalar = self.d / (N * self.eps**2)
        gram = Z_norm.T @ Z_norm
        R_exp = 0.5 * stable_logdet(I + scalar * gram)

        # Compression: sum over clusters
        R_comp = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
        for j in range(Pi.shape[1]):
            pi_j = Pi[:, j]
            size_j = pi_j.sum() + 1e-8

            if size_j < 1.0:
                continue

            scalar_j = self.d / (size_j * self.eps**2)
            sqrt_pi = torch.sqrt(pi_j + 1e-8).unsqueeze(1)
            Z_weighted = sqrt_pi * Z_norm
            gram_j = Z_weighted.T @ Z_weighted

            logdet_j = stable_logdet(I + scalar_j * gram_j)
            R_comp = R_comp + (size_j / N) * 0.5 * logdet_j

        loss = -R_exp + self.gamma * R_comp

        return loss, {
            "expansion": R_exp.item(),
            "compression": R_comp.item() if isinstance(R_comp, Tensor) else R_comp,
        }


def cluster_balance_loss(Pi: Tensor) -> Tensor:
    """KL divergence from uniform distribution for balanced clusters.

    Args:
        Pi: Soft cluster assignments [N, k].

    Returns:
        KL divergence loss.

    """
    avg_pi = Pi.mean(dim=0)
    target = torch.ones_like(avg_pi) / Pi.shape[1]
    # Add epsilon to avoid log(0)
    return F.kl_div(
        (avg_pi + 1e-8).log(),
        target,
        reduction="sum",
    )


# ============================================================================
# Training Loop
# ============================================================================


def move_batch_to_device(batch: DGBatch, device: str) -> DGBatch:
    """Move batch to device and remap global IDs to batch-local indices.

    Args:
        batch: TGM batch.
        device: Target device.

    Returns:
        Batch with tensors on device and remapped edge_index.

    """
    batch.src = batch.src.to(device)
    batch.dst = batch.dst.to(device)
    batch.time = batch.time.to(device)
    batch.edge_feats = batch.edge_feats.to(device)
    batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
    batch.node_ids = batch.node_ids.to(device)

    # Remap global node IDs to batch-local indices
    src_np = batch.src.cpu().numpy()
    dst_np = batch.dst.cpu().numpy()
    node_ids_np = batch.node_ids.cpu().numpy()

    node_id_to_pos = {int(nid): i for i, nid in enumerate(node_ids_np)}

    # Filter edges where both endpoints are in batch
    valid_edges = []
    valid_edge_indices = []
    for i, (s, d) in enumerate(zip(src_np, dst_np, strict=True)):
        s_int, d_int = int(s), int(d)
        if s_int in node_id_to_pos and d_int in node_id_to_pos:
            valid_edges.append((node_id_to_pos[s_int], node_id_to_pos[d_int]))
            valid_edge_indices.append(i)

    if valid_edges:
        src_local = torch.tensor(
            [s for s, _ in valid_edges], device=device, dtype=torch.long
        )
        dst_local = torch.tensor(
            [d for _, d in valid_edges], device=device, dtype=torch.long
        )
        batch.edge_index = torch.stack([src_local, dst_local], dim=0)

        if len(valid_edge_indices) < len(src_np):
            indices_tensor = torch.tensor(valid_edge_indices, dtype=torch.long)
            batch.edge_feats = batch.edge_feats[indices_tensor].to(device)
    else:
        batch.edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
        batch.edge_feats = torch.zeros(
            (0, batch.edge_feats.shape[1]), device=device, dtype=batch.edge_feats.dtype
        )

    return batch


def train_epoch(
    model: BipartiteMCR2Model,
    dataloader: DGDataLoader,
    mcr2_loss_fn: MCR2Loss,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: BipartiteMCR2Model.
        dataloader: TGM data loader.
        mcr2_loss_fn: MCR² loss function.
        optimizer: Optimizer.
        device: Device.

    Returns:
        Dictionary of training metrics.

    """
    model.train()

    total_loss = 0.0
    total_expansion = 0.0
    total_compression = 0.0
    total_balance = 0.0
    num_batches = 0

    for batch in dataloader:
        if batch.src.shape[0] == 0:
            continue

        batch = move_batch_to_device(batch, device)

        if batch.edge_index.shape[1] == 0:
            continue

        optimizer.zero_grad()

        # Forward pass
        Z, Pi = model(
            node_feats=batch.dynamic_node_feats,
            node_ids=batch.node_ids,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_feats,
        )

        # MCR² loss
        mcr2_loss, mcr2_metrics = mcr2_loss_fn(Z, Pi)

        # Cluster balance regularization
        balance_loss = cluster_balance_loss(Pi)

        # Total loss
        loss = mcr2_loss + CLUSTER_BALANCE_WEIGHT * balance_loss

        # Skip if loss is NaN
        if torch.isnan(loss):
            continue

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_expansion += mcr2_metrics["expansion"]
        total_compression += mcr2_metrics["compression"]
        total_balance += balance_loss.item()
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "expansion": 0.0, "compression": 0.0, "balance": 0.0}

    return {
        "loss": total_loss / num_batches,
        "expansion": total_expansion / num_batches,
        "compression": total_compression / num_batches,
        "balance": total_balance / num_batches,
    }


def validate(
    model: BipartiteMCR2Model,
    dataloader: DGDataLoader,
    mcr2_loss_fn: MCR2Loss,
    device: str,
) -> dict[str, float]:
    """Validate model.

    Args:
        model: BipartiteMCR2Model.
        dataloader: TGM data loader.
        mcr2_loss_fn: MCR² loss function.
        device: Device.

    Returns:
        Dictionary of validation metrics.

    """
    model.eval()

    all_Z = []
    all_Pi = []

    with torch.no_grad():
        for batch in dataloader:
            if batch.src.shape[0] == 0:
                continue

            batch = move_batch_to_device(batch, device)

            if batch.edge_index.shape[1] == 0:
                continue

            Z, Pi = model(
                node_feats=batch.dynamic_node_feats,
                node_ids=batch.node_ids,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_feats,
            )

            all_Z.append(Z.cpu())
            all_Pi.append(Pi.cpu())

    if not all_Z:
        return {"val_loss": 0.0, "silhouette": 0.0}

    Z_cat = torch.cat(all_Z, dim=0)
    Pi_cat = torch.cat(all_Pi, dim=0)

    # MCR² loss on validation
    mcr2_loss, mcr2_metrics = mcr2_loss_fn(Z_cat, Pi_cat)

    # Silhouette score on hard clusters
    hard_clusters = Pi_cat.argmax(dim=1).numpy()
    Z_np = Z_cat.numpy()

    # Need at least 2 clusters with samples
    unique_clusters = np.unique(hard_clusters)
    if len(unique_clusters) >= 2 and len(Z_np) >= 2:
        try:
            sil = silhouette_score(Z_np, hard_clusters)
        except ValueError:
            sil = 0.0
    else:
        sil = 0.0

    return {
        "val_loss": mcr2_loss.item(),
        "val_expansion": mcr2_metrics["expansion"],
        "val_compression": mcr2_metrics["compression"],
        "silhouette": sil,
    }


def extract_final_embeddings(
    model: BipartiteMCR2Model,
    dataloader: DGDataLoader,
    device: str,
    num_tokens: int,
    num_pools: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract final embeddings via last-seen aggregation.

    Args:
        model: Trained model.
        dataloader: Data loader.
        device: Device.
        num_tokens: Number of tokens.
        num_pools: Number of pools.

    Returns:
        Tuple of (token_Z, pool_Z, token_Pi, pool_Pi).

    """
    logger.info("Extracting final embeddings...")
    model.eval()

    # Track last-seen embeddings for each node
    token_Z = {}
    pool_Z = {}
    token_Pi = {}
    pool_Pi = {}

    with torch.no_grad():
        for batch in dataloader:
            if batch.src.shape[0] == 0:
                continue

            batch = move_batch_to_device(batch, device)

            if batch.edge_index.shape[1] == 0:
                continue

            Z, Pi = model(
                node_feats=batch.dynamic_node_feats,
                node_ids=batch.node_ids,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_feats,
            )

            # Map back to global node IDs
            node_ids_np = batch.node_ids.cpu().numpy()
            Z_np = Z.cpu().numpy()
            Pi_np = Pi.cpu().numpy()

            for i, nid in enumerate(node_ids_np):
                if nid < num_tokens:
                    token_Z[nid] = Z_np[i]
                    token_Pi[nid] = Pi_np[i]
                else:
                    pool_id = nid - num_tokens
                    pool_Z[pool_id] = Z_np[i]
                    pool_Pi[pool_id] = Pi_np[i]

    # Convert to arrays (fill missing with zeros)
    token_Z_arr = np.zeros((num_tokens, model.embed_dim))
    token_Pi_arr = np.zeros((num_tokens, Pi.shape[1]))
    for tid, emb in token_Z.items():
        token_Z_arr[tid] = emb
    for tid, pi in token_Pi.items():
        token_Pi_arr[tid] = pi

    pool_Z_arr = np.zeros((num_pools, model.embed_dim))
    pool_Pi_arr = np.zeros((num_pools, Pi.shape[1]))
    for pid, emb in pool_Z.items():
        if pid < num_pools:
            pool_Z_arr[pid] = emb
    for pid, pi in pool_Pi.items():
        if pid < num_pools:
            pool_Pi_arr[pid] = pi

    logger.info("  Tokens with embeddings: %s / %s", len(token_Z), num_tokens)
    logger.info("  Pools with embeddings: %s / %s", len(pool_Z), num_pools)

    return token_Z_arr, pool_Z_arr, token_Pi_arr, pool_Pi_arr


# ============================================================================
# CLI
# ============================================================================

app = typer.Typer()


@app.command()
def main(
    epochs: int = typer.Option(50, help="Maximum training epochs"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    num_clusters: int = typer.Option(8, help="Number of clusters for MCR²"),
    embed_dim: int = typer.Option(64, help="Embedding dimension"),
    num_heads: int = typer.Option(4, help="Attention heads per layer"),
    num_layers: int = typer.Option(2, help="Number of GATv2-GRU layers"),
) -> None:
    """Train bipartite GNN with MCR² loss for LDR embeddings."""
    logger.info("=" * 70)
    logger.info("BIPARTITE GNN WITH MCR² LOSS FOR LDR EMBEDDINGS")
    logger.info("=" * 70)

    # 1. Load data
    df, pool_meta = load_data()

    # 2. Build bipartite topology
    topology = build_bipartite_topology(df)

    # 3. Temporal split (70% train, 15% val, 15% test)
    total_rows = df.shape[0]
    train_size = int(total_rows * 0.7)
    val_size = int(total_rows * 0.15)

    train_df = df.slice(0, train_size)
    val_df = df.slice(train_size, val_size)
    test_df = df.slice(train_size + val_size, total_rows - train_size - val_size)

    logger.info("Data splits:")
    logger.info("  Train: %s rows (70%%)", f"{train_df.shape[0]:,}")
    logger.info("  Val: %s rows (15%%)", f"{val_df.shape[0]:,}")
    logger.info("  Test: %s rows (15%%)", f"{test_df.shape[0]:,}")

    # 4. Build DGData objects
    train_data = build_streaming_data(train_df, topology, pool_meta)
    val_data = build_streaming_data(val_df, topology, pool_meta)
    test_data = build_streaming_data(test_df, topology, pool_meta)

    # 5. Create data loaders
    train_dg = DGraph(train_data, device=DEVICE)
    train_loader = DGDataLoader(train_dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    val_dg = DGraph(val_data, device=DEVICE)
    val_loader = DGDataLoader(val_dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    # 6. Initialize model
    model = BipartiteMCR2Model(
        num_nodes=topology["num_nodes"],
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_clusters=num_clusters,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized with %s parameters", f"{total_params:,}")

    # 7. Loss and optimizer
    mcr2_loss_fn = MCR2Loss(embed_dim=embed_dim, eps=MCR2_EPS, gamma=MCR2_GAMMA)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    # 8. Training loop
    history: dict[str, list[float]] = {
        "train_loss": [],
        "expansion": [],
        "compression": [],
        "val_loss": [],
        "silhouette": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training (%s epochs, patience=%s)...", epochs, patience)

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Training", total=epochs)

        for epoch in range(epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, mcr2_loss_fn, optimizer, DEVICE
            )

            # Validate
            val_metrics = validate(model, val_loader, mcr2_loss_fn, DEVICE)

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["expansion"].append(train_metrics["expansion"])
            history["compression"].append(train_metrics["compression"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["silhouette"].append(val_metrics["silhouette"])

            # Log progress
            logger.info(
                "Epoch %s/%s | Loss: %.4f | Exp: %.2f | Comp: %.2f | "
                "Val: %.4f | Sil: %.4f",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                train_metrics["expansion"],
                train_metrics["compression"],
                val_metrics["val_loss"],
                val_metrics["silhouette"],
            )

            # Early stopping
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                logger.info("  ✅ New best validation loss: %.4f", best_val_loss)
            else:
                patience_counter += 1
                logger.info(
                    "  ⚠️  No improvement (patience: %s/%s)", patience_counter, patience
                )

            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %s", epoch + 1)
                break

            progress.update(task, advance=1)

    training_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds", training_time)

    # 9. Extract embeddings from test set
    test_dg = DGraph(test_data, device=DEVICE)
    test_loader = DGDataLoader(test_dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    token_Z, pool_Z, token_Pi, pool_Pi = extract_final_embeddings(
        model,
        test_loader,
        DEVICE,
        topology["num_tokens"],
        topology["num_pools"],
    )

    # 10. Save results
    timestamp = int(time.time())
    output_path = OUTPUT_DIR / f"mcr2_bipartite_{timestamp}.npz"

    metadata = {
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "final_silhouette": history["silhouette"][-1] if history["silhouette"] else 0.0,
        "training_time_sec": training_time,
        "embed_dim": embed_dim,
        "num_clusters": num_clusters,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_tokens": topology["num_tokens"],
        "num_pools": topology["num_pools"],
        "mcr2_eps": MCR2_EPS,
        "mcr2_gamma": MCR2_GAMMA,
    }

    np.savez_compressed(
        output_path,
        token_embeddings=token_Z,
        pool_embeddings=pool_Z,
        token_clusters=token_Pi,
        pool_clusters=pool_Pi,
        token_addresses=topology["token_encoder"].classes_,
        pool_addresses=topology["pool_encoder"].classes_,
        train_loss_history=np.array(history["train_loss"]),
        expansion_history=np.array(history["expansion"]),
        compression_history=np.array(history["compression"]),
        val_loss_history=np.array(history["val_loss"]),
        silhouette_history=np.array(history["silhouette"]),
        **{f"meta_{k}": v for k, v in metadata.items()},
    )

    logger.info("=" * 70)
    logger.info("✅ Embeddings saved to %s", output_path)
    logger.info("=" * 70)
    logger.info("Summary:")
    logger.info("  Epochs: %s", metadata["epochs_trained"])
    logger.info("  Best Val Loss: %.4f", best_val_loss)
    logger.info("  Final Silhouette: %.4f", metadata["final_silhouette"])
    logger.info("  Token embeddings: [%s, %s]", token_Z.shape[0], token_Z.shape[1])
    logger.info("  Pool embeddings: [%s, %s]", pool_Z.shape[0], pool_Z.shape[1])
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
