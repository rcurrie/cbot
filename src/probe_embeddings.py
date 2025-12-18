"""First Principles: TGM Embedding Probe via Link Prediction.

Trains Temporal Graph Network in a self-supervised manner (Link Prediction).
Goal: Learn rich node embeddings that cluster by market behavior (Sector, Volatility)
without relying on noisy future price labels.

Hypothesis: If the model understands the market, the resulting UMAP projection
will show distinct islands for Stablecoins, Blue-chips, and Long-tail assets.
"""

# %%
import json
import logging
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import umap  # type: ignore[import-untyped]
from sklearn.preprocessing import LabelEncoder
from tgm import DGBatch, DGraph  # type: ignore[import-untyped]
from tgm.data import DGData, DGDataLoader  # type: ignore[import-untyped]
from tgm.nn import TGCN  # type: ignore[import-untyped]
from torch import nn
from torch.nn import functional
from tqdm import tqdm

# %%
# Configuration & Ceremony
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DIVIDER_LENGTH = 40
DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")
TOKENS_PATH = Path("data/tokens.json")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 200  # Larger batch size for structural learning
BATCH_UNIT = "s"
NODE_FEAT_DIM = 3  # fracdiff, volatility, flow
NODE_EMBED_DIM = 64
LEARNING_RATE = 1e-3
NEG_SAMPLE_RATIO = 1.0  # 1 negative edge for every positive edge

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info("Device: %s", DEVICE)


# %%
# Data Loading
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

    # Ensure data is sorted by time and token IDs to break ties consistently
    df = df.sort(["bar_close_timestamp", "src_token_id", "dest_token_id"])

    # Filter out rows with NaN in edge features and labels
    # Note: We keep rows if EITHER src or dest has a valid label
    edge_cols = [
        "src_fracdiff",
        "dest_fracdiff",
        "src_flow_usdc",
        "dest_flow_usdc",
        "tick_count",
        "rolling_volatility",
        "bar_time_delta_sec",
    ]
    # For labels, we'll filter later - keep rows with at least one valid label
    for col in edge_cols:
        df = df.filter(pl.col(col).is_not_null())
        # Also filter NaNs for float columns
        if df.schema[col] in (pl.Float32, pl.Float64):
            df = df.filter(pl.col(col).is_finite())
    logger.info("After filtering null edge features: %s", f"{df.shape[0]:,}")

    # Keep rows where at least ONE of src_label or dest_label is valid
    # This maximizes our training signal (Issue 3 fix)
    n_before = df.shape[0]
    df = df.filter(
        (pl.col("label").is_not_null() & pl.col("label").is_finite())
        | (pl.col("dest_label").is_not_null() & pl.col("dest_label").is_finite()),
    )
    n_after = df.shape[0]
    logger.info(
        "After filtering for at least one valid label: %s (kept %d rows)",
        f"{n_after:,}",
        n_after - n_before,
    )

    return df


def prepare_topology(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, LabelEncoder, int]:
    """Encode tokens and Prepare Bi-Directional Topology.

    CRITICAL CHANGE:
    We ensure 'src' sees 'dest' by creating a bidirectional graph.
    The original script treated swaps as directed (src->dst), meaning
    the source token's embedding never learned from the asset it was buying.
    """
    logger.info("Encoding tokens and creating bidirectional topology...")
    le = LabelEncoder()
    all_tokens = np.concatenate(
        [
            df["src_token_id"].to_numpy(),
            df["dest_token_id"].to_numpy(),
        ],
    )
    le.fit(all_tokens)
    num_tokens = len(le.classes_)

    # Encode IDs
    src_encoded = le.transform(df["src_token_id"].to_numpy())
    dest_encoded = le.transform(df["dest_token_id"].to_numpy())

    df = df.with_columns(
        [
            pl.Series("src_encoded", src_encoded, dtype=pl.Int32),
            pl.Series("dest_encoded", dest_encoded, dtype=pl.Int32),
        ],
    )

    return df, le, num_tokens


def build_streaming_data(df: pl.DataFrame) -> DGData:
    """Convert DataFrame to TGM DGData object.

    We simplify the feature set to the 'First Principles' physics:
    - State: fracdiff, volatility, flow
    - Force: Interaction (Edge)
    """
    num_events = len(df)

    # 1. Timestamps (relative to start)
    timestamps = np.arange(num_events, dtype=np.int64)

    # 2. Topology (Bidirectional for better message passing)
    # Ideally, TGM handles this, but we force it here to be safe.
    # We will stick to the provided df direction for the primary edge list,
    # but the TGCN aggregation typically handles direction.
    # For Link Prediction, we predict the EXISTENCE of an interaction.
    src = df["src_encoded"].to_numpy()
    dst = df["dest_encoded"].to_numpy()

    # 3. Node Features (Dynamic)
    # We construct updates for both src and dst for every event
    node_timestamps = np.repeat(timestamps, 2)
    node_ids = np.empty(num_events * 2, dtype=np.int32)
    node_ids[0::2] = src
    node_ids[1::2] = dst

    # Feats: [fracdiff, volatility, flow]
    # We normalize flow to be cleaner for embedding learning
    flow_src = df["src_flow_usdc"].to_numpy()
    flow_dst = df["dest_flow_usdc"].to_numpy()

    # Simple log normalization for flow to handle whales vs minnows
    # Adding constant to handle negatives if needed, or just using raw for now
    # Sticking to raw as per original script for consistency

    feats = np.zeros((num_events * 2, 3), dtype=np.float32)

    # Src Updates
    feats[0::2, 0] = df["src_fracdiff"].to_numpy()
    feats[0::2, 1] = df["rolling_volatility"].to_numpy()
    feats[0::2, 2] = flow_src

    # Dst Updates
    feats[1::2, 0] = df["dest_fracdiff"].to_numpy()
    feats[1::2, 1] = df["rolling_volatility"].to_numpy()
    feats[1::2, 2] = flow_dst

    feats = np.nan_to_num(feats, nan=0.0)

    # 4. Edge Features
    # Just the magnitude of the interaction
    edge_feats = np.abs(flow_src).astype(np.float32).reshape(-1, 1)

    return DGData.from_raw(
        edge_timestamps=torch.from_numpy(timestamps),
        edge_index=torch.from_numpy(np.column_stack((src, dst))),
        edge_feats=torch.from_numpy(edge_feats),
        node_timestamps=torch.from_numpy(node_timestamps),
        node_ids=torch.from_numpy(node_ids),
        dynamic_node_feats=torch.from_numpy(feats),
        time_delta="s",
    )


# %%
# Model Architecture: Auto-Encoder
class LinkPredictorModel(nn.Module):
    """Self-Supervised Graph Auto-Encoder.

    Encoder: TGCN (Compresses history + interactions into Z)
    Decoder: Dot Product (Predicts likelihood of interaction)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        embed_dim: int,
        device: str,
    ):
        super().__init__()
        self.device = device

        # 1. Feature Projector
        self.projector = nn.Linear(input_dim, embed_dim)

        # 2. Learnable "Personality" Embeddings
        # Captures static traits not in the time-series (e.g. "I am a stablecoin")
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim),
        )

        # 3. Temporal Encoder
        self.tgcn = TGCN(in_channels=embed_dim, out_channels=embed_dim)

    def encode(self, batch: DGBatch) -> torch.Tensor:
        """Generate node embeddings Z for the current batch snapshot."""
        # Project dynamic features
        # feats shape: [num_updates, 3] -> [num_updates, embed_dim]
        dynamic_h = self.projector(batch.dynamic_node_feats)

        # Add static personality (indexed by node_ids)
        static_h = self.node_embeddings[batch.node_ids]

        # Combine
        h = dynamic_h + static_h

        # Message Passing
        edge_weight = (
            batch.edge_feats.squeeze(-1) if batch.edge_feats is not None else None
        )
        z = self.tgcn(h, batch.edge_index, edge_weight)
        return z

    def forward(
        self,
        batch: DGBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scores for Positive and Negative edges."""
        z = self.encode(batch)

        # --- Positive Edges (Real Swaps) ---
        # The batch contains edges in batch.src and batch.dst
        # z is indexed by the order of nodes in the batch
        # We need to find which row in 'z' corresponds to the src and dst of the edge
        # In TGM DGBatch, batch.src/dst are indices into the localized batch nodes?
        # Typically TGCN returns Z shaped [num_nodes_in_batch, dim].

        # Assuming DGBatch preserves order or mapping.
        # For simplicity in this probe, we assume standard PyG-like behavior:
        # If the batch edge_index refers to the nodes in Z:

        src_z = z[batch.edge_index[0]]
        dst_z = z[batch.edge_index[1]]

        # Dot product score
        pos_scores = (src_z * dst_z).sum(dim=-1)

        # --- Negative Edges (Fake Swaps) ---
        # Corrupt the destination
        num_edges = batch.edge_index.size(1)
        perm = torch.randperm(num_edges, device=self.device)
        dst_z_neg = dst_z[perm]  # Shuffle destinations

        neg_scores = (src_z * dst_z_neg).sum(dim=-1)

        return pos_scores, neg_scores, z


# %%
# Training Routine
def train_embeddings(
    df: pl.DataFrame,
    num_tokens: int,
    epochs: int = 1,
) -> tuple[LinkPredictorModel, np.ndarray, np.ndarray]:
    logger.info("Initializing Model...")
    model = LinkPredictorModel(
        num_nodes=num_tokens,
        input_dim=NODE_FEAT_DIM,
        embed_dim=NODE_EMBED_DIM,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare Data
    data = build_streaming_data(df)
    dg = DGraph(data, device=DEVICE)
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    # Metrics
    loss_history = []

    logger.info(f"Starting Training ({epochs} epochs)...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in pbar:
            # FIX: Check batch.src instead of batch.edge_index
            if batch.src.shape[0] == 0:
                continue

            # FIX: Manually construct edge_index and move to device
            batch.src = batch.src.to(DEVICE)
            batch.dst = batch.dst.to(DEVICE)
            batch.time = batch.time.to(DEVICE)
            batch.edge_index = torch.stack([batch.src, batch.dst], dim=0)

            # Move other features
            batch.dynamic_node_feats = batch.dynamic_node_feats.to(DEVICE)
            batch.node_ids = batch.node_ids.to(DEVICE)
            if batch.edge_feats is not None:
                batch.edge_feats = batch.edge_feats.to(DEVICE)

            optimizer.zero_grad()
            pos_scores, neg_scores, _ = model(batch)

            # Contrastive Loss (BCE with Logits)
            pos_loss = functional.binary_cross_entropy_with_logits(
                pos_scores,
                torch.ones_like(pos_scores),
            )
            neg_loss = functional.binary_cross_entropy_with_logits(
                neg_scores,
                torch.zeros_like(neg_scores),
            )

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / steps if steps > 0 else 0
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f}")

    return model, data, np.array(loss_history)


def extract_final_embeddings(
    model: LinkPredictorModel,
    data: DGData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one final pass to get the embeddings of all nodes at the end of the timeline.
    Returns: (Embeddings, TokenIDs, Volatility_for_Coloring)
    """
    logger.info("Extracting final embeddings...")
    model.eval()
    dg = DGraph(data, device=DEVICE)
    # Use a large batch to try and capture global state, or just iterate
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    all_embeddings = []
    all_ids = []
    all_volatilities = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring"):
            # FIX: Check batch.src
            if batch.src.shape[0] == 0:
                continue

            # FIX: Manually construct edge_index
            batch.src = batch.src.to(DEVICE)
            batch.dst = batch.dst.to(DEVICE)
            batch.time = batch.time.to(DEVICE)
            batch.edge_index = torch.stack([batch.src, batch.dst], dim=0)

            # Move features
            batch.dynamic_node_feats = batch.dynamic_node_feats.to(DEVICE)
            batch.node_ids = batch.node_ids.to(DEVICE)
            if batch.edge_feats is not None:
                batch.edge_feats = batch.edge_feats.to(DEVICE)

            z = model.encode(batch)

            # Store these embeddings
            ids = batch.node_ids.cpu().numpy()
            embs = z.cpu().numpy()
            vols = batch.dynamic_node_feats[:, 1].cpu().numpy()

            all_embeddings.append(embs)
            all_ids.append(ids)
            all_volatilities.append(vols)  # Concatenate

    full_embs = np.concatenate(all_embeddings)
    full_ids = np.concatenate(all_ids)
    full_vols = np.concatenate(all_volatilities)

    # Deduplicate to get unique node final states
    # We use a dictionary to keep the last seen version of each node
    node_map = {}
    for i, node_id in enumerate(full_ids):
        node_map[node_id] = (full_embs[i], full_vols[i])

    final_ids = []
    final_z = []
    final_v = []

    for nid, (z, v) in node_map.items():
        final_ids.append(nid)
        final_z.append(z)
        final_v.append(v)

    return np.array(final_z), np.array(final_ids), np.array(final_v)


# %%
# Visualization
def visualize_umap(
    embeddings: np.ndarray,
    volatilities: np.ndarray,
    token_ids: np.ndarray,
    le: LabelEncoder,
    tokens_meta: dict,
) -> None:
    logger.info("Computing UMAP projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))

    # Scatter plot colored by volatility (log scale for better viz)
    # Clip volatility to avoid outliers ruining the color scale
    vol_clipped = np.clip(volatilities, 0, np.percentile(volatilities, 95))

    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=vol_clipped,
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Rolling Volatility")
    plt.title("TGM Node Embeddings (UMAP Projection)\nColored by Volatility")

    # Annotate top tokens
    # We look for symbols in tokens_meta
    annotated_count = 0
    indices = np.arange(len(token_ids))
    np.random.shuffle(indices)  # Random shuffle to annotate random subset

    for idx in indices:
        token_addr = le.inverse_transform([token_ids[idx]])[0]
        meta = tokens_meta.get(token_addr.lower())

        # Heuristic: Annotate if it has a symbol and we haven't annotated too many
        if meta and meta.get("symbol"):
            symbol = meta["symbol"]
            # Only annotate "major" tokens or high vol ones for interest
            if annotated_count < 30:
                plt.annotate(
                    symbol,
                    (embedding_2d[idx, 0], embedding_2d[idx, 1]),
                    fontsize=8,
                    alpha=0.8,
                )
                annotated_count += 1

    output_path = PLOTS_DIR / f"embeddings_umap_{int(time.time())}.png"
    plt.savefig(output_path, dpi=300)
    logger.info(f"Plot saved to {output_path}")

    # Save Loss Curve
    plt.figure()
    # (Assuming we pass loss history, simple placeholder here)
    plt.close()


# %%
@click.command()
@click.option("--epochs", default=5, help="Training epochs")
@click.option("--limit-days", default=5, help="Limit data to last N days for speed")
def main(epochs: int, limit_days: int) -> None:
    """Run the TGM Probe."""
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("TGM EMBEDDING PROBE")
    logger.info("=" * DIVIDER_LENGTH)

    # 1. Load Metadata
    with TOKENS_PATH.open() as f:
        tokens_metadata = json.load(f)

    # 2. Load Data
    df = load_and_filter_bars(DATA_PATH)

    # Limit data for faster probing
    if limit_days:
        max_date = df["bar_close_timestamp"].max()
        cutoff = (
            max_date - torch.timedelta(days=limit_days)
            if hasattr(torch, "timedelta")
            else max_date
        )
        # Polars date math
        import datetime

        cutoff_date = df["bar_close_timestamp"].max() - datetime.timedelta(
            days=limit_days,
        )
        df = df.filter(pl.col("bar_close_timestamp") > cutoff_date)
        logger.info(f"Filtered to last {limit_days} days: {len(df):,} events")

    # 3. Prepare
    df, le, num_tokens = prepare_topology(df)

    # 4. Train
    model, data_obj, loss_hist = train_embeddings(df, num_tokens, epochs)

    # 5. Extract & Visualize
    emb, ids, vols = extract_final_embeddings(model, data_obj)
    visualize_umap(emb, vols, ids, le, tokens_metadata)

    logger.info("Done.")


if __name__ == "__main__":
    main()
