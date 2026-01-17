"""Temporal Graph Convolutional Network (TGCN) trader with daily sliding backtest.

WHY: Traditional time-series models ignore the network effects and contagion
patterns that dominate crypto markets. Price movements propagate through the token
graph via swap events - when ETH moves, tokens paired with ETH are affected. TGCN
captures this by modeling the temporal dynamics of the token swap graph (nodes =
tokens, edges = swaps).

This approach combines Prado's rigorous ML methodology (fractional differentiation,
triple-barrier labeling, concurrency weighting) with graph neural networks to model
information flow through the DEX ecosystem. We use a sliding backtest where the
model is retrained daily from scratch to prevent look-ahead bias and simulate real
trading.

WHAT: Daily sliding backtest implementing:
1. Train TGCN on 5 days of historical data + morning data (up to 9am)
2. Predict top N tokens with highest probability of upward movement at 9am
3. Execute equal-weighted long positions from 9am to 5pm
4. Apply risk management (stop-loss at -20%, take-profit at +200%)
5. Calculate returns and slide forward 1 day, retraining from scratch
6. Aggregate performance metrics: Sharpe ratio, max drawdown, win rate

Binary classification strategy: Model predicts up (invest) vs down (don't invest).
Original triple-barrier labels (up/stay/down) are merged: stay+up → up (invest),
down → down (no invest). This focuses the model on avoiding losses.

HOW:
1. Build temporal graph from labeled bars with bidirectional edges
2. Node features: fracdiff price, rolling volatility, flow magnitude (dynamic)
3. Edge features: flow magnitude as attention weights
4. TGCN architecture: learnable embeddings + feature projection + message passing
5. Training: Binary cross-entropy with class weights and Prado sample weights
6. Prediction: Softmax probabilities, invest in tokens with P(up) > 0.5
7. Returns: Track intraday performance with stop-loss/take-profit exits

INPUT: data/labeled_log_fracdiff_price.parquet (final labeled training data)
       data/tokens.json (token metadata for display)
OUTPUT: Console backtest report with daily returns and summary statistics

References:
- Prado AFML Ch. 3: Triple-Barrier Method and Meta-Labeling
- Prado AFML Ch. 4: Sample Weights from Label Concurrency
- Prado AFML Ch. 5: Fractional Differentiation
- Prado AFML Ch. 7: Cross-Validation for Financial ML
- TGCN: Temporal Graph Convolutional Networks for message passing on dynamic graphs

"""

import json
import logging
import time
from datetime import date
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
from sklearn.preprocessing import LabelEncoder
from tgm import DGBatch, DGraph  # type: ignore[import-untyped]
from tgm.data import DGData, DGDataLoader  # type: ignore[import-untyped]
from tgm.nn import TGCN, NodePredictor  # type: ignore[import-untyped]
from torch import nn

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Number of '=' characters for logging (consistent with preprocessing scripts)
DIVIDER_LENGTH = 70

# Create Typer app for CLI
app = typer.Typer(
    help="Daily sliding backtest for TGCN token price predictions.",
)

# Backtest parameters
TRAIN_WINDOW_DAYS = 5
TRADE_WINDOW_HOURS = (9, 17)  # 9am to 5pm EST
TOP_N_TOKENS = 5
SLIDE_STEP_DAYS = 1

# Risk management
STOP_LOSS_PCT = 0.20  # Exit if position drops 20%
TAKE_PROFIT_PCT = 2.0  # Exit if position gains 200%

# Model hyperparameters
DYNAMIC_NODE_FEAT_DIM = 5  # fracdiff, volatility, flow, label, weight
NODE_INPUT_DIM = 3  # fracdiff, volatility, flow (first 3 of dynamic features)
NODE_EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 2  # Binary classification: down (0), up (1)
LABEL_UP = 1
BATCH_SIZE = 100
BATCH_UNIT = "s"  # Batch by 100 seconds (which are events in our case)
LEARNING_RATE = 1e-3
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
        logger.error("  ❌ ERROR: Input file not found: %s", data_path)
        msg = f"Input file not found: {data_path}"
        raise FileNotFoundError(msg)

    logger.info("  ✅ OK: Input file exists: %s", data_path)

    # Quick read to check basic structure
    df = pl.read_parquet(data_path)
    required_cols = [
        "bar_close_timestamp",
        "src_token_id",
        "dest_token_id",
        "src_fracdiff",
        "dest_fracdiff",
        "label",
        "dest_label",
        "sample_weight",
        "dest_sample_weight",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("  ❌ ERROR: Missing required columns: %s", missing_cols)
        msg = f"Missing required columns: {missing_cols}"
        raise ValueError(msg)

    logger.info("  ✅ OK: All required columns present")
    logger.info(
        "  ℹ️ Dataset shape: %s rows x %d columns",
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


def prepare_data(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, LabelEncoder, int, pl.DataFrame]:
    """Encode token IDs and create date index.

    Args:
        df: DataFrame with swap bars (already sorted).

    Returns:
        Tuple of (df with trade_date, label encoder, num_tokens, unique_dates).

    """
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

    # Create date index for windowing
    df = df.with_columns(
        pl.col("bar_close_timestamp").dt.date().alias("trade_date"),
    )

    # Add monotonic event index (since df is already sorted)
    # This respects dollar bar "information time" ordering
    df = df.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.Int64).alias("event_index"),
    )

    # Add encoded token IDs (avoid re-computing in build_window)
    src_encoded = le.transform(df["src_token_id"].to_numpy())
    dest_encoded = le.transform(df["dest_token_id"].to_numpy())
    df = df.with_columns(
        [
            pl.Series("src_token_encoded", src_encoded, dtype=pl.Int32),
            pl.Series("dest_token_encoded", dest_encoded, dtype=pl.Int32),
        ],
    )

    unique_dates = df.select("trade_date").unique().sort("trade_date")
    logger.info(
        "Date range: %s to %s | Total days: %d",
        unique_dates["trade_date"][0],
        unique_dates["trade_date"][-1],
        len(unique_dates),
    )

    return df, le, num_tokens, unique_dates


# Model architecture
class TokenPredictorModel(nn.Module):
    """TGCN encoder + decoder for token price movement prediction.

    Architecture:
    - Node features: fracdiff, volatility, flow (from aggregated swap data)
    - Feature projection: Linear layer to map features to embedding space
    - Node embeddings: Learnable parameters combined with projected features
    - TGCN encoder: Refines combined embeddings using temporal swaps
    - Predictor: Maps refined embeddings to price predictions (supervised)
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        node_embed_dim: int,
        output_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize token predictor model.

        Args:
            num_nodes: Total number of tokens in graph.
            node_feat_dim: Dimension of input node features (fracdiff, etc).
            node_embed_dim: Embedding dimension (used for TGCN).
            output_dim: Number of output classes (3: down/stay/up).
            device: Device to place tensors on.

        """
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.node_embed_dim = node_embed_dim

        # Project node features to embedding space
        self.feature_projection = nn.Linear(
            node_feat_dim,
            node_embed_dim,
            device=device,
        )

        # Learnable node embeddings - initialized randomly, updated via backprop
        # These complement the projected features with learnable representations
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, node_embed_dim, device=device),
        )

        # TGCN encoder: refines embeddings using temporal swap graph structure
        self.tgcn = TGCN(
            in_channels=node_embed_dim,
            out_channels=node_embed_dim,
        )

        # Predictor: maps refined embeddings to class predictions
        self.predictor = NodePredictor(
            in_dim=node_embed_dim,
            out_dim=output_dim,
            hidden_dim=node_embed_dim,
        )

    def forward(
        self,
        batch: "DGBatch",
    ) -> torch.Tensor | None:
        """Forward pass through encoder and predictor.

        Args:
            batch: DGBatch with dynamic_node_feats, edge_index, edge_feats.

        Returns:
            logits: [num_nodes_in_batch, output_dim] or None if no nodes.

        """
        # Extract node features: [fracdiff, volatility, flow]
        # Labels and weights ([:, 3:5]) are used during training
        node_features = batch.dynamic_node_feats[:, :3]

        # Project node features to embedding space
        node_feats_projected = self.feature_projection(node_features)

        # Combine projected features with learnable embeddings
        # node_ids tells us which tokens are updated at each event
        # We add the learnable embedding for each token
        x_combined = node_feats_projected + self.node_embeddings[batch.node_ids]

        # TGCN refines embeddings via message passing on swap graph
        # Extract edge weights from edge_feats (single feature: flow magnitude)
        edge_weight = (
            batch.edge_feats.squeeze(-1) if batch.edge_feats is not None else None
        )
        z = self.tgcn(x_combined, batch.edge_index, edge_weight)

        # For prediction, we only use nodes that have labels
        # Labels are in dynamic_node_feats[:, 3]
        # Filter for labeled nodes (exclude -999 marker for unlabeled dest tokens)
        has_label = batch.dynamic_node_feats[:, 3] != -999
        if has_label.any():
            z_labeled = z[has_label]
            result: torch.Tensor = self.predictor(z_labeled)
            return result
        return None


def build_window(
    df: pl.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    start_hour: int | None = None,
    end_hour: int | None = None,
) -> DGData:
    """Build a temporal window from the full DataFrame.

    This function filters the DataFrame by date and hour ranges to create a window
    of events, then constructs a DGData object for that window. Dollar bars are
    volume-based events with variable timing, but we use calendar time for windowing
    to support training on "N days" and predicting during specific hours.

    Each swap event creates TWO node updates (one for src, one for dest) with
    dynamic features that evolve over time as swaps occur.

    Args:
        df: Full DataFrame with all events (must have trade_date,
            bar_close_timestamp, and pre-computed encoded tokens and event_index).
        start_date: Start date for filtering (inclusive), or None for no limit.
        end_date: End date for filtering (inclusive), or None for no limit.
        start_hour: Start hour (0-23) for filtering (inclusive), or None.
        end_hour: End hour (0-23) for filtering (exclusive), or None.

    Returns:
        DGData object for the filtered temporal window.

    """
    df_window = df

    # Filter by date range
    if start_date is not None and end_date is not None:
        df_window = df_window.filter(
            (pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date),
        )
    elif start_date is not None:
        df_window = df_window.filter(pl.col("trade_date") >= start_date)
    elif end_date is not None:
        df_window = df_window.filter(pl.col("trade_date") <= end_date)

    # Filter by hour range (for intraday windowing)
    if start_hour is not None:
        df_window = df_window.filter(
            pl.col("bar_close_timestamp").dt.hour() >= start_hour,
        )
    if end_hour is not None:
        df_window = df_window.filter(
            pl.col("bar_close_timestamp").dt.hour() < end_hour,
        )

    # Use pre-computed encoded token IDs (already Int32 from prepare_data)
    src = df_window["src_token_encoded"].to_numpy()
    dst = df_window["dest_token_encoded"].to_numpy()
    num_events = len(df_window)

    # Use real time deltas from window start (Phase 5 Issue 4 - Option A)
    # This preserves temporal dynamics and Prado's "information time" concept
    # Windows still start at t=0 (relative) but gaps reflect actual time differences
    # Active periods: many bars/small time gaps; Inactive periods: few bars/large gaps
    timestamps = df_window["bar_close_timestamp"].to_numpy()
    first_timestamp = timestamps[0]
    # Convert to milliseconds from window start for uniqueness/sorting stability
    # Milliseconds prevent collisions when multiple swaps occur in same second
    # while staying within TGM's int32 timestamp limits (~24 days at ms resolution)
    edge_timestamps = (
        (timestamps - first_timestamp) / np.timedelta64(1, "ms")
    ).astype(np.int64)

    # Extract features for each swap event
    src_fracdiff = df_window["src_fracdiff"].to_numpy()
    dest_fracdiff = df_window["dest_fracdiff"].to_numpy()
    src_flow = df_window["src_flow_usdc"].to_numpy()
    dest_flow = df_window["dest_flow_usdc"].to_numpy()
    volatility = df_window["rolling_volatility"].to_numpy()

    # Binary classification: merge stay into up (0 or 1 -> 1 for up, -1 -> 0 for down)
    # Original labels: -1=down, 0=stay, 1=up
    # New binary labels: 0=down (no invest), 1=up (consider invest)
    # Fill null with -999 to mark unlabeled nodes (cleaned upstream in pipeline)

    # Src token labels (always present after pipeline filtering)
    raw_src_labels = df_window["label"].fill_null(-999).to_numpy().astype(np.int64)
    src_labels = np.where(
        raw_src_labels == -999,
        -999,
        np.where(raw_src_labels >= 0, 1, 0),
    )
    src_weights = df_window["sample_weight"].to_numpy()

    # Dest token labels (may be null - not all dest tokens have labels)
    raw_dest_labels = (
        df_window["dest_label"].fill_null(-999).to_numpy().astype(np.int64)
    )
    dest_labels = np.where(
        raw_dest_labels == -999,
        -999,
        np.where(raw_dest_labels >= 0, 1, 0),
    )
    dest_weights = df_window["dest_sample_weight"].fill_null(0.0).to_numpy()

    # Each swap event updates TWO nodes: src and dest
    # We create 2 node updates per edge event
    node_timestamps = np.repeat(edge_timestamps, 2)  # [0,0, 1,1, 2,2, ...]
    node_ids = np.empty(num_events * 2, dtype=np.int32)
    node_ids[0::2] = src  # Even indices: src tokens
    node_ids[1::2] = dst  # Odd indices: dest tokens

    # Dynamic node features: [fracdiff, volatility, flow, label, weight]
    # Array shape will be [num_events * 2, 5] for src and dest nodes
    dynamic_node_feats = np.zeros((num_events * 2, 5), dtype=np.float32)

    # Src node updates (even indices)
    dynamic_node_feats[0::2, 0] = src_fracdiff  # fracdiff
    dynamic_node_feats[0::2, 1] = volatility  # volatility
    dynamic_node_feats[0::2, 2] = src_flow  # flow
    dynamic_node_feats[0::2, 3] = src_labels  # label
    dynamic_node_feats[0::2, 4] = src_weights  # weight

    # Dest node updates (odd indices) - NOW WITH LABELS (Issue 3 fix)
    dynamic_node_feats[1::2, 0] = dest_fracdiff  # fracdiff
    dynamic_node_feats[1::2, 1] = volatility  # volatility
    dynamic_node_feats[1::2, 2] = dest_flow  # flow
    dynamic_node_feats[1::2, 3] = dest_labels  # label (dest tokens now labeled!)
    dynamic_node_feats[1::2, 4] = dest_weights  # weight (dest tokens now weighted!)

    # Handle NaN values in features (set to 0)
    dynamic_node_feats = np.nan_to_num(dynamic_node_feats, nan=0.0)

    # Edge features: use absolute flow magnitude (importance weighting)
    # Create bidirectional edges with direction-specific flow features
    # Forward edge (src->dst) uses src_flow, reverse edge (dst->src) uses dest_flow
    edge_feats_forward = np.abs(src_flow).astype(np.float32).reshape(-1, 1)
    edge_feats_reverse = np.abs(dest_flow).astype(np.float32).reshape(-1, 1)

    # Bi-directional edge index: (src, dst) and (dst, src)
    edges_forward = np.column_stack((src, dst))
    edges_reverse = np.column_stack((dst, src))
    edge_index = np.vstack([edges_forward, edges_reverse])

    # Combine edge features with direction-specific flows
    edge_feats = np.concatenate([edge_feats_forward, edge_feats_reverse])
    edge_timestamps = np.concatenate([edge_timestamps, edge_timestamps])

    return DGData.from_raw(
        edge_timestamps=torch.from_numpy(edge_timestamps),
        edge_index=torch.from_numpy(edge_index),
        edge_feats=torch.from_numpy(edge_feats),
        node_timestamps=torch.from_numpy(node_timestamps),
        node_ids=torch.from_numpy(node_ids),
        dynamic_node_feats=torch.from_numpy(dynamic_node_feats),
        time_delta="ms",  # Milliseconds for temporal resolution + uniqueness
    )


def compute_class_weights(
    loader: DGDataLoader,
    device: str,
) -> tuple[torch.Tensor, dict[int, int]]:
    """Compute class weights from label distribution.

    Args:
        loader: Data loader to iterate through.
        device: Device for tensors.

    Returns:
        Tuple of (class_weights tensor, label_counts dict).

    """
    label_counts = {0: 0, 1: 0}
    for batch in loader:
        if batch.src.shape[0] == 0 or batch.node_ids is None:
            continue
        batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
        has_label = batch.dynamic_node_feats[:, 3] != -999
        if has_label.any():
            labels_batch = batch.dynamic_node_feats[has_label, 3].long()
            for label in labels_batch.cpu().numpy():
                label_counts[int(label)] = label_counts.get(int(label), 0) + 1

    # Compute class weights from full distribution
    total = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES, device=device)
    if total > 0:
        for i in range(NUM_CLASSES):
            if label_counts[i] > 0:
                class_weights[i] = total / (NUM_CLASSES * label_counts[i])
        logger.info(
            "Class weights: Down=%.2f, Up=%.2f",
            class_weights[0],
            class_weights[1],
        )

    return class_weights, label_counts


def move_batch_to_device(batch: DGBatch, device: str) -> DGBatch:
    """Move all batch tensors to specified device.

    Args:
        batch: Batch to move.
        device: Target device.

    Returns:
        Batch with tensors on device.

    """
    batch.src = batch.src.to(device)
    batch.dst = batch.dst.to(device)
    batch.time = batch.time.to(device)
    edge_index = torch.stack([batch.src, batch.dst], dim=0)
    batch.edge_index = edge_index
    batch.edge_feats = batch.edge_feats.to(device)
    batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
    batch.node_ids = batch.node_ids.to(device)
    return batch


def train_model(
    model: TokenPredictorModel,
    data: DGData,
    device: str,
    epochs: int = 1,
    patience: int = 3,
    min_delta: float = 0.001,
) -> None:
    """Train model with Prado concurrency weighting and early stopping.

    Args:
        model: TokenPredictorModel instance.
        data: DGData with training events.
        device: Device for training.
        epochs: Maximum number of epochs (default 1 for sliding backtest).
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in loss to qualify as improvement.

    """
    model.train()
    dg = DGraph(data, device=device)
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Compute class weights from label distribution
    temp_loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)
    class_weights, label_counts = compute_class_weights(temp_loader, device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    # Track overall training statistics
    overall_start_time = time.time()
    first_epoch_avg_loss = None
    overall_final_loss = None
    total_batches = sum(1 for _ in loader)

    # Early stopping tracking
    best_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    # Create nested progress bars: one for epochs, one for current epoch batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        epoch_task = progress.add_task(
            f"Training (0/{epochs} epochs)",
            total=epochs,
        )
        batch_task = progress.add_task(
            "Processing batches",
            total=total_batches,
        )

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # Reset batch progress for new epoch
            progress.reset(
                batch_task,
                description=f"Epoch {epoch + 1}/{epochs}",
                total=total_batches,
            )

            for batch_data in loader:
                if batch_data.src.shape[0] == 0 or batch_data.node_ids is None:
                    progress.update(batch_task, advance=1)
                    continue

                move_batch_to_device(batch_data, device)

                optimizer.zero_grad()
                logits = model(batch_data)

                if logits is None:
                    progress.update(batch_task, advance=1)
                    continue

                # Extract labels and weights
                # Binary labels: 0=down (no invest), 1=up (consider invest)
                has_label = batch_data.dynamic_node_feats[:, 3] != -999
                y_true = batch_data.dynamic_node_feats[has_label, 3].long()
                batch_weights = batch_data.dynamic_node_feats[has_label, 4].float()

                loss_per_node = criterion(logits, y_true)
                weighted_loss = (loss_per_node * batch_weights).mean()

                weighted_loss.backward()
                optimizer.step()

                total_loss += weighted_loss.item()
                num_batches += 1

                progress.update(batch_task, advance=1)

            # Calculate average epoch loss
            if num_batches > 0:
                epoch_avg_loss = total_loss / num_batches
                overall_final_loss = epoch_avg_loss

                # Track first epoch average for comparison
                if first_epoch_avg_loss is None:
                    first_epoch_avg_loss = epoch_avg_loss

                # Check for improvement
                if epoch_avg_loss < best_loss - min_delta:
                    best_loss = epoch_avg_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if epochs_without_improvement >= patience:
                    stopped_early = True
                    progress.update(
                        epoch_task,
                        advance=epoch + 1,
                        description=f"Early stop at {epoch + 1}/{epochs} epochs",
                    )
                    break

            # Update epoch progress
            loss_str = f"{epoch_avg_loss:.4f}"
            epoch_desc = f"Training ({epoch + 1}/{epochs} epochs, loss: {loss_str})"
            progress.update(
                epoch_task,
                advance=1,
                description=epoch_desc,
            )

    # Log summary after training complete
    overall_time = time.time() - overall_start_time
    start_loss = first_epoch_avg_loss if first_epoch_avg_loss is not None else 0.0
    end_loss = overall_final_loss if overall_final_loss is not None else 0.0
    total_labels = sum(label_counts.values())

    stop_reason = "early stopped" if stopped_early else "completed"

    # Check if loss increased overall (warning sign)
    has_valid_losses = start_loss > 0 and end_loss > 0
    loss_increased = end_loss > start_loss if has_valid_losses else False

    if loss_increased:
        logger.warning(
            "⚠️  Training %s in %.2fs - Loss INCREASED: %.4f → %.4f (Best: %.4f)",
            stop_reason,
            overall_time,
            start_loss,
            end_loss,
            best_loss,
        )
        logger.warning(
            "    This suggests the model is unstable or learning rate is too high!",
        )
    else:
        logger.info(
            "Training %s in %.2fs - Start loss: %.4f, End loss: %.4f, Best loss: %.4f",
            stop_reason,
            overall_time,
            start_loss,
            end_loss,
            best_loss,
        )

    if total_labels > 0:
        down_pct = label_counts[0] / total_labels * 100
        up_pct = label_counts[1] / total_labels * 100
        logger.info(
            "Label distribution - Down: %d (%.1f%%), Up: %d (%.1f%%)",
            label_counts[0],
            down_pct,
            label_counts[1],
            up_pct,
        )


def predict_top_tokens(
    model: TokenPredictorModel,
    data: DGData,
    le: LabelEncoder,
    device: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Predict top N tokens by bullish score.

    Args:
        model: TokenPredictorModel instance.
        data: DGData for inference.
        le: LabelEncoder for decoding token IDs.
        device: Device for inference.
        top_n: Number of top tokens to return.

    Returns:
        List of (token_address, bullish_score) tuples, sorted descending.

    """
    model.eval()
    dg = DGraph(data, device=device)
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    # Store probabilities instead of hard predictions for more robust confidence
    token_probabilities: dict[int, list[float]] = {}
    total_batches = sum(1 for _ in loader)

    with (
        torch.no_grad(),
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress,
    ):
            task = progress.add_task("Predicting token movements", total=total_batches)

            for batch_data in loader:
                if batch_data.src.shape[0] == 0 or batch_data.node_ids is None:
                    progress.update(task, advance=1)
                    continue

                move_batch_to_device(batch_data, device)

                logits = model(batch_data)
                if logits is None:
                    progress.update(task, advance=1)
                    continue

                # Convert logits to probabilities using softmax
                probs = torch.softmax(logits, dim=1)
                # Extract probability of "up" class (index 1 in binary classification)
                up_probs = probs[:, LABEL_UP].cpu().numpy()

                # Get node_ids for labeled nodes only
                has_label = batch_data.dynamic_node_feats[:, 3] != -999
                labeled_node_ids = batch_data.node_ids[has_label].cpu().numpy()

                for node_id, prob in zip(
                    labeled_node_ids,
                    up_probs,
                    strict=True,
                ):
                    if node_id not in token_probabilities:
                        token_probabilities[node_id] = []
                    token_probabilities[node_id].append(prob)

                progress.update(task, advance=1)

    recommendations: list[tuple[str, float]] = []
    for node_id, prob_list in token_probabilities.items():
        # Average probability of "up" across all predictions for this token
        bullish_score = np.mean(prob_list)
        token_addr = le.inverse_transform([node_id])[0]
        recommendations.append((token_addr, bullish_score))

    # Sort by bullish score descending
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Filter to only tokens predicted to go UP (probability > 0.5)
    # Then take top N among those with positive predictions
    bullish_recommendations = [
        (addr, score) for addr, score in recommendations if score > 0.5
    ]

    return bullish_recommendations[:top_n]


def calculate_daily_returns(
    df_trade: pl.DataFrame,
    top_tokens: list[tuple[str, float]],
) -> dict[str, float]:
    """Calculate daily returns for predicted tokens with stop loss and take profit.

    For each token predicted to go long, simulate intraday position:
    - Exit if price drops by STOP_LOSS_PCT
    - Exit if price gains by TAKE_PROFIT_PCT
    - Otherwise hold until 5pm

    Args:
        df_trade: DataFrame with swap events for the trade date.
        top_tokens: List of (token_address, bullish_score) tuples.

    Returns:
        Dict mapping token_address to daily_return (as decimal, e.g., 0.05 = 5%).

    """
    returns: dict[str, float] = {}

    for token_addr, _ in top_tokens:
        # Find all swaps where this token is src_token (we go long when swapping it out)
        token_events = df_trade.filter(
            pl.col("src_token_id") == token_addr,
        ).sort("bar_close_timestamp")

        if len(token_events) == 0:
            logger.debug("No events for token %s", token_addr[:10])
            returns[token_addr] = 0.0
            continue

        # Get price series during the day
        prices = token_events.select("src_price_usdc").to_numpy().flatten()
        prices = prices[~np.isnan(prices)]

        if len(prices) < 2:
            returns[token_addr] = 0.0
            continue

        start_price = prices[0]
        stop_loss_price = start_price * (1 - STOP_LOSS_PCT)
        take_profit_price = start_price * (1 + TAKE_PROFIT_PCT)

        # Simulate intraday position with stop loss and take profit
        position_closed = False
        final_return = 0.0

        for price in prices[1:]:
            # Check if stop loss hit
            if price <= stop_loss_price:
                final_return = -STOP_LOSS_PCT
                position_closed = True
                break

            # Check if take profit hit
            if price >= take_profit_price:
                final_return = TAKE_PROFIT_PCT
                position_closed = True
                break

        # If neither limit hit, use end-of-day price
        if not position_closed:
            end_price = prices[-1]
            if start_price > 0:
                final_return = (end_price - start_price) / start_price
            else:
                final_return = 0.0

        returns[token_addr] = final_return

    return returns


def compute_backtest_summary(results: list[dict]) -> None:
    """Compute and log comprehensive backtest performance summary.

    Extract portfolio returns from results and calculate key metrics: total return,
    Sharpe ratio, win rate, max drawdown, and best/worst trading days.
    """
    if not results:
        logger.warning("No results to summarize")
        return

    # Extract portfolio returns
    portfolio_returns = [r["portfolio_return"] for r in results]

    # Calculate statistics
    total_return = np.prod([1 + r for r in portfolio_returns]) - 1
    cumulative_return = np.sum(portfolio_returns)
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0

    # Win/loss statistics
    winning_days = sum(1 for r in portfolio_returns if r > 0)
    losing_days = sum(1 for r in portfolio_returns if r < 0)
    win_rate = winning_days / len(portfolio_returns) if portfolio_returns else 0.0

    # Max drawdown calculation
    # Convert returns to cumulative wealth (starting from 1.0)
    cumulative_wealth = np.cumprod([1 + r for r in portfolio_returns])
    running_max_wealth = np.maximum.accumulate(cumulative_wealth)

    # Drawdown is the percentage decline from running peak
    drawdown = (cumulative_wealth - running_max_wealth) / running_max_wealth
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Best/worst days
    best_day_idx = np.argmax(portfolio_returns) if portfolio_returns else 0
    worst_day_idx = np.argmin(portfolio_returns) if portfolio_returns else 0
    best_day_return = portfolio_returns[best_day_idx]
    worst_day_return = portfolio_returns[worst_day_idx]
    best_day_date = results[best_day_idx]["trade_date"]
    worst_day_date = results[worst_day_idx]["trade_date"]

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("  ℹ️ Total trading days: %d", len(results))
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("RETURNS SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)

    # Determine return performance status
    return_icon = "✅" if total_return > 0 else "❌" if total_return < 0 else "⚠️"
    logger.info("  %s Total Return: %+.2f%%", return_icon, total_return * 100)
    logger.info("  ℹ️ Cumulative Return: %+.2f%%", cumulative_return * 100)
    logger.info("  ℹ️ Mean Daily Return: %+.2f%%", mean_return * 100)
    logger.info("  ℹ️ Std Dev: %.2f%%", std_return * 100)

    # Sharpe ratio status
    sharpe_icon = "✅" if sharpe_ratio > 1.0 else "⚠️" if sharpe_ratio > 0 else "❌"
    logger.info("  %s Sharpe Ratio (annualized): %.2f", sharpe_icon, sharpe_ratio)

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("WIN/LOSS STATISTICS")
    logger.info("=" * DIVIDER_LENGTH)

    # Win rate status
    win_rate_icon = "✅" if win_rate > 0.5 else "⚠️" if win_rate > 0.4 else "❌"
    logger.info(
        "  %s Winning Days: %d / %d (%.1f%%)",
        win_rate_icon,
        winning_days,
        len(results),
        win_rate * 100,
    )
    logger.info("  ℹ️ Losing Days: %d", losing_days)
    logger.info(
        "  ℹ️ Break-even Days: %d",
        len(results) - winning_days - losing_days,
    )

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("DRAWDOWN ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)

    # Drawdown status
    dd_icon = "✅" if max_drawdown > -0.1 else "⚠️" if max_drawdown > -0.2 else "❌"
    logger.info("  %s Max Drawdown: %.2f%%", dd_icon, max_drawdown * 100)

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EXTREME DAYS")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("  ℹ️ Best Day: %s (%+.2f%%)", best_day_date, best_day_return * 100)
    logger.info("  ℹ️ Worst Day: %s (%+.2f%%)", worst_day_date, worst_day_return * 100)
    logger.info("=" * DIVIDER_LENGTH)

    # Overall assessment
    logger.info("")
    if total_return > 0 and sharpe_ratio > 1.0 and win_rate > 0.5:
        logger.info("✅ Strategy shows strong performance!")
    elif total_return > 0:
        logger.info("⚠️  Strategy is profitable but could be improved")
    else:
        logger.info("❌ Strategy underperformed - needs adjustment")
    logger.info("=" * DIVIDER_LENGTH)


def backtest_slide(
    df: pl.DataFrame,
    le: LabelEncoder,
    num_tokens: int,
    unique_dates: pl.DataFrame,
    tokens_metadata: dict[str, dict[str, Any]],
    epochs: int,
    trading_days: int | None = None,
) -> None:
    """Run sliding backtest: daily retrain and trade.

    Each day:
    1. Train on 5 days of data up to 9am on trade day
    2. Predict top 5 tokens at 9am
    3. Trade from 9am to 5pm
    4. Calculate returns
    5. Slide forward 1 day and retrain from scratch

    Args:
        df: DataFrame with swap bars and trade_date column.
        le: LabelEncoder for token IDs.
        num_tokens: Total number of unique tokens.
        unique_dates: DataFrame with unique trade dates.
        tokens_metadata: Dictionary mapping token address to metadata.
        epochs: Number of training epochs per day.
        trading_days: Number of days to trade (None = all available days).

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("SLIDING BACKTEST - DAILY RETRAIN")
    logger.info("=" * DIVIDER_LENGTH)

    # Extract unique dates for sliding windows
    dates_array = unique_dates["trade_date"].to_list()

    # Determine number of trading days
    max_trading_days = len(dates_array) - TRAIN_WINDOW_DAYS
    if trading_days is None:
        trading_days = max_trading_days
    else:
        trading_days = min(trading_days, max_trading_days)

    logger.info("Total available days: %d", len(dates_array))
    logger.info("Training window: %d days", TRAIN_WINDOW_DAYS)
    logger.info("Trading days: %d", trading_days)

    # Results storage
    results: list[dict[str, Any]] = []

    # Start from day 5 (need 5 days of training history)
    # Trade for specified number of days
    for day_idx in range(TRAIN_WINDOW_DAYS, TRAIN_WINDOW_DAYS + trading_days):
        trade_date = dates_array[day_idx]

        # Training window: 5 days before trade_date, up to 9am on trade_date
        train_start_date = dates_array[day_idx - TRAIN_WINDOW_DAYS]
        train_end_date = dates_array[day_idx - 1]  # Day before trade_date

        logger.info("")
        logger.info("=" * DIVIDER_LENGTH)
        logger.info("Trade Day: %s", trade_date)
        logger.info("=" * DIVIDER_LENGTH)
        logger.info(
            "Training on: %s to %s (full days)",
            train_start_date,
            train_end_date,
        )
        logger.info("Plus morning data on %s (up to 9am)", trade_date)

        # Initialize fresh model for this day
        model = TokenPredictorModel(
            num_nodes=num_tokens,
            node_feat_dim=NODE_INPUT_DIM,
            node_embed_dim=NODE_EMBED_DIM,
            output_dim=NUM_CLASSES,
            device=DEVICE,
        ).to(DEVICE)

        # Count events for logging
        df_train_history = df.filter(
            (pl.col("trade_date") >= train_start_date)
            & (pl.col("trade_date") <= train_end_date),
        )
        df_train_morning = df.filter(
            (pl.col("trade_date") == trade_date)
            & (pl.col("bar_close_timestamp").dt.hour() < TRADE_WINDOW_HOURS[0]),
        )

        total_train_events = len(df_train_history) + len(df_train_morning)
        logger.info(
            "Training events: %s (history) + %s (morning) = %s total",
            f"{len(df_train_history):,}",
            f"{len(df_train_morning):,}",
            f"{total_train_events:,}",
        )

        # For training, we'll use the combined window
        data_train = build_window(
            df,
            start_date=train_start_date,
            end_date=trade_date,
            end_hour=TRADE_WINDOW_HOURS[0],  # Up to 9am
        )

        # Train model from scratch
        train_model(
            model,
            data_train,
            DEVICE,
            epochs=epochs,
        )

        # Predict at 9am using all training data (including morning)
        top_tokens = predict_top_tokens(
            model,
            data_train,  # Use same data we trained on for prediction
            le,
            DEVICE,
            top_n=TOP_N_TOKENS,
        )

        logger.info(
            "Top tokens predicted to go UP (score > 50%%) at 9am: %d",
            len(top_tokens),
        )

        # Skip if no tokens meet the bullish threshold
        if len(top_tokens) == 0:
            logger.info("No tokens predicted to go up, skipping trading day")
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

        # Calculate returns during trading window (9am-5pm)
        df_trading = df.filter(
            (pl.col("trade_date") == trade_date)
            & (pl.col("bar_close_timestamp").dt.hour() >= TRADE_WINDOW_HOURS[0])
            & (pl.col("bar_close_timestamp").dt.hour() < TRADE_WINDOW_HOURS[1]),
        )

        logger.info("Trading window events (9am-5pm): %s", f"{len(df_trading):,}")

        if len(df_trading) == 0:
            logger.warning("No trading data for %s, skipping", trade_date)
            continue

        # Calculate token returns with stop loss and take profit
        token_returns = calculate_daily_returns(df_trading, top_tokens)

        for rank, (token_addr, score) in enumerate(top_tokens, 1):
            token_addr_lower = token_addr.lower()
            token_meta = tokens_metadata.get(token_addr_lower, {})
            symbol = token_meta.get("symbol", token_addr[:10])
            daily_return = token_returns.get(token_addr, 0.0)
            logger.info(
                "  %d. %s | Score: %.2f%% | Return: %+.2f%%",
                rank,
                symbol,
                score * 100,
                daily_return * 100,
            )

        # Calculate equal-weighted portfolio return
        portfolio_return = (
            sum(token_returns.values()) / len(token_returns)
            if len(token_returns) > 0
            else 0.0
        )
        logger.info("Portfolio return: %+.2f%%", portfolio_return * 100)

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
) -> None:
    """Run daily sliding backtest for TGCN token predictions.

    Each day:
    - Train on 5 days up to 9am
    - Predict top 5 tokens at 9am
    - Trade from 9am to 5pm
    - Slide forward 1 day and retrain from scratch
    """
    # Validate input data
    validate_input_data(data_path)

    # Load token metadata
    logger.info("Loading token metadata from %s", tokens_path)
    tokens_metadata = load_token_metadata(tokens_path)
    logger.info("Loaded metadata for %d tokens", len(tokens_metadata))

    # Load and prepare data
    df = load_and_filter_bars(data_path)
    df, le, num_tokens, unique_dates = prepare_data(df)

    # Run backtest
    backtest_slide(
        df,
        le,
        num_tokens,
        unique_dates,
        tokens_metadata,
        epochs,
        trading_days,
    )


if __name__ == "__main__":
    app()
