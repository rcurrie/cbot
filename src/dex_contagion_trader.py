"""Daily sliding backtest for TGCN token price predictions on DEX swaps.

Daily Trading Cycle:
1. Train on 5 days of data up to 9am on trade day
2. Predict top 5 tokens to go long at 9am
3. Trade from 9am to 5pm
4. Calculate returns and close positions
5. Slide forward 1 day and retrain from scratch

TGCN Architecture:
- Nodes: Tokens
- Edges: Swaps between tokens with temporal and flow information
- Node labels: Triple-barrier method outputs (up/down/stay predictions)
- Node features (dynamic): fracdiff, volatility, flow (evolve with each swap)
- Edge features: flow magnitude (importance weighting)
"""

# %%
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from tgm import DGBatch, DGraph  # type: ignore[import-untyped]
from tgm.data import DGData, DGDataLoader  # type: ignore[import-untyped]
from tgm.nn import TGCN, NodePredictor  # type: ignore[import-untyped]
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

# Number of '==' characters for logging
DIVIDER_LENGTH = 40

# Storage locations
DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")
TOKENS_PATH = Path("data/tokens.json")

CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
NUM_CLASSES = 3  # down (0), stay (1), up (2)
LABEL_UP = 2
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

# %%
# Load token metadata from tokens.json
with TOKENS_PATH.open() as f:
    tokens_metadata: dict[str, dict[str, Any]] = json.load(f)


# %%
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
    edge_cols = [
        "src_fracdiff",
        "dest_fracdiff",
        "src_flow_usdc",
        "dest_flow_usdc",
        "tick_count",
        "rolling_volatility",
        "bar_time_delta_sec",
        "label",
        "sample_weight",
    ]
    for col in edge_cols:
        df = df.filter(pl.col(col).is_not_null())
        # Also filter NaNs for float columns
        if df.schema[col] in (pl.Float32, pl.Float64):
            df = df.filter(pl.col(col).is_finite())
    logger.info("After filtering null edge features: %s", f"{df.shape[0]:,}")

    return df


# %%
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


# %%
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
        # Extract node features from dynamic_node_feats
        # dynamic_node_feats[:, 0:3] = [fracdiff, volatility, flow]
        # dynamic_node_feats[:, 3:5] = [label, weight] (used for training)
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
            batch.edge_feats.squeeze(-1)
            if batch.edge_feats is not None
            else None
        )
        z = self.tgcn(x_combined, batch.edge_index, edge_weight)

        # For prediction, we only use nodes that have labels
        # Labels are in dynamic_node_feats[:, 3]
        # Filter for nodes with non-zero labels (src tokens)
        has_label = batch.dynamic_node_feats[:, 3] > 0
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

    # For each window, create local timestamps starting from 0
    # This preserves relative timing within the window while making each window
    # independent. Note: event_index maintains global ordering, but we renumber
    # for local window time
    edge_timestamps = np.arange(num_events, dtype=np.int64)

    # Extract features for each swap event
    src_fracdiff = df_window["src_fracdiff"].to_numpy()
    dest_fracdiff = df_window["dest_fracdiff"].to_numpy()
    src_flow = df_window["src_flow_usdc"].to_numpy()
    dest_flow = df_window["dest_flow_usdc"].to_numpy()
    volatility = df_window["rolling_volatility"].to_numpy()
    labels = df_window["label"].fill_null(0).to_numpy().astype(np.int64) + 1
    weights = df_window["sample_weight"].to_numpy()

    # Each swap event updates TWO nodes: src and dest
    # We create 2 node updates per edge event
    node_timestamps = np.repeat(edge_timestamps, 2)  # [0,0, 1,1, 2,2, ...]
    node_ids = np.empty(num_events * 2, dtype=np.int32)
    node_ids[0::2] = src  # Even indices: src tokens
    node_ids[1::2] = dst  # Odd indices: dest tokens

    # Dynamic node features: [fracdiff, volatility, flow, label, weight]
    # Shape: [num_events * 2, 5]
    dynamic_node_feats = np.zeros((num_events * 2, 5), dtype=np.float32)

    # Src node updates (even indices)
    dynamic_node_feats[0::2, 0] = src_fracdiff   # fracdiff
    dynamic_node_feats[0::2, 1] = volatility      # volatility
    dynamic_node_feats[0::2, 2] = src_flow        # flow
    dynamic_node_feats[0::2, 3] = labels          # label (only src has labels)
    dynamic_node_feats[0::2, 4] = weights         # weight

    # Dest node updates (odd indices)
    dynamic_node_feats[1::2, 0] = dest_fracdiff   # fracdiff
    dynamic_node_feats[1::2, 1] = volatility      # volatility
    dynamic_node_feats[1::2, 2] = dest_flow       # flow
    dynamic_node_feats[1::2, 3] = 0               # no label for dest (yet)
    dynamic_node_feats[1::2, 4] = 0               # no weight for dest (yet)

    # Handle NaN values in features (set to 0)
    dynamic_node_feats = np.nan_to_num(dynamic_node_feats, nan=0.0)

    # Edge features: use absolute flow magnitude (importance weighting)
    # Shape: [num_events, 1] for single feature
    edge_feats = np.abs(src_flow).astype(np.float32).reshape(-1, 1)

    return DGData.from_raw(
        edge_timestamps=torch.from_numpy(edge_timestamps),
        edge_index=torch.from_numpy(np.column_stack((src, dst))),
        edge_feats=torch.from_numpy(edge_feats),
        node_timestamps=torch.from_numpy(node_timestamps),
        node_ids=torch.from_numpy(node_ids),
        dynamic_node_feats=torch.from_numpy(dynamic_node_feats),
        time_delta="s",
    )


# %%
def train_model(
    model: TokenPredictorModel,
    data: DGData,
    device: str,
    epochs: int = 1,
) -> None:
    """Train model with Prado concurrency weighting.

    Args:
        model: TokenPredictorModel instance.
        data: DGData with training events.
        device: Device for training.
        epochs: Number of epochs (default 1 for sliding backtest).

    """
    model.train()
    dg = DGraph(data, device=device)
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            if batch.src.shape[0] == 0 or batch.node_ids is None:
                continue

            batch.src = batch.src.to(device)
            batch.dst = batch.dst.to(device)
            batch.time = batch.time.to(device)
            edge_index = torch.stack([batch.src, batch.dst], dim=0)
            batch.edge_index = edge_index

            # Move features to device
            batch.edge_feats = batch.edge_feats.to(device)
            batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
            batch.node_ids = batch.node_ids.to(device)

            optimizer.zero_grad()
            logits = model(batch)

            if logits is None:
                continue

            # Extract labels and weights from dynamic_node_feats
            # Only use nodes with labels (src tokens, where label > 0)
            has_label = batch.dynamic_node_feats[:, 3] > 0
            y_true = batch.dynamic_node_feats[has_label, 3].long()
            batch_weights = batch.dynamic_node_feats[has_label, 4].float()

            loss_per_node = criterion(logits, y_true)
            weighted_loss = (loss_per_node * batch_weights).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            num_batches += 1

            # Update progress bar with current average loss
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})


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

    token_predictions: dict[int, list[int]] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            if batch.src.shape[0] == 0 or batch.node_ids is None:
                continue

            batch.src = batch.src.to(device)
            batch.dst = batch.dst.to(device)
            batch.time = batch.time.to(device)
            edge_index = torch.stack([batch.src, batch.dst], dim=0)
            batch.edge_index = edge_index

            # Move features to device
            batch.edge_feats = batch.edge_feats.to(device)
            batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
            batch.node_ids = batch.node_ids.to(device)

            logits = model(batch)
            if logits is None:
                continue

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Get node_ids for labeled nodes only (src tokens)
            has_label = batch.dynamic_node_feats[:, 3] > 0
            labeled_node_ids = batch.node_ids[has_label].cpu().numpy()

            for node_id, pred in zip(labeled_node_ids, preds):
                if node_id not in token_predictions:
                    token_predictions[node_id] = []
                token_predictions[node_id].append(pred)

    recommendations: list[tuple[str, float]] = []
    for node_id, pred_list in token_predictions.items():
        up_count = sum(1 for p in pred_list if p == LABEL_UP)
        bullish_score = up_count / len(pred_list)
        token_addr = le.inverse_transform([node_id])[0]
        recommendations.append((token_addr, bullish_score))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


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


def should_retrain(
    predictions: np.ndarray,
    adf_threshold: float = 0.05,
) -> bool:
    """Check if predictions are stationary using ADF test.

    Args:
        predictions: Array of model predictions.
        adf_threshold: P-value threshold for ADF test (default: 0.05).

    Returns:
        False (never retrain by default as per requirements).

    """
    # Stub: Always return False (no retraining) per requirements
    # In future: compute ADF on predictions, retrain if p > threshold
    return False


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

    logger.info("BACKTEST SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("Total trading days: %d", len(results))
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("RETURNS SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("Total Return: %+.2f%%", total_return * 100)
    logger.info("Cumulative Return: %+.2f%%", cumulative_return * 100)
    logger.info("Mean Daily Return: %+.2f%%", mean_return * 100)
    logger.info("Std Dev: %.2f%%", std_return * 100)
    logger.info("Sharpe Ratio (annualized): %.2f", sharpe_ratio)
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("WIN/LOSS STATISTICS")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info(
        "Winning Days: %d / %d (%.1f%%)",
        winning_days,
        len(results),
        win_rate * 100,
    )
    logger.info("Losing Days: %d", losing_days)
    logger.info(
        "Break-even Days: %d",
        len(results) - winning_days - losing_days,
    )
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("DRAWDOWN ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("Max Drawdown: %.2f%%", max_drawdown * 100)
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EXTREME DAYS")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("Best Day: %s (%+.2f%%)", best_day_date, best_day_return * 100)
    logger.info("Worst Day: %s (%+.2f%%)", worst_day_date, worst_day_return * 100)
    logger.info("=" * DIVIDER_LENGTH)


def backtest_slide(
    df: pl.DataFrame,
    le: LabelEncoder,
    num_tokens: int,
    unique_dates: pl.DataFrame,
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

        logger.info("Top 5 tokens predicted at 9am:")

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
        portfolio_return = sum(token_returns.values()) / len(token_returns)
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


@click.command()
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Number of training epochs per day (default: 10)",
)
@click.option(
    "--trading-days",
    default=None,
    type=int,
    help="Number of days to trade (default: all available days)",
)
def main(epochs: int, trading_days: int | None) -> None:
    """Run daily sliding backtest for TGCN token predictions.

    Each day:
    - Train on 5 days up to 9am
    - Predict top 5 tokens at 9am
    - Trade from 9am to 5pm
    - Slide forward 1 day and retrain
    """
    df = load_and_filter_bars(DATA_PATH)
    df, le, num_tokens, unique_dates = prepare_data(df)

    # Run backtest
    backtest_slide(df, le, num_tokens, unique_dates, epochs, trading_days)


if __name__ == "__main__":
    main()
