"""Sliding backtest for TGCN token price predictions on DEX swaps.

Architecture:
- Train on 5 days of events (up to 9am EST)
- Predict top-5 tokens to go long (1-8 hours ahead)
- Close positions at 5pm EST
- Report daily returns
- Slide forward 1 day and repeat
- Retraining decision based on ADF stationarity test on predictions

Nodes: Tokens
Edges: Swaps between tokens with temporal and flow information
Node labels: Triple-barrier method outputs (up/down/stay predictions)
Edge features: src_fracdiff, dest_fracdiff, flows, volatility, time
"""

# %%
import json
import logging
from pathlib import Path
from typing import Any

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
TRADING_DAYS = 3

# Model hyperparameters
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

EPOCHS = 10

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
# Load and prepare data
logger.info("Loading labeled bars from %s", DATA_PATH)
df = pl.read_parquet(DATA_PATH).sort("bar_close_timestamp")
logger.info("Bars loaded: %s", f"{df.shape[0]:,}")

# Filter out rows where dest_fracdiff is null
df = df.filter(pl.col("dest_fracdiff").is_not_null())
logger.info("After filtering null dest_fracdiff: %s", f"{df.shape[0]:,}")

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

# Create date index for windowing
df = df.with_columns(
    pl.col("bar_close_timestamp").dt.date().alias("trade_date"),
)
unique_dates = df.select("trade_date").unique().sort("trade_date")
logger.info(
    "Date range: %s to %s | Total days: %d",
    unique_dates["trade_date"][0],
    unique_dates["trade_date"][-1],
    len(unique_dates),
)


# %%
# Model architecture
class TokenPredictorModel(nn.Module):
    """TGCN encoder + decoder for token price movement prediction.

    Architecture:
    - Node embeddings: Learnable parameters initialized randomly, updated via backprop
    - TGCN encoder: Refines embeddings using temporal swaps (unsupervised)
    - Predictor: Maps refined embeddings to price predictions (supervised via labels)
    """

    def __init__(
        self,
        num_nodes: int,
        node_embed_dim: int,
        output_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialize token predictor model.

        Args:
            num_nodes: Total number of tokens in graph.
            node_embed_dim: Embedding dimension (used for node embeddings and TGCN).
            output_dim: Number of output classes (3: down/stay/up).
            device: Device to place tensors on.

        """
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # Learnable node embeddings - initialized randomly, updated via backprop
        # Refined by TGCN via message passing on the swap graph
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
            batch: DGBatch with src, dst, edge_index, node_ids.

        Returns:
            logits: [num_nodes_in_batch, output_dim] or None if no nodes.

        """
        # TGCN takes self.node_embeddings as INPUT
        # and outputs UPDATED embeddings based on graph structure + messages
        z = self.tgcn(self.node_embeddings, batch.edge_index)

        if batch.node_ids is not None:
            # Select embeddings for nodes that have labels
            z_node = z[batch.node_ids]
            # Predictor: predict price movement from refined embeddings
            result: torch.Tensor = self.predictor(z_node)
            return result
        return None


# %%
def build_window(
    df_window: pl.DataFrame,
    le: LabelEncoder,
) -> DGData:
    """Build a window of swap bars.

    Args:
        df_window: DataFrame with swap bars for training window.
        le: LabelEncoder for token IDs.

    Returns:
        DGData object with labels and sample_weights packed in dynamic_node_feats.

    """
    src = np.asarray(le.transform(df_window["src_token_id"].to_numpy()), dtype=np.int32)
    dst = np.asarray(
        le.transform(df_window["dest_token_id"].to_numpy()),
        dtype=np.int32,
    )

    # Use event index as timestamp to respect dollar bar "information time"
    timestamps_sec = np.arange(len(df_window), dtype=np.int32)

    edge_feats = (
        df_window.select(
            [
                "src_fracdiff",
                "dest_fracdiff",
                "src_flow_usdc",
                "dest_flow_usdc",
                "tick_count",
                "rolling_volatility",
                "bar_time_delta_sec",
            ],
        )
        .to_numpy()
        .astype(np.float32)
    )
    assert not np.isnan(edge_feats).any(), "NaN values found in edge features"

    node_timestamps = timestamps_sec
    node_ids = src

    # Pack labels and weights into dynamic_node_feats
    labels = (df_window["label"].fill_null(0).to_numpy().astype(np.int64) + 1).astype(
        np.float32,
    )
    weights = df_window["sample_weight"].to_numpy().astype(np.float32)
    dynamic_node_feats = np.stack((labels, weights), axis=-1)

    return DGData.from_raw(
        edge_timestamps=torch.from_numpy(timestamps_sec),
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

        for batch in tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False):
            if batch.src.shape[0] == 0 or batch.node_ids is None:
                continue

            batch.src = batch.src.to(device)
            batch.dst = batch.dst.to(device)
            batch.time = batch.time.to(device)
            edge_index = torch.stack([batch.src, batch.dst], dim=0)
            batch.edge_index = edge_index

            batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
            batch.node_ids = batch.node_ids.to(device)

            optimizer.zero_grad()
            logits = model(batch)

            if logits is None:
                continue

            # Unpack labels and weights from dynamic_node_feats
            y_true = batch.dynamic_node_feats[:, 0].long()
            batch_weights = batch.dynamic_node_feats[:, 1].float()

            loss_per_node = criterion(logits, y_true)
            weighted_loss = (loss_per_node * batch_weights).mean()

            weighted_loss.backward()
            optimizer.step()

            total_loss += weighted_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)


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

            batch.dynamic_node_feats = batch.dynamic_node_feats.to(device)
            batch.node_ids = batch.node_ids.to(device)

            logits = model(batch)  # <-- No longer pass node_embeddings
            if logits is None:
                continue

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            node_ids = batch.node_ids.cpu().numpy()

            for node_id, pred in zip(node_ids, preds):
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
    le: LabelEncoder,
) -> dict[str, float]:
    """Calculate daily returns for predicted tokens.

    For each token predicted to go long, compute return as:
    (median_price_end - median_price_start) / median_price_start

    Args:
        df_trade: DataFrame with swap events for the trade date.
        top_tokens: List of (token_address, bullish_score) tuples.
        le: LabelEncoder for token ID conversion.

    Returns:
        Dict mapping token_address to daily_return (as decimal, e.g., 0.05 = 5%).

    """
    returns: dict[str, float] = {}

    for token_addr, _ in top_tokens:
        # Get encoded token ID
        try:
            token_id_encoded = le.transform([token_addr])[0]
        except ValueError:
            logger.warning("Token %s not in encoder", token_addr[:10])
            returns[token_addr] = 0.0
            continue

        # Find all swaps where this token is src_token (we go long when swapping it out)
        token_events = df_trade.filter(
            pl.col("src_token_id") == token_addr,
        )

        if len(token_events) == 0:
            logger.debug("No events for token %s", token_addr[:10])
            returns[token_addr] = 0.0
            continue

        # Get first and last prices during the day
        prices = token_events.select("src_price_usdc").to_numpy().flatten()
        prices = prices[~np.isnan(prices)]

        if len(prices) < 2:
            returns[token_addr] = 0.0
            continue

        start_price = prices[0]
        end_price = prices[-1]

        if start_price > 0:
            daily_return = (end_price - start_price) / start_price
            returns[token_addr] = daily_return
        else:
            returns[token_addr] = 0.0

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

    # Max drawdown (simplified)
    cumsum = np.cumsum(portfolio_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - running_max) / (running_max + 1e-8)
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Best/worst days
    best_day_idx = np.argmax(portfolio_returns) if portfolio_returns else 0
    worst_day_idx = np.argmin(portfolio_returns) if portfolio_returns else 0
    best_day_return = portfolio_returns[best_day_idx]
    worst_day_return = portfolio_returns[worst_day_idx]
    best_day_date = results[best_day_idx]["trade_date"]
    worst_day_date = results[worst_day_idx]["trade_date"]

    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 80)
    logger.info("Total trading days: %d", len(results))
    logger.info("=" * 80)
    logger.info("RETURNS SUMMARY")
    logger.info("=" * 80)
    logger.info("Total Return: %+.2f%%", total_return * 100)
    logger.info("Cumulative Return: %+.2f%%", cumulative_return * 100)
    logger.info("Mean Daily Return: %+.2f%%", mean_return * 100)
    logger.info("Std Dev: %.2f%%", std_return * 100)
    logger.info("Sharpe Ratio (annualized): %.2f", sharpe_ratio)
    logger.info("=" * 80)
    logger.info("WIN/LOSS STATISTICS")
    logger.info("=" * 80)
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
    logger.info("=" * 80)
    logger.info("DRAWDOWN ANALYSIS")
    logger.info("=" * 80)
    logger.info("Max Drawdown: %.2f%%", max_drawdown * 100)
    logger.info("=" * 80)
    logger.info("EXTREME DAYS")
    logger.info("=" * 80)
    logger.info("Best Day: %s (%+.2f%%)", best_day_date, best_day_return * 100)
    logger.info("Worst Day: %s (%+.2f%%)", worst_day_date, worst_day_return * 100)
    logger.info("=" * 80)


def backtest_slide() -> None:
    """Run sliding backtest: train 5 days, trade 5 days, slide 1 day."""
    logger.info("=" * 80)
    logger.info("SLIDING BACKTEST")
    logger.info("=" * 80)

    # Initialize model (node embeddings are now part of the model)
    model = TokenPredictorModel(
        num_nodes=num_tokens,
        node_embed_dim=NODE_EMBED_DIM,  # <-- Simplified: no separate static_feat_dim
        output_dim=NUM_CLASSES,
        device=DEVICE,
    ).to(DEVICE)

    # Extract unique dates for sliding windows
    dates_array = unique_dates["trade_date"].to_list()

    # Sliding window: train on 5 days, then trade for next TRADING_DAYS days
    results: list[dict[str, Any]] = []

    for i in range(1):
        train_start_date = dates_array[i]
        train_end_date = dates_array[i + TRAIN_WINDOW_DAYS - 1]

        logger.info("Training window: %s to %s", train_start_date, train_end_date)

        df_train = df.filter(
            (pl.col("trade_date") >= train_start_date)
            & (pl.col("trade_date") <= train_end_date),
        )
        logger.info("Training events: %s", f"{len(df_train):,}")

        data_train = build_window(df_train, le)

        # Train model - node embeddings updated via backprop
        train_model(
            model,
            data_train,
            DEVICE,
            epochs=EPOCHS,
        )

        # Trade for next TRADING_DAYS days
        for trade_offset in range(TRADING_DAYS):
            if i + TRAIN_WINDOW_DAYS + trade_offset >= len(dates_array):
                break

            trade_date = dates_array[i + TRAIN_WINDOW_DAYS + trade_offset]
            df_day = df.filter(pl.col("trade_date") == trade_date)

            if len(df_day) == 0:
                logger.warning("No data for trade date %s", trade_date)
                continue

            # Split into prediction context (pre-9am) and trading window (9am-5pm)
            trade_start_hour = TRADE_WINDOW_HOURS[0]
            trade_end_hour = TRADE_WINDOW_HOURS[1]

            df_morning = df_day.filter(
                pl.col("bar_close_timestamp").dt.hour() < trade_start_hour,
            )
            df_trading = df_day.filter(
                (pl.col("bar_close_timestamp").dt.hour() >= trade_start_hour)
                & (pl.col("bar_close_timestamp").dt.hour() < trade_end_hour),
            )

            logger.info(
                "\nTrade date: %s | Morning events: %s | Trading events: %s",
                trade_date,
                f"{len(df_morning):,}",
                f"{len(df_trading):,}",
            )

            if len(df_morning) == 0:
                logger.warning("No morning data for %s, skipping", trade_date)
                continue

            data_pred = build_window(df_morning, le)
            top_tokens = predict_top_tokens(
                model,
                data_pred,
                le,
                DEVICE,
                top_n=TOP_N_TOKENS,
            )

            logger.info(
                "Top 5 tokens for %s (based on pre-%d:00 data):",
                trade_date,
                trade_start_hour,
            )

            token_returns = calculate_daily_returns(df_trading, top_tokens, le)

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

            portfolio_return = np.mean(list(token_returns.values()))

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


def run_training_validation() -> None:
    """Run a simple training loop for validation."""
    logger.info("=" * 80)
    logger.info("TRAINING VALIDATION RUN")
    logger.info("=" * 80)

    # Initialize model
    model = TokenPredictorModel(
        num_nodes=num_tokens,
        node_embed_dim=NODE_EMBED_DIM,
        output_dim=NUM_CLASSES,
        device=DEVICE,
    ).to(DEVICE)

    # Use first window
    train_start_date = unique_dates["trade_date"][0]
    train_end_date = unique_dates["trade_date"][TRAIN_WINDOW_DAYS - 1]

    logger.info("Training window: %s to %s", train_start_date, train_end_date)

    df_train = df.filter(
        (pl.col("trade_date") >= train_start_date)
        & (pl.col("trade_date") <= train_end_date),
    )
    logger.info("Training events: %s", f"{len(df_train):,}")

    data_train = build_window(df_train, le)

    # Train model for 1 epoch
    train_model(
        model,
        data_train,
        DEVICE,
        epochs=1,
    )

    logger.info("Training validation complete.")


if __name__ == "__main__":
    # Default to validation run
    run_training_validation()
    # To run full backtest:
    # backtest_slide()
