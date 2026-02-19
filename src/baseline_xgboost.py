"""XGBoost baseline for DEX token price prediction.

WHY: Establish whether tabular features alone can predict token price
movements, without graph structure. If XGBoost matches or exceeds the
GNN (ldr_tgn_trader.py), the graph adds no value and simpler models
should be preferred.

WHAT: Walk-forward backtest using the same protocol as ldr_tgn_trader.py:
1. Train on N days of labeled dollar bars
2. Predict on the next day's morning data
3. Trade top-N tokens with strongest signals
4. Calculate portfolio returns with stop-loss/take-profit

HOW: XGBoost multi-class classifier on per-event tabular features:
- src_fracdiff, dest_fracdiff (trend signal)
- rolling_volatility (risk)
- src_flow_usdc, dest_flow_usdc (momentum)
- bar_time_delta_sec (liquidity proxy)
- tick_count, tick_delta (microstructure)
- liquidity_close (depth)
- Prado sample weights applied natively via XGBoost sample_weight

INPUT: data/labeled_log_fracdiff_price.parquet
       data/tokens.json
OUTPUT: Console backtest report (same format as ldr_tgn_trader.py)

References:
- Prado AFML Ch. 3-5: Triple-Barrier, Sample Weights
- Chen & Guestrin "XGBoost" (KDD 2016)

"""

# ruff: noqa: PLR0915, N806

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
import xgboost as xgb

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DIVIDER_LENGTH = 70

app = typer.Typer(
    help="XGBoost baseline for DEX token price prediction.",
)

# Backtest parameters (match ldr_tgn_trader.py)
TRAIN_WINDOW_DAYS = 30
TRADE_WINDOW_HOURS = (9, 17)
TOP_N_TOKENS = 5
NUM_CLASSES = 3  # down (0), neutral (1), up (2)

# Risk management (match ldr_tgn_trader.py)
STOP_LOSS_PCT = 0.20
TAKE_PROFIT_PCT = 2.0

# Feature columns for XGBoost
FEATURE_COLS = [
    "src_fracdiff",
    "dest_fracdiff",
    "rolling_volatility",
    "log_src_flow",
    "log_dest_flow",
    "buying_pressure",
    "bar_time_delta_sec",
    "log_tick_count",
    "src_tick_delta",
    "dest_tick_delta",
    "log_src_liquidity",
    "log_dest_liquidity",
]


def load_token_metadata(tokens_path: Path) -> dict[str, dict[str, Any]]:
    """Load token metadata from tokens.json."""
    with tokens_path.open() as f:
        metadata: dict[str, dict[str, Any]] = json.load(f)
    return metadata


def load_and_prepare(
    data_path: Path,
) -> pl.DataFrame:
    """Load parquet and engineer features for XGBoost."""
    logger.info("Loading labeled bars from %s", data_path)
    df = pl.read_parquet(data_path)
    logger.info("Bars loaded: %s", f"{df.shape[0]:,}")

    # Sort by time
    df = df.sort(["bar_close_timestamp", "src_token_id", "dest_token_id"])

    # Filter null labels
    df = df.filter(
        pl.col("label").is_not_null() & pl.col("label").is_finite(),
    )
    logger.info("After filtering null labels: %s", f"{df.shape[0]:,}")

    # Add trade date
    df = df.with_columns(
        pl.col("bar_close_timestamp").dt.date().alias("trade_date"),
    )

    # Map labels: {-1, 0, 1} -> {0, 1, 2}
    df = df.with_columns(
        (pl.col("label").cast(pl.Int32) + 1).alias("label_encoded"),
    )

    # Engineer features (log-transform skewed features)
    df = df.with_columns(
        [
            pl.col("src_flow_usdc")
            .abs()
            .log1p()
            .fill_null(0)
            .alias("log_src_flow"),
            pl.col("dest_flow_usdc")
            .abs()
            .log1p()
            .fill_null(0)
            .alias("log_dest_flow"),
            (
                pl.col("src_flow_usdc").abs()
                / (
                    pl.col("src_flow_usdc").abs()
                    + pl.col("dest_flow_usdc").abs()
                    + 1e-8
                )
            )
            .fill_null(0.5)
            .alias("buying_pressure"),
            pl.col("tick_count")
            .fill_null(1)
            .cast(pl.Float64)
            .log1p()
            .alias("log_tick_count"),
            pl.col("src_liquidity_close")
            .fill_null(0)
            .cast(pl.Float64)
            .log1p()
            .alias("log_src_liquidity"),
            pl.col("dest_liquidity_close")
            .fill_null(0)
            .cast(pl.Float64)
            .log1p()
            .alias("log_dest_liquidity"),
            pl.col("src_fracdiff").fill_null(0),
            pl.col("dest_fracdiff").fill_null(0),
            pl.col("rolling_volatility").fill_null(0),
            pl.col("bar_time_delta_sec").fill_null(0),
            pl.col("src_tick_delta").fill_null(0),
            pl.col("dest_tick_delta").fill_null(0),
        ],
    )

    # Log feature stats
    for col in FEATURE_COLS:
        vals = df[col].to_numpy()
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            logger.info(
                "  Feature %-20s: min=%.3f, max=%.3f,"
                " mean=%.3f, std=%.3f",
                col,
                np.min(finite),
                np.max(finite),
                np.mean(finite),
                np.std(finite),
            )

    unique_dates = df.select("trade_date").unique().sort("trade_date")
    logger.info(
        "Date range: %s to %s | Total days: %d",
        unique_dates["trade_date"][0],
        unique_dates["trade_date"][-1],
        len(unique_dates),
    )

    return df


def train_xgb(
    df_train: pl.DataFrame,
) -> xgb.Booster:
    """Train XGBoost on labeled bar features with Prado sample weights."""
    X = df_train.select(FEATURE_COLS).to_numpy()
    y = df_train["label_encoded"].to_numpy()
    w = df_train["sample_weight"].to_numpy()

    # Replace any NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Class weights to handle imbalance (inverse frequency)
    class_counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1)
    total = class_counts.sum()
    # Scale sample weights by inverse class frequency
    class_weight = total / (NUM_CLASSES * class_counts)
    scaled_weights = w * class_weight[y]

    dtrain = xgb.DMatrix(X, label=y, weight=scaled_weights)

    params = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }

    return xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )


def predict_signals(
    model: xgb.Booster,
    df_pred: pl.DataFrame,
    top_n: int = TOP_N_TOKENS,
) -> list[tuple[str, float, int]]:
    """Predict token signals using XGBoost probabilities.

    Signal = P(UP) - P(DOWN) per token, averaged over events.
    """
    if len(df_pred) == 0:
        return []

    X = df_pred.select(FEATURE_COLS).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    dmat = xgb.DMatrix(X)
    probs = model.predict(dmat)  # [n_events, 3]

    # Aggregate per src_token
    token_signals: dict[str, list[float]] = {}
    src_tokens = df_pred["src_token_id"].to_list()

    for i, token_addr in enumerate(src_tokens):
        p_down, _p_neutral, p_up = probs[i]
        signal = float(p_up - p_down)
        if token_addr not in token_signals:
            token_signals[token_addr] = []
        token_signals[token_addr].append(signal)

    # Average signal per token
    signals = []
    for token_addr, sigs in token_signals.items():
        avg_signal = np.mean(sigs)
        pred_class = 2 if avg_signal > 0 else 0  # UP or DOWN
        signals.append((token_addr, float(avg_signal), pred_class))

    # Sort by signal descending, filter positive
    signals.sort(key=lambda x: x[1], reverse=True)
    long_signals = [s for s in signals if s[1] > 0]

    return long_signals[:top_n]


def calculate_daily_returns(
    df_trade: pl.DataFrame,
    top_tokens: list[tuple[str, float, int]],
) -> dict[str, tuple[float, float]]:
    """Calculate daily returns for selected tokens (same as ldr_tgn_trader)."""
    returns: dict[str, tuple[float, float]] = {}

    for token_addr, signal_strength, _ in top_tokens:
        position_size = min(abs(signal_strength), 0.5)

        token_events = df_trade.filter(
            pl.col("src_token_id") == token_addr,
        ).sort("bar_close_timestamp")

        if len(token_events) == 0:
            returns[token_addr] = (0.0, position_size)
            continue

        prices = (
            token_events.select("src_price_usdc").to_numpy().flatten()
        )
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


def compute_backtest_summary(
    results: list[dict[str, Any]],
) -> None:
    """Compute and log backtest summary (same format as ldr_tgn_trader)."""
    if not results:
        logger.warning("No results to summarize")
        return

    portfolio_returns = [r["portfolio_return"] for r in results]

    total_return = float(np.prod([1 + r for r in portfolio_returns]) - 1)
    cumulative_return = float(np.sum(portfolio_returns))
    mean_return = float(np.mean(portfolio_returns))
    std_return = float(np.std(portfolio_returns))
    sharpe = (
        mean_return / std_return * math.sqrt(252)
        if std_return > 0
        else 0.0
    )

    winning_days = sum(1 for r in portfolio_returns if r > 0)
    losing_days = sum(1 for r in portfolio_returns if r < 0)
    win_rate = (
        winning_days / len(portfolio_returns)
        if portfolio_returns
        else 0.0
    )

    cumulative_wealth = np.cumprod([1 + r for r in portfolio_returns])
    running_max = np.maximum.accumulate(cumulative_wealth)
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    best_idx = int(np.argmax(portfolio_returns))
    worst_idx = int(np.argmin(portfolio_returns))

    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("BACKTEST SUMMARY (XGBoost Baseline)")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("  Total trading days: %d", len(results))
    logger.info("")

    ret_icon = (
        "✅" if total_return > 0 else "⚠️" if total_return == 0 else "❌"
    )
    logger.info(
        "  %s Total Return: %+.2f%%", ret_icon, total_return * 100,
    )
    logger.info(
        "  Cumulative Return: %+.2f%%", cumulative_return * 100,
    )
    logger.info("  Mean Daily Return: %+.2f%%", mean_return * 100)
    logger.info("  Std Dev: %.2f%%", std_return * 100)

    sharpe_icon = (
        "✅" if sharpe > 1.0 else "⚠️" if sharpe > 0 else "❌"
    )
    logger.info(
        "  %s Sharpe Ratio (annualized): %.2f", sharpe_icon, sharpe,
    )

    logger.info("")
    win_icon = (
        "✅" if win_rate > 0.5 else "⚠️" if win_rate > 0.4 else "❌"
    )
    logger.info(
        "  %s Win Rate: %d / %d (%.1f%%)",
        win_icon,
        winning_days,
        len(results),
        win_rate * 100,
    )
    logger.info("  Losing Days: %d", losing_days)

    dd_icon = (
        "✅"
        if max_drawdown > -0.1
        else "⚠️"
        if max_drawdown > -0.2
        else "❌"
    )
    logger.info("  %s Max Drawdown: %.2f%%", dd_icon, max_drawdown * 100)

    logger.info("")
    logger.info(
        "  Best Day: %s (%+.2f%%)",
        results[best_idx]["trade_date"],
        portfolio_returns[best_idx] * 100,
    )
    logger.info(
        "  Worst Day: %s (%+.2f%%)",
        results[worst_idx]["trade_date"],
        portfolio_returns[worst_idx] * 100,
    )
    logger.info("=" * DIVIDER_LENGTH)


def backtest_slide(
    df: pl.DataFrame,
    tokens_metadata: dict[str, dict[str, Any]],
    trading_days: int | None = None,
    train_window_days: int = TRAIN_WINDOW_DAYS,
) -> None:
    """Run sliding backtest with walk-forward validation."""
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("XGBOOST BASELINE - WALK-FORWARD BACKTEST")
    logger.info("=" * DIVIDER_LENGTH)

    dates_array = (
        df.select("trade_date")
        .unique()
        .sort("trade_date")["trade_date"]
        .to_list()
    )

    max_trading_days = len(dates_array) - train_window_days
    if trading_days is None:
        trading_days = max_trading_days
    else:
        trading_days = min(trading_days, max_trading_days)

    logger.info("Total available days: %d", len(dates_array))
    logger.info("Training window: %d days", train_window_days)
    logger.info("Trading days: %d", trading_days)

    results: list[dict[str, Any]] = []

    for day_idx in range(
        train_window_days, train_window_days + trading_days,
    ):
        trade_date = dates_array[day_idx]
        train_start = dates_array[day_idx - train_window_days]
        train_end = dates_array[day_idx - 1]

        logger.info("")
        logger.info("=" * DIVIDER_LENGTH)
        logger.info("Trade Day: %s", trade_date)
        logger.info("=" * DIVIDER_LENGTH)
        logger.info("Training on: %s to %s", train_start, train_end)

        # Training data
        df_train = df.filter(
            (pl.col("trade_date") >= train_start)
            & (pl.col("trade_date") <= train_end),
        )
        logger.info("Training events: %s", f"{len(df_train):,}")

        if len(df_train) < 100:
            logger.warning("Insufficient training data, skipping")
            continue

        # Train XGBoost
        model = train_xgb(df_train)

        # Morning data for prediction
        df_morning = df.filter(
            (pl.col("trade_date") == trade_date)
            & (
                pl.col("bar_close_timestamp").dt.hour()
                < TRADE_WINDOW_HOURS[0]
            ),
        )

        # Predict signals
        pred_data = df_morning if len(df_morning) > 0 else df_train
        top_tokens = predict_signals(model, pred_data)

        logger.info(
            "Top tokens with positive signals: %d", len(top_tokens),
        )

        if len(top_tokens) == 0:
            logger.info("No positive signals, skipping")
            results.append(
                {
                    "trade_date": trade_date,
                    "top_tokens": [],
                    "token_returns": {},
                    "portfolio_return": 0.0,
                },
            )
            continue

        # Trading window data
        df_trading = df.filter(
            (pl.col("trade_date") == trade_date)
            & (
                pl.col("bar_close_timestamp").dt.hour()
                >= TRADE_WINDOW_HOURS[0]
            )
            & (
                pl.col("bar_close_timestamp").dt.hour()
                < TRADE_WINDOW_HOURS[1]
            ),
        )

        logger.info(
            "Trading window events: %s", f"{len(df_trading):,}",
        )

        if len(df_trading) == 0:
            logger.warning("No trading data for %s", trade_date)
            continue

        # Calculate returns
        token_returns = calculate_daily_returns(df_trading, top_tokens)

        for rank, (token_addr, signal_strength, pred_class) in enumerate(
            top_tokens, 1,
        ):
            token_meta = tokens_metadata.get(
                token_addr.lower(), {},
            )
            symbol = token_meta.get("symbol", token_addr[:10])
            daily_return, position_size = token_returns.get(
                token_addr, (0.0, abs(signal_strength)),
            )
            class_name = ["DOWN", "NEUTRAL", "UP"][pred_class]
            logger.info(
                "  %d. %s | Signal: %.3f (%s) |"
                " Size: %.1f%% | Return: %+.2f%%",
                rank,
                symbol,
                signal_strength,
                class_name,
                position_size * 100,
                daily_return * 100,
            )

        # Portfolio return
        total_size = sum(pos for _, pos in token_returns.values())
        weighted_ret = sum(
            ret * pos for ret, pos in token_returns.values()
        )
        portfolio_return = (
            weighted_ret / total_size if total_size > 0 else 0.0
        )

        logger.info(
            "Portfolio return: %+.2f%% (%.1f%% capital deployed)",
            portfolio_return * 100,
            total_size * 100,
        )

        results.append(
            {
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
    trading_days: int | None = typer.Option(
        None,
        "--trading-days",
        help="Number of days to trade (default: all available)",
    ),
    train_days: int = typer.Option(
        TRAIN_WINDOW_DAYS,
        "--train-days",
        help="Number of days in training window",
    ),
) -> None:
    """Run XGBoost baseline backtest for comparison with GNN models."""
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("XGBOOST BASELINE TRADER")
    logger.info("=" * DIVIDER_LENGTH)

    if not data_path.exists():
        logger.error("Input file not found: %s", data_path)
        msg = f"Input file not found: {data_path}"
        raise FileNotFoundError(msg)

    tokens_metadata = load_token_metadata(tokens_path)
    logger.info("Loaded metadata for %d tokens", len(tokens_metadata))

    df = load_and_prepare(data_path)

    backtest_slide(
        df,
        tokens_metadata,
        trading_days,
        train_days,
    )


if __name__ == "__main__":
    app()
