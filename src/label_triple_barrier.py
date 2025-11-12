"""Apply Triple-Barrier Method (TBM) labeling with concurrency handling.

This script implements Milestone 1 of Phase 3:
- Load log_fracdiff_price.parquet
- Group by token_id
- Calculate rolling volatility for each token
- Apply triple-barrier method to generate labels
- Handle label overlap/concurrency with sample weights
- Output labeled_log_fracdiff_price.parquet
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import typer

logger = logging.getLogger(__name__)


def calculate_rolling_volatility(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """Calculate rolling standard deviation.

    Args:
        series: Time series data (y_target_fracdiff).
        window: Rolling window size.

    Returns:
        Rolling volatility (same length as series, NaNs at start).

    """
    volatility = np.full(len(series), np.nan)

    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1 : i + 1]
        volatility[i] = np.std(window_data, ddof=1)

    return volatility


def apply_triple_barrier(
    prices: np.ndarray,
    volatilities: np.ndarray,
    upper_multiple: float,
    lower_multiple: float,
    vertical_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply triple-barrier method to generate labels.

    Args:
        prices: Price series (y_target_fracdiff).
        volatilities: Rolling volatility series.
        upper_multiple: Multiplier for upper barrier (C1).
        lower_multiple: Multiplier for lower barrier (C2).
        vertical_bars: Number of bars for time limit (N).

    Returns:
        Tuple of (labels, barrier_touch_bars):
            - labels: Array of labels (-1, 0, +1), NaN where not computable.
            - barrier_touch_bars: Number of bars until barrier touched, NaN
              where not computable.

    """
    n = len(prices)
    labels = np.full(n, np.nan)
    barrier_touch_bars = np.full(n, np.nan)

    for t in range(n):
        # Skip if we don't have volatility or can't look forward
        if np.isnan(volatilities[t]) or t + vertical_bars >= n:
            continue

        price_t = prices[t]
        vol_t = volatilities[t]

        # Skip if volatility is zero or invalid
        if vol_t <= 0 or np.isnan(vol_t):
            continue

        # Set up barriers
        upper_barrier = price_t + (vol_t * upper_multiple)
        lower_barrier = price_t - (vol_t * lower_multiple)

        # Look forward to find which barrier is hit first
        label_found = False
        for delta in range(1, vertical_bars + 1):
            if t + delta >= n:
                break

            future_price = prices[t + delta]

            # Check upper barrier (take profit)
            if future_price >= upper_barrier:
                labels[t] = 1
                barrier_touch_bars[t] = delta
                label_found = True
                break

            # Check lower barrier (stop loss)
            if future_price <= lower_barrier:
                labels[t] = -1
                barrier_touch_bars[t] = delta
                label_found = True
                break

        # If no barrier hit, vertical barrier is hit (time limit)
        if not label_found:
            labels[t] = 0
            barrier_touch_bars[t] = vertical_bars

    return labels, barrier_touch_bars


def calculate_sample_weights(
    barrier_touch_bars: np.ndarray,
) -> np.ndarray:
    """Calculate sample weights based on label concurrency/overlap.

    Args:
        barrier_touch_bars: Array indicating how many bars until barrier
            touched for each sample.

    Returns:
        Sample weights (1.0 / number of concurrent labels at each time).

    """
    n = len(barrier_touch_bars)
    weights = np.full(n, np.nan)

    # For each bar, count how many labels have it in their forward window
    for t in range(n):
        if np.isnan(barrier_touch_bars[t]):
            continue

        # Count concurrent labels: how many other labels include time t in their window?
        concurrent_count = 0

        # Look backward: which previous labels have t in their forward window?
        for prev_t in range(max(0, t - int(np.nanmax(barrier_touch_bars))), t + 1):
            if prev_t >= n or np.isnan(barrier_touch_bars[prev_t]):
                continue

            # Does label at prev_t extend to time t?
            if prev_t + barrier_touch_bars[prev_t] >= t:
                concurrent_count += 1

        if concurrent_count > 0:
            weights[t] = 1.0 / concurrent_count
        else:
            weights[t] = 1.0

    return weights


def _log_labeling_statistics(
    token_stats: list[dict[str, int | float | str]],
    n_before: int,
    n_dropped: int,
    verbose: bool,
) -> None:
    """Log comprehensive statistics about labeling results.

    Args:
        token_stats: List of per-token statistics.
        n_before: Number of rows before filtering.
        n_dropped: Number of rows dropped.
        verbose: Whether to log detailed statistics.

    """
    logger.info(
        "Dropped %s rows with NaN labels (%.1f%%)",
        f"{n_dropped:,}",
        100 * n_dropped / n_before if n_before > 0 else 0,
    )

    if not token_stats or not verbose:
        return

    stats_df = pl.DataFrame(token_stats)
    logger.info("\nLabel Distribution Statistics:")
    logger.info("  Total tokens processed: %d", len(stats_df))
    logger.info("  Total valid labels: %s", f"{stats_df['n_valid_labels'].sum():,}")
    logger.info(
        "  Positive labels (+1): %s (%.1f%%)",
        f"{stats_df['n_positive'].sum():,}",
        100 * stats_df["n_positive"].sum() / stats_df["n_valid_labels"].sum(),
    )
    logger.info(
        "  Negative labels (-1): %s (%.1f%%)",
        f"{stats_df['n_negative'].sum():,}",
        100 * stats_df["n_negative"].sum() / stats_df["n_valid_labels"].sum(),
    )
    logger.info(
        "  Neutral labels (0): %s (%.1f%%)",
        f"{stats_df['n_neutral'].sum():,}",
        100 * stats_df["n_neutral"].sum() / stats_df["n_valid_labels"].sum(),
    )
    logger.info("  Mean sample weight: %.3f", stats_df["mean_sample_weight"].mean())

    logger.info("\nExample tokens:")
    examples = stats_df.head(5)
    for row in examples.iter_rows(named=True):
        logger.info(
            "  Token %s: n=%d, +1=%d, -1=%d, 0=%d, weight=%.3f",
            row["token_id"][:10] + "...",
            row["n_valid_labels"],
            row["n_positive"],
            row["n_negative"],
            row["n_neutral"],
            row["mean_sample_weight"],
        )


def _log_output_statistics(result_df: pl.DataFrame, verbose: bool) -> None:
    """Log final output dataset and distribution statistics.

    Args:
        result_df: The final labeled dataframe.
        verbose: Whether to log detailed statistics.

    """
    logger.info("\nOutput Dataset:")
    logger.info("  Total labeled messages: %s", f"{len(result_df):,}")
    logger.info("  Unique tokens: %d", result_df["token_id"].n_unique())
    logger.info("  Unique pools: %d", result_df["pool_id"].n_unique())

    if not verbose:
        return

    logger.info(
        "  Date range: %s to %s",
        result_df["bar_close_timestamp"].min(),
        result_df["bar_close_timestamp"].max(),
    )

    label_counts = result_df.group_by("label").agg(pl.len().alias("count"))
    logger.info("\nFinal Label Counts:")
    for row in label_counts.iter_rows(named=True):
        logger.info("  Label %+d: %s", int(row["label"]), f"{row['count']:,}")

    logger.info("\nSample Weight Distribution:")
    logger.info("  Mean: %.3f", result_df["sample_weight"].mean())
    logger.info("  Median: %.3f", result_df["sample_weight"].median())
    logger.info("  Min: %.3f", result_df["sample_weight"].min())
    logger.info("  Max: %.3f", result_df["sample_weight"].max())


def label_triple_barrier(
    input_file: Path,
    output_file: Path,
    upper_multiple: float,
    lower_multiple: float,
    vertical_bars: int,
    volatility_window: int,
    verbose: bool = False,
) -> None:
    """Apply triple-barrier method labeling with concurrency handling.

    Args:
        input_file: Path to log_fracdiff_price.parquet.
        output_file: Path to output labeled_log_fracdiff_price.parquet.
        upper_multiple: Multiplier for upper barrier (take profit).
        lower_multiple: Multiplier for lower barrier (stop loss).
        vertical_bars: Number of bars for time limit.
        volatility_window: Rolling window for volatility calculation.
        verbose: Whether to log detailed statistics.

    """
    logger.info("Loading data from %s...", input_file)
    df = pl.read_parquet(input_file)
    logger.info(
        "Loaded %s messages for %d tokens",
        f"{len(df):,}",
        df["token_id"].n_unique(),
    )

    logger.info("\nTriple-Barrier Parameters:")
    logger.info("  Upper barrier multiplier (C1): %.2f", upper_multiple)
    logger.info("  Lower barrier multiplier (C2): %.2f", lower_multiple)
    logger.info("  Vertical bars (N): %d", vertical_bars)
    logger.info("  Volatility window: %d", volatility_window)

    # Sort by token_id and timestamp
    df = df.sort(["token_id", "bar_close_timestamp"])

    # Process each token group
    logger.info("\nProcessing tokens to generate labels...")
    all_labeled_rows = []
    token_stats = []

    for token_tuple, token_group in df.group_by("token_id", maintain_order=True):
        token_id = token_tuple[0]
        token_df = token_group.sort("bar_close_timestamp")

        # Get price series (stationary target)
        prices = token_df["y_target_fracdiff"].to_numpy()

        # Check for invalid data
        if len(prices) < volatility_window + vertical_bars:
            logger.warning(
                "Token %s has insufficient data (%d bars), skipping",
                token_id[:10] + "...",
                len(prices),
            )
            continue

        # Calculate rolling volatility
        volatilities = calculate_rolling_volatility(prices, volatility_window)

        # Apply triple-barrier method
        labels, barrier_touch_bars = apply_triple_barrier(
            prices,
            volatilities,
            upper_multiple,
            lower_multiple,
            vertical_bars,
        )

        # Calculate sample weights based on concurrency
        sample_weights = calculate_sample_weights(barrier_touch_bars)

        # Count valid labels
        valid_mask = ~np.isnan(labels)
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            logger.warning(
                "Token %s produced no valid labels, skipping",
                token_id[:10] + "...",
            )
            continue

        # Count label distribution
        n_positive = np.sum(labels[valid_mask] == 1)
        n_negative = np.sum(labels[valid_mask] == -1)
        n_neutral = np.sum(labels[valid_mask] == 0)

        # Store statistics
        token_stats.append(
            {
                "token_id": token_id,
                "n_total": len(prices),
                "n_valid_labels": n_valid,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_neutral": n_neutral,
                "mean_sample_weight": np.nanmean(sample_weights[valid_mask]),
            },
        )

        # Add new columns to token_df
        all_labeled_rows.append(
            token_df.with_columns(
                [
                    pl.Series("rolling_volatility", volatilities),
                    pl.Series("label", labels),
                    pl.Series("barrier_touch_bars", barrier_touch_bars),
                    pl.Series("sample_weight", sample_weights),
                ],
            ),
        )

    # Combine all token dataframes
    logger.info("\nCombining results from all tokens...")
    result_df = pl.concat(all_labeled_rows)

    # Drop rows with NaN labels
    n_before = len(result_df)
    result_df = result_df.filter(pl.col("label").is_not_nan())
    n_after = len(result_df)
    n_dropped = n_before - n_after

    # Log statistics
    _log_labeling_statistics(token_stats, n_before, n_dropped, verbose)

    # Log output statistics
    _log_output_statistics(result_df, verbose)

    # Save output
    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s labeled messages to %s",
        f"{len(result_df):,}",
        output_file.name,
    )


def validate_output(output_file: Path, verbose: bool = False) -> None:
    """Validate the output parquet file.

    Args:
        output_file: Path to the output parquet file.
        verbose: Whether to log detailed validation information.

    """
    logger.info("Validating output file: %s", output_file)

    # Check file exists and is readable
    if not output_file.exists():
        msg = f"Output file not found: {output_file}"
        raise FileNotFoundError(msg)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("  File size: %.2f MB", file_size_mb)

    # Read and validate parquet structure
    try:
        df = pl.read_parquet(output_file)
    except Exception as e:
        msg = "Failed to read parquet file"
        raise ValueError(msg) from e

    # Verify required columns
    required_columns = {
        "token_id",
        "pool_id",
        "bar_close_timestamp",
        "y_target_fracdiff",
        "label",
        "sample_weight",
        "rolling_volatility",
        "barrier_touch_bars",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    logger.info("  Rows: %s", f"{len(df):,}")
    logger.info("  Unique tokens: %d", df["token_id"].n_unique())
    logger.info("  Columns: %d", len(df.columns))

    if not verbose:
        logger.info("  Validation passed!")
        return

    # Validate label column
    label_counts = df.group_by("label").agg(pl.len().alias("count")).sort("label")
    logger.info("  Label distribution:")
    for row in label_counts.iter_rows(named=True):
        logger.info("    Label %+d: %s", int(row["label"]), f"{row['count']:,}")

    # Check for NaN values in critical columns
    critical_columns = ["label", "sample_weight", "rolling_volatility"]
    for col in critical_columns:
        nan_count = df[col].null_count()
        if nan_count > 0:
            logger.warning("    Column %s has %d NaN values", col, nan_count)

    logger.info("  Validation passed!")


def main(
    input_file: Path = typer.Option(
        Path("data/log_fracdiff_price.parquet"),
        help="Input log_fracdiff_price.parquet file",
    ),
    output_file: Path = typer.Option(
        Path("data/labeled_log_fracdiff_price.parquet"),
        help="Output parquet file path",
    ),
    upper_multiple: float = typer.Option(
        2.0,
        help="Multiplier for upper barrier (take profit, C1)",
    ),
    lower_multiple: float = typer.Option(
        1.0,
        help="Multiplier for lower barrier (stop loss, C2)",
    ),
    vertical_bars: int = typer.Option(
        25,
        help="Number of bars for time limit (N)",
    ),
    volatility_window: int = typer.Option(
        20,
        help="Rolling window for volatility calculation",
    ),
    validate: bool = typer.Option(
        False,
        help="Validate the output parquet file",
    ),
    verbose: bool = typer.Option(
        False,
        help="Enable verbose logging",
    ),
) -> None:
    """Apply triple-barrier method labeling with concurrency handling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    label_triple_barrier(
        input_file,
        output_file,
        upper_multiple,
        lower_multiple,
        vertical_bars,
        volatility_window,
        verbose=verbose,
    )

    if validate:
        validate_output(output_file, verbose=verbose)


if __name__ == "__main__":
    typer.run(main)
