"""Generate training labels using Triple-Barrier Method with concurrency weighting.

WHY: Traditional labeling uses fixed-horizon returns (e.g., 1-day forward return),
which introduces look-ahead bias and ignores path dynamics. Prado's Triple-Barrier
Method (AFML Ch. 3) addresses this by defining three barriers: upper (profit-take),
lower (stop-loss), and vertical (time limit). Labels are assigned when the FIRST
barrier is touched, capturing actual trade outcomes while respecting market dynamics.

This approach is critical for crypto trading where:
1. Volatility varies dramatically across tokens
2. Holding period should adapt to price movement speed
3. We need to avoid "toxic labels" where the path crosses both barriers

Sample weights account for label overlap (concurrency) - when multiple training
samples span the same time period, they're not independent and should be downweighted
to prevent overfitting (Prado Ch. 4).

WHAT: For each token's price series:
1. Calculate rolling volatility (window=20 bars by default)
2. Set dynamic barriers: upper = price + C1*volatility, lower = price - C2*volatility
3. Set vertical barrier based on token's average bar frequency (adaptive)
4. Look forward to find which barrier is hit first
5. Assign labels: +1 (upper), -1 (lower), 0 (time limit)
6. Calculate sample weights based on concurrent label count

This produces high-quality labels that respect market microstructure and token-specific
dynamics while preventing data leakage.

HOW:
1. Group bars by token and calculate rolling volatility
2. Determine dynamic vertical barrier (fraction of daily bars per token)
3. For each bar, look forward up to vertical barrier or until price hits threshold
4. Track which bar each label extends to (for concurrency calculation)
5. Weight samples: w = 1 / (number of concurrent labels at that time)
6. Generate labels for both src and dest tokens independently

INPUT: data/log_fracdiff_price.parquet (stationary features)
OUTPUT: data/labeled_log_fracdiff_price.parquet (bars with labels and weights)

References:
- Prado AFML Ch. 3.2: Triple-Barrier Method
- Prado AFML Ch. 4.3: Sample Weights from Label Concurrency

"""

# ruff: noqa: PLR0915, PLR0912, C901

import logging
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


def calculate_rolling_volatility(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """Calculate rolling standard deviation.

    Args:
        series: Time series data (src_fracdiff).
        window: Rolling window size.

    Returns:
        Rolling volatility (same length as series, NaNs at start).

    """
    volatility = np.full(len(series), np.nan)

    for i in range(window - 1, len(series)):
        window_data = series[i - window + 1 : i + 1]
        volatility[i] = np.std(window_data, ddof=1)

    return volatility


def calculate_dynamic_vertical_barrier(
    timestamps: pl.Series,
    barrier_fraction: float = 0.1,
    min_bars: int = 1,
    max_bars: int = 1000,
) -> int:
    """Calculate dynamic vertical barrier based on average daily volume (bars).

    Args:
        timestamps: Series of bar timestamps.
        barrier_fraction: Fraction of daily volume to use as barrier.
        min_bars: Minimum number of bars for the barrier.
        max_bars: Maximum number of bars for the barrier.

    Returns:
        Number of bars to use as vertical barrier.

    """
    min_timestamps_for_duration = 2
    if len(timestamps) < min_timestamps_for_duration:
        return min_bars

    # Calculate total duration in days
    duration = timestamps.max() - timestamps.min()
    duration_days = duration.total_seconds() / 86400.0

    if duration_days <= 0:
        return min_bars

    # Average bars per day
    avg_daily_bars = len(timestamps) / duration_days

    # Calculate barrier
    barrier = int(avg_daily_bars * barrier_fraction)

    # Clamp
    return max(min_bars, min(barrier, max_bars))


def apply_triple_barrier(
    prices: np.ndarray,
    volatilities: np.ndarray,
    upper_multiple: float,
    lower_multiple: float,
    vertical_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply triple-barrier method to generate labels.

    Args:
        prices: Price series (src_fracdiff).
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
    drop_pct = 100 * n_dropped / n_before if n_before > 0 else 0
    high_drop_threshold = 50
    if drop_pct > high_drop_threshold:
        logger.warning(
            "  ⚠️  WARN: Dropped %s rows with NaN labels (%.1f%%)",
            f"{n_dropped:,}",
            drop_pct,
        )
    else:
        logger.info(
            "  ✅ OK: Dropped %s rows with NaN labels (%.1f%%)",
            f"{n_dropped:,}",
            drop_pct,
        )

    if not token_stats or not verbose:
        return

    stats_df = pl.DataFrame(token_stats)
    logger.info("\n📊 Label Distribution Statistics:")
    logger.info("  Total tokens processed: %d", len(stats_df))
    logger.info("  Total valid labels: %s", f"{stats_df['n_valid_labels'].sum():,}")

    total_labels = stats_df["n_valid_labels"].sum()
    pos_pct = 100 * stats_df["n_positive"].sum() / total_labels
    neg_pct = 100 * stats_df["n_negative"].sum() / total_labels
    neu_pct = 100 * stats_df["n_neutral"].sum() / total_labels

    logger.info(
        "  Positive labels (+1): %s (%.1f%%)",
        f"{stats_df['n_positive'].sum():,}",
        pos_pct,
    )
    logger.info(
        "  Negative labels (-1): %s (%.1f%%)",
        f"{stats_df['n_negative'].sum():,}",
        neg_pct,
    )
    logger.info(
        "  Neutral labels (0): %s (%.1f%%)",
        f"{stats_df['n_neutral'].sum():,}",
        neu_pct,
    )

    # Check for severe class imbalance
    severe_imbalance_threshold = 80
    max_pct = max(pos_pct, neg_pct, neu_pct)
    if max_pct > severe_imbalance_threshold:
        logger.warning(
            "  ⚠️  WARN: Severe class imbalance detected (%.1f%% in one class)",
            max_pct,
        )
    else:
        logger.info("  ✅ OK: Reasonable class balance")

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
    logger.info("\n" + "=" * 70)
    logger.info("OUTPUT DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info("  Total labeled messages: %s", f"{len(result_df):,}")
    logger.info("  Unique tokens: %d", result_df["src_token_id"].n_unique())
    logger.info("  Unique pools: %d", result_df["pool_id"].n_unique())

    if not verbose:
        return

    logger.info(
        "  Date range: %s to %s",
        result_df["bar_close_timestamp"].min(),
        result_df["bar_close_timestamp"].max(),
    )

    label_counts = (
        result_df.group_by("label").agg(pl.len().alias("count")).sort("label")
    )
    logger.info("\nFinal Label Counts:")
    for row in label_counts.iter_rows(named=True):
        logger.info("  Label %+d: %s", int(row["label"]), f"{row['count']:,}")

    logger.info("\nSample Weight Distribution:")
    logger.info("  Mean: %.3f", result_df["sample_weight"].mean())
    logger.info("  Median: %.3f", result_df["sample_weight"].median())
    logger.info("  Min: %.3f", result_df["sample_weight"].min())
    logger.info("  Max: %.3f", result_df["sample_weight"].max())
    logger.info("=" * 70)


def _process_token_labels(
    df: pl.DataFrame,
    token_col: str,
    fracdiff_col: str,
    barrier_fraction: float,
    volatility_window: int,
    upper_multiple: float,
    lower_multiple: float,
) -> pl.DataFrame:
    """Process labels for a specific token column (src or dest).

    Args:
        df: Input DataFrame.
        token_col: Column name for token ID (src_token_id or dest_token_id).
        fracdiff_col: Column name for fracdiff price (src_fracdiff or dest_fracdiff).
        barrier_fraction: Fraction of daily volume for vertical barrier.
        volatility_window: Rolling window for volatility calculation.
        upper_multiple: Multiplier for upper barrier (take profit).
        lower_multiple: Multiplier for lower barrier (stop loss).

    Returns:
        DataFrame with columns: token_col, bar_close_timestamp, label,
        sample_weight, etc.

    """
    # Sort by token and timestamp
    df_sorted = df.sort([token_col, "bar_close_timestamp"])

    # Store results as list of dataframes
    result_dfs = []

    # Get unique tokens for progress tracking
    token_groups = list(df_sorted.group_by(token_col, maintain_order=True))
    total_tokens = len(token_groups)

    token_type = "src" if "src" in token_col else "dest"
    desc = f"Processing {token_type.upper()} tokens"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(desc, total=total_tokens)

        for token_tuple, token_group in token_groups:
            token_tuple[0]
            token_df = token_group.sort("bar_close_timestamp")

            # Calculate dynamic vertical barrier
            vertical_bars = calculate_dynamic_vertical_barrier(
                token_df["bar_close_timestamp"],
                barrier_fraction=barrier_fraction,
            )

            # Get price series (stationary target)
            prices = token_df[fracdiff_col].to_numpy()

            # Check for invalid data
            if len(prices) < volatility_window + vertical_bars:
                progress.update(task, advance=1)
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

            # Add computed columns to this token's dataframe
            token_df = token_df.with_columns(
                [
                    pl.Series("_label", labels),
                    pl.Series("_sample_weight", sample_weights),
                    pl.Series("_volatility", volatilities),
                    pl.Series("_barrier_touch_bars", barrier_touch_bars),
                ],
            )

            result_dfs.append(token_df)
            progress.update(task, advance=1)

    # Concatenate all results
    if result_dfs:
        return pl.concat(result_dfs)
    return pl.DataFrame()


def label_triple_barrier(
    input_file: Path,
    output_file: Path,
    upper_multiple: float,
    lower_multiple: float,
    barrier_fraction: float,
    volatility_window: int,
    verbose: bool = False,
) -> None:
    """Apply triple-barrier method labeling with concurrency handling.

    Args:
        input_file: Path to log_fracdiff_price.parquet.
        output_file: Path to output labeled_log_fracdiff_price.parquet.
        upper_multiple: Multiplier for upper barrier (take profit).
        lower_multiple: Multiplier for lower barrier (stop loss).
        barrier_fraction: Fraction of daily volume for vertical barrier.
        volatility_window: Rolling window for volatility calculation.
        verbose: Whether to log detailed statistics.

    """
    logger.info("=" * 70)
    logger.info("TRIPLE-BARRIER LABELING")
    logger.info("=" * 70)
    logger.info("Loading data from %s", input_file)
    df = pl.read_parquet(input_file)
    logger.info(
        "✅ Loaded %s messages for %d unique src tokens, %d unique dest tokens",
        f"{len(df):,}",
        df["src_token_id"].n_unique(),
        df["dest_token_id"].n_unique(),
    )

    logger.info("\n📐 Triple-Barrier Parameters:")
    logger.info("  Upper barrier multiplier (C1): %.2f", upper_multiple)
    logger.info("  Lower barrier multiplier (C2): %.2f", lower_multiple)
    logger.info("  Barrier fraction (daily vol): %.2f", barrier_fraction)
    logger.info("  Volatility window: %d bars", volatility_window)

    # Process src tokens to generate labels
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Processing SRC tokens")
    logger.info("=" * 70)
    src_labels_df = _process_token_labels(
        df,
        "src_token_id",
        "src_fracdiff",
        barrier_fraction,
        volatility_window,
        upper_multiple,
        lower_multiple,
    )
    if len(src_labels_df) > 0:
        n_src = src_labels_df["src_token_id"].n_unique()
        logger.info("✅ Processed %d src tokens", n_src)
    else:
        logger.warning("⚠️  No src tokens processed")

    # Process dest tokens to generate labels
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Processing DEST tokens")
    logger.info("=" * 70)
    dest_labels_df = _process_token_labels(
        df,
        "dest_token_id",
        "dest_fracdiff",
        barrier_fraction,
        volatility_window,
        upper_multiple,
        lower_multiple,
    )
    if len(dest_labels_df) > 0:
        n_dest = dest_labels_df["dest_token_id"].n_unique()
        logger.info("✅ Processed %d dest tokens", n_dest)
    else:
        logger.warning("⚠️  No dest tokens processed")

    # Now assign labels back to the original dataframe
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Assigning labels to swap events")
    logger.info("=" * 70)

    # Join src labels
    if len(src_labels_df) > 0:
        # Select only the columns we need for joining
        src_join = src_labels_df.select([
            "src_token_id",
            "bar_close_timestamp",
            pl.col("_label").alias("label"),
            pl.col("_sample_weight").alias("sample_weight"),
            pl.col("_volatility").alias("rolling_volatility"),
            pl.col("_barrier_touch_bars").alias("barrier_touch_bars"),
        ])
        result_df = df.join(
            src_join,
            on=["src_token_id", "bar_close_timestamp"],
            how="left",
        )
    else:
        result_df = df.with_columns([
            pl.lit(np.nan).alias("label"),
            pl.lit(np.nan).alias("sample_weight"),
            pl.lit(np.nan).alias("rolling_volatility"),
            pl.lit(np.nan).alias("barrier_touch_bars"),
        ])

    # Join dest labels
    if len(dest_labels_df) > 0:
        dest_join = dest_labels_df.select([
            "dest_token_id",
            "bar_close_timestamp",
            pl.col("_label").alias("dest_label"),
            pl.col("_sample_weight").alias("dest_sample_weight"),
            pl.col("_barrier_touch_bars").alias("dest_barrier_touch_bars"),
        ])
        result_df = result_df.join(
            dest_join,
            on=["dest_token_id", "bar_close_timestamp"],
            how="left",
        )
    else:
        result_df = result_df.with_columns([
            pl.lit(np.nan).alias("dest_label"),
            pl.lit(np.nan).alias("dest_sample_weight"),
            pl.lit(np.nan).alias("dest_barrier_touch_bars"),
        ])

    # Gather token statistics from both src and dest
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Gathering statistics")
    logger.info("=" * 70)
    token_stats = []

    # Get stats from src labels
    if len(src_labels_df) > 0:
        src_stats = (
            src_labels_df
            .filter(pl.col("_label").is_not_nan())
            .group_by("src_token_id")
            .agg([
                pl.len().alias("n_total"),
                (pl.col("_label") == 1).sum().alias("n_positive"),
                (pl.col("_label") == -1).sum().alias("n_negative"),
                (pl.col("_label") == 0).sum().alias("n_neutral"),
                pl.col("_sample_weight").mean().alias("mean_sample_weight"),
            ])
        )
        token_stats.extend([
            {
                "token_id": row["src_token_id"],
                "n_total": row["n_total"],
                "n_valid_labels": row["n_total"],
                "n_positive": row["n_positive"],
                "n_negative": row["n_negative"],
                "n_neutral": row["n_neutral"],
                "mean_sample_weight": row["mean_sample_weight"],
            }
            for row in src_stats.iter_rows(named=True)
        ])

    # Add stats from dest labels (combine with existing src stats if token
    # appears in both)
    if len(dest_labels_df) > 0:
        dest_stats = (
            dest_labels_df
            .filter(pl.col("_label").is_not_nan())
            .group_by("dest_token_id")
            .agg([
                pl.len().alias("n_total"),
                (pl.col("_label") == 1).sum().alias("n_positive"),
                (pl.col("_label") == -1).sum().alias("n_negative"),
                (pl.col("_label") == 0).sum().alias("n_neutral"),
                pl.col("_sample_weight").mean().alias("mean_sample_weight"),
            ])
        )
        # Merge with existing stats or add new ones
        token_dict = {s["token_id"]: s for s in token_stats}
        for row in dest_stats.iter_rows(named=True):
            token_id = row["dest_token_id"]
            if token_id in token_dict:
                # Combine stats
                existing = token_dict[token_id]
                existing["n_total"] += row["n_total"]
                existing["n_valid_labels"] += row["n_total"]
                existing["n_positive"] += row["n_positive"]
                existing["n_negative"] += row["n_negative"]
                existing["n_neutral"] += row["n_neutral"]
                # Average the weights
                existing["mean_sample_weight"] = (
                    existing["mean_sample_weight"] + row["mean_sample_weight"]
                ) / 2
            else:
                token_dict[token_id] = {
                    "token_id": token_id,
                    "n_total": row["n_total"],
                    "n_valid_labels": row["n_total"],
                    "n_positive": row["n_positive"],
                    "n_negative": row["n_negative"],
                    "n_neutral": row["n_neutral"],
                    "mean_sample_weight": row["mean_sample_weight"],
                }
        token_stats = list(token_dict.values())

    # Filtering rows with valid labels
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Filtering rows with valid labels")
    logger.info("=" * 70)

    # Drop rows with NaN src labels (required)
    # Replace NaN dest labels with null (dest labels are optional)
    n_before = len(result_df)
    result_df = result_df.filter(pl.col("label").is_not_nan())

    # Replace NaN with null in dest columns for cleaner downstream handling
    # NaN values cause issues when casting to int64, null is safer
    result_df = result_df.with_columns([
        pl.when(pl.col("dest_label").is_nan())
        .then(pl.lit(None))
        .otherwise(pl.col("dest_label"))
        .alias("dest_label"),
        pl.when(pl.col("dest_sample_weight").is_nan())
        .then(pl.lit(None))
        .otherwise(pl.col("dest_sample_weight"))
        .alias("dest_sample_weight"),
        pl.when(pl.col("dest_barrier_touch_bars").is_nan())
        .then(pl.lit(None))
        .otherwise(pl.col("dest_barrier_touch_bars"))
        .alias("dest_barrier_touch_bars"),
    ])

    n_after = len(result_df)
    n_dropped = n_before - n_after

    # Log statistics
    _log_labeling_statistics(token_stats, n_before, n_dropped, verbose)

    # Log output statistics
    _log_output_statistics(result_df, verbose)

    # Save output
    logger.info("\nSaving to %s", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)

    logger.info(
        "✅ Done! Saved %s labeled messages to %s",
        f"{len(result_df):,}",
        output_file.name,
    )


def validate_output(output_file: Path, verbose: bool = False) -> None:
    """Validate the output parquet file.

    Args:
        output_file: Path to the output parquet file.
        verbose: Whether to log detailed validation information.

    """
    logger.info("=" * 70)
    logger.info("OUTPUT VALIDATION")
    logger.info("=" * 70)
    logger.info("Validating output file: %s", output_file)

    has_issues = False

    # 1. Check file exists and is readable
    logger.info("\n1. Checking file existence and readability...")
    if not output_file.exists():
        logger.error("  ❌ ERROR: Output file not found: %s", output_file)
        msg = f"Output file not found: {output_file}"
        raise FileNotFoundError(msg)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("  ✅ OK: File exists (%.2f MB)", file_size_mb)

    # 2. Read and validate parquet structure
    logger.info("\n2. Reading parquet file...")
    try:
        df = pl.read_parquet(output_file)
        logger.info("  ✅ OK: File is readable")
    except Exception as e:
        logger.exception("  ❌ ERROR: Failed to read parquet file")
        msg = "Failed to read parquet file"
        raise ValueError(msg) from e

    # 3. Verify required columns
    logger.info("\n3. Checking required columns...")
    required_columns = {
        "src_token_id",
        "pool_id",
        "bar_close_timestamp",
        "src_fracdiff",
        "dest_fracdiff",
        "label",
        "sample_weight",
        "rolling_volatility",
        "barrier_touch_bars",
        "dest_label",
        "dest_sample_weight",
        "dest_barrier_touch_bars",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error("  ❌ ERROR: Missing required columns: %s", missing_columns)
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)
    logger.info("  ✅ OK: All required columns present")

    # 4. Basic dataset statistics
    logger.info("\n4. Dataset statistics...")
    logger.info("  Rows: %s", f"{len(df):,}")
    logger.info("  Unique tokens: %d", df["src_token_id"].n_unique())
    logger.info("  Columns: %d", len(df.columns))
    logger.info("  ✅ OK: Dataset loaded successfully")

    if not verbose:
        logger.info("\n" + "=" * 70)
        logger.info("✅ All validation checks passed!")
        logger.info("=" * 70)
        return

    # 5. Validate label distribution
    logger.info("\n5. Checking label distribution...")
    label_counts = (
        df.group_by("label").agg(pl.len().alias("count")).sort("label")
    )
    logger.info("  Label distribution:")
    for row in label_counts.iter_rows(named=True):
        logger.info("    Label %+d: %s", int(row["label"]), f"{row['count']:,}")

    # Check for severe imbalance
    severe_imbalance_threshold = 80
    total_labels = df.select(pl.len()).item()
    for row in label_counts.iter_rows(named=True):
        pct = 100 * row["count"] / total_labels
        if pct > severe_imbalance_threshold:
            logger.warning(
                "  ⚠️  WARN: Severe class imbalance (%.1f%% in label %+d)",
                pct,
                int(row["label"]),
            )
            has_issues = True
    if not has_issues:
        logger.info("  ✅ OK: Reasonable label distribution")

    # 6. Check for NaN values in critical columns
    logger.info("\n6. Checking for NaN values in critical columns...")
    critical_columns = ["label", "sample_weight", "rolling_volatility"]
    nan_found = False
    for col in critical_columns:
        nan_count = df[col].null_count()
        if nan_count > 0:
            logger.warning("  ⚠️  WARN: Column %s has %d NaN values", col, nan_count)
            nan_found = True
            has_issues = True
    if not nan_found:
        logger.info("  ✅ OK: No NaN values in critical columns")

    # 7. Sample weight validation
    logger.info("\n7. Validating sample weights...")
    weight_stats = df.select([
        pl.col("sample_weight").min().alias("min"),
        pl.col("sample_weight").max().alias("max"),
        pl.col("sample_weight").mean().alias("mean"),
        pl.col("sample_weight").median().alias("median"),
    ]).to_dicts()[0]

    logger.info("  Min: %.3f", weight_stats["min"])
    logger.info("  Max: %.3f", weight_stats["max"])
    logger.info("  Mean: %.3f", weight_stats["mean"])
    logger.info("  Median: %.3f", weight_stats["median"])

    if weight_stats["min"] < 0 or weight_stats["max"] > 1:
        logger.warning("  ⚠️  WARN: Sample weights outside expected range [0, 1]")
        has_issues = True
    else:
        logger.info("  ✅ OK: Sample weights in valid range")

    # 8. Time range validation
    logger.info("\n8. Checking time range...")
    time_min = df["bar_close_timestamp"].min()
    time_max = df["bar_close_timestamp"].max()
    logger.info("  Date range: %s to %s", time_min, time_max)
    logger.info("  ✅ OK: Time range validated")

    # Summary
    logger.info("\n" + "=" * 70)
    if has_issues:
        logger.warning("⚠️  Validation completed with warnings")
    else:
        logger.info("✅ All validation checks passed!")
    logger.info("=" * 70)


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
    barrier_fraction: float = typer.Option(
        0.1,
        help="Fraction of daily volume for vertical barrier (default 0.1 = 10%)",
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
        barrier_fraction,
        volatility_window,
        verbose=verbose,
    )

    if validate:
        validate_output(output_file, verbose=verbose)


if __name__ == "__main__":
    typer.run(main)
