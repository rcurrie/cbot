"""Make target variable stationary using fractional differentiation.

This script implements Milestone 2 of Phase 2:
- Load usdc_bars.parquet
- Group by token_id
- Apply log transformation to token_close_price_usdc
- Find minimum fractional differentiation order d for each token
- Apply fractional differentiation to achieve stationarity
- Output log_fracdiff_price.parquet
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import typer
from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def frac_diff_fixed(
    series: np.ndarray,
    d: float,
    threshold: float = 1e-5,
) -> np.ndarray:
    """Apply fixed-width window fractional differentiation.

    Args:
        series: Time series to differentiate.
        d: Fractional differentiation order (0 < d < 1).
        threshold: Minimum weight threshold for computational efficiency.

    Returns:
        Fractionally differentiated series (same length, NaNs at start).

    """
    # Compute weights
    weights: list[float] = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    weights_arr = np.array(weights[::-1])  # Reverse for convolution
    width = len(weights_arr)

    # Apply fractional differentiation
    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights_arr, series[i - width + 1 : i + 1])

    return result


MIN_ADF_OBSERVATIONS = 20


def find_min_d_for_stationarity(
    log_price: np.ndarray,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    p_value_threshold: float = 0.05,
) -> tuple[float, bool]:
    """Find minimum fractional differentiation order for stationarity.

    Args:
        log_price: Log-transformed price series.
        d_min: Minimum d to search.
        d_max: Maximum d to search.
        d_step: Step size for search.
        p_value_threshold: ADF test p-value threshold (default 0.05).

    Returns:
        Tuple of (minimum_d, is_stationary).
            - minimum_d: Minimum d achieving stationarity (d_max if none).
            - is_stationary: True if stationarity was achieved.

    """
    d_values = np.arange(d_min, d_max + d_step, d_step)

    for d_val in d_values:
        # Test original log_price series or apply fractional differentiation
        fracdiff_series = (
            log_price if float(d_val) == 0 else frac_diff_fixed(log_price, float(d_val))
        )

        # Remove NaNs for ADF test
        fracdiff_clean = fracdiff_series[~np.isnan(fracdiff_series)]

        # Need sufficient data for ADF test
        if len(fracdiff_clean) < MIN_ADF_OBSERVATIONS:
            continue

        try:
            # Run ADF test
            adf_result = adfuller(fracdiff_clean, autolag="AIC")
            p_value = adf_result[1]

            if p_value < p_value_threshold:
                # Stationary! Return this d
                return float(d_val), True
        except (ValueError, RuntimeError) as e:
            logger.warning("ADF test failed for d=%.2f: %s", d_val, e)
            continue

    # No stationary d found
    return float(d_max), False


def _process_token_group(
    token_id: str,
    token_df: pl.DataFrame,
    d_min: float,
    d_max: float,
    d_step: float,
) -> tuple[pl.DataFrame | None, dict[str, object]]:
    """Process a single token group for stationarity.

    Args:
        token_id: Token identifier.
        token_df: DataFrame for this token (rows sorted by timestamp).
        d_min: Minimum d to search.
        d_max: Maximum d to search.
        d_step: Step size for d search.

    Returns:
        Tuple of (processed_df, token_stats_dict). processed_df is None if token
        should be skipped (non-stationary or invalid data).

    """
    # Get price series
    prices = token_df["price"].to_numpy()

    # Apply log transformation
    log_prices = np.log(prices)

    # Check if log_prices are valid
    if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
        logger.warning("Token %s has invalid log prices, skipping", token_id)
        return (
            None,
            {
                "token_id": token_id,
                "n_observations": len(token_df),
                "min_d": 0.0,
                "is_stationary": False,
                "n_valid_after_fracdiff": 0,
                "dropped": True,
            },
        )

    # Find minimum d for stationarity
    min_d, is_stationary = find_min_d_for_stationarity(
        log_prices,
        d_min=d_min,
        d_max=d_max,
        d_step=d_step,
    )

    stats: dict[str, object] = {
        "token_id": token_id,
        "n_observations": len(token_df),
        "min_d": float(min_d),
        "is_stationary": is_stationary,
        "dropped": False,
    }

    if not is_stationary:
        logger.info(
            "Token %s not stationary (d=%.2f)",
            token_id[:10] + "...",
            min_d,
        )
        stats["n_valid_after_fracdiff"] = 0
        stats["dropped"] = True
        return None, stats

    # Apply fractional differentiation with the found d
    if min_d == 0:
        fracdiff_log_price = log_prices
    else:
        fracdiff_log_price = frac_diff_fixed(log_prices, min_d)

    # Store statistics
    stats["n_valid_after_fracdiff"] = int(np.sum(~np.isnan(fracdiff_log_price)))

    # Add fracdiff values to token_df
    result_df = token_df.with_columns(
        [pl.Series("fracdiff", fracdiff_log_price)],
    )

    return result_df, stats


def make_stationary(
    input_file: Path,
    output_file: Path,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    *,
    drop_non_stationary: bool = False,
) -> None:
    """Make target variable stationary using fractional differentiation.

    Args:
        input_file: Path to usdc_bars.parquet.
        output_file: Path to output log_fracdiff_price.parquet.
        d_min: Minimum fractional differentiation order to search.
        d_max: Maximum fractional differentiation order to search.
        d_step: Step size for d search.
        drop_non_stationary: If True, drop all tokens that don't achieve stationarity.

    """
    logger.info("Loading data from %s...", input_file)
    df = pl.read_parquet(input_file)
    logger.info(
        "Loaded %s bar events for %d source tokens",
        f"{len(df):,}",
        df["src_token_id"].n_unique(),
    )

    # Extract unique price observations for all tokens (both src and dest)
    logger.info("Extracting price series for all tokens...")

    # Get prices from source side
    df_src = df.select(
        pl.col("src_token_id").alias("token_id"),
        pl.col("bar_close_timestamp"),
        pl.col("src_price_usdc").alias("price"),
    )

    # Get prices from dest side
    df_dest = df.select(
        pl.col("dest_token_id").alias("token_id"),
        pl.col("bar_close_timestamp"),
        pl.col("dest_price_usdc").alias("price"),
    )

    # Combine and deduplicate (taking mean price if multiple pools have bars at same
    # timestamp)
    df_prices = (
        pl.concat([df_src, df_dest])
        .group_by(["token_id", "bar_close_timestamp"])
        .agg(pl.col("price").mean())
        .sort(["token_id", "bar_close_timestamp"])
    )

    logger.info(
        "Extracted %s unique price observations for %d tokens",
        f"{len(df_prices):,}",
        df_prices["token_id"].n_unique(),
    )

    logger.info("\nProcessing tokens to achieve stationarity...")
    logger.info(
        "Testing d values from %.2f to %.2f (step=%.2f)",
        d_min,
        d_max,
        d_step,
    )

    # Process each token group
    all_fracdiff_rows: list[pl.DataFrame] = []
    token_stats: list[dict[str, object]] = []

    for token_tuple, token_group in df_prices.group_by("token_id", maintain_order=True):
        token_id = token_tuple[0]  # Extract scalar from tuple
        token_df = token_group.sort("bar_close_timestamp")

        result_df, stats = _process_token_group(
            token_id,
            token_df,
            d_min,
            d_max,
            d_step,
        )
        token_stats.append(stats)

        should_include = result_df is not None and (
            not drop_non_stationary or stats["is_stationary"]
        )
        if should_include:
            assert result_df is not None
            all_fracdiff_rows.append(result_df)

    # Combine all token fracdiff series
    logger.info("\nCombining results from all tokens...")
    df_fracdiff = pl.concat(all_fracdiff_rows)

    # Join fracdiff values back to original bars (Double Lookup)
    logger.info("Joining fracdiff values back to bars (Double Lookup)...")

    # 1. Join for Source Token
    result_df = df.join(
        df_fracdiff.select(["token_id", "bar_close_timestamp", "fracdiff"]),
        left_on=["src_token_id", "bar_close_timestamp"],
        right_on=["token_id", "bar_close_timestamp"],
        how="left",
    ).rename({"fracdiff": "src_fracdiff"})

    # 2. Join for Destination Token
    result_df = result_df.join(
        df_fracdiff.select(["token_id", "bar_close_timestamp", "fracdiff"]),
        left_on=["dest_token_id", "bar_close_timestamp"],
        right_on=["token_id", "bar_close_timestamp"],
        how="left",
    ).rename({"fracdiff": "dest_fracdiff"})

    # Drop rows with NaN in src_fracdiff (we need at least the source to be valid)
    # Note: dest_fracdiff might be NaN if dest token was dropped due to non-stationarity
    n_before = len(result_df)
    result_df = result_df.filter(pl.col("src_fracdiff").is_not_nan())
    n_after = len(result_df)
    n_dropped = n_before - n_after

    logger.info(
        "Dropped %s bar events with NaN src_fracdiff values (%.1f%%)",
        f"{n_dropped:,}",
        100 * n_dropped / n_before if n_before > 0 else 0,
    )

    # Check dest_fracdiff coverage
    n_dest_nan = result_df.filter(pl.col("dest_fracdiff").is_null()).height
    if n_dest_nan > 0:
        logger.warning(
            "  %s bars have NaN dest_fracdiff (%.1f%%)",
            f"{n_dest_nan:,}",
            100 * n_dest_nan / len(result_df),
        )

    # Log statistics
    _log_stationarity_stats(token_stats, drop_non_stationary)
    _log_output_stats(result_df)

    # Save output
    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s bar events with stationary targets to %s",
        f"{len(result_df):,}",
        output_file.name,
    )


def _log_stationarity_stats(
    token_stats: list[dict[str, object]],
    drop_non_stationary: bool,
) -> None:
    """Log stationarity statistics."""
    stats_df = pl.DataFrame(token_stats)
    logger.info("\nStationarity Statistics:")
    logger.info("  Total tokens processed: %d", len(stats_df))
    logger.info(
        "  Tokens achieving stationarity: %d (%.1f%%)",
        stats_df["is_stationary"].sum(),
        100 * stats_df["is_stationary"].sum() / len(stats_df),
    )
    if drop_non_stationary:
        n_dropped_tokens = stats_df["dropped"].sum()
        logger.info(
            "  Tokens dropped (non-stationary): %d (%.1f%%)",
            n_dropped_tokens,
            100 * n_dropped_tokens / len(stats_df),
        )
    logger.info("  Mean d: %.3f", stats_df["min_d"].mean())
    logger.info("  Median d: %.3f", stats_df["min_d"].median())

    # Show some example tokens
    logger.info("\nExample tokens (sorted by d):")
    examples = stats_df.sort("min_d").head(5)
    for row in examples.iter_rows(named=True):
        logger.info(
            "  Token %s: d=%.2f, n=%d, stationary=%s",
            row["token_id"][:10] + "...",
            row["min_d"],
            row["n_observations"],
            row["is_stationary"],
        )


def _log_output_stats(result_df: pl.DataFrame) -> None:
    """Log final output dataset statistics."""
    logger.info("\nOutput Dataset:")
    logger.info("  Total bar events: %s", f"{len(result_df):,}")
    logger.info("  Unique source tokens: %d", result_df["src_token_id"].n_unique())
    logger.info("  Unique pools: %d", result_df["pool_id"].n_unique())
    logger.info(
        "  Date range: %s to %s",
        result_df["bar_close_timestamp"].min(),
        result_df["bar_close_timestamp"].max(),
    )


def validate_output(
    output_file: Path,
) -> None:
    """Validate the generated output parquet file.

    Checks:
    - File exists
    - Has required columns
    - Data types are correct
    - No unexpected NaN values
    - Statistics look reasonable

    Args:
        output_file: Path to the output parquet file.

    Raises:
        RuntimeError: If validation fails.

    """
    logger.info("Validating output file %s...", output_file)

    if not output_file.exists():
        msg = f"Output file does not exist: {output_file}"
        raise RuntimeError(msg)

    df = pl.read_parquet(output_file)
    logger.info("  Loaded %s rows", f"{len(df):,}")

    # Check required columns
    required_cols = {
        "src_token_id",
        "pool_id",
        "bar_close_timestamp",
        "src_price_usdc",
        "src_fracdiff",
        "dest_fracdiff",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        raise RuntimeError(msg)
    logger.info("  ✓ All required columns present")

    # Check data types
    logger.info("  Column data types:")
    for col in ["src_fracdiff", "dest_fracdiff", "src_price_usdc"]:
        dtype = df.select(col).dtypes[0]
        logger.info("    %s: %s", col, dtype)

    # Check for NaNs in critical columns
    nan_counts: dict[str, int] = {}
    for col_name in ["src_token_id", "pool_id", "src_fracdiff"]:
        nan_count = df.filter(pl.col(col_name).is_null()).height
        nan_counts[col_name] = nan_count
        if nan_count > 0:
            logger.warning("    %s has %d NaN values", col_name, nan_count)
    logger.info("  ✓ NaN check complete: %s", nan_counts)

    # Check src_fracdiff statistics
    fracdiff_stats = df.select(
        [
            pl.col("src_fracdiff").min().alias("min"),
            pl.col("src_fracdiff").max().alias("max"),
            pl.col("src_fracdiff").mean().alias("mean"),
            pl.col("src_fracdiff").std().alias("std"),
        ],
    )
    logger.info("  src_fracdiff statistics:")
    for row in fracdiff_stats.iter_rows(named=True):
        logger.info(
            "    min=%.4f, max=%.4f, mean=%.6f, std=%.6f",
            row["min"],
            row["max"],
            row["mean"],
            row["std"],
        )

    # Check unique token counts
    n_tokens = df["src_token_id"].n_unique()
    n_pools = df["pool_id"].n_unique()
    logger.info("  ✓ Unique source tokens: %d", n_tokens)
    logger.info("  ✓ Unique pools: %d", n_pools)

    # Check date range
    date_min = df["bar_close_timestamp"].min()
    date_max = df["bar_close_timestamp"].max()
    logger.info("  ✓ Date range: %s to %s", date_min, date_max)

    logger.info("✓ Validation passed!")


def main(
    input_file: Path = typer.Option(
        Path("data/usdc_bars.parquet"),
        help="Input usdc_bars.parquet file",
    ),
    output_file: Path = typer.Option(
        Path("data/log_fracdiff_price.parquet"),
        help="Output parquet file path",
    ),
    d_min: float = typer.Option(
        0.0,
        help="Minimum fractional differentiation order",
    ),
    d_max: float = typer.Option(
        1.0,
        help="Maximum fractional differentiation order",
    ),
    d_step: float = typer.Option(
        0.05,
        help="Step size for d search",
    ),
    drop_non_stationary: bool = typer.Option(
        False,
        help="Drop tokens that don't achieve stationarity",
    ),
    validate: bool = typer.Option(
        False,
        help="Run validation on the generated output",
    ),
    verbose: bool = typer.Option(
        False,
        help="Enable verbose logging",
    ),
) -> None:
    """Make target variable stationary using fractional differentiation."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    make_stationary(
        input_file,
        output_file,
        d_min,
        d_max,
        d_step,
        drop_non_stationary=drop_non_stationary,
    )

    if validate:
        validate_output(output_file)


if __name__ == "__main__":
    typer.run(main)
