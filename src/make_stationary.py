"""Make target variable stationary using fractional differentiation.

This script implements Milestone 2 of Phase 2:
- Load master_message_log.parquet
- Group by token_id
- Apply log transformation to token_close_price_usdc
- Find minimum fractional differentiation order d for each token
- Apply fractional differentiation to achieve stationarity
- Output log_fracdiff_price.parquet
"""

import logging
from pathlib import Path

import click
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller

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
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    weights = np.array(weights[::-1])  # Reverse for convolution
    width = len(weights)

    # Apply fractional differentiation
    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1 : i + 1])

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

    for d in d_values:
        # Test original log_price series or apply fractional differentiation
        fracdiff_series = (
            log_price if d == 0 else frac_diff_fixed(log_price, d)
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
                return d, True
        except Exception as e:
            logger.warning("ADF test failed for d=%.2f: %s", d, e)
            continue

    # No stationary d found
    return d_max, False


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
        input_file: Path to master_message_log.parquet.
        output_file: Path to output log_fracdiff_price.parquet.
        d_min: Minimum fractional differentiation order to search.
        d_max: Maximum fractional differentiation order to search.
        d_step: Step size for d search.
        drop_non_stationary: If True, drop all tokens that don't achieve stationarity.

    """
    logger.info("Loading data from %s...", input_file)
    df = pl.read_parquet(input_file)
    logger.info(
        "Loaded %s messages for %d tokens",
        f"{len(df):,}",
        df["token_id"].n_unique(),
    )

    # Create empty column for y_target_fracdiff
    df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias("y_target_fracdiff")])

    # Sort by token_id and timestamp for proper time series processing
    df = df.sort(["token_id", "bar_close_timestamp"])

    logger.info("\nProcessing tokens to achieve stationarity...")
    logger.info(
        "Testing d values from %.2f to %.2f (step=%.2f)",
        d_min,
        d_max,
        d_step,
    )

    # Process each token group
    all_fracdiff_rows = []
    token_stats = []

    for token_tuple, token_group in df.group_by("token_id", maintain_order=True):
        token_id = token_tuple[0]  # Extract scalar from tuple
        token_df = token_group.sort("bar_close_timestamp")

        # Get price series
        prices = token_df["token_close_price_usdc"].to_numpy()

        # Apply log transformation
        log_prices = np.log(prices)

        # Check if log_prices are valid
        if np.any(np.isnan(log_prices)) or np.any(np.isinf(log_prices)):
            logger.warning("Token %s has invalid log prices, skipping", token_id)
            continue

        # Find minimum d for stationarity
        min_d, is_stationary = find_min_d_for_stationarity(
            log_prices,
            d_min=d_min,
            d_max=d_max,
            d_step=d_step,
        )

        # Skip token if not stationary and drop_non_stationary is True
        if drop_non_stationary and not is_stationary:
            logger.info(
                "Dropping token %s (not stationary, d=%.2f)",
                token_id[:10] + "...",
                min_d,
            )
            token_stats.append(
                {
                    "token_id": token_id,
                    "n_observations": len(token_df),
                    "min_d": min_d,
                    "is_stationary": is_stationary,
                    "n_valid_after_fracdiff": 0,
                    "dropped": True,
                }
            )
            continue

        # Apply fractional differentiation with the found d
        if min_d == 0:
            fracdiff_log_price = log_prices
        else:
            fracdiff_log_price = frac_diff_fixed(log_prices, min_d)

        # Store statistics
        token_stats.append(
            {
                "token_id": token_id,
                "n_observations": len(token_df),
                "min_d": min_d,
                "is_stationary": is_stationary,
                "n_valid_after_fracdiff": np.sum(~np.isnan(fracdiff_log_price)),
                "dropped": False,
            }
        )

        # Add fracdiff values to token_df and collect rows
        all_fracdiff_rows.append(
            token_df.with_columns(
                [pl.Series("y_target_fracdiff", fracdiff_log_price)],
            ),
        )

    # Combine all token dataframes
    logger.info("\nCombining results from all tokens...")
    result_df = pl.concat(all_fracdiff_rows)

    # Drop rows with NaN in y_target_fracdiff
    n_before = len(result_df)
    result_df = result_df.filter(pl.col("y_target_fracdiff").is_not_nan())
    n_after = len(result_df)
    n_dropped = n_before - n_after

    logger.info(
        "Dropped %s rows with NaN values (%.1f%%)",
        f"{n_dropped:,}",
        100 * n_dropped / n_before,
    )

    # Log statistics
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

    # Final output statistics
    logger.info("\nOutput Dataset:")
    logger.info("  Total messages: %s", f"{len(result_df):,}")
    logger.info("  Unique tokens: %d", result_df["token_id"].n_unique())
    logger.info("  Unique pools: %d", result_df["pool_id"].n_unique())
    logger.info(
        "  Date range: %s to %s",
        result_df["bar_close_timestamp"].min(),
        result_df["bar_close_timestamp"].max(),
    )

    # Save output
    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s messages to %s",
        f"{len(result_df):,}",
        output_file.name,
    )


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/master_message_log.parquet"),
    help="Input master_message_log.parquet file",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/log_fracdiff_price.parquet"),
    help="Output parquet file path",
)
@click.option(
    "--d-min",
    type=float,
    default=0.0,
    help="Minimum fractional differentiation order",
)
@click.option(
    "--d-max",
    type=float,
    default=1.0,
    help="Maximum fractional differentiation order",
)
@click.option(
    "--d-step",
    type=float,
    default=0.05,
    help="Step size for d search",
)
@click.option(
    "--drop-non-stationary",
    is_flag=True,
    help="Drop tokens that don't achieve stationarity",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    input_file: Path,
    output_file: Path,
    d_min: float,
    d_max: float,
    d_step: float,
    *,
    drop_non_stationary: bool,
    verbose: bool,
) -> None:
    """Make target variable stationary using fractional differentiation."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    make_stationary(
        input_file,
        output_file,
        d_min,
        d_max,
        d_step,
        drop_non_stationary=drop_non_stationary,
    )


if __name__ == "__main__":
    main()
