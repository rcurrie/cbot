"""Calculate USDC prices from decoded swap data.

Process swaps chronologically and calculate the price of each token relative to USDC.
For USDC-paired swaps, calculate direct price from swap amounts.
"""

import logging
from pathlib import Path

import click
import polars as pl

logger = logging.getLogger(__name__)

# USDC contract address on Ethereum mainnet
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"


def decode_swap_amounts(data_hex: str) -> tuple[int, int]:
    """Decode amount0 and amount1 from Uniswap V3 Swap event data.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        Tuple of (amount0, amount1).

    """
    # Remove 0x prefix
    data = data_hex.removeprefix("0x")

    # Extract 32-byte chunks for amount0 and amount1
    amount0_hex = data[0:64]
    amount1_hex = data[64:128]

    # Decode as signed integers
    amount0 = int.from_bytes(bytes.fromhex(amount0_hex), "big", signed=True)
    amount1 = int.from_bytes(bytes.fromhex(amount1_hex), "big", signed=True)

    return amount0, amount1


def calculate_direct_price_from_swap(
    amount0: int,
    amount1: int,
    token0_addr: str,
    token1_addr: str,
    token0_decimals: int,
    token1_decimals: int,
) -> list[dict]:
    """Calculate token prices in USDC from swap amounts for direct USDC pairs.

    Args:
        amount0: Amount of token0 in the swap.
        amount1: Amount of token1 in the swap.
        token0_addr: Token0 address.
        token1_addr: Token1 address.
        token0_decimals: Token0 decimals.
        token1_decimals: Token1 decimals.

    Returns:
        List of price observation dicts. One observation per token in the swap.

    """
    # Minimum USDC volume threshold (0.01 USDC) to avoid extreme prices from dust swaps
    min_usdc_volume = 0.01
    results = []

    if token0_addr == USDC_ADDRESS.lower():
        # USDC is token0, price token1 in terms of USDC
        if amount1 != 0:
            usdc_amount = abs(amount0) / (10**token0_decimals)

            # Only include if above dust threshold
            if usdc_amount >= min_usdc_volume:
                price_in_usdc = (
                    abs(amount0)
                    * (10**token1_decimals)
                    / (abs(amount1) * (10**token0_decimals))
                )
                results.append({
                    "token_address": token1_addr,
                    "price_in_usdc": price_in_usdc,
                    "usdc_volume": usdc_amount,
                })

    elif token1_addr == USDC_ADDRESS.lower() and amount0 != 0:
        # USDC is token1, price token0 in terms of USDC
        usdc_amount = abs(amount1) / (10**token1_decimals)

        # Only include if above dust threshold
        if usdc_amount >= min_usdc_volume:
            price_in_usdc = (
                abs(amount1)
                * (10**token0_decimals)
                / (abs(amount0) * (10**token1_decimals))
            )
            results.append({
                "token_address": token0_addr,
                "price_in_usdc": price_in_usdc,
                "usdc_volume": usdc_amount,
            })

    return results


def filter_price_outliers(df_prices: pl.DataFrame) -> pl.DataFrame:
    """Filter extreme price outliers per token using IQR method.

    Remove prices that fall outside 3x the IQR (Interquartile Range) for each token.
    This catches both high and low outliers, and is more robust than percentile-based.

    Args:
        df_prices: DataFrame with price observations.

    Returns:
        Filtered DataFrame with outliers removed.

    """
    # Calculate IQR-based thresholds per token
    token_thresholds = df_prices.group_by("token_address").agg(
        [
            pl.col("price_in_usdc").quantile(0.25).alias("q1"),
            pl.col("price_in_usdc").quantile(0.75).alias("q3"),
        ],
    ).with_columns(
        [
            (pl.col("q3") - pl.col("q1")).alias("iqr"),
        ],
    ).with_columns(
        [
            (pl.col("q1") - 3 * pl.col("iqr")).alias("lower_bound"),
            (pl.col("q3") + 3 * pl.col("iqr")).alias("upper_bound"),
        ],
    )

    # Join thresholds back to prices
    df_with_thresholds = df_prices.join(
        token_thresholds,
        on="token_address",
        how="left",
    )

    # Identify outliers (both high and low)
    df_outliers = df_with_thresholds.filter(
        (pl.col("price_in_usdc") < pl.col("lower_bound"))
        | (pl.col("price_in_usdc") > pl.col("upper_bound")),
    )

    # Report outliers by token
    if len(df_outliers) > 0:
        outlier_summary = (
            df_outliers.group_by("token_address")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("price_in_usdc").min().alias("min_outlier"),
                    pl.col("price_in_usdc").max().alias("max_outlier"),
                    pl.col("lower_bound").first().alias("lower_bound"),
                    pl.col("upper_bound").first().alias("upper_bound"),
                ],
            )
            .sort("count", descending=True)
        )

        logger.info("Found outliers in %d tokens:", len(outlier_summary))
        for row in outlier_summary.iter_rows(named=True):
            logger.info(
                "  %s: %d outliers (bounds: %.2e - %.2e, range: %.2e - %.2e)",
                row["token_address"][:10] + "...",
                row["count"],
                row["lower_bound"],
                row["upper_bound"],
                row["min_outlier"],
                row["max_outlier"],
            )

    # Filter out outliers
    df_filtered = df_with_thresholds.filter(
        (pl.col("price_in_usdc") >= pl.col("lower_bound"))
        & (pl.col("price_in_usdc") <= pl.col("upper_bound")),
    ).drop(["q1", "q3", "iqr", "lower_bound", "upper_bound"])

    n_removed = len(df_prices) - len(df_filtered)
    logger.info(
        "Filtered %d outliers (%.3f%% of data)",
        n_removed,
        n_removed / len(df_prices) * 100,
    )

    return df_filtered


def calculate_usdc_prices(  # noqa: PLR0915
    input_file: Path,
    output_file: Path,
    *,
    filter_outliers: bool = True,
) -> None:
    """Calculate USDC prices from decoded swap data.

    Processes swaps chronologically to build a price cache. For direct USDC swaps,
    calculates prices directly. For indirect swaps (token A <-> token B), uses
    the most recent USDC prices for both tokens to infer their USDC prices.

    Args:
        input_file: Input parquet file with decoded swaps.
        output_file: Output parquet file for price time series.
        filter_outliers: Remove price outliers (IQR method).

    """
    logger.info("Loading decoded swaps from %s...", input_file)
    df = pl.read_parquet(input_file)
    logger.info("Loaded %s swaps", f"{len(df):,}")

    # Analyze USDC coverage in input data
    logger.info("\nInput Data Analysis:")
    swaps_with_usdc = df.filter(
        (pl.col("token0") == USDC_ADDRESS.lower())
        | (pl.col("token1") == USDC_ADDRESS.lower()),
    )
    logger.info(
        "  Swaps with direct USDC: %s (%.2f%%)",
        f"{len(swaps_with_usdc):,}",
        len(swaps_with_usdc) / len(df) * 100 if len(df) > 0 else 0,
    )

    # Data is already sorted by timestamp in filter_and_decode_swaps.py
    logger.info("Calculating prices from decoded swap amounts...")
    logger.info("Building price cache for indirect swaps...")

    # Price cache: token_address -> most recent USDC price
    price_cache: dict[str, float] = {}
    price_cache[USDC_ADDRESS.lower()] = 1.0  # USDC = 1 USDC

    # Process in batches for memory efficiency
    results = []
    batch_size = 100_000
    indirect_swap_count = 0
    direct_swap_count = 0
    skipped_swap_count = 0

    for i in range(0, len(df), batch_size):
        batch = df.slice(i, batch_size)
        logger.info(
            "Processing batch %d/%d...",
            i // batch_size + 1,
            (len(df) + batch_size - 1) // batch_size,
        )

        # Calculate prices for this batch
        decoded_data = []
        for row in batch.iter_rows(named=True):
            # Decode swap amounts on-demand
            try:
                amount0, amount1 = decode_swap_amounts(row["data"])
            except (ValueError, IndexError) as e:
                logger.warning(
                    "Failed to decode swap at block %s: %s",
                    row["block_number"],
                    e,
                )
                skipped_swap_count += 1
                continue

            # Get token info
            token0_addr = row["token0"]
            token1_addr = row["token1"]
            token0_decimals = int(row["token0_decimals"])
            token1_decimals = int(row["token1_decimals"])

            # Check if this is a direct USDC swap
            usdc_lower = USDC_ADDRESS.lower()
            is_direct_usdc = usdc_lower in (token0_addr, token1_addr)

            if is_direct_usdc:
                # Direct USDC swap - calculate price directly
                price_results = calculate_direct_price_from_swap(
                    amount0,
                    amount1,
                    token0_addr,
                    token1_addr,
                    token0_decimals,
                    token1_decimals,
                )

                for price_result in price_results:
                    # Update price cache
                    price_cache[price_result["token_address"]] = price_result[
                        "price_in_usdc"
                    ]

                    # Add to results
                    decoded_data.append({
                        "block_timestamp": row["block_timestamp"],
                        "block_number": row["block_number"],
                        "transaction_hash": row["transaction_hash"],
                        "pool": row["pool"],
                        "token_address": price_result["token_address"],
                        "price_in_usdc": price_result["price_in_usdc"],
                        "usdc_volume": price_result["usdc_volume"],
                    })
                    direct_swap_count += 1

            # Indirect swap - use price cache to infer prices
            # Both tokens should have USDC prices in cache
            elif token0_addr in price_cache and token1_addr in price_cache:
                # Calculate the exchange rate from the swap
                if amount0 != 0 and amount1 != 0:
                    # Amount in human-readable form
                    amount0_real = abs(amount0) / (10**token0_decimals)
                    amount1_real = abs(amount1) / (10**token1_decimals)

                    # Price of token1 in terms of token0
                    token1_in_token0 = amount0_real / amount1_real

                    # Use cached USDC price for token0 to infer token1's USDC price
                    token0_usdc_price = price_cache[token0_addr]
                    inferred_token1_price = token1_in_token0 * token0_usdc_price

                    # Update cache with inferred price
                    price_cache[token1_addr] = inferred_token1_price

                    # Estimate USDC volume using token0's price
                    usdc_volume_estimate = amount0_real * token0_usdc_price

                    # Add price observations for both tokens
                    decoded_data.append({
                        "block_timestamp": row["block_timestamp"],
                        "block_number": row["block_number"],
                        "transaction_hash": row["transaction_hash"],
                        "pool": row["pool"],
                        "token_address": token1_addr,
                        "price_in_usdc": inferred_token1_price,
                        "usdc_volume": usdc_volume_estimate,
                    })
                    indirect_swap_count += 1
            else:
                # Can't price this swap yet - missing price data
                skipped_swap_count += 1

        if decoded_data:
            results.append(pl.DataFrame(decoded_data))

    # Concatenate all batches
    logger.info("Concatenating results...")
    df_prices = pl.concat(results) if results else pl.DataFrame()

    logger.info("Calculated %s price observations", f"{len(df_prices):,}")
    logger.info("  Direct USDC swaps: %s", f"{direct_swap_count:,}")
    logger.info("  Indirect swaps (inferred): %s", f"{indirect_swap_count:,}")
    logger.info("  Skipped (no price data): %s", f"{skipped_swap_count:,}")

    # Filter outliers if requested
    if filter_outliers:
        logger.info("Detecting and filtering outliers...")
        df_prices = filter_price_outliers(df_prices)

    # Save the price time series
    logger.info("Saving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_prices.write_parquet(output_file)

    # Print summary statistics
    logger.info("\nSummary:")
    logger.info("  Unique tokens: %d", df_prices["token_address"].n_unique())
    logger.info(
        "  Date range: %s to %s",
        df_prices["block_timestamp"].min(),
        df_prices["block_timestamp"].max(),
    )
    logger.info("  Total USDC volume: %.2f", df_prices["usdc_volume"].sum())

    logger.info("Done!")


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/usdc_paired_swaps.parquet"),
    help="Input parquet file with decoded swaps (from filter_and_decode_swaps.py)",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/usdc_prices_timeseries.parquet"),
    help="Output parquet file for price time series",
)
@click.option(
    "--filter-outliers/--no-filter-outliers",
    default=True,
    help="Filter extreme price outliers (default: enabled)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    input_file: Path,
    output_file: Path,
    *,
    filter_outliers: bool,
    verbose: bool,
) -> None:
    """Calculate USDC prices from decoded swap data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    calculate_usdc_prices(
        input_file,
        output_file,
        filter_outliers=filter_outliers,
    )


if __name__ == "__main__":
    main()
