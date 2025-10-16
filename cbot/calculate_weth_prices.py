"""Calculate WETH prices from decoded swap data.

Process swaps chronologically and calculate the price of each token relative to WETH.
For WETH-paired swaps, calculate direct price from swap amounts.
"""

import json
import logging
from pathlib import Path

import click
import polars as pl

logger = logging.getLogger(__name__)

# WETH contract address on Ethereum mainnet
WETH_ADDRESS = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"

# Constants for int24 decoding (3-byte signed integer)
INT24_SIGN_BIT = 0x800000  # 2^23
INT24_MAX = 0x1000000  # 2^24


def load_token_decimals(pools_file: Path) -> dict[str, int]:
    """Load token decimals from WETH pools file.

    Args:
        pools_file: Path to weth_pools_by_address.json file.

    Returns:
        Dictionary mapping token_address -> decimals.

    """
    with pools_file.open() as f:
        weth_pools = json.load(f)

    decimals_map = {}
    for pool in weth_pools.values():
        for token in pool["tokens"]:
            addr = token["address"].lower()
            decimals_map[addr] = int(token["decimals"])

    logger.info("Loaded decimals for %d tokens", len(decimals_map))
    return decimals_map


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


def calculate_price_from_swap(  # noqa: PLR0913
    amount0: int,
    amount1: int,
    token0_addr: str,
    token1_addr: str,
    token0_decimals: int,
    token1_decimals: int,
) -> tuple[str, float, float] | None:
    """Calculate token price in WETH from swap amounts.

    Args:
        amount0: Amount of token0 in the swap.
        amount1: Amount of token1 in the swap.
        token0_addr: Token0 address.
        token1_addr: Token1 address.
        token0_decimals: Token0 decimals.
        token1_decimals: Token1 decimals.

    Returns:
        Tuple of (token_address, price_in_weth, weth_volume) or None if invalid.

    """
    if token0_addr == WETH_ADDRESS.lower():
        # WETH is token0, price token1 in terms of WETH
        if amount1 == 0:
            return None
        price_in_weth = (
            abs(amount0) * (10 ** token1_decimals) /
            (abs(amount1) * (10 ** token0_decimals))
        )
        weth_amount = abs(amount0) / (10 ** token0_decimals)
        return token1_addr, price_in_weth, weth_amount

    if token1_addr == WETH_ADDRESS.lower():
        # WETH is token1, price token0 in terms of WETH
        if amount0 == 0:
            return None
        price_in_weth = (
            abs(amount1) * (10 ** token0_decimals) /
            (abs(amount0) * (10 ** token1_decimals))
        )
        weth_amount = abs(amount1) / (10 ** token1_decimals)
        return token0_addr, price_in_weth, weth_amount

    return None


def filter_price_outliers(df_prices: pl.DataFrame) -> pl.DataFrame:
    """Filter extreme price outliers per token.

    Remove prices that are > 3x the 99th percentile for each token.
    Reports outliers found for each token.

    Args:
        df_prices: DataFrame with price observations.

    Returns:
        Filtered DataFrame with outliers removed.

    """
    # Calculate outlier threshold per token (3x 99th percentile)
    token_thresholds = (
        df_prices.group_by("token_address")
        .agg(pl.col("price_in_weth").quantile(0.99).alias("p99"))
        .with_columns((pl.col("p99") * 3).alias("outlier_threshold"))
    )

    # Join thresholds back to prices
    df_with_thresholds = df_prices.join(
        token_thresholds,
        on="token_address",
        how="left",
    )

    # Identify outliers
    df_outliers = df_with_thresholds.filter(
        pl.col("price_in_weth") > pl.col("outlier_threshold"),
    )

    # Report outliers by token
    if len(df_outliers) > 0:
        outlier_summary = (
            df_outliers.group_by("token_address")
            .agg([
                pl.count().alias("count"),
                pl.col("price_in_weth").min().alias("min_outlier"),
                pl.col("price_in_weth").max().alias("max_outlier"),
                pl.col("outlier_threshold").first().alias("threshold"),
            ])
            .sort("count", descending=True)
        )

        logger.info("Found outliers in %d tokens:", len(outlier_summary))
        for row in outlier_summary.iter_rows(named=True):
            logger.info(
                "  %s: %d outliers (threshold: %.6f, range: %.2e - %.2e)",
                row["token_address"][:10] + "...",
                row["count"],
                row["threshold"],
                row["min_outlier"],
                row["max_outlier"],
            )

    # Filter out outliers
    df_filtered = df_with_thresholds.filter(
        pl.col("price_in_weth") <= pl.col("outlier_threshold"),
    ).drop(["p99", "outlier_threshold"])

    n_removed = len(df_prices) - len(df_filtered)
    logger.info(
        "Filtered %d outliers (%.3f%% of data)",
        n_removed,
        n_removed / len(df_prices) * 100,
    )

    return df_filtered


def calculate_weth_prices(
    input_file: Path,
    output_file: Path,
    pools_file: Path,
    *,
    filter_outliers: bool = True,
) -> None:
    """Calculate WETH prices from swap data.

    Args:
        input_file: Input parquet file with WETH-paired swaps.
        output_file: Output parquet file for price time series.
        pools_file: JSON file with WETH pool information (for decimals).
        filter_outliers: Remove price outliers (> 3x 99th percentile per token).

    """
    logger.info("Loading token decimals from %s...", pools_file)
    decimals_map = load_token_decimals(pools_file)

    logger.info("Loading WETH-paired swaps from %s...", input_file)
    df = pl.read_parquet(input_file)
    logger.info("Loaded %s swaps", f"{len(df):,}")

    # Sort by timestamp to process chronologically
    logger.info("Sorting swaps chronologically...")
    df = df.sort("block_timestamp")

    logger.info("Decoding swap amounts and calculating prices...")

    # Decode amounts using polars expressions for efficiency
    # We'll use a Python UDF for the decoding, then calculate prices
    df_with_amounts = df.with_columns([
        # Extract token addresses for decimal lookup
        pl.col("token0").alias("token0_addr"),
        pl.col("token1").alias("token1_addr"),
    ])

    # Process in batches for memory efficiency
    results = []
    batch_size = 100_000

    for i in range(0, len(df_with_amounts), batch_size):
        batch = df_with_amounts.slice(i, batch_size)
        logger.info("Processing batch %d/%d...", i // batch_size + 1,
                    (len(df_with_amounts) + batch_size - 1) // batch_size)

        # Decode amounts for this batch
        decoded_data = []
        for row in batch.iter_rows(named=True):
            amount0, amount1 = decode_swap_amounts(row["data"])

            # Get token info
            token0_addr = row["token0_addr"]
            token1_addr = row["token1_addr"]
            token0_decimals = decimals_map.get(token0_addr, 18)
            token1_decimals = decimals_map.get(token1_addr, 18)

            # Calculate price
            result = calculate_price_from_swap(
                amount0, amount1,
                token0_addr, token1_addr,
                token0_decimals, token1_decimals,
            )

            if result is not None:
                token_address, price_in_weth, weth_amount = result
                decoded_data.append({
                    "block_timestamp": row["block_timestamp"],
                    "block_number": row["block_number"],
                    "transaction_hash": row["transaction_hash"],
                    "pool": row["pool"],
                    "token_address": token_address,
                    "price_in_weth": price_in_weth,
                    "weth_volume": weth_amount,
                })

        results.append(pl.DataFrame(decoded_data))

    # Concatenate all batches
    logger.info("Concatenating results...")
    df_prices = pl.concat(results)

    logger.info("Calculated %s price observations", f"{len(df_prices):,}")

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
    logger.info("  Date range: %s to %s",
                df_prices["block_timestamp"].min(),
                df_prices["block_timestamp"].max())
    logger.info("  Total WETH volume: %.2f",
                df_prices["weth_volume"].sum())

    logger.info("Done!")


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_paired_swaps.parquet"),
    help="Input parquet file with WETH-paired swaps",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/weth_prices_timeseries.parquet"),
    help="Output parquet file for price time series",
)
@click.option(
    "--pools-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_pools_by_address.json"),
    help="JSON file with WETH pool information",
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
    pools_file: Path,
    *,
    filter_outliers: bool,
    verbose: bool,
) -> None:
    """Calculate WETH prices from swap data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    calculate_weth_prices(
        input_file,
        output_file,
        pools_file,
        filter_outliers=filter_outliers,
    )


if __name__ == "__main__":
    main()
