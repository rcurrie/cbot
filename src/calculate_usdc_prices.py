"""Calculate USDC prices from decoded swap data.

Process swaps chronologically and calculate the price of each token relative to USDC.
For USDC-paired swaps, calculate direct price from swap amounts.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import requests
import typer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class PriceObservation(TypedDict):
    """Price observation for a token."""

    token_address: str
    price_in_usdc: float
    usdc_volume: float


# USDC contract address on Ethereum mainnet
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

# CoinGecko API configuration
load_dotenv()
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
VALIDATION_TOKENS_FILE = Path("validation_tokens.json")


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
) -> list[PriceObservation]:
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
    results: list[PriceObservation] = []

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
                results.append(
                    cast(
                        "PriceObservation",
                        {
                            "token_address": token1_addr,
                            "price_in_usdc": price_in_usdc,
                            "usdc_volume": usdc_amount,
                        },
                    ),
                )

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
            results.append(
                cast(
                    "PriceObservation",
                    {
                        "token_address": token0_addr,
                        "price_in_usdc": price_in_usdc,
                        "usdc_volume": usdc_amount,
                    },
                ),
            )

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
    token_thresholds = (
        df_prices.group_by("token_address")
        .agg(
            [
                pl.col("price_in_usdc").quantile(0.25).alias("q1"),
                pl.col("price_in_usdc").quantile(0.75).alias("q3"),
            ],
        )
        .with_columns(
            [
                (pl.col("q3") - pl.col("q1")).alias("iqr"),
            ],
        )
        .with_columns(
            [
                (pl.col("q1") - 3 * pl.col("iqr")).alias("lower_bound"),
                (pl.col("q3") + 3 * pl.col("iqr")).alias("upper_bound"),
            ],
        )
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


def fetch_coingecko_prices(
    token_id: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch historical prices from CoinGecko API with caching.

    Args:
        token_id: CoinGecko token ID (e.g., 'weth').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        Polars DataFrame with timestamps and prices in USD.

    """
    # Create cache directory if it doesn't exist
    cache_dir = Path("data/prices")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache filename based on token_id and date range
    cache_file = cache_dir / f"{token_id}_{start_date}_{end_date}.parquet"

    # Check if cached file exists
    if cache_file.exists():
        logger.info("Loading from cache: %s", cache_file)
        return pl.read_parquet(cache_file)

    logger.info("Fetching from CoinGecko API for %s...", token_id)

    # Convert dates to timestamps
    start_ts = int(
        datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp(),
    )
    end_ts = int(
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp(),
    )

    # Use standard API endpoint with Demo API key header
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
    params: dict[str, str | int] = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts,
    }
    headers = {
        "x-cg-demo-api-key": COINGECKO_API_KEY,  # Demo API key uses this header
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    prices = data.get("prices", [])

    # Convert to polars DataFrame
    df = pl.DataFrame(
        {
            "timestamp": [datetime.fromtimestamp(p[0] / 1000, tz=UTC) for p in prices],
            "price_usd": [p[1] for p in prices],
        },
    )

    # Cache the results
    df.write_parquet(cache_file)
    logger.info("Cached to: %s", cache_file)

    return df


def validate_prices(
    prices_file: Path = Path("data/usdc_prices_timeseries.parquet"),
    start_date: str = "2025-07-01",
    end_date: str = "2025-09-01",
) -> None:
    """Validate calculated USDC prices against CoinGecko data.

    Compare our calculated swap-derived USDC prices against CoinGecko historical
    data for tokens specified in validation_tokens.json.

    Args:
        prices_file: Path to calculated USDC prices parquet file.
        start_date: Start date for CoinGecko data fetch (YYYY-MM-DD).
        end_date: End date for CoinGecko data fetch (YYYY-MM-DD).

    """
    # Load validation tokens configuration
    if not VALIDATION_TOKENS_FILE.exists():
        logger.warning("Validation tokens file not found: %s", VALIDATION_TOKENS_FILE)
        return

    with VALIDATION_TOKENS_FILE.open() as f:
        validation_config = json.load(f)

    # Filter to only validation tokens
    tokens_to_validate = {
        addr: info for addr, info in validation_config.items() if "coingecko_id" in info
    }

    if not tokens_to_validate:
        logger.warning(
            "No tokens with coingecko_id found in %s",
            VALIDATION_TOKENS_FILE,
        )
        return

    logger.info("Validating prices for %d tokens...", len(tokens_to_validate))

    # Load our calculated prices
    if not prices_file.exists():
        logger.warning("Prices file not found: %s", prices_file)
        return

    df_prices = pl.read_parquet(prices_file)

    logger.info("Loaded %d price observations", len(df_prices))
    logger.info("Start: %s", df_prices["block_timestamp"].min())
    logger.info("End: %s", df_prices["block_timestamp"].max())
    logger.info("Unique tokens: %d", df_prices["token_address"].n_unique())

    # Fetch CoinGecko prices for validation tokens
    coingecko_prices: dict[str, pl.DataFrame] = {}

    for address, info in tokens_to_validate.items():
        symbol = info.get("symbol", address[:10])
        try:
            df_cg = fetch_coingecko_prices(info["coingecko_id"], start_date, end_date)
            coingecko_prices[address] = df_cg
            logger.info("  %s: Got %d price points", symbol, len(df_cg))
        except Exception:
            logger.exception("Error fetching %s from CoinGecko", symbol)

    # Calculate correlation and error metrics
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    for address, info in tokens_to_validate.items():
        symbol = info.get("symbol", address[:10])

        df_our = df_prices.filter(pl.col("token_address") == address)

        if len(df_our) == 0:
            logger.info("%s: No swap data available", symbol)
            continue

        if address not in coingecko_prices:
            logger.info("%s: No CoinGecko data available", symbol)
            continue

        # Resample both to hourly and join
        df_our_hourly = (
            df_our.sort("block_timestamp")
            .group_by_dynamic("block_timestamp", every="1h")
            .agg(pl.col("price_in_usdc").mean().alias("swap_price"))
            .with_columns(
                pl.col("block_timestamp").dt.truncate("1h").alias("hour"),
            )
        )

        df_cg = coingecko_prices[address]
        df_cg_hourly = (
            df_cg.with_columns(
                pl.col("timestamp").dt.truncate("1h").alias("hour"),
            )
            .group_by("hour")
            .agg(pl.col("price_usd").mean().alias("cg_price"))
        )

        # Join on hour
        df_joined = df_our_hourly.join(df_cg_hourly, on="hour", how="inner")

        if len(df_joined) > 0:
            # Calculate metrics
            correlation_result = df_joined.select(
                pl.corr("swap_price", "cg_price").alias("corr"),
            )["corr"][0]
            correlation: float = float(
                correlation_result if correlation_result is not None else 0.0,
            )

            # Mean absolute percentage error
            mape_value = (  # type: ignore[operator]
                (df_joined["swap_price"] - df_joined["cg_price"]).abs()
                / df_joined["cg_price"]
            ).mean() * 100
            mape = float(mape_value if mape_value is not None else 0.0)  # type: ignore[arg-type]

            logger.info("%s:", symbol)
            logger.info("  Correlation: %.4f", correlation)
            logger.info("  MAPE: %.2f%%", mape)
            logger.info("  Matched hours: %d", len(df_joined))


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
                    decoded_data.append(
                        {
                            "block_timestamp": row["block_timestamp"],
                            "block_number": row["block_number"],
                            "transaction_hash": row["transaction_hash"],
                            "pool": row["pool"],
                            "token_address": price_result["token_address"],
                            "price_in_usdc": price_result["price_in_usdc"],
                            "usdc_volume": price_result["usdc_volume"],
                        },
                    )
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
                    decoded_data.append(
                        {
                            "block_timestamp": row["block_timestamp"],
                            "block_number": row["block_number"],
                            "transaction_hash": row["transaction_hash"],
                            "pool": row["pool"],
                            "token_address": token1_addr,
                            "price_in_usdc": inferred_token1_price,
                            "usdc_volume": usdc_volume_estimate,
                        },
                    )
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


def main(
    input_file: Path = typer.Option(
        Path("data/usdc_paired_swaps.parquet"),
        "--input-file",
        help="Input parquet file with decoded swaps (from filter_and_decode_swaps.py)",
    ),
    output_file: Path = typer.Option(
        Path("data/usdc_prices_timeseries.parquet"),
        "--output-file",
        help="Output parquet file for price time series",
    ),
    filter_outliers: bool = typer.Option(
        True,
        "--filter-outliers/--no-filter-outliers",
        help="Filter extreme price outliers (default: enabled)",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Validate prices against CoinGecko data after calculation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
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

    # Run validation if requested
    if validate:
        logger.info("Running price validation...")
        validate_prices(output_file)


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(main)
    app()
