"""Calculate USDC-denominated prices for all tokens from swap events.

WHY: We need consistent, temporally-aligned prices for all tokens to enable financial
machine learning. Direct USDC pairs provide ground truth, but many important tokens
trade primarily against ETH or WBTC. By maintaining a chronological price cache, we can
infer prices for indirect pairs while preserving temporal consistency.

This approach follows Prado's emphasis on information-driven sampling (AFML Ch. 2) - we
sample prices based on actual swap events (information flow) rather than fixed time
intervals, capturing market microstructure more accurately.

WHAT: Build a complete price time series for all tokens by:
1. Direct calculation: For USDC-paired swaps, calculate token price from swap amounts
2. Cache maintenance: Update price cache with each direct USDC observation
3. Inference: For indirect swaps (e.g., ETH/WBTC), use cached USDC prices to infer
   both tokens
4. Outlier filtering: Remove extreme prices using IQR method to catch data errors

HOW:
1. Process swaps in chronological order (critical for cache consistency)
2. For direct USDC swaps: price = usdc_amount / token_amount (adjusted for decimals)
3. For indirect swaps: use cached prices to cross-reference and infer prices
4. Filter outliers using token-specific IQR bounds and absolute price limits
5. Validate against CoinGecko data for major tokens

INPUT: data/usdc_paired_swaps.parquet (filtered swap events with hex data)
OUTPUT: data/usdc_priced_swaps.parquet (price observations with USDC volume)

References:
- Prado AFML Ch. 2: Information-Driven Bars
- Uniswap V3 pricing: sqrtPriceX96 = sqrt(token1/token0) * 2^96

"""

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict, cast

import polars as pl
import requests
import typer
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

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
    """Filter extreme price outliers per token using IQR method with absolute bounds.

    Remove prices that fall outside 3x the IQR (Interquartile Range) for each token.
    Also applies absolute min/max price bounds to catch extreme outliers that pass
    through IQR filtering (e.g., when Q1-3*IQR is negative).

    Args:
        df_prices: DataFrame with price observations.

    Returns:
        Filtered DataFrame with outliers removed.

    """
    # Absolute price bounds (independent of token-specific IQR)
    # Min: 1e-9 USDC (0.000000001 USDC) - catches extreme low outliers
    # Max: 1e7 USDC (10 million) - catches extreme high outliers
    absolute_min_price = 1e-9
    absolute_max_price = 1e7

    # Absolute volume bounds to catch miscalculations
    # Max: 1e9 USDC (1 billion) per swap - no single swap should exceed this
    absolute_max_volume = 1e9

    # First pass: Apply absolute bounds on both price and volume
    df_abs_filtered = df_prices.filter(
        (pl.col("price_in_usdc") >= absolute_min_price)
        & (pl.col("price_in_usdc") <= absolute_max_price)
        & (pl.col("usdc_volume") <= absolute_max_volume),
    )

    n_abs_removed = len(df_prices) - len(df_abs_filtered)
    if n_abs_removed > 0:
        logger.info(
            "Removed %d observations outside absolute bounds",
            n_abs_removed,
        )
        logger.info(
            "  Price bounds: [%.2e, %.2e]",
            absolute_min_price,
            absolute_max_price,
        )
        logger.info("  Volume bound: %.2e", absolute_max_volume)

        # Report which tokens had extreme outliers
        extreme_outliers = df_prices.filter(
            (pl.col("price_in_usdc") < absolute_min_price)
            | (pl.col("price_in_usdc") > absolute_max_price)
            | (pl.col("usdc_volume") > absolute_max_volume),
        )
        extreme_summary = (
            extreme_outliers.group_by("token_address")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("price_in_usdc").min().alias("min_price"),
                    pl.col("price_in_usdc").max().alias("max_price"),
                ],
            )
            .sort("count", descending=True)
        )

        logger.info("  Tokens with extreme outliers:")
        for row in extreme_summary.head(10).iter_rows(named=True):
            logger.info(
                "    %s: %d outliers (range: %.2e - %.2e)",
                row["token_address"][:10] + "...",
                row["count"],
                row["min_price"],
                row["max_price"],
            )

    # Second pass: Calculate IQR-based thresholds per token
    token_thresholds = (
        df_abs_filtered.group_by("token_address")
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
                # Apply max() to ensure lower_bound is never negative
                pl.max_horizontal(
                    [
                        pl.col("q1") - 3 * pl.col("iqr"),
                        pl.lit(absolute_min_price),
                    ],
                ).alias("lower_bound"),
                pl.min_horizontal(
                    [
                        pl.col("q3") + 3 * pl.col("iqr"),
                        pl.lit(absolute_max_price),
                    ],
                ).alias("upper_bound"),
            ],
        )
    )

    # Join thresholds back to prices
    df_with_thresholds = df_abs_filtered.join(
        token_thresholds,
        on="token_address",
        how="left",
    )

    # Identify IQR outliers
    df_outliers = df_with_thresholds.filter(
        (pl.col("price_in_usdc") < pl.col("lower_bound"))
        | (pl.col("price_in_usdc") > pl.col("upper_bound")),
    )

    # Report IQR outliers by token
    if len(df_outliers) > 0:
        outlier_summary = (
            df_outliers.group_by("token_address")
            .agg(
                [
                    pl.len().alias("count"),
                    pl.col("price_in_usdc").min().alias("min_outlier"),
                    pl.col("price_in_usdc").max().alias("max_outlier"),
                    pl.col("lower_bound").first().alias("lower_bound"),
                    pl.col("upper_bound").first().alias("upper_bound"),
                ],
            )
            .sort("count", descending=True)
        )

        logger.info("Found IQR outliers in %d tokens:", len(outlier_summary))
        for row in outlier_summary.head(10).iter_rows(named=True):
            logger.info(
                "  %s: %d outliers (bounds: %.2e - %.2e, range: %.2e - %.2e)",
                row["token_address"][:10] + "...",
                row["count"],
                row["lower_bound"],
                row["upper_bound"],
                row["min_outlier"],
                row["max_outlier"],
            )

    # Filter out IQR outliers
    df_filtered = df_with_thresholds.filter(
        (pl.col("price_in_usdc") >= pl.col("lower_bound"))
        & (pl.col("price_in_usdc") <= pl.col("upper_bound")),
    ).drop(["q1", "q3", "iqr", "lower_bound", "upper_bound"])

    n_iqr_removed = len(df_abs_filtered) - len(df_filtered)
    total_removed = len(df_prices) - len(df_filtered)
    logger.info(
        "Filtered %d IQR outliers (%.3f%% of data after absolute filtering)",
        n_iqr_removed,
        n_iqr_removed / len(df_abs_filtered) * 100 if len(df_abs_filtered) > 0 else 0,
    )
    logger.info(
        "Total filtered: %d (%.3f%% of original data)",
        total_removed,
        total_removed / len(df_prices) * 100,
    )

    return df_filtered


def validate_output(  # noqa: C901, PLR0912, PLR0915
    df_prices: pl.DataFrame,
) -> None:
    """Validate the output price dataset for quality issues.

    Args:
        df_prices: DataFrame with calculated USDC prices.

    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PRICE DATA VALIDATION")
    logger.info("=" * 70)

    # 1. Check for null values
    logger.info("\n1. Checking for null values...")
    null_counts = df_prices.null_count()
    has_nulls = False
    for col in null_counts.columns:
        null_count = null_counts[col][0]
        if null_count > 0:
            logger.warning("  ⚠️ WARN: %s has %d null values", col, null_count)
            has_nulls = True
    if not has_nulls:
        logger.info("  ✅ OK: No null values found")

    # 2. Check timestamp ordering
    logger.info("\n2. Checking timestamp ordering...")
    sorted_check = df_prices["block_timestamp"].is_sorted()
    if sorted_check:
        logger.info("  ✅ OK: Timestamps are sorted")
    else:
        logger.warning("  ⚠️ WARN: Timestamps not sorted")

    # 3. Check for time gaps
    logger.info("\n3. Checking for large time gaps...")
    time_diffs = df_prices["block_timestamp"].diff().dt.total_seconds()
    max_gap = time_diffs.max()
    if max_gap is not None and max_gap > 3600:  # noqa: PLR2004 More than 1 hour
        max_gap_hours = max_gap / 3600
        logger.info(
            "  i INFO: Max gap between price observations: %.1f hours",
            max_gap_hours,
        )
    else:
        logger.info("  ✅ OK: No significant time gaps")

    # 4. Price range validation
    logger.info("\n4. Validating price ranges...")
    price_stats = df_prices.select(
        [
            pl.col("price_in_usdc").min().alias("min_price"),
            pl.col("price_in_usdc").max().alias("max_price"),
            pl.col("price_in_usdc").median().alias("median_price"),
            pl.col("price_in_usdc").mean().alias("mean_price"),
        ],
    ).row(0, named=True)

    logger.info(
        "  Price range: [%.2e, %.2e]",
        price_stats["min_price"],
        price_stats["max_price"],
    )
    logger.info(
        "  Median: %.2f, Mean: %.2f",
        price_stats["median_price"],
        price_stats["mean_price"],
    )

    # Check for suspiciously extreme prices
    has_price_issues = False
    if price_stats["min_price"] < 1e-6:  # noqa: PLR2004
        logger.warning(
            "  ⚠️  WARN: Minimum price (%.2e) is very low "
            "- possible scam/honeypot tokens",
            price_stats["min_price"],
        )
        has_price_issues = True
    if price_stats["max_price"] > 1e6:  # noqa: PLR2004
        logger.warning(
            "  ⚠️  WARN: Maximum price (%.2e) is very high - review for errors",
            price_stats["max_price"],
        )
        has_price_issues = True

    if not has_price_issues:
        logger.info("  ✅ OK: Price ranges are reasonable")

    # 5. Volume validation
    logger.info("\n5. Validating USDC volumes...")
    volume_stats = df_prices.select(
        [
            pl.col("usdc_volume").min().alias("min_volume"),
            pl.col("usdc_volume").max().alias("max_volume"),
            pl.col("usdc_volume").median().alias("median_volume"),
            pl.col("usdc_volume").sum().alias("total_volume"),
        ],
    ).row(0, named=True)

    logger.info(
        "  Volume range: [%.2f, %.2e]",
        volume_stats["min_volume"],
        volume_stats["max_volume"],
    )
    logger.info(
        "  Median: %.2f, Total: %.2e",
        volume_stats["median_volume"],
        volume_stats["total_volume"],
    )
    logger.info("  ✅ OK: Volume statistics computed successfully")

    # 6. Token-level price volatility analysis
    logger.info("\n6. Analyzing price volatility by token...")
    token_volatility = (
        df_prices.group_by("token_address")
        .agg(
            [
                pl.len().alias("obs_count"),
                pl.col("price_in_usdc").min().alias("min_price"),
                pl.col("price_in_usdc").max().alias("max_price"),
                pl.col("price_in_usdc").mean().alias("mean_price"),
                pl.col("price_in_usdc").std().alias("std_price"),
            ],
        )
        .with_columns(
            [
                # Coefficient of variation (std/mean) - measures relative volatility
                (pl.col("std_price") / pl.col("mean_price")).alias("cv"),
                # Price range ratio (max/min)
                (pl.col("max_price") / pl.col("min_price")).alias("price_range_ratio"),
            ],
        )
        .filter(pl.col("obs_count") >= 10)  # noqa: PLR2004 tokens w/ observations
        .sort("cv", descending=True)
    )

    # Report tokens with extreme volatility
    extreme_volatility = token_volatility.filter(pl.col("cv") > 2.0)  # noqa: PLR2004
    if len(extreme_volatility) > 0:
        logger.warning(
            "  ⚠️ WARN: %d tokens have CV > 2.0 (extreme volatility)",
            len(extreme_volatility),
        )
        for row in extreme_volatility.head(5).iter_rows(named=True):
            logger.warning(
                "    %s: CV=%.2f, range ratio=%.1fx (%.2e - %.2e USDC)",
                row["token_address"][:10] + "...",
                row["cv"],
                row["price_range_ratio"],
                row["min_price"],
                row["max_price"],
            )
    else:
        logger.info("  ✅ OK: No tokens with extreme volatility")

    # 7. Identify tokens with very low prices (potential scam tokens)
    logger.info("\n7. Identifying low-value tokens...")
    low_price_tokens = (
        df_prices.filter(pl.col("price_in_usdc") < 0.0001)  # noqa: PLR2004
        .group_by("token_address")
        .agg(
            [
                pl.len().alias("obs_count"),
                pl.col("price_in_usdc").mean().alias("mean_price"),
                pl.col("usdc_volume").sum().alias("total_volume"),
            ],
        )
        .sort("total_volume", descending=True)
    )

    if len(low_price_tokens) > 0:
        logger.info(
            "  i INFO: %d tokens with price < $0.0001 "
            "(potential scam/meme tokens)",
            len(low_price_tokens),
        )
        logger.info("    Top by volume:")
        for row in low_price_tokens.head(5).iter_rows(named=True):
            logger.info(
                "      %s: avg price %.2e, total volume $%.2f",
                row["token_address"][:10] + "...",
                row["mean_price"],
                row["total_volume"],
            )

    # 8. Distribution summary
    logger.info("\n8. Token distribution summary...")
    logger.info("  Total unique tokens: %d", df_prices["token_address"].n_unique())
    logger.info("  Total price observations: %s", f"{len(df_prices):,}")

    obs_per_token = df_prices.group_by("token_address").agg(pl.len().alias("count"))
    logger.info("  Observations per token:")
    logger.info("    Min: %d", obs_per_token["count"].min())
    logger.info("    Median: %d", obs_per_token["count"].median())
    logger.info("    Max: %d", obs_per_token["count"].max())
    logger.info("  ✅ OK: Distribution analysis complete")

    # Overall validation summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    # Count validation results
    validation_checks = 8
    warnings_found = (
        (1 if has_nulls else 0)
        + (0 if sorted_check else 1)
        + (1 if has_price_issues else 0)
        + (1 if len(extreme_volatility) > 0 else 0)
    )

    if warnings_found == 0:
        logger.info("✅ All %d validation checks passed!", validation_checks)
    else:
        logger.info(
            "⚠️  %d validation checks passed with %d warning(s)",
            validation_checks,
            warnings_found,
        )

    logger.info("")
    logger.info("=" * 70)


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


def calculate_usdc_prices(  # noqa: PLR0915, C901, PLR0912
    input_file: Path,
    output_file: Path,
    *,
    filter_outliers: bool = True,
    use_indirect_swaps: bool = True,
    price_cache_ttl_hours: float = 1.0,
    dry_run: bool = False,
) -> None:
    """Calculate USDC prices from decoded swap data.

    Processes swaps chronologically to build a price cache. For direct USDC swaps,
    calculates prices directly. For indirect swaps (token A <-> token B), uses
    the most recent USDC prices for both tokens to infer their USDC prices.

    Args:
        input_file: Input parquet file with decoded swaps.
        output_file: Output parquet file for price time series.
        filter_outliers: Remove price outliers (IQR method).
        use_indirect_swaps: Enable price inference for indirect swaps (default: True).
        price_cache_ttl_hours: Time-to-live for cached prices in hours (default: 1.0).
        dry_run: Preview processing without saving output (default: False).

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
    if use_indirect_swaps:
        logger.info(
            "Building price cache for indirect swaps (TTL: %.1fh)...",
            price_cache_ttl_hours,
        )
    else:
        logger.info("Processing direct USDC swaps only (indirect inference disabled)")

    # Price cache: token_address -> (USDC price, timestamp)
    # Includes timestamp to detect stale prices
    price_cache: dict[str, tuple[float, datetime]] = {}
    # USDC = 1 USDC
    price_cache[USDC_ADDRESS.lower()] = (1.0, datetime.min.replace(tzinfo=UTC))

    # Sanity check bounds for prices (uppercase for constants)
    MIN_SANE_PRICE = 1e-9  # noqa: N806 0.000000001 USDC
    MAX_SANE_PRICE = 1e7  # noqa: N806 10 million USDC
    MAX_SANE_VOLUME = 1e9  # noqa: N806 1 billion USDC per swap

    # TTL for price cache
    price_cache_ttl = timedelta(hours=price_cache_ttl_hours)

    # Process in batches for memory efficiency
    results = []
    batch_size = 100_000
    indirect_swap_count = 0
    direct_swap_count = 0
    skipped_swap_count = 0
    stale_cache_count = 0
    sanity_check_failures = 0

    num_batches = (len(df) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing swaps...", total=num_batches)

        for i in range(0, len(df), batch_size):
            batch = df.slice(i, batch_size)
            progress.update(task, advance=1)

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
                current_timestamp = row["block_timestamp"]

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
                        price = price_result["price_in_usdc"]
                        volume = price_result["usdc_volume"]

                        # Sanity check: reject extreme prices and volumes
                        if not (MIN_SANE_PRICE <= price <= MAX_SANE_PRICE):
                            sanity_check_failures += 1
                            continue

                        if volume > MAX_SANE_VOLUME:
                            sanity_check_failures += 1
                            continue

                        # Update price cache with timestamp
                        price_cache[price_result["token_address"]] = (
                            price,
                            current_timestamp,
                        )

                        # Add to results
                        decoded_data.append(
                            {
                                "block_timestamp": current_timestamp,
                                "block_number": row["block_number"],
                                "transaction_hash": row["transaction_hash"],
                                "pool": row["pool"],
                                "token_address": price_result["token_address"],
                                "price_in_usdc": price,
                                "usdc_volume": volume,
                            },
                        )
                        direct_swap_count += 1

                # Indirect swap - use price cache to infer prices (if enabled)
                # Both tokens should have USDC prices in cache
                elif (
                    use_indirect_swaps
                    and token0_addr in price_cache
                    and token1_addr in price_cache
                ):
                    # Check if cached prices are still fresh (not stale)
                    token0_price, token0_time = price_cache[token0_addr]
                    token1_price, token1_time = price_cache[token1_addr]

                    time_since_token0 = current_timestamp - token0_time
                    time_since_token1 = current_timestamp - token1_time

                    # Skip if either cached price is stale
                    if (
                        time_since_token0 > price_cache_ttl
                        or time_since_token1 > price_cache_ttl
                    ):
                        stale_cache_count += 1
                        skipped_swap_count += 1
                        continue

                    # Calculate the exchange rate from the swap
                    if amount0 != 0 and amount1 != 0:
                        # Amount in human-readable form
                        amount0_real = abs(amount0) / (10**token0_decimals)
                        amount1_real = abs(amount1) / (10**token1_decimals)

                        # Price of token1 in terms of token0
                        token1_in_token0 = amount0_real / amount1_real

                        # Use cached USDC price for token0 to infer token1's USDC price
                        inferred_token1_price = token1_in_token0 * token0_price

                        # Price of token0 in terms of token1
                        token0_in_token1 = amount1_real / amount0_real

                        # Use cached USDC price for token1 to infer token0's USDC price
                        inferred_token0_price = token0_in_token1 * token1_price

                        # Sanity check inferred prices
                        if not (
                            MIN_SANE_PRICE <= inferred_token0_price <= MAX_SANE_PRICE
                        ):
                            sanity_check_failures += 1
                            skipped_swap_count += 1
                            continue

                        if not (
                            MIN_SANE_PRICE <= inferred_token1_price <= MAX_SANE_PRICE
                        ):
                            sanity_check_failures += 1
                            skipped_swap_count += 1
                            continue

                        # Estimate USDC volume using token0 side only
                        # (avoid double-counting)
                        usdc_volume_estimate = amount0_real * inferred_token0_price

                        # Sanity check volume
                        if usdc_volume_estimate > MAX_SANE_VOLUME:
                            sanity_check_failures += 1
                            skipped_swap_count += 1
                            continue

                        # Update cache with inferred prices and timestamp
                        price_cache[token1_addr] = (
                            inferred_token1_price,
                            current_timestamp,
                        )
                        price_cache[token0_addr] = (
                            inferred_token0_price,
                            current_timestamp,
                        )

                        # Add price observations for BOTH tokens
                        decoded_data.append(
                            {
                                "block_timestamp": current_timestamp,
                                "block_number": row["block_number"],
                                "transaction_hash": row["transaction_hash"],
                                "pool": row["pool"],
                                "token_address": token0_addr,
                                "price_in_usdc": inferred_token0_price,
                                "usdc_volume": usdc_volume_estimate,
                            },
                        )
                        decoded_data.append(
                            {
                                "block_timestamp": current_timestamp,
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
                    # Can't price this swap yet
                    # - missing price data or indirect disabled
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
    logger.info("  Rejected (stale cache): %s", f"{stale_cache_count:,}")
    logger.info("  Rejected (sanity checks): %s", f"{sanity_check_failures:,}")

    # Filter outliers if requested
    if filter_outliers:
        logger.info("Detecting and filtering outliers...")
        df_prices = filter_price_outliers(df_prices)

    # Validate output before saving
    validate_output(df_prices)

    if dry_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DRY RUN - Output not saved")
        logger.info("=" * 70)
        logger.info("Would save to: %s", output_file)
        logger.info("Preview of first 5 rows:")
        print(df_prices.head(5))
        logger.info("")
        logger.info("Preview of last 5 rows:")
        print(df_prices.tail(5))
    else:
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

    logger.info("Done!" if not dry_run else "Dry run completed!")


def main(
    input_file: Path = typer.Option(
        Path("data/usdc_paired_swaps.parquet"),
        "--input-file",
        help="Input parquet file with decoded swaps (from filter_and_decode_swaps.py)",
    ),
    output_file: Path = typer.Option(
        Path("data/usdc_priced_swaps.parquet"),
        "--output-file",
        help="Output parquet file for price time series",
    ),
    filter_outliers: bool = typer.Option(
        True,
        "--filter-outliers/--no-filter-outliers",
        help="Filter extreme price outliers (default: enabled)",
    ),
    use_indirect_swaps: bool = typer.Option(
        True,
        "--use-indirect-swaps/--no-indirect-swaps",
        help="Enable price inference for indirect swaps (default: enabled)",
    ),
    price_cache_ttl_hours: float = typer.Option(
        1.0,
        "--price-cache-ttl-hours",
        help="Time-to-live for cached prices in hours (default: 1.0)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview processing without saving output",
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
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    calculate_usdc_prices(
        input_file,
        output_file,
        filter_outliers=filter_outliers,
        use_indirect_swaps=use_indirect_swaps,
        price_cache_ttl_hours=price_cache_ttl_hours,
        dry_run=dry_run,
    )

    # Run validation if requested
    if validate and not dry_run:
        logger.info("Running price validation...")
        validate_prices(output_file)


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(main)
    app()
