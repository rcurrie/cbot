"""Generate pool-level usdc bars with signed net flows from USDC-paired swaps.

This script implements Milestone 1 of Phase 2:
- Groups swaps by pool_id
- Accumulates swaps into dollar bars based on target_usdc_bar_size
- Calculates signed net flows for each token in the bar
- Determines primary flow direction (src -> dest) based on dominant flow
- Generates one output row per bar with src and dest token information
- Outputs usdc_bars.parquet with bar statistics and signed flows
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests
import typer
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# CoinGecko API configuration
load_dotenv()
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
VALIDATION_TOKENS_FILE = Path("validation_tokens.json")

# USDC address (used as numeraire, always $1.00)
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

# Create Typer app for CLI
app = typer.Typer(help="Generate pool-level bars with signed net flows.")


def decode_swap_amounts(data_hex: str) -> tuple[int, int]:
    """Decode amount0 and amount1 from Uniswap V3 Swap event data.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        Tuple of (amount0, amount1) as signed integers.

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


def load_pool_info(pools_file: Path) -> dict[str, dict[str, float | str]]:
    """Load pool information including fee levels.

    Args:
        pools_file: Path to pools.json file.

    Returns:
        Dictionary mapping pool_address -> {fee, token0, token1}.

    """
    with pools_file.open() as f:
        data = json.load(f)
    pool_info = {}
    all_pools = data.get("data", [])

    # Filter to Uniswap V3 Ethereum pools
    uniswap_v3_pools = [
        p
        for p in all_pools
        if p.get("protocol") == "usp3" and p.get("blockchain") == "ethereum"
    ]

    logger.info("Found %d Uni V3 Ethereum pools in pools.json", len(uniswap_v3_pools))

    for pool in uniswap_v3_pools:
        pool_addr = pool["address"].lower()
        tokens = pool.get("tokens", [])

        if len(tokens) == 2:  # noqa: PLR2004
            pool_info[pool_addr] = {
                "fee": float(pool.get("fee", 0.003)),
                "token0": tokens[0]["address"].lower(),
                "token1": tokens[1]["address"].lower(),
            }

    logger.info("Loaded fee info for %d pools", len(pool_info))
    return pool_info


def generate_pool_bars(  # noqa: C901, PLR0912, PLR0915
    swaps_file: Path,
    prices_file: Path,
    pools_file: Path,
    output_file: Path,
    target_usdc_bar_size: float,
) -> None:
    """Generate pool-level bars with signed net flows.

    Args:
        swaps_file: Path to usdc_paired_swaps.parquet (needed for flow direction).
        prices_file: Path to usdc_prices_timeseries.parquet.
        pools_file: Path to pools.json.
        output_file: Path to output usdc_bars.parquet.
        target_usdc_bar_size: Target dollar volume per bar (e.g., 100000 for $100k).

    """
    logger.info("Loading pool information...")
    pool_info = load_pool_info(pools_file)

    logger.info("Loading prices data from %s...", prices_file)
    prices_df = pl.read_parquet(prices_file)
    logger.info("Loaded %s price records", f"{len(prices_df):,}")

    logger.info("Loading swaps data from %s...", swaps_file)
    swaps_df = pl.read_parquet(swaps_file)
    logger.info("Loaded %s swaps", f"{len(swaps_df):,}")

    # Decode swap amounts to determine flow direction (buy vs sell)
    logger.info("Decoding swap amounts for flow direction...")

    def decode_amounts_to_dict(data_hex: str) -> dict[str, int]:
        """Decode amounts and return sign indicators."""
        amount0, amount1 = decode_swap_amounts(data_hex)
        return {
            "amount0_sign": 1 if amount0 > 0 else -1,
            "amount1_sign": 1 if amount1 > 0 else -1,
        }

    decoded = [decode_amounts_to_dict(data) for data in swaps_df["data"]]
    swaps_df = swaps_df.with_columns(
        [
            pl.Series(
                "amount0_sign",
                [d["amount0_sign"] for d in decoded],
                dtype=pl.Int8,
            ),
            pl.Series(
                "amount1_sign",
                [d["amount1_sign"] for d in decoded],
                dtype=pl.Int8,
            ),
        ],
    )

    # Join prices with swaps to get token addresses and flow direction
    data = prices_df.join(
        swaps_df.select(
            [
                "pool",
                "block_timestamp",
                "transaction_hash",
                "token0",
                "token1",
                "amount0_sign",
                "amount1_sign",
            ],
        ),
        on=["pool", "block_timestamp", "transaction_hash"],
        how="inner",
    )

    logger.info("Joined %s price records with swap data", f"{len(data):,}")

    # Calculate signed volume (positive = buying, negative = selling)
    # If token_address == token0, use amount0_sign; otherwise use amount1_sign
    data = data.with_columns(
        [
            pl.when(pl.col("token_address") == pl.col("token0"))
            .then(pl.col("amount0_sign"))
            .otherwise(pl.col("amount1_sign"))
            .alias("flow_sign"),
        ],
    )

    data = data.with_columns(
        [
            (pl.col("usdc_volume") * pl.col("flow_sign")).alias("signed_volume"),
        ],
    )

    logger.info("Calculated flow directions for %s price records", f"{len(data):,}")

    # Sort by pool and timestamp for bar generation
    data = data.sort(["pool", "block_timestamp"])

    logger.info("Generating pool-level bars...")

    # Group by pool and process each pool separately
    all_bar_rows = []
    pools_processed = 0

    for pool_id_tuple, pool_data_df in data.group_by("pool", maintain_order=True):
        pools_processed += 1
        if pools_processed % 1000 == 0:
            logger.info("Processed %d pools...", pools_processed)

        pool_id = pool_id_tuple[0]  # Extract scalar from tuple
        pool_data = pool_data_df.sort("block_timestamp")

        # Get token addresses for this pool
        if pool_id not in pool_info:
            continue

        token_a = pool_info[pool_id]["token0"]
        token_b = pool_info[pool_id]["token1"]

        # Initialize bar accumulation variables
        bar_usdc_volume = 0.0
        bar_price_records = []  # Store price records for the bar
        seen_swaps = set()  # Track unique swaps for tick count
        last_bar_timestamp = None

        for row in pool_data.iter_rows(named=True):
            swap_key = (row["block_timestamp"], row["transaction_hash"])

            # Add this price record to the bar
            bar_price_records.append(row)

            # Add volume for this price record
            bar_usdc_volume += row["usdc_volume"]

            # Track unique swaps for tick count
            seen_swaps.add(swap_key)

            # Check if bar is complete
            if bar_usdc_volume >= target_usdc_bar_size:
                # Bar is complete - generate output rows
                bar_close_timestamp = row["block_timestamp"]
                tick_count = len(seen_swaps)

                # Calculate bar_time_delta_sec
                if last_bar_timestamp is not None:
                    bar_time_delta_sec = (
                        bar_close_timestamp - last_bar_timestamp
                    ).total_seconds()
                else:
                    bar_time_delta_sec = 0.0

                # Calculate signed net flows for each token
                net_flow_a = sum(
                    rec["signed_volume"]
                    for rec in bar_price_records
                    if rec["token_address"] == token_a
                )
                net_flow_b = sum(
                    rec["signed_volume"]
                    for rec in bar_price_records
                    if rec["token_address"] == token_b
                )

                # Get final prices (last price for each token in the bar)
                # USDC is always $1.00 (it's the numeraire)
                token_a_price = 1.0 if token_a == USDC_ADDRESS else None
                token_b_price = 1.0 if token_b == USDC_ADDRESS else None

                for rec in reversed(bar_price_records):
                    if rec["token_address"] == token_a and token_a_price is None:
                        token_a_price = rec["price_in_usdc"]
                    if rec["token_address"] == token_b and token_b_price is None:
                        token_b_price = rec["price_in_usdc"]
                    if token_a_price is not None and token_b_price is not None:
                        break

                # Determine flow direction (src -> dest) based on dominant flow
                # The token with larger absolute flow is the source
                if abs(net_flow_a) > abs(net_flow_b):
                    src_token = token_a
                    dest_token = token_b
                    src_flow = net_flow_a
                    dest_flow = net_flow_b
                    src_price = token_a_price
                    dest_price = token_b_price
                else:
                    src_token = token_b
                    dest_token = token_a
                    src_flow = net_flow_b
                    dest_flow = net_flow_a
                    src_price = token_b_price
                    dest_price = token_a_price

                # Generate single row for this bar (only if we have both prices)
                if src_price is not None and dest_price is not None:
                    all_bar_rows.append(
                        {
                            "bar_close_timestamp": bar_close_timestamp,
                            "pool_id": pool_id,
                            "src_token_id": src_token,
                            "dest_token_id": dest_token,
                            "src_flow_usdc": src_flow,
                            "dest_flow_usdc": dest_flow,
                            "src_price_usdc": src_price,
                            "dest_price_usdc": dest_price,
                            "bar_time_delta_sec": bar_time_delta_sec,
                            "tick_count": tick_count,
                        },
                    )

                # Reset for next bar
                last_bar_timestamp = bar_close_timestamp
                bar_usdc_volume = 0.0
                bar_price_records = []
                seen_swaps = set()

    logger.info(
        "Generated %d bar messages from %d pools",
        len(all_bar_rows),
        pools_processed,
    )

    # Check if we generated any bars
    if len(all_bar_rows) == 0:
        logger.error(
            "No bars were generated! Check your input data and target_usdc_bar_size.",
        )
        msg = "No bars generated - possibly target_usdc_bar_size is too large for the data"
        raise ValueError(msg)

    # Create output DataFrame
    output_df = pl.DataFrame(all_bar_rows)

    # Ensure flow columns are Float64 (both src and dest should be consistent)
    # Note: dest_flow can be zero, which causes Polars to infer Int64; we explicitly cast to Float64
    output_df = output_df.with_columns(
        [
            pl.col("src_flow_usdc").cast(pl.Float64),
            pl.col("dest_flow_usdc").cast(pl.Float64),
        ],
    )

    # Sort by timestamp
    output_df = output_df.sort("bar_close_timestamp")

    logger.info("\nBar Statistics:")
    logger.info("  Total messages: %s", f"{len(output_df):,}")
    logger.info("  Unique pools: %d", output_df["pool_id"].n_unique())
    # Count unique tokens (both src and dest)
    unique_tokens = set(output_df["src_token_id"].unique()) | set(
        output_df["dest_token_id"].unique(),
    )
    logger.info("  Unique tokens: %d", len(unique_tokens))
    logger.info(
        "  Date range: %s to %s",
        output_df["bar_close_timestamp"].min(),
        output_df["bar_close_timestamp"].max(),
    )

    # Analyze coverage of indirect (non-USDC) swaps
    _log_indirect_swap_analysis(prices_file, swaps_file, output_df)

    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s messages to usdc_bars.parquet",
        f"{len(output_df):,}",
    )


def _log_indirect_swap_analysis(
    prices_file: Path,
    swaps_file: Path,
    output_df: pl.DataFrame,
) -> None:
    """Analyze coverage of indirect (non-USDC) swaps in generated bars.

    Check if bars are missing indirect swap data and quantify the gap.

    Args:
        prices_file: Path to prices file.
        swaps_file: Path to swaps file.
        output_df: Generated bars DataFrame.

    """
    logger.info("\n=== INDIRECT SWAP COVERAGE ANALYSIS ===")

    usdc_lower = USDC_ADDRESS.lower()

    # Load data
    prices_df = pl.read_parquet(prices_file)
    swaps_df = pl.read_parquet(swaps_file)

    # Join prices with swaps to see all price records and their swap contexts
    joined = prices_df.join(
        swaps_df.select(
            ["pool", "block_timestamp", "transaction_hash", "token0", "token1"],
        ),
        on=["pool", "block_timestamp", "transaction_hash"],
        how="inner",
    )

    # Classify by swap type
    direct_usdc = joined.filter(
        (pl.col("token0") == usdc_lower) | (pl.col("token1") == usdc_lower),
    )
    indirect_only = joined.filter(
        (pl.col("token0") != usdc_lower) & (pl.col("token1") != usdc_lower),
    )

    logger.info(
        "Price records from direct USDC swaps: %s",
        f"{len(direct_usdc):,}",
    )
    logger.info(
        "Price records from indirect swaps (non-USDC pairs): %s",
        f"{len(indirect_only):,}",
    )

    if len(indirect_only) > 0:
        indirect_pool_ids = set(indirect_only["pool"].unique())
        bars_pool_ids = set(output_df["pool_id"].unique())
        overlap = indirect_pool_ids & bars_pool_ids

        logger.info(
            "  Pools with indirect swaps: %d",
            len(indirect_pool_ids),
        )
        logger.info(
            "  Pools that generated bars: %d",
            len(bars_pool_ids),
        )
        logger.info(
            "  Overlap (indirect swaps in pools with bars): %d",
            len(overlap),
        )

        if len(overlap) == 0:
            logger.warning(
                "  ⚠ NO INDIRECT SWAP POOLS generated bars! "
                "All %s indirect swap price records were lost.",
                f"{len(indirect_only):,}",
            )
            logger.warning(
                "  This suggests all bars come exclusively from USDC-paired swaps.",
            )
        else:
            logger.info(
                "  ✓ Some indirect swap pools have bars: %d pools",
                len(overlap),
            )

    # Token pair coverage
    logger.info("\n=== TOKEN PAIR TYPES IN BARS ===")
    bars_with_usdc = len(
        output_df.filter(
            (pl.col("src_token_id") == usdc_lower)
            | (pl.col("dest_token_id") == usdc_lower),
        ),
    )
    bars_without_usdc = len(
        output_df.filter(
            (pl.col("src_token_id") != usdc_lower)
            & (pl.col("dest_token_id") != usdc_lower),
        ),
    )

    logger.info(
        "  Bars with USDC: %s (%.1f%%)",
        f"{bars_with_usdc:,}",
        bars_with_usdc / len(output_df) * 100 if len(output_df) > 0 else 0,
    )
    logger.info(
        "  Bars without USDC: %s (%.1f%%)",
        f"{bars_without_usdc:,}",
        bars_without_usdc / len(output_df) * 100 if len(output_df) > 0 else 0,
    )

    if bars_without_usdc == 0 and len(indirect_only) > 0:
        logger.warning(
            "  ⚠ No bars between non-USDC token pairs despite %s "
            "indirect swap price records available!",
            f"{len(indirect_only):,}",
        )
    elif bars_without_usdc > 0:
        logger.info("  ✓ Non-USDC token pair bars present: good for TGNN!")


def _log_null_checks(df_bars: pl.DataFrame) -> bool:
    """Log null value checks and return True if nulls found.

    Args:
        df_bars: DataFrame to check.

    Returns:
        True if any nulls found, False otherwise.

    """
    logger.info("=== NULL VALUE CHECKS ===")
    null_counts = df_bars.null_count()
    has_nulls = False
    for col in null_counts.columns:
        count = null_counts[col][0]
        if count > 0:
            logger.warning("  ⚠ %s: %d nulls", col, count)
            has_nulls = True
        else:
            logger.info("  ✓ %s: no nulls", col)
    return has_nulls


def _log_volume_checks(df_bars: pl.DataFrame) -> bool:
    """Log volume validation checks.

    Args:
        df_bars: DataFrame to check.

    Returns:
        True if critical issues found (all zeros or NaN), False otherwise.

    """
    logger.info("\n=== VOLUME CHECKS ===")

    # Check src flow distribution
    src_positive = df_bars.filter(pl.col("src_flow_usdc") > 0)
    src_negative = df_bars.filter(pl.col("src_flow_usdc") < 0)
    src_zero = df_bars.filter(pl.col("src_flow_usdc") == 0)

    logger.info("  Source flow distribution:")
    logger.info(
        "    Positive: %d (%.2f%%)",
        len(src_positive),
        len(src_positive) / len(df_bars) * 100,
    )
    logger.info(
        "    Negative: %d (%.2f%%)",
        len(src_negative),
        len(src_negative) / len(df_bars) * 100,
    )
    logger.info(
        "    Zero: %d (%.2f%%)",
        len(src_zero),
        len(src_zero) / len(df_bars) * 100,
    )

    # Check dest flow distribution
    dest_positive = df_bars.filter(pl.col("dest_flow_usdc") > 0)
    dest_negative = df_bars.filter(pl.col("dest_flow_usdc") < 0)
    dest_zero = df_bars.filter(pl.col("dest_flow_usdc") == 0)

    logger.info("  Destination flow distribution:")
    logger.info(
        "    Positive: %d (%.2f%%)",
        len(dest_positive),
        len(dest_positive) / len(df_bars) * 100,
    )
    logger.info(
        "    Negative: %d (%.2f%%)",
        len(dest_negative),
        len(dest_negative) / len(df_bars) * 100,
    )
    logger.info(
        "    Zero: %d (%.2f%%)",
        len(dest_zero),
        len(dest_zero) / len(df_bars) * 100,
    )

    # Critical issue: all flows are zero (no trading activity)
    has_critical_issue = len(src_zero) == len(df_bars) and len(dest_zero) == len(
        df_bars,
    )
    if has_critical_issue:
        logger.warning("  ⚠ All bars have zero flows (no trading activity)")
    else:
        logger.info("  ✓ Trading activity detected in bars")

    return has_critical_issue


def _log_price_checks(df_bars: pl.DataFrame) -> bool:
    """Log price validation checks.

    Args:
        df_bars: DataFrame to check.

    Returns:
        True if negative prices found, False otherwise.

    """
    negative_src_prices = df_bars.filter(pl.col("src_price_usdc") < 0)
    negative_dest_prices = df_bars.filter(pl.col("dest_price_usdc") < 0)
    has_negative_prices = len(negative_src_prices) > 0 or len(negative_dest_prices) > 0

    if has_negative_prices:
        logger.warning(
            "  ⚠ Found %d bars with negative src prices, %d with negative dest prices",
            len(negative_src_prices),
            len(negative_dest_prices),
        )
    else:
        logger.info("  ✓ All prices are non-negative")

    return has_negative_prices


def _log_statistics(df_bars: pl.DataFrame) -> None:
    """Log price and volume statistics.

    Args:
        df_bars: DataFrame to analyze.

    """
    logger.info("\n=== PRICE STATISTICS ===")
    logger.info("  Source token prices:")
    src_price_stats = df_bars.select(
        [
            pl.col("src_price_usdc").min().alias("min"),
            pl.col("src_price_usdc").max().alias("max"),
            pl.col("src_price_usdc").mean().alias("mean"),
            pl.col("src_price_usdc").std().alias("std"),
        ],
    )
    for col in src_price_stats.columns:
        val = src_price_stats[col][0]
        logger.info("    %s: %.6f", col, val)

    logger.info("  Destination token prices:")
    dest_price_stats = df_bars.select(
        [
            pl.col("dest_price_usdc").min().alias("min"),
            pl.col("dest_price_usdc").max().alias("max"),
            pl.col("dest_price_usdc").mean().alias("mean"),
            pl.col("dest_price_usdc").std().alias("std"),
        ],
    )
    for col in dest_price_stats.columns:
        val = dest_price_stats[col][0]
        logger.info("    %s: %.6f", col, val)

    logger.info("\n=== VOLUME STATISTICS ===")
    logger.info("  Source flow:")
    src_vol_stats = df_bars.select(
        [
            pl.col("src_flow_usdc").min().alias("min"),
            pl.col("src_flow_usdc").max().alias("max"),
            pl.col("src_flow_usdc").mean().alias("mean"),
            pl.col("src_flow_usdc").median().alias("median"),
            pl.col("src_flow_usdc").std().alias("std"),
        ],
    )
    for col in src_vol_stats.columns:
        val = src_vol_stats[col][0]
        logger.info("    %s: %.6f", col, val)

    logger.info("  Destination flow:")
    dest_vol_stats = df_bars.select(
        [
            pl.col("dest_flow_usdc").min().alias("min"),
            pl.col("dest_flow_usdc").max().alias("max"),
            pl.col("dest_flow_usdc").mean().alias("mean"),
            pl.col("dest_flow_usdc").median().alias("median"),
            pl.col("dest_flow_usdc").std().alias("std"),
        ],
    )
    for col in dest_vol_stats.columns:
        val = dest_vol_stats[col][0]
        logger.info("    %s: %.6f", col, val)


def _log_summary(df_bars: pl.DataFrame) -> None:
    """Log summary statistics.

    Args:
        df_bars: DataFrame to summarize.

    """
    logger.info("\n=== SUMMARY ===")
    logger.info("  Total records: %s", f"{len(df_bars):,}")
    logger.info("  Unique pools: %d", df_bars["pool_id"].n_unique())
    # Count unique tokens (both src and dest)
    unique_tokens = set(df_bars["src_token_id"].unique()) | set(
        df_bars["dest_token_id"].unique(),
    )
    logger.info("  Unique tokens: %d", len(unique_tokens))
    logger.info(
        "  Date range: %s to %s",
        df_bars["bar_close_timestamp"].min(),
        df_bars["bar_close_timestamp"].max(),
    )


def fetch_coingecko_prices(
    token_id: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    """Fetch historical prices from CoinGecko API with caching.

    Args:
        token_id: CoinGecko token ID (e.g., 'usd-coin').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with timestamp and price_usd columns.

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

    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
    params: dict[str, int | str] = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts,
    }
    headers = {
        "x-cg-demo-api-key": COINGECKO_API_KEY,
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


def _validate_against_coingecko(
    df_bars: pl.DataFrame,
    tokens_to_validate: dict[str, dict[str, str | float]],
) -> None:
    """Validate volume bars against CoinGecko data.

    Args:
        df_bars: DataFrame with bar data.
        tokens_to_validate: Dict of token addresses to validation config.

    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("VOLUME CORRELATION WITH COINGECKO")
    logger.info("=" * 70)

    coingecko_prices: dict[str, pl.DataFrame] = {}
    for address, info in tokens_to_validate.items():
        symbol = info.get("symbol", address[:10])
        try:
            df_cg = fetch_coingecko_prices(
                str(info["coingecko_id"]),
                "2025-07-01",
                "2025-09-01",
            )
            coingecko_prices[address] = df_cg
            logger.info("  %s: Got %d price points", symbol, len(df_cg))
        except Exception:
            logger.exception("Error fetching %s from CoinGecko", symbol)

    # Compare volumes for each token
    logger.info("")
    for address, info in tokens_to_validate.items():
        token_name = str(info.get("symbol", address[:10]))
        logger.info("%s:", token_name)

        # Get our volume bars for this token (check both src and dest)
        df_our_bars_src = df_bars.filter(pl.col("src_token_id") == address)
        df_our_bars_dest = df_bars.filter(pl.col("dest_token_id") == address)

        if len(df_our_bars_src) == 0 and len(df_our_bars_dest) == 0:
            logger.info("  ⚠ No volume bars in our data")
            continue

        if address not in coingecko_prices:
            logger.info("  ⚠ No CoinGecko data fetched")
            continue

        df_cg = coingecko_prices[address]

        # Aggregate our bars to calendar-day volumes (combine src and dest)
        df_our_daily_src = (
            df_our_bars_src.with_columns(
                pl.col("bar_close_timestamp").dt.truncate("1d").alias("date"),
            )
            .group_by("date")
            .agg(
                [
                    pl.col("src_price_usdc").mean().alias("avg_price"),
                ],
            )
            .with_columns(pl.col("date").cast(pl.Date))
        )

        df_our_daily_dest = (
            df_our_bars_dest.with_columns(
                pl.col("bar_close_timestamp").dt.truncate("1d").alias("date"),
            )
            .group_by("date")
            .agg(
                [
                    pl.col("dest_price_usdc").mean().alias("avg_price"),
                ],
            )
            .with_columns(pl.col("date").cast(pl.Date))
        )

        # Combine and average prices from both src and dest appearances
        df_our_daily = (
            pl.concat([df_our_daily_src, df_our_daily_dest])
            .group_by("date")
            .agg(
                pl.col("avg_price").mean().alias("avg_price"),
            )
            .sort("date")
        )

        # Aggregate CoinGecko to calendar-day
        df_cg_daily = (
            df_cg.with_columns(
                pl.col("timestamp").dt.truncate("1d").alias("date"),
            )
            .group_by("date")
            .agg(
                [
                    pl.col("price_usd").mean().alias("cg_price"),
                ],
            )
            .with_columns(pl.col("date").cast(pl.Date))
            .sort("date")
        )

        # Join on calendar date
        df_joined = df_our_daily.join(df_cg_daily, on="date", how="inner")

        if len(df_joined) == 0:
            logger.info("  ⚠ No overlapping dates")
            continue

        # Price correlation (our avg price vs CoinGecko price)
        price_corr_result = df_joined.select(
            pl.corr("avg_price", "cg_price").alias("corr"),
        )["corr"][0]
        price_corr: float = (
            float(price_corr_result) if price_corr_result is not None else 0.0
        )

        # Price MAPE
        price_mape_calc = (
            (df_joined["avg_price"] - df_joined["cg_price"]).abs()
            / (df_joined["cg_price"] + 1e-9)
            * 100
        ).mean()
        price_mape: float = price_mape_calc if price_mape_calc is not None else 0.0  # type: ignore[assignment]

        min_date = df_joined["date"].min()
        max_date = df_joined["date"].max()
        logger.info("  Period: %s to %s", min_date, max_date)
        logger.info("  Days matched: %d", len(df_joined))
        logger.info("  Price correlation: %.4f", price_corr)
        logger.info("  Price MAPE: %.2f%%", price_mape)

    logger.info("=" * 70)
    logger.info("VALIDATION INTERPRETATION")
    logger.info("=" * 70)
    logger.info("  - Price correlation > 0.95: prices align ✓")
    logger.info("  - Price MAPE < 5%%: prices are accurate ✓")


def validate_volume_bars(output_file: Path) -> None:
    """Validate generated volume bars for data quality and external correlation.

    Performs both internal data quality checks and validation against CoinGecko
    data for tokens specified in validation_tokens.json.

    Args:
        output_file: Path to the generated usdc_bars.parquet file.

    """
    logger.info("Validating volume bars...")

    if not output_file.exists():
        logger.error("Output file %s not found", output_file)
        msg = f"Output file not found: {output_file}"
        raise FileNotFoundError(msg)

    df_bars = pl.read_parquet(output_file)
    logger.info("Loaded %d bar records for validation", len(df_bars))

    # Internal data quality checks
    has_nulls = _log_null_checks(df_bars)
    has_volume_issues = _log_volume_checks(df_bars)
    has_negative_prices = _log_price_checks(df_bars)
    _log_statistics(df_bars)
    _log_summary(df_bars)

    # External correlation validation with CoinGecko
    if not VALIDATION_TOKENS_FILE.exists():
        logger.warning("Validation tokens file not found: %s", VALIDATION_TOKENS_FILE)
        validation_ok = not (has_nulls or has_volume_issues or has_negative_prices)
        if validation_ok:
            logger.info("\n✓ All internal validation checks passed!")
        else:
            logger.warning("\n⚠ Some internal validation checks failed")
        return

    with VALIDATION_TOKENS_FILE.open() as f:
        validation_config = json.load(f)

    # Filter to only validation tokens with CoinGecko IDs
    tokens_to_validate = {
        addr: info for addr, info in validation_config.items() if "coingecko_id" in info
    }

    if not tokens_to_validate:
        logger.warning(
            "No tokens with coingecko_id found in %s",
            VALIDATION_TOKENS_FILE,
        )
        validation_ok = not (has_nulls or has_volume_issues or has_negative_prices)
        if validation_ok:
            logger.info("\n✓ All internal validation checks passed!")
        else:
            logger.warning("\n⚠ Some internal validation checks failed")
        return

    logger.info(
        "\n=== COINGECKO CORRELATION VALIDATION ===",
    )
    logger.info("Validating %d tokens against CoinGecko...", len(tokens_to_validate))

    _validate_against_coingecko(df_bars, tokens_to_validate)

    validation_ok = not (has_nulls or has_volume_issues or has_negative_prices)
    if validation_ok:
        logger.info("\n✓ All validation checks passed!")
    else:
        logger.warning("\n⚠ Some validation checks failed")


@app.command()
def main(
    swaps_file: Path = typer.Option(
        Path("data/usdc_paired_swaps.parquet"),
        "--swaps-file",
        help="Input swaps parquet file",
    ),
    prices_file: Path = typer.Option(
        Path("data/usdc_priced_swaps.parquet"),
        "--prices-file",
        help="Input prices parquet file",
    ),
    pools_file: Path = typer.Option(
        Path("data/pools.json"),
        "--pools-file",
        help="JSON file with pool information",
    ),
    output_file: Path = typer.Option(
        Path("data/usdc_bars.parquet"),
        "--output-file",
        help="Output parquet file path",
    ),
    target_usdc_bar_size: float = typer.Option(
        100000.0,
        "--target-usdc-bar-size",
        help="Target USDC volume per bar (default: $100k)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Run validation checks on generated bars after generation",
    ),
) -> None:
    """Generate pool-level bars with signed net flows."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_pool_bars(
        swaps_file,
        prices_file,
        pools_file,
        output_file,
        target_usdc_bar_size,
    )

    if validate:
        validate_volume_bars(output_file)


if __name__ == "__main__":
    app()
