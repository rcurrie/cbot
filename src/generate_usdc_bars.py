"""Generate information-driven dollar bars with directional flow from swap events.

WHY: Traditional time-based sampling (e.g., daily OHLCV) ignores information arrival
rate. Prado (AFML Ch. 2) advocates for information-driven bars that sample when
meaningful activity occurs. Dollar bars accumulate swaps until a threshold dollar
volume is reached, creating bars that are information-homogeneous rather than
time-homogeneous. This is crucial for ML as each bar represents similar "information
content" regardless of calendar time elapsed.

WHAT: Transform swap-level price observations into pool-level dollar bars where:
1. Bars close when cumulative USDC volume reaches ADAPTIVE threshold (0.1% daily volume)
2. Each bar captures signed net flows (buy/sell pressure) for both tokens
3. Primary flow direction (src → dest) determined by dominant volume
4. Time delta between bars captures liquidity dynamics
5. Token-specific thresholds ensure information homogeneity across market caps

**ADAPTIVE BAR SIZING (Phase 1 Improvement)**:
Unlike fixed $100k thresholds, adaptive bars scale with token liquidity:
- High-volume tokens (WETH, USDC): Large bars ($200k-$1M) = minutes
- Low-volume tokens (small-caps): Small bars ($10k-$50k) = hours
This prevents heterogeneity where USDC bars = noise, small-cap bars = gaps.

This creates a feature-rich representation for temporal graph neural networks, where
each bar becomes a directed edge in the token graph with flow magnitudes as features.

HOW:
1. Calculate token daily volumes from historical data
2. Set pool threshold = 0.1% of max(token_a_volume, token_b_volume)
3. Clamp thresholds to [$10k, $1M] for stability
4. Group price observations by pool and process chronologically
5. Accumulate USDC volume and signed flows until pool-specific threshold reached
6. Calculate bar features: time delta, tick count, net flows, closing prices
7. Assign src/dest based on which token has larger absolute flow
8. Filter illiquid tokens (>20% bars with >1 day time delta)
9. Validate output and optionally compare to CoinGecko ground truth

INPUT: data/usdc_priced_swaps.parquet (price time series)
OUTPUT: data/usdc_bars.parquet (dollar bars with directional flows)

References:
- Prado AFML Ch. 2.3: Dollar Bars (information-driven sampling)
- Prado AFML Ch. 2.4: Imbalance Bars (buy/sell flow asymmetry)
- Phase 1 Plan: Adaptive thresholds for information homogeneity

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

# Time delta constants (in seconds)
MAX_BAR_TIME_DELTA_SEC = 604800  # 7 days - cap for extreme deltas
ILLIQUIDITY_TIME_THRESHOLD_SEC = 86400  # 1 day - threshold for illiquid tokens
ILLIQUIDITY_PCT_THRESHOLD = 0.20  # 20% - percent of bars with extreme deltas to filter

# Adaptive bar threshold constants
DEFAULT_BAR_SIZE = 100_000.0  # Default $100k bar size (fallback)
ADAPTIVE_BAR_FRACTION = 0.001  # 0.1% of daily volume per bar
MIN_BAR_SIZE = 10_000.0  # Minimum $10k bar (prevent too-small bars)
MAX_BAR_SIZE = 1_000_000.0  # Maximum $1M bar (prevent too-large bars)

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

    Raises:
        FileNotFoundError: If pools_file doesn't exist.
        ValueError: If pools.json has invalid structure.
        json.JSONDecodeError: If pools.json is not valid JSON.

    """
    if not pools_file.exists():
        msg = f"Pools file not found: {pools_file}"
        raise FileNotFoundError(msg)

    try:
        with pools_file.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.exception("Failed to parse pools.json - file may be corrupted")
        raise

    # Validate structure
    if not isinstance(data, dict):
        msg = f"Invalid pools.json structure: expected dict, got {type(data)}"
        raise TypeError(msg)

    if "data" not in data:
        msg = "Invalid pools.json structure: missing 'data' key"
        raise ValueError(msg)

    if not isinstance(data["data"], list):
        data_type = type(data["data"])
        msg = f"Invalid pools.json structure: 'data' should be list, got {data_type}"
        raise TypeError(msg)

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

        if len(tokens) == 2:
            pool_info[pool_addr] = {
                "fee": float(pool.get("fee", 0.003)),
                "token0": tokens[0]["address"].lower(),
                "token1": tokens[1]["address"].lower(),
            }

    logger.info("Loaded fee info for %d pools", len(pool_info))
    return pool_info


def calculate_token_daily_volumes(prices_df: pl.DataFrame) -> dict[str, float]:
    """Calculate average daily USDC volume per token across the dataset.

    This provides token-level liquidity statistics used for adaptive bar sizing.
    Higher volume tokens get larger bar thresholds to maintain information
    homogeneity across the temporal graph (Prado AFML Ch. 2.3).

    Args:
        prices_df: DataFrame with token_address and usdc_volume columns.

    Returns:
        Dictionary mapping token_address -> avg_daily_volume_usdc.

    """
    logger.info("Calculating token daily volumes for adaptive bar sizing...")

    # Group by token and date, sum volume per day
    daily_volumes = (
        prices_df.with_columns(
            pl.col("block_timestamp").dt.truncate("1d").alias("date"),
        )
        .group_by(["token_address", "date"])
        .agg(pl.col("usdc_volume").sum().alias("daily_volume"))
    )

    # Calculate average daily volume per token
    token_avg_volumes = (
        daily_volumes.group_by("token_address")
        .agg(pl.col("daily_volume").mean().alias("avg_daily_volume"))
        .to_dict(as_series=False)
    )

    # Convert to dictionary
    volume_map = dict(
        zip(
            token_avg_volumes["token_address"],
            token_avg_volumes["avg_daily_volume"],
            strict=False,
        ),
    )

    # Log statistics
    volumes = list(volume_map.values())
    if len(volumes) > 0:
        logger.info("  Tokens analyzed: %d", len(volumes))
        logger.info("  Min daily volume: $%s", f"{min(volumes):,.0f}")
        logger.info("  Max daily volume: $%s", f"{max(volumes):,.0f}")
        median_vol = sorted(volumes)[len(volumes) // 2]
        logger.info("  Median daily volume: $%s", f"{median_vol:,.0f}")

    return volume_map


def get_adaptive_bar_threshold(
    token_a: str,
    token_b: str,
    token_volumes: dict[str, float],
) -> float:
    """Calculate adaptive dollar bar threshold for a pool based on token liquidity.

    Uses token-specific volume to create information-homogeneous bars. High-volume
    tokens (like WETH) get larger bar thresholds, while low-volume tokens get smaller
    thresholds. This ensures each bar represents similar "information content"
    regardless of the token's market cap (Prado AFML Ch. 2.3).

    Formula: threshold = ADAPTIVE_BAR_FRACTION * max(volume_a, volume_b)
    Clamped to [MIN_BAR_SIZE, MAX_BAR_SIZE] for stability.

    Args:
        token_a: First token address.
        token_b: Second token address.
        token_volumes: Dictionary mapping token_address -> avg_daily_volume.

    Returns:
        Dollar bar threshold in USDC for this pool.

    """
    # Get volumes for both tokens (default to 0 if not found)
    vol_a = token_volumes.get(token_a, 0.0)
    vol_b = token_volumes.get(token_b, 0.0)

    # Use the higher volume token to determine bar size
    max_volume = max(vol_a, vol_b)

    if max_volume == 0:
        # No volume data - use default
        return DEFAULT_BAR_SIZE

    # Calculate adaptive threshold: 0.1% of daily volume
    threshold = max_volume * ADAPTIVE_BAR_FRACTION

    # Clamp to reasonable range
    return max(MIN_BAR_SIZE, min(MAX_BAR_SIZE, threshold))



def generate_pool_bars(  # noqa: C901, PLR0912, PLR0915
    swaps_file: Path,
    prices_file: Path,
    pools_file: Path,
    output_file: Path,
    target_usdc_bar_size: float | None = None,
    use_adaptive_bars: bool = True,
) -> None:
    """Generate pool-level bars with signed net flows.

    Args:
        swaps_file: Path to usdc_paired_swaps.parquet (needed for flow direction).
        prices_file: Path to usdc_prices_timeseries.parquet.
        pools_file: Path to pools.json.
        output_file: Path to output usdc_bars.parquet.
        target_usdc_bar_size: Fixed dollar volume per bar (e.g., 100000 for $100k).
            If None and use_adaptive_bars=True, uses adaptive sizing.
        use_adaptive_bars: If True, calculate adaptive bar thresholds per pool.
            If False, use target_usdc_bar_size for all pools.

    """
    logger.info("Loading pool information...")
    pool_info = load_pool_info(pools_file)

    logger.info("Loading prices data from %s...", prices_file)
    prices_df = pl.read_parquet(prices_file)
    logger.info("Loaded %s price records", f"{len(prices_df):,}")

    # Calculate token daily volumes for adaptive bar sizing
    token_volumes: dict[str, float] = {}
    if use_adaptive_bars:
        token_volumes = calculate_token_daily_volumes(prices_df)
        logger.info("✅ Adaptive bar sizing enabled (0.1%% of daily volume per bar)")
    else:
        if target_usdc_bar_size is None:
            target_usdc_bar_size = DEFAULT_BAR_SIZE
        logger.info("Using fixed bar size: $%s", f"{target_usdc_bar_size:,.0f}")

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
    # Track thresholds per pool for logging
    adaptive_thresholds_used: dict[str, float] = {}

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

        # Type narrowing: ensure tokens are strings
        assert isinstance(token_a, str)
        assert isinstance(token_b, str)

        # Determine bar threshold for this pool (adaptive or fixed)
        if use_adaptive_bars:
            pool_bar_threshold = get_adaptive_bar_threshold(
                token_a, token_b, token_volumes,
            )
            adaptive_thresholds_used[pool_id] = pool_bar_threshold
        else:
            pool_bar_threshold = target_usdc_bar_size or DEFAULT_BAR_SIZE

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

            # Check if bar is complete (use pool-specific threshold)
            if bar_usdc_volume >= pool_bar_threshold:
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

    # Log adaptive threshold statistics
    if use_adaptive_bars and len(adaptive_thresholds_used) > 0:
        thresholds = list(adaptive_thresholds_used.values())
        logger.info("\n=== ADAPTIVE BAR THRESHOLD STATISTICS ===")
        logger.info("  Pools with adaptive thresholds: %d", len(thresholds))
        logger.info("  Min threshold: $%s", f"{min(thresholds):,.0f}")
        logger.info("  Max threshold: $%s", f"{max(thresholds):,.0f}")
        median_threshold = sorted(thresholds)[len(thresholds) // 2]
        logger.info("  Median threshold: $%s", f"{median_threshold:,.0f}")
        logger.info("  Mean threshold: $%s", f"{sum(thresholds)/len(thresholds):,.0f}")

        # Show distribution
        small_bars = sum(1 for t in thresholds if t <= 50_000)
        medium_bars = sum(1 for t in thresholds if 50_000 < t <= 200_000)
        large_bars = sum(1 for t in thresholds if t > 200_000)
        total = len(thresholds)
        logger.info("  Distribution:")
        logger.info(
            "    Small bars ($10k-$50k): %d (%.1f%%)",
            small_bars,
            small_bars / total * 100,
        )
        logger.info(
            "    Medium bars ($50k-$200k): %d (%.1f%%)",
            medium_bars,
            medium_bars / total * 100,
        )
        logger.info(
            "    Large bars ($200k+): %d (%.1f%%)",
            large_bars,
            large_bars / total * 100,
        )

    # Check if we generated any bars
    if len(all_bar_rows) == 0:
        logger.error(
            "No bars were generated! Check input data and bar sizing parameters.",
        )
        msg = "No bars generated - bar threshold may be too large"
        raise ValueError(msg)

    # Create output DataFrame
    output_df = pl.DataFrame(all_bar_rows)

    # Ensure flow columns are Float64 (both src and dest should be consistent)
    # Note: dest_flow can be zero, which causes Polars to infer Int64;
    # we explicitly cast to Float64
    output_df = output_df.with_columns(
        [
            pl.col("src_flow_usdc").cast(pl.Float64),
            pl.col("dest_flow_usdc").cast(pl.Float64),
        ],
    )

    # Sort by timestamp
    output_df = output_df.sort("bar_close_timestamp")

    # Deduplicate bars (same timestamp + pool + tokens)
    # This can happen if a pool switches flow direction during bar generation
    logger.info("Deduplicating bars...")
    rows_before = len(output_df)
    output_df = output_df.unique(
        subset=["bar_close_timestamp", "pool_id", "src_token_id", "dest_token_id"],
        keep="first",
    )
    rows_after = len(output_df)
    if rows_before > rows_after:
        logger.warning(
            "Removed %d duplicate bars (%.2f%%)",
            rows_before - rows_after,
            ((rows_before - rows_after) / rows_before) * 100,
        )
    else:
        logger.info("No duplicates found")

    # Re-sort after deduplication to ensure ordering
    output_df = output_df.sort("bar_close_timestamp")

    # Filter out extremely illiquid tokens before capping time deltas
    # This removes tokens where >20% of bars have extreme time deltas (>1 day)
    logger.info("Filtering extremely illiquid tokens...")
    output_df = filter_illiquid_tokens(output_df)

    # Cap bar_time_delta_sec at a reasonable maximum
    # Extremely long deltas occur for illiquid pools and can skew downstream analysis
    extreme_deltas = output_df.filter(
        pl.col("bar_time_delta_sec") > MAX_BAR_TIME_DELTA_SEC,
    )
    if len(extreme_deltas) > 0:
        logger.info(
            "Capping %d bars with time delta > 7 days (%.2f%%)",
            len(extreme_deltas),
            (len(extreme_deltas) / len(output_df)) * 100,
        )
        output_df = output_df.with_columns(
            pl.when(pl.col("bar_time_delta_sec") > MAX_BAR_TIME_DELTA_SEC)
            .then(MAX_BAR_TIME_DELTA_SEC)
            .otherwise(pl.col("bar_time_delta_sec"))
            .alias("bar_time_delta_sec"),
        )

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

    # Validate output before saving
    validate_output(output_df)

    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s messages to usdc_bars.parquet",
        f"{len(output_df):,}",
    )


def filter_illiquid_tokens(df_bars: pl.DataFrame) -> pl.DataFrame:
    """Filter out extremely illiquid tokens that have poor data quality.

    Removes tokens where >20% of their bars have extreme time deltas (>1 day).
    These tokens are too illiquid to provide reliable training signals and
    often represent scam/honeypot tokens with little real trading value.

    Args:
        df_bars: DataFrame with generated bars.

    Returns:
        Filtered DataFrame with illiquid tokens removed.

    """
    # Analyze source tokens for extreme time deltas
    extreme_deltas = df_bars.filter(
        pl.col("bar_time_delta_sec") > ILLIQUIDITY_TIME_THRESHOLD_SEC,
    )

    src_analysis = (
        extreme_deltas.group_by("src_token_id")
        .agg(pl.len().alias("extreme_bars"))
        .join(
            df_bars.group_by("src_token_id").agg(pl.len().alias("total_bars")),
            on="src_token_id",
        )
        .with_columns(
            (pl.col("extreme_bars") / pl.col("total_bars")).alias("pct_extreme"),
        )
        .filter(pl.col("pct_extreme") > ILLIQUIDITY_PCT_THRESHOLD)
    )

    # Also check destination tokens
    dest_analysis = (
        extreme_deltas.group_by("dest_token_id")
        .agg(pl.len().alias("extreme_bars"))
        .join(
            df_bars.group_by("dest_token_id").agg(pl.len().alias("total_bars")),
            on="dest_token_id",
        )
        .with_columns(
            (pl.col("extreme_bars") / pl.col("total_bars")).alias("pct_extreme"),
        )
        .filter(pl.col("pct_extreme") > ILLIQUIDITY_PCT_THRESHOLD)
    )

    # Combine tokens to filter
    tokens_to_remove = set(src_analysis["src_token_id"].to_list()) | set(
        dest_analysis["dest_token_id"].to_list(),
    )

    if len(tokens_to_remove) > 0:
        logger.info(
            "Identified %d illiquid tokens (>%.0f%% of bars with time delta > 1 day)",
            len(tokens_to_remove),
            ILLIQUIDITY_PCT_THRESHOLD * 100,
        )

        # Show top 10 most problematic tokens
        logger.info("  Top 10 most illiquid tokens:")
        top_illiquid = (
            src_analysis.sort("pct_extreme", descending=True)
            .head(10)
            .join(
                df_bars.group_by("src_token_id").agg(
                    [
                        pl.col("src_price_usdc").mean().alias("avg_price"),
                        pl.col("src_flow_usdc").abs().sum().alias("total_volume"),
                    ],
                ),
                left_on="src_token_id",
                right_on="src_token_id",
            )
        )

        for row in top_illiquid.iter_rows(named=True):
            logger.info(
                "    %s: %.1f%% extreme (avg price: $%.2e, total volume: $%.0f)",
                row["src_token_id"][:10] + "...",
                row["pct_extreme"] * 100,
                row["avg_price"],
                row["total_volume"],
            )

        # Filter out bars containing these tokens
        bars_before = len(df_bars)
        df_filtered = df_bars.filter(
            ~pl.col("src_token_id").is_in(tokens_to_remove)
            & ~pl.col("dest_token_id").is_in(tokens_to_remove),
        )
        bars_removed = bars_before - len(df_filtered)

        logger.info(
            "Removed %d bars (%.2f%%) containing illiquid tokens",
            bars_removed,
            (bars_removed / bars_before) * 100,
        )

        return df_filtered

    logger.info("No extremely illiquid tokens found")
    return df_bars


def validate_output(df_bars: pl.DataFrame) -> None:  # noqa: C901, PLR0912, PLR0915
    """Validate the output bar dataset for quality issues.

    Performs comprehensive validation including:
    - Duplicate detection
    - Null value checks
    - Timestamp ordering
    - Time delta ranges
    - Tick count validation
    - Flow value validation
    - Price value validation
    - Price continuity checks
    - Pool and token distribution

    Args:
        df_bars: DataFrame with generated bars.

    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("BAR DATA VALIDATION")
    logger.info("=" * 70)

    # 1. Check for duplicates
    logger.info("\n1. Checking for duplicate bars...")
    dup_count = (
        df_bars.group_by(
            ["bar_close_timestamp", "pool_id", "src_token_id", "dest_token_id"],
        )
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .shape[0]
    )
    if dup_count > 0:
        dup_pct = (dup_count / len(df_bars)) * 100
        logger.warning(
            "  ⚠️ WARN: %d duplicate bars (%.2f%%)",
            dup_count,
            dup_pct,
        )
    else:
        logger.info("  ✅ OK: No duplicate bars found")

    # 2. Check for null values
    logger.info("\n2. Checking for null values...")
    null_counts = df_bars.null_count()
    has_nulls = False
    for col in null_counts.columns:
        null_count = null_counts[col][0]
        if null_count > 0:
            logger.warning("  ⚠️ WARN: %s has %d null values", col, null_count)
            has_nulls = True
    if not has_nulls:
        logger.info("  ✅ OK: No null values found")

    # 3. Check timestamp ordering
    logger.info("\n3. Checking timestamp ordering...")
    sorted_check = df_bars["bar_close_timestamp"].is_sorted()
    if sorted_check:
        logger.info("  ✅ OK: Timestamps are sorted")
    else:
        logger.warning("  ⚠️ WARN: Timestamps not sorted")

    # 4. Validate bar_time_delta_sec
    logger.info("\n4. Validating bar_time_delta_sec...")
    time_delta_stats = df_bars.select(
        [
            pl.col("bar_time_delta_sec").min().alias("min"),
            pl.col("bar_time_delta_sec").max().alias("max"),
            pl.col("bar_time_delta_sec").mean().alias("mean"),
            pl.col("bar_time_delta_sec").median().alias("median"),
        ],
    ).row(0, named=True)

    logger.info(
        "  Range: [%.1f, %.1f] seconds",
        time_delta_stats["min"],
        time_delta_stats["max"],
    )
    logger.info(
        "  Mean: %.1f, Median: %.1f",
        time_delta_stats["mean"],
        time_delta_stats["median"],
    )

    # Max should be reasonable (< 1 day)
    if time_delta_stats["max"] > ILLIQUIDITY_TIME_THRESHOLD_SEC:
        max_days = time_delta_stats["max"] / 86400
        logger.warning(
            "  ⚠️ WARN: Max time delta (%.1f days) exceeds expected (1 day)",
            max_days,
        )
        # Find bars with extreme time deltas
        extreme_deltas = df_bars.filter(
            pl.col("bar_time_delta_sec") > ILLIQUIDITY_TIME_THRESHOLD_SEC,
        )
        logger.warning(
            "    Found %d bars with time delta > 1 day",
            len(extreme_deltas),
        )
        if len(extreme_deltas) > 0:
            logger.info("    Top 5 pools with extreme time deltas:")
            pool_delta_summary = (
                extreme_deltas.group_by("pool_id")
                .agg(
                    [
                        pl.len().alias("count"),
                        pl.col("bar_time_delta_sec").max().alias("max_delta"),
                    ],
                )
                .sort("max_delta", descending=True)
            )
            for row in pool_delta_summary.head(5).iter_rows(named=True):
                logger.info(
                    "      %s: %d bars, max delta=%.1f days",
                    row["pool_id"][:10] + "...",
                    row["count"],
                    row["max_delta"] / 86400,
                )
    else:
        logger.info("  ✅ OK: All time deltas within expected range")

    # 5. Validate tick_count
    logger.info("\n5. Validating tick_count...")
    tick_stats = df_bars.select(
        [
            pl.col("tick_count").min().alias("min"),
            pl.col("tick_count").max().alias("max"),
            pl.col("tick_count").mean().alias("mean"),
            pl.col("tick_count").median().alias("median"),
        ],
    ).row(0, named=True)

    logger.info("  Range: [%d, %d]", tick_stats["min"], tick_stats["max"])
    logger.info("  Mean: %.1f, Median: %.1f", tick_stats["mean"], tick_stats["median"])

    # Check for bars with only 1 tick (might indicate data issues)
    single_tick = df_bars.filter(pl.col("tick_count") == 1)
    if len(single_tick) > 0:
        single_tick_pct = (len(single_tick) / len(df_bars)) * 100
        logger.info(
            "  INFO: %d bars (%.2f%%) have only 1 tick",
            len(single_tick),
            single_tick_pct,
        )

    # 6. Validate flow values
    logger.info("\n6. Validating flow values...")
    src_flow_stats = df_bars.select(
        [
            pl.col("src_flow_usdc").min().alias("min"),
            pl.col("src_flow_usdc").max().alias("max"),
            pl.col("src_flow_usdc").mean().alias("mean"),
        ],
    ).row(0, named=True)

    dest_flow_stats = df_bars.select(
        [
            pl.col("dest_flow_usdc").min().alias("min"),
            pl.col("dest_flow_usdc").max().alias("max"),
            pl.col("dest_flow_usdc").mean().alias("mean"),
        ],
    ).row(0, named=True)

    logger.info(
        "  src_flow_usdc: [%.2e, %.2e], mean=%.2e",
        src_flow_stats["min"],
        src_flow_stats["max"],
        src_flow_stats["mean"],
    )
    logger.info(
        "  dest_flow_usdc: [%.2e, %.2e], mean=%.2e",
        dest_flow_stats["min"],
        dest_flow_stats["max"],
        dest_flow_stats["mean"],
    )

    # Check for all-zero flows (indicates no trading activity)
    zero_src = df_bars.filter(pl.col("src_flow_usdc") == 0)
    zero_dest = df_bars.filter(pl.col("dest_flow_usdc") == 0)

    if len(zero_src) > 0:
        logger.info(
            "  INFO: %d bars (%.2f%%) have zero src_flow",
            len(zero_src),
            (len(zero_src) / len(df_bars)) * 100,
        )

    if len(zero_dest) > 0:
        logger.info(
            "  INFO: %d bars (%.2f%%) have zero dest_flow",
            len(zero_dest),
            (len(zero_dest) / len(df_bars)) * 100,
        )
        # NOTE: Zero dest_flow is expected when dest_token is USDC
        # USDC has no price records (it's the numeraire at $1.00)
        # so net_flow calculations only occur for non-USDC tokens

    # 7. Validate price values
    logger.info("\n7. Validating price values...")
    src_price_stats = df_bars.select(
        [
            pl.col("src_price_usdc").min().alias("min"),
            pl.col("src_price_usdc").max().alias("max"),
        ],
    ).row(0, named=True)

    dest_price_stats = df_bars.select(
        [
            pl.col("dest_price_usdc").min().alias("min"),
            pl.col("dest_price_usdc").max().alias("max"),
        ],
    ).row(0, named=True)

    logger.info(
        "  src_price_usdc: [%.2e, %.2e]",
        src_price_stats["min"],
        src_price_stats["max"],
    )
    logger.info(
        "  dest_price_usdc: [%.2e, %.2e]",
        dest_price_stats["min"],
        dest_price_stats["max"],
    )

    # Check for negative prices
    neg_src = df_bars.filter(pl.col("src_price_usdc") < 0)
    neg_dest = df_bars.filter(pl.col("dest_price_usdc") < 0)

    if len(neg_src) > 0 or len(neg_dest) > 0:
        logger.warning(
            "  WARN: Found %d bars with negative src price, %d with negative dest",
            len(neg_src),
            len(neg_dest),
        )
    else:
        logger.info("  ✅ OK: All prices are non-negative")

    # 8. Validate price continuity
    logger.info("\n8. Validating price continuity...")
    # Check for extreme price jumps between consecutive bars for same token
    # Group by token and check price changes
    extreme_price_jumps = 0
    price_jump_threshold = 0.50  # 50% price change

    for token_col, price_col in [
        ("src_token_id", "src_price_usdc"),
        ("dest_token_id", "dest_price_usdc"),
    ]:
        token_bars = (
            df_bars.select([token_col, "bar_close_timestamp", price_col])
            .sort([token_col, "bar_close_timestamp"])
            .with_columns(
                [
                    pl.col(price_col).shift(1).over(token_col).alias("prev_price"),
                ],
            )
            .filter(pl.col("prev_price").is_not_null())
        )

        # Calculate price change percentage
        price_changes = token_bars.with_columns(
            [
                (
                    (pl.col(price_col) - pl.col("prev_price")).abs()
                    / (pl.col("prev_price") + 1e-9)
                ).alias("pct_change"),
            ],
        )

        extreme = price_changes.filter(pl.col("pct_change") > price_jump_threshold)
        extreme_price_jumps += len(extreme)

    if extreme_price_jumps > 0:
        logger.warning(
            "  ⚠️ WARN: Found %d bars with >50%% price change from previous bar",
            extreme_price_jumps,
        )
        logger.warning("    (This may indicate data quality issues or high volatility)")
    else:
        logger.info("  ✅ OK: No extreme price jumps detected")

    # 9. Check pool and token distribution
    logger.info("\n9. Pool and token distribution...")
    logger.info("  Unique pools: %d", df_bars["pool_id"].n_unique())
    unique_src_tokens = df_bars["src_token_id"].n_unique()
    unique_dest_tokens = df_bars["dest_token_id"].n_unique()
    unique_tokens = set(df_bars["src_token_id"].unique()) | set(
        df_bars["dest_token_id"].unique(),
    )
    logger.info("  Unique tokens (src): %d", unique_src_tokens)
    logger.info("  Unique tokens (dest): %d", unique_dest_tokens)
    logger.info("  Unique tokens (combined): %d", len(unique_tokens))

    logger.info("")
    logger.info("=" * 70)


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
                "  ✅ Some indirect swap pools have bars: %d pools",
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
        logger.info("  ✅ Non-USDC token pair bars present: good for TGNN!")


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
    logger.info("  - Price correlation > 0.95: prices align ✅")
    logger.info("  - Price MAPE < 5%%: prices are accurate ✅")


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

    # External correlation validation with CoinGecko
    if not VALIDATION_TOKENS_FILE.exists():
        logger.warning(
            "Validation tokens file not found: %s (skipping CoinGecko validation)",
            VALIDATION_TOKENS_FILE,
        )
        return

    with VALIDATION_TOKENS_FILE.open() as f:
        validation_config = json.load(f)

    # Filter to only validation tokens with CoinGecko IDs
    tokens_to_validate = {
        addr: info for addr, info in validation_config.items() if "coingecko_id" in info
    }

    if not tokens_to_validate:
        logger.warning(
            "No tokens with coingecko_id found in %s (skipping validation)",
            VALIDATION_TOKENS_FILE,
        )
        return

    logger.info("\n=== COINGECKO CORRELATION VALIDATION ===")
    logger.info("Validating %d tokens against CoinGecko...", len(tokens_to_validate))

    _validate_against_coingecko(df_bars, tokens_to_validate)


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
    target_usdc_bar_size: float | None = typer.Option(
        None,
        "--target-usdc-bar-size",
        help=(
            "Fixed USDC volume per bar (e.g., 100000 for $100k). "
            "If not specified, uses adaptive sizing."
        ),
    ),
    use_adaptive_bars: bool = typer.Option(
        True,
        "--adaptive/--fixed",
        help="Use adaptive bar sizing (0.1%% daily volume) or fixed size",
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
    """Generate pool-level bars with signed net flows.

    By default, uses ADAPTIVE bar sizing where each pool gets a threshold
    based on 0.1% of its tokens' daily volume. This creates information-
    homogeneous bars where high-volume tokens (WETH, USDC) get larger bars
    and low-volume tokens get smaller bars.

    Use --fixed to disable adaptive sizing and use a constant threshold.
    """
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
        use_adaptive_bars,
    )

    if validate:
        validate_volume_bars(output_file)


if __name__ == "__main__":
    app()
