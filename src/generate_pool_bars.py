"""Generate pool-level bars with signed net flows from USDC-paired swaps.

This script implements Milestone 1 of Phase 2:
- Groups swaps by pool_id
- Accumulates swaps into dollar bars based on target_usdc_bar_size
- Calculates signed net flows for each token in the bar
- Generates two output rows per bar (one per token)
- Outputs master_message_log.parquet with bar statistics and signed flows
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
        output_file: Path to output master_message_log.parquet.
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
                token_a_price = None
                token_b_price = None
                for rec in reversed(bar_price_records):
                    if rec["token_address"] == token_a and token_a_price is None:
                        token_a_price = rec["price_in_usdc"]
                    if rec["token_address"] == token_b and token_b_price is None:
                        token_b_price = rec["price_in_usdc"]
                    if token_a_price is not None and token_b_price is not None:
                        break

                # Generate rows for both tokens (if we have prices for both)
                if token_a_price is not None:
                    all_bar_rows.append(
                        {
                            "bar_close_timestamp": bar_close_timestamp,
                            "pool_id": pool_id,
                            "token_id": token_a,
                            "net_flow_usdc": net_flow_a,
                            "token_close_price_usdc": token_a_price,
                            "bar_time_delta_sec": bar_time_delta_sec,
                            "tick_count": tick_count,
                        },
                    )

                if token_b_price is not None:
                    all_bar_rows.append(
                        {
                            "bar_close_timestamp": bar_close_timestamp,
                            "pool_id": pool_id,
                            "token_id": token_b,
                            "net_flow_usdc": net_flow_b,
                            "token_close_price_usdc": token_b_price,
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

    # Create output DataFrame
    output_df = pl.DataFrame(all_bar_rows)

    # Sort by timestamp
    output_df = output_df.sort("bar_close_timestamp")

    logger.info("\nBar Statistics:")
    logger.info("  Total messages: %s", f"{len(output_df):,}")
    logger.info("  Unique pools: %d", output_df["pool_id"].n_unique())
    logger.info("  Unique tokens: %d", output_df["token_id"].n_unique())
    logger.info(
        "  Date range: %s to %s",
        output_df["bar_close_timestamp"].min(),
        output_df["bar_close_timestamp"].max(),
    )

    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_file)

    logger.info(
        "Done! Saved %s messages to master_message_log.parquet",
        f"{len(output_df):,}",
    )


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

    # Net flow can be positive (buying token) or negative (selling token/USDC)
    # This is expected behavior and represents flow direction
    positive_flows = df_bars.filter(pl.col("net_flow_usdc") > 0)
    negative_flows = df_bars.filter(pl.col("net_flow_usdc") < 0)
    zero_flows = df_bars.filter(pl.col("net_flow_usdc") == 0)

    pos_pct = len(positive_flows) / len(df_bars) * 100
    neg_pct = len(negative_flows) / len(df_bars) * 100
    zero_pct = len(zero_flows) / len(df_bars) * 100

    logger.info("  Net flow distribution:")
    logger.info(
        "    Positive (buying token): %d (%.2f%%)",
        len(positive_flows),
        pos_pct,
    )
    logger.info(
        "    Negative (selling token): %d (%.2f%%)",
        len(negative_flows),
        neg_pct,
    )
    logger.info("    Zero-flow bars: %d (%.2f%%)", len(zero_flows), zero_pct)

    # Critical issue: all flows are zero (no trading activity)
    has_critical_issue = len(zero_flows) == len(df_bars)
    if has_critical_issue:
        logger.warning("  ⚠ All bars have zero net_flow_usdc (no trading activity)")
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
    negative_prices = df_bars.filter(pl.col("token_close_price_usdc") < 0)
    has_negative_prices = len(negative_prices) > 0

    if has_negative_prices:
        logger.warning(
            "  ⚠ Found %d bars with negative prices",
            len(negative_prices),
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
    price_stats = df_bars.select(
        [
            pl.col("token_close_price_usdc").min().alias("min"),
            pl.col("token_close_price_usdc").max().alias("max"),
            pl.col("token_close_price_usdc").mean().alias("mean"),
            pl.col("token_close_price_usdc").std().alias("std"),
        ],
    )
    for col in price_stats.columns:
        val = price_stats[col][0]
        logger.info("  %s: %.6f", col, val)

    logger.info("\n=== VOLUME STATISTICS ===")
    vol_stats = df_bars.select(
        [
            pl.col("net_flow_usdc").min().alias("min"),
            pl.col("net_flow_usdc").max().alias("max"),
            pl.col("net_flow_usdc").mean().alias("mean"),
            pl.col("net_flow_usdc").median().alias("median"),
            pl.col("net_flow_usdc").std().alias("std"),
            (pl.col("net_flow_usdc") > 0).sum().alias("non_zero_count"),
        ],
    )
    for col in vol_stats.columns:
        val = vol_stats[col][0]
        logger.info("  %s: %.6f", col, val)


def _log_summary(df_bars: pl.DataFrame) -> None:
    """Log summary statistics.

    Args:
        df_bars: DataFrame to summarize.

    """
    logger.info("\n=== SUMMARY ===")
    logger.info("  Total records: %s", f"{len(df_bars):,}")
    logger.info("  Unique pools: %d", df_bars["pool_id"].n_unique())
    logger.info("  Unique tokens: %d", df_bars["token_id"].n_unique())
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
    """Fetch historical prices from CoinGecko API.

    Args:
        token_id: CoinGecko token ID (e.g., 'usd-coin').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with timestamp and price_usd columns.

    """
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
    return pl.DataFrame(
        {
            "timestamp": [datetime.fromtimestamp(p[0] / 1000, tz=UTC) for p in prices],
            "price_usd": [p[1] for p in prices],
        },
    )


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

        # Get our volume bars for this token
        df_our_bars = df_bars.filter(pl.col("token_id") == address)

        if len(df_our_bars) == 0:
            logger.info("  ⚠ No volume bars in our data")
            continue

        if address not in coingecko_prices:
            logger.info("  ⚠ No CoinGecko data fetched")
            continue

        df_cg = coingecko_prices[address]

        # Aggregate our bars to calendar-day volumes
        df_our_daily = (
            df_our_bars.with_columns(
                pl.col("bar_close_timestamp").dt.truncate("1d").alias("date"),
            )
            .group_by("date")
            .agg(
                [
                    pl.col("net_flow_usdc").sum().alias("our_volume"),
                    pl.col("token_close_price_usdc").mean().alias("avg_price"),
                ],
            )
            .with_columns(pl.col("date").cast(pl.Date))
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
        output_file: Path to the generated master_message_log.parquet file.

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
        Path("data/usdc_prices_timeseries.parquet"),
        "--prices-file",
        help="Input prices parquet file",
    ),
    pools_file: Path = typer.Option(
        Path("data/pools.json"),
        "--pools-file",
        help="JSON file with pool information",
    ),
    output_file: Path = typer.Option(
        Path("data/master_message_log.parquet"),
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
