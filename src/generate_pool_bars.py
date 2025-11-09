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
from pathlib import Path

import click
import polars as pl

logger = logging.getLogger(__name__)


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


def load_pool_info(pools_file: Path) -> dict[str, dict]:
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


def generate_pool_bars(
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

    def decode_amounts_to_dict(data_hex: str) -> dict:
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
                "amount0_sign", [d["amount0_sign"] for d in decoded], dtype=pl.Int8
            ),
            pl.Series(
                "amount1_sign", [d["amount1_sign"] for d in decoded], dtype=pl.Int8
            ),
        ]
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
            ]
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
        ]
    )

    data = data.with_columns(
        [
            (pl.col("usdc_volume") * pl.col("flow_sign")).alias("signed_volume"),
        ]
    )

    logger.info("Calculated flow directions for %s price records", f"{len(data):,}")

    # Sort by pool and timestamp for bar generation
    data = data.sort(["pool", "block_timestamp"])

    logger.info("Generating pool-level bars...")

    # Group by pool and process each pool separately
    all_bar_rows = []
    pools_processed = 0

    for pool_id, pool_data in data.group_by("pool", maintain_order=True):
        pools_processed += 1
        if pools_processed % 1000 == 0:
            logger.info("Processed %d pools...", pools_processed)

        pool_id = pool_id[0]  # Extract scalar from tuple
        pool_data = pool_data.sort("block_timestamp")

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
                        }
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
                        }
                    )

                # Reset for next bar
                last_bar_timestamp = bar_close_timestamp
                bar_usdc_volume = 0.0
                bar_price_records = []
                seen_swaps = set()

    logger.info(
        "Generated %d bar messages from %d pools", len(all_bar_rows), pools_processed
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
        "Done! Saved %s messages to master_message_log.parquet", f"{len(output_df):,}"
    )


@click.command()
@click.option(
    "--swaps-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/usdc_paired_swaps.parquet"),
    help="Input swaps parquet file",
)
@click.option(
    "--prices-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/usdc_prices_timeseries.parquet"),
    help="Input prices parquet file",
)
@click.option(
    "--pools-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/pools.json"),
    help="JSON file with pool information",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/master_message_log.parquet"),
    help="Output parquet file path",
)
@click.option(
    "--target-usdc-bar-size",
    type=float,
    default=100000.0,
    help="Target USDC volume per bar (default: $100k)",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    swaps_file: Path,
    prices_file: Path,
    pools_file: Path,
    output_file: Path,
    target_usdc_bar_size: float,
    *,
    verbose: bool,
) -> None:
    """Generate pool-level bars with signed net flows."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generate_pool_bars(
        swaps_file,
        prices_file,
        pools_file,
        output_file,
        target_usdc_bar_size,
    )


if __name__ == "__main__":
    main()
