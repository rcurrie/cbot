"""Decode WETH-paired swaps from Uniswap V3 data.

Decode V3 swap events and filter to only include swaps where at least one
token is paired with WETH in a liquidity pool.
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


def load_all_pools(
    pools_file: Path,
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    """Load all Uniswap V3 pool information.

    Args:
        pools_file: Path to pools.json file (all pools, not just WETH).

    Returns:
        Tuple of (weth_paired_tokens, all_pool_tokens_map).
        weth_paired_tokens is set of tokens that have a pool with WETH.
        all_pool_tokens_map maps all pool_address -> (token0, token1).

    """
    with pools_file.open() as f:
        data = json.load(f)

    # Extract Uniswap V3 Ethereum pools
    all_pools = data.get("data", [])
    uniswap_v3_pools = [
        p
        for p in all_pools
        if p.get("protocol") == "usp3" and p.get("blockchain") == "ethereum"
    ]

    logger.info("Found %d Uniswap V3 Ethereum pools", len(uniswap_v3_pools))

    # Build mapping of all pool addresses to token pairs
    all_pool_tokens_map = {}
    weth_paired_tokens = set()

    for pool in uniswap_v3_pools:
        pool_addr = pool["address"].lower()
        tokens = pool.get("tokens", [])

        if len(tokens) == 2:  # noqa: PLR2004
            token0_addr = tokens[0]["address"].lower()
            token1_addr = tokens[1]["address"].lower()

            all_pool_tokens_map[pool_addr] = (token0_addr, token1_addr)

            # Track tokens that have a pool with WETH
            if token0_addr == WETH_ADDRESS.lower():
                weth_paired_tokens.add(token1_addr)
            elif token1_addr == WETH_ADDRESS.lower():
                weth_paired_tokens.add(token0_addr)

    logger.info("Found %d unique tokens with WETH pools", len(weth_paired_tokens))

    return weth_paired_tokens, all_pool_tokens_map


def decode_swap_data(data_hex: str) -> dict[str, int]:
    """Decode Uniswap V3 Swap event data field.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        Dictionary with decoded swap parameters.

    """
    # Remove 0x prefix
    data = data_hex.removeprefix("0x")

    # Extract 32-byte chunks for first 4 params
    amount0_hex = data[0:64]
    amount1_hex = data[64:128]
    sqrtpricex96_hex = data[128:192]
    liquidity_hex = data[192:256]

    # Last parameter is int24 (3 bytes)
    tick_hex = data[256:]

    # Decode as integers
    amount0 = int.from_bytes(bytes.fromhex(amount0_hex), "big", signed=True)
    amount1 = int.from_bytes(bytes.fromhex(amount1_hex), "big", signed=True)
    sqrtpricex96 = int(sqrtpricex96_hex, 16)
    liquidity = int(liquidity_hex, 16)

    # Decode int24 (3 bytes = 6 hex chars)
    tick_int = int(tick_hex[-6:], 16)
    if tick_int >= INT24_SIGN_BIT:
        tick_int -= INT24_MAX

    return {
        "amount0": amount0,
        "amount1": amount1,
        "sqrtpricex96": sqrtpricex96,
        "liquidity": liquidity,
        "tick": tick_int,
    }


def filter_and_decode_weth_swaps(
    input_file: Path,
    output_file: Path,
) -> None:
    """Filter swaps to WETH-paired tokens and decode swap data.

    Filter strategy: Keep swaps where at least one token has a WETH pool.
    This includes:
    - Direct WETH swaps (token <-> WETH)
    - Indirect swaps (tokenA <-> tokenB) where both have WETH pools

    Args:
        input_file: Input parquet file with V3 swaps.
        output_file: Output parquet file for filtered and decoded swaps.

    """
    logger.info("Loading pool information from data/pools.json...")
    weth_paired_tokens, all_pool_tokens_map = load_all_pools(Path("data/pools.json"))

    logger.info("Identified %d tokens with WETH pools", len(weth_paired_tokens))

    # Create a polars dataframe with ALL pool token mappings for joining
    pool_tokens_df = pl.DataFrame(
        {
            "pool_address": list(all_pool_tokens_map.keys()),
            "token0": [tokens[0] for tokens in all_pool_tokens_map.values()],
            "token1": [tokens[1] for tokens in all_pool_tokens_map.values()],
        },
    )

    logger.info("Loading swap data from %s...", input_file)

    # Read the swaps data
    df = pl.read_parquet(input_file)
    logger.info("Loaded %s total V3 swaps", f"{len(df):,}")

    # Get all pool info by joining with pool tokens
    df_with_pools = df.join(
        pool_tokens_df,
        left_on=pl.col("pool_or_manager_address").str.to_lowercase(),
        right_on="pool_address",
        how="left",
    )

    # Filter to swaps where at least one token has a WETH pool
    # This includes both direct WETH swaps and swaps between WETH-paired tokens
    logger.info("Filtering to swaps with WETH-paired tokens...")
    df_filtered = df_with_pools.filter(
        pl.col("token0").is_in(weth_paired_tokens | {WETH_ADDRESS.lower()})
        | pl.col("token1").is_in(weth_paired_tokens | {WETH_ADDRESS.lower()}),
    )

    logger.info("Filtered to %s swaps with WETH-paired tokens", f"{len(df_filtered):,}")

    # Decode swap data
    logger.info("Decoding swap events...")

    # Extract sender, recipient, and decode amount0/amount1 from swap data
    # Using hex string slicing to extract the relevant parts
    df_decoded = df_filtered.with_columns([
        # Extract sender and recipient from topics (skip 0x and padding)
        pl.col("topics").list.get(1).str.slice(26).str.to_lowercase().alias("sender"),
        pl.col("topics").list.get(2).str.slice(26).str.to_lowercase().alias("recipient"),
        # Keep pool address in normalized form for joining
        pl.col("pool_or_manager_address").str.to_lowercase().alias("pool"),
    ])

    # Join with pool token information
    df_decoded = df_decoded.join(
        pool_tokens_df,
        left_on="pool",
        right_on="pool_address",
        how="left",
    )

    # Select final columns
    df_decoded = df_decoded.select([
        "block_timestamp",
        "block_number",
        "transaction_hash",
        "pool",
        "token0",
        "token1",
        "sender",
        "recipient",
        "data",
    ])

    # Analyze WETH coverage
    logger.info("\nWETH Coverage Analysis:")
    swaps_with_weth = df_decoded.filter(
        (pl.col("token0") == WETH_ADDRESS.lower())
        | (pl.col("token1") == WETH_ADDRESS.lower()),
    )
    swaps_without_weth = df_decoded.filter(
        (pl.col("token0") != WETH_ADDRESS.lower())
        & (pl.col("token1") != WETH_ADDRESS.lower()),
    )

    logger.info("  Total swaps: %s", f"{len(df_decoded):,}")
    logger.info(
        "  Swaps with WETH: %s (%.2f%%)",
        f"{len(swaps_with_weth):,}",
        len(swaps_with_weth) / len(df_decoded) * 100,
    )
    logger.info(
        "  Swaps between non-WETH tokens: %s (%.2f%%)",
        f"{len(swaps_without_weth):,}",
        len(swaps_without_weth) / len(df_decoded) * 100,
    )

    weth_as_token0 = len(df_decoded.filter(pl.col("token0") == WETH_ADDRESS.lower()))
    weth_as_token1 = len(df_decoded.filter(pl.col("token1") == WETH_ADDRESS.lower()))
    logger.info("  WETH as token0: %s", f"{weth_as_token0:,}")
    logger.info("  WETH as token1: %s", f"{weth_as_token1:,}")

    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.write_parquet(output_file)

    logger.info("Done! Saved %s swaps with WETH-paired tokens", f"{len(df_decoded):,}")


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/uniswap_v3_swaps.parquet"),
    help="Input parquet file with V3 swaps",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/weth_paired_swaps.parquet"),
    help="Output parquet file path",
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
    verbose: bool,
) -> None:
    """Filter and decode WETH-paired swaps from Uniswap V3 data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    filter_and_decode_weth_swaps(input_file, output_file)


if __name__ == "__main__":
    main()
