"""Filter and decode WETH-paired swaps from Uniswap V3 data.

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


def load_weth_pools(
    pools_file: Path,
) -> tuple[set[str], set[str], dict[str, tuple[str, str]]]:
    """Load WETH pool information and extract pool addresses and token addresses.

    Args:
        pools_file: Path to weth_pools_by_address.json file.

    Returns:
        Tuple of (weth_pool_addresses, weth_paired_token_addresses, pool_tokens_map).
        pool_tokens_map is a dict mapping pool_address -> (token0_address,
        token1_address).

    """
    with pools_file.open() as f:
        weth_pools = json.load(f)

    pool_addresses = set(weth_pools.keys())

    # Extract all unique token addresses from WETH pools (excluding WETH itself)
    token_addresses = set()
    # Map pool address to its token pair (token0, token1)
    pool_tokens_map = {}

    for pool_addr, pool in weth_pools.items():
        tokens = pool["tokens"]
        # Tokens are stored in order (token0, token1) based on address
        token0_addr = tokens[0]["address"].lower()
        token1_addr = tokens[1]["address"].lower()

        pool_tokens_map[pool_addr.lower()] = (token0_addr, token1_addr)

        # Add non-WETH tokens to our set
        for token in tokens:
            addr = token["address"].lower()
            if addr != WETH_ADDRESS.lower():
                token_addresses.add(addr)

    logger.info("Loaded %d WETH pools", len(pool_addresses))
    logger.info("Found %d unique tokens paired with WETH", len(token_addresses))

    return pool_addresses, token_addresses, pool_tokens_map


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
    pools_file: Path,
) -> None:
    """Filter swaps to WETH-paired tokens and decode swap data.

    Args:
        input_file: Input parquet file with V3 swaps.
        output_file: Output parquet file for filtered and decoded swaps.
        pools_file: JSON file with WETH pool information.

    """
    logger.info("Loading WETH pool information from %s...", pools_file)
    weth_pool_addresses, _weth_paired_tokens, pool_tokens_map = load_weth_pools(
        pools_file,
    )

    # Create a polars dataframe with pool token mappings for joining
    pool_tokens_df = pl.DataFrame(
        {
            "pool_address": list(pool_tokens_map.keys()),
            "token0": [tokens[0] for tokens in pool_tokens_map.values()],
            "token1": [tokens[1] for tokens in pool_tokens_map.values()],
        },
    )

    logger.info("Loading swap data from %s...", input_file)

    # Read the swaps data
    df = pl.read_parquet(input_file)
    logger.info("Loaded %s total V3 swaps", f"{len(df):,}")

    # Filter to only swaps in WETH-paired pools
    logger.info("Filtering to WETH-paired pools...")
    df_filtered = df.filter(
        pl.col("pool_or_manager_address").str.to_lowercase().is_in(weth_pool_addresses),
    )

    logger.info("Filtered to %s swaps in WETH-paired pools", f"{len(df_filtered):,}")

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

    logger.info("Saving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.write_parquet(output_file)

    logger.info("Done! Saved %s WETH-paired swaps", f"{len(df_decoded):,}")


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
    "--pools-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_pools_by_address.json"),
    help="JSON file with WETH pool information",
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
    verbose: bool,
) -> None:
    """Filter and decode WETH-paired swaps from Uniswap V3 data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    filter_and_decode_weth_swaps(input_file, output_file, pools_file)


if __name__ == "__main__":
    main()
