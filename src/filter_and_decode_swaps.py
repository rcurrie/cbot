"""Filter and decode USDC-paired swaps from raw swap data.

Load all parquet files from data/swaps/, filter to only Uniswap V3 Swap events
that involve USDC-paired tokens, and decode the swap data.
"""

import json
import logging
from pathlib import Path

import click
import polars as pl
from web3 import Web3

logger = logging.getLogger(__name__)

# USDC contract address on Ethereum mainnet
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

# Uniswap V3 Swap event signature
# Swap(address indexed sender, address indexed recipient,
#      int256 amount0, int256 amount1, uint160 sqrtPriceX96,
#      uint128 liquidity, int24 tick)
V3_SWAP_SIGNATURE = Web3.keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)",
).hex()


def load_all_pools(
    pools_file: Path,
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    """Load all Uniswap V3 pool information.

    Args:
        pools_file: Path to pools.json file (all pools, not just USDC).

    Returns:
        Tuple of (usdc_paired_tokens, all_pool_tokens_map).
        usdc_paired_tokens is set of tokens that have a pool with USDC.
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
    usdc_paired_tokens = set()

    for pool in uniswap_v3_pools:
        pool_addr = pool["address"].lower()
        tokens = pool.get("tokens", [])

        if len(tokens) == 2:  # noqa: PLR2004
            token0_addr = tokens[0]["address"].lower()
            token1_addr = tokens[1]["address"].lower()

            all_pool_tokens_map[pool_addr] = (token0_addr, token1_addr)

            # Track tokens that have a pool with USDC
            if token0_addr == USDC_ADDRESS.lower():
                usdc_paired_tokens.add(token1_addr)
            elif token1_addr == USDC_ADDRESS.lower():
                usdc_paired_tokens.add(token0_addr)

    logger.info("Found %d unique tokens with USDC pools", len(usdc_paired_tokens))

    return usdc_paired_tokens, all_pool_tokens_map


def filter_and_decode_usdc_swaps(
    input_dir: Path,
    output_file: Path,
    pools_file: Path,
) -> None:
    """Filter and decode USDC-paired swaps from raw swap data.

    Filter strategy: Keep swaps where at least one token has a USDC pool.
    This includes:
    - Direct USDC swaps (token <-> USDC)
    - Indirect swaps (tokenA <-> tokenB) where both have USDC pools

    Args:
        input_dir: Directory containing input parquet files.
        output_file: Output parquet file for filtered and decoded swaps.
        pools_file: Path to pools.json file.

    """
    logger.info("Loading pool information from %s...", pools_file)
    usdc_paired_tokens, all_pool_tokens_map = load_all_pools(pools_file)

    logger.info("Identified %d tokens with USDC pools", len(usdc_paired_tokens))

    # Create a polars dataframe with ALL pool token mappings for joining
    pool_tokens_df = pl.DataFrame(
        {
            "pool_address": list(all_pool_tokens_map.keys()),
            "token0": [tokens[0] for tokens in all_pool_tokens_map.values()],
            "token1": [tokens[1] for tokens in all_pool_tokens_map.values()],
        },
    )

    logger.info("Loading and filtering swap data from %s...", input_dir)

    # Use lazy evaluation to efficiently:
    # 1. Read all parquet files
    # 2. Filter to V3 swaps
    # 3. Join with pool info
    # 4. Filter to USDC-paired swaps
    df = (
        pl.scan_parquet(input_dir / "*.parquet")
        .with_columns(
            # Extract first topic (event signature)
            pl.col("topics").list.get(0).alias("event_signature"),
        )
        .filter(
            # Filter to V3 swaps only
            pl.col("event_signature") == f"0x{V3_SWAP_SIGNATURE}",
        )
        .drop("event_signature")  # Drop temporary column
        .collect()  # Execute the lazy query
    )

    logger.info("Filtered to %s Uniswap V3 swap events", f"{len(df):,}")

    # Get all pool info by joining with pool tokens
    df_with_pools = df.join(
        pool_tokens_df,
        left_on=pl.col("pool_or_manager_address").str.to_lowercase(),
        right_on="pool_address",
        how="left",
    )

    # Filter to swaps where at least one token has a USDC pool
    # This includes both direct USDC swaps and swaps between USDC-paired tokens
    logger.info("Filtering to swaps with USDC-paired tokens...")
    df_filtered = df_with_pools.filter(
        pl.col("token0").is_in(usdc_paired_tokens | {USDC_ADDRESS.lower()})
        | pl.col("token1").is_in(usdc_paired_tokens | {USDC_ADDRESS.lower()}),
    )

    logger.info("Filtered to %s swaps with USDC-paired tokens", f"{len(df_filtered):,}")

    # Decode swap data
    logger.info("Decoding swap events...")

    # Extract sender, recipient, and keep pool address in normalized form
    df_decoded = df_filtered.with_columns([
        # Extract sender and recipient from topics (skip 0x and padding)
        pl.col("topics").list.get(1).str.slice(26).str.to_lowercase().alias("sender"),
        pl.col("topics").list.get(2).str.slice(26).str.to_lowercase().alias("recipient"),
        # Keep pool address in normalized form
        pl.col("pool_or_manager_address").str.to_lowercase().alias("pool"),
    ])

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

    # Analyze USDC coverage
    logger.info("\nUSDC Coverage Analysis:")
    swaps_with_usdc = df_decoded.filter(
        (pl.col("token0") == USDC_ADDRESS.lower())
        | (pl.col("token1") == USDC_ADDRESS.lower()),
    )
    swaps_without_usdc = df_decoded.filter(
        (pl.col("token0") != USDC_ADDRESS.lower())
        & (pl.col("token1") != USDC_ADDRESS.lower()),
    )

    logger.info("  Total swaps: %s", f"{len(df_decoded):,}")
    logger.info(
        "  Swaps with USDC: %s (%.2f%%)",
        f"{len(swaps_with_usdc):,}",
        len(swaps_with_usdc) / len(df_decoded) * 100,
    )
    logger.info(
        "  Swaps between non-USDC tokens: %s (%.2f%%)",
        f"{len(swaps_without_usdc):,}",
        len(swaps_without_usdc) / len(df_decoded) * 100,
    )

    usdc_as_token0 = len(df_decoded.filter(pl.col("token0") == USDC_ADDRESS.lower()))
    usdc_as_token1 = len(df_decoded.filter(pl.col("token1") == USDC_ADDRESS.lower()))
    logger.info("  USDC as token0: %s", f"{usdc_as_token0:,}")
    logger.info("  USDC as token1: %s", f"{usdc_as_token1:,}")

    logger.info("\nSaving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.write_parquet(output_file)

    logger.info("Done! Saved %s swaps with USDC-paired tokens", f"{len(df_decoded):,}")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/swaps"),
    help="Directory containing input parquet files",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/usdc_paired_swaps.parquet"),
    help="Output parquet file path",
)
@click.option(
    "--pools-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/pools.json"),
    help="JSON file with pool information",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    input_dir: Path,
    output_file: Path,
    pools_file: Path,
    *,
    verbose: bool,
) -> None:
    """Filter and decode USDC-paired swaps from raw swap data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    filter_and_decode_usdc_swaps(input_dir, output_file, pools_file)


if __name__ == "__main__":
    main()
