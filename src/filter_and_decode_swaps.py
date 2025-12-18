"""Filter USDC-paired swaps from raw swap data.

Load all parquet files from data/swaps/, filter to only Uniswap V3 Swap events
that involve USDC-paired tokens.

This script outputs a parquet file with one row per swap, keeping the raw data
field for on-demand decoding. It filters to include:
1. Swaps that have USDC as one of the tokens
2. Swaps between tokens A and B where both A and B have pools with USDC
"""

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import typer
from web3 import Web3

app = typer.Typer()

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

# Constants for int24 decoding (3-byte signed integer)
INT24_SIGN_BIT = 0x800000  # 2^23
INT24_MAX = 0x1000000  # 2^24


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


def decode_sqrt_price_x96(data_hex: str) -> int:
    """Decode sqrtPriceX96 from Uniswap V3 Swap event data.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        sqrtPriceX96 as an unsigned integer.

    """
    # Remove 0x prefix
    data = data_hex.removeprefix("0x")

    # Extract sqrtPriceX96 (bytes 128-192, third 32-byte chunk)
    sqrt_price_hex = data[128:192]

    # Decode as unsigned integer (uint160)
    return int.from_bytes(bytes.fromhex(sqrt_price_hex), "big", signed=False)


def decode_liquidity(data_hex: str) -> int:
    """Decode liquidity from Uniswap V3 Swap event data.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        Liquidity as an unsigned integer.

    """
    # Remove 0x prefix
    data = data_hex.removeprefix("0x")

    # Extract liquidity (bytes 192-256, fourth 32-byte chunk)
    liquidity_hex = data[192:256]

    # Decode as unsigned integer (uint128)
    return int.from_bytes(bytes.fromhex(liquidity_hex), "big", signed=False)


def decode_tick(data_hex: str) -> int:
    """Decode tick from Uniswap V3 Swap event data.

    Args:
        data_hex: Hex string of swap event data (with 0x prefix).

    Returns:
        Tick as a signed 24-bit integer.

    """
    # Remove 0x prefix
    data = data_hex.removeprefix("0x")

    # Extract tick (last 3 bytes of the fifth 32-byte chunk)
    # The tick is int24, stored in the last 3 bytes
    tick_hex = data[256 + 58 : 256 + 64]  # Last 3 bytes (6 hex chars)

    # Decode as signed 24-bit integer
    tick_unsigned = int.from_bytes(bytes.fromhex(tick_hex), "big", signed=False)

    # Convert to signed int24
    if tick_unsigned >= INT24_SIGN_BIT:
        return tick_unsigned - INT24_MAX
    return tick_unsigned


def load_token_decimals(pools: list[dict[str, Any]]) -> dict[str, int]:
    """Load token decimals from pool data.

    Args:
        pools: List of pool dictionaries with token information.

    Returns:
        Dictionary mapping token_address -> decimals.

    """
    decimals_map = {}
    for pool in pools:
        for token in pool.get("tokens", []):
            addr = token["address"].lower()
            decimals_map[addr] = int(token["decimals"])

    logger.info("Loaded decimals for %d tokens", len(decimals_map))
    return decimals_map


def load_all_pools(
    pools_file: Path,
) -> tuple[set[str], dict[str, tuple[str, str]], list[dict[str, Any]]]:
    """Load all Uniswap V3 pool information.

    Args:
        pools_file: Path to pools.json file (all pools, not just USDC).

    Returns:
        Tuple of (usdc_paired_tokens, all_pool_tokens_map, uniswap_v3_pools).
        usdc_paired_tokens is set of tokens that have a pool with USDC.
        all_pool_tokens_map maps all pool_address -> (token0, token1).
        uniswap_v3_pools is the list of all Uniswap V3 pools for decimals.

    """
    logger.info("Loading pool information from %s...", pools_file)
    with pools_file.open() as f:
        data = json.load(f)

    # Extract Uniswap V3 Ethereum pools
    all_pools = data.get("data", [])
    uniswap_v3_pools = [
        p
        for p in all_pools
        if p.get("protocol") == "usp3" and p.get("blockchain") == "ethereum"
    ]

    logger.info("  %d Uniswap V3 Ethereum pools", len(uniswap_v3_pools))

    # Build mapping of all pool addresses to token pairs
    all_pool_tokens_map = {}
    usdc_paired_tokens = set()

    for pool in uniswap_v3_pools:
        pool_addr = pool["address"].lower()
        tokens = pool.get("tokens", [])
        assert len(tokens) == 2, f"Pool {pool_addr} does not have exactly 2 tokens"

        token0_addr = tokens[0]["address"].lower()
        token1_addr = tokens[1]["address"].lower()

        all_pool_tokens_map[pool_addr] = (token0_addr, token1_addr)

        # Track tokens that have a pool with USDC
        if token0_addr == USDC_ADDRESS.lower():
            usdc_paired_tokens.add(token1_addr)
        elif token1_addr == USDC_ADDRESS.lower():
            usdc_paired_tokens.add(token0_addr)

    logger.info("  %d unique tokens with USDC pools", len(usdc_paired_tokens))

    # Build token symbol mapping
    token_symbols = {}
    for pool in uniswap_v3_pools:
        for token in pool.get("tokens", []):
            addr = token["address"].lower()
            symbol = token.get("symbol", "UNKNOWN")
            token_symbols[addr] = symbol

    # Count how many pools each token is a member of
    token_pool_counts: dict[str, int] = {}
    for token0_addr, token1_addr in all_pool_tokens_map.values():
        token_pool_counts[token0_addr] = token_pool_counts.get(token0_addr, 0) + 1
        token_pool_counts[token1_addr] = token_pool_counts.get(token1_addr, 0) + 1

    # Build histogram: how many tokens have N pools
    # Also track which tokens for single-token cases
    pools_per_token_hist: dict[int, int] = {}
    pools_to_tokens: dict[int, list[str]] = {}
    for token_addr, num_pools in token_pool_counts.items():
        pools_per_token_hist[num_pools] = pools_per_token_hist.get(num_pools, 0) + 1
        if num_pools not in pools_to_tokens:
            pools_to_tokens[num_pools] = []
        pools_to_tokens[num_pools].append(token_addr)

    logger.info("  Pools per token histogram:")
    for num_pools in sorted(pools_per_token_hist.keys()):
        num_tokens = pools_per_token_hist[num_pools]
        if num_tokens == 1:
            token_addr = pools_to_tokens[num_pools][0]
            symbol = token_symbols.get(token_addr, "UNKNOWN")
            logger.info(
                "    %d token with %d pools (%s)",
                num_tokens,
                num_pools,
                symbol,
            )
        else:
            logger.info("    %d tokens with %d pools", num_tokens, num_pools)

    # Build histogram of pools by fee level
    fee_pool_counts: dict[str, int] = {}
    for pool in uniswap_v3_pools:
        fee = pool.get("fee", "0")
        fee_pool_counts[fee] = fee_pool_counts.get(fee, 0) + 1

    logger.info("  Pools by fee level histogram:")
    for fee in sorted(fee_pool_counts.keys()):
        logger.info("    %s fee: %d pools", fee, fee_pool_counts[fee])

    return usdc_paired_tokens, all_pool_tokens_map, uniswap_v3_pools


def validate_output(df: pl.DataFrame) -> None:
    """Validate the output dataset for common issues.

    This function checks for data quality issues that could cause problems
    downstream in the pipeline, particularly issues that were found in the
    final training data validation.

    Args:
        df: The filtered and decoded swaps DataFrame.

    """
    logger.info("=" * 60)
    logger.info("VALIDATING OUTPUT DATA")
    logger.info("=" * 60)

    issues_found = []

    # 1. Check for duplicate swaps (same transaction + pool)
    logger.info("Checking for duplicate swaps...")
    dup_count = (
        df.group_by(["transaction_hash", "pool"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .shape[0]
    )
    if dup_count > 0:
        issues_found.append(f"Found {dup_count:,} duplicate swaps (same tx + pool)")
        logger.warning("  ✗ WARN: %d duplicate swaps detected", dup_count)
    else:
        logger.info("  ✓ PASS: No duplicate swaps")

    # 2. Check for null/missing values in critical columns
    logger.info("Checking for null values in critical columns...")
    critical_cols = [
        "block_timestamp",
        "transaction_hash",
        "pool",
        "token0",
        "token1",
        "data",
    ]
    for col in critical_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            pct = (null_count / df.shape[0]) * 100
            issues_found.append(f"{col} has {null_count:,} nulls ({pct:.2f}%)")
            logger.warning("  ✗ WARN: %s has %d nulls (%.2f%%)", col, null_count, pct)
        else:
            logger.info("  ✓ PASS: %s has no nulls", col)

    # 3. Validate hex data format (should start with 0x and have correct length)
    logger.info("Validating hex data format...")
    invalid_hex_count = df.filter(
        ~pl.col("data").str.starts_with("0x")
        | (pl.col("data").str.len_chars() < 10),  # At least some data
    ).shape[0]
    if invalid_hex_count > 0:
        pct = (invalid_hex_count / df.shape[0]) * 100
        issues_found.append(
            f"{invalid_hex_count:,} swaps have invalid hex data ({pct:.2f}%)",
        )
        logger.warning("  ✗ WARN: %d swaps with invalid hex data", invalid_hex_count)
    else:
        logger.info("  ✓ PASS: All hex data properly formatted")

    # 4. Validate token addresses (should be 42 chars with 0x prefix)
    logger.info("Validating token addresses...")
    invalid_token0 = df.filter(pl.col("token0").str.len_chars() != 42).shape[0]
    invalid_token1 = df.filter(pl.col("token1").str.len_chars() != 42).shape[0]
    if invalid_token0 > 0 or invalid_token1 > 0:
        issues_found.append(
            f"{invalid_token0:,} invalid token0 addresses, "
            f"{invalid_token1:,} invalid token1 addresses",
        )
        logger.warning(
            "  ✗ WARN: %d invalid token0, %d invalid token1 addresses",
            invalid_token0,
            invalid_token1,
        )
    else:
        logger.info("  ✓ PASS: All token addresses valid")

    # 5. Validate decimals (should be 0-18, typically)
    logger.info("Validating token decimals...")
    invalid_decimals0 = df.filter(
        (pl.col("token0_decimals") < 0) | (pl.col("token0_decimals") > 77),
    ).shape[0]
    invalid_decimals1 = df.filter(
        (pl.col("token1_decimals") < 0) | (pl.col("token1_decimals") > 77),
    ).shape[0]
    if invalid_decimals0 > 0 or invalid_decimals1 > 0:
        issues_found.append(
            f"{invalid_decimals0:,} invalid token0 decimals, "
            f"{invalid_decimals1:,} invalid token1 decimals",
        )
        logger.warning(
            "  ✗ WARN: %d invalid token0, %d invalid token1 decimals",
            invalid_decimals0,
            invalid_decimals1,
        )
    else:
        logger.info("  ✓ PASS: All token decimals valid")

    # Log unusual decimal values
    logger.info("  Token decimals distribution (token0):")
    decimals0_counts = df["token0_decimals"].value_counts().sort("token0_decimals")
    for row in decimals0_counts.iter_rows(named=True):
        logger.info(
            "    %d decimals: %d occurrences",
            row["token0_decimals"],
            row["count"],
        )

    # 6. Check timestamp ordering
    logger.info("Checking timestamp ordering...")
    is_sorted = df["block_timestamp"].is_sorted()
    if not is_sorted:
        issues_found.append("Timestamps are not sorted")
        logger.warning("  ✗ WARN: Timestamps not sorted")
    else:
        logger.info("  ✓ PASS: Timestamps sorted")

    # 7. Check for large time gaps
    logger.info("Checking for large time gaps...")
    time_diffs = df.select(pl.col("block_timestamp").diff().alias("time_diff")).filter(
        pl.col("time_diff").is_not_null(),
    )
    if time_diffs.shape[0] > 0:
        max_gap = time_diffs["time_diff"].max()
        max_gap_seconds = max_gap.total_seconds() if max_gap else 0
        if max_gap_seconds > 3600:  # 1 hour
            logger.warning(
                "  ✗ WARN: Large time gap detected: %s (%.0f seconds)",
                max_gap,
                max_gap_seconds,
            )
            issues_found.append(f"Max time gap: {max_gap_seconds:.0f} seconds")
        else:
            logger.info(
                "  ✓ PASS: No large time gaps (max: %.0f seconds)",
                max_gap_seconds,
            )

    # 8. Validate that we can decode at least a sample of swap data
    logger.info("Validating swap data decoding on sample...")
    sample_size = min(100, df.shape[0])
    sample_df = df.head(sample_size)
    decode_failures = 0

    for row in sample_df.iter_rows(named=True):
        try:
            # Try to decode amounts
            decode_swap_amounts(row["data"])
            # Try to decode other fields
            decode_sqrt_price_x96(row["data"])
            decode_liquidity(row["data"])
            decode_tick(row["data"])
        except Exception as e:
            decode_failures += 1
            if decode_failures == 1:  # Log first failure
                logger.warning("  Decode failure example: %s", str(e))

    if decode_failures > 0:
        pct = (decode_failures / sample_size) * 100
        issues_found.append(
            f"{decode_failures}/{sample_size} swaps failed to decode ({pct:.1f}%)",
        )
        logger.warning(
            "  ✗ WARN: %d/%d swaps failed to decode (%.1f%%)",
            decode_failures,
            sample_size,
            pct,
        )
    else:
        logger.info("  ✓ PASS: All sampled swaps decode successfully")

    # 9. Check for unusual patterns that might indicate data issues
    logger.info("Checking for unusual patterns...")

    # Same sender and recipient - normal for DEX swaps but check concentration
    same_addr_df = df.filter(pl.col("sender") == pl.col("recipient"))
    same_sender_recipient = same_addr_df.shape[0]

    if same_sender_recipient > 0:
        pct = (same_sender_recipient / df.shape[0]) * 100
        unique_addrs = same_addr_df["sender"].n_unique()

        # Check concentration (top 10 and top 100)
        addr_counts = (
            same_addr_df.group_by("sender")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        top_10_count = addr_counts.head(10)["count"].sum()
        top_100_count = addr_counts.head(100)["count"].sum()

        conc_10 = (
            (top_10_count / same_sender_recipient * 100) if same_sender_recipient > 0 else 0
        )
        conc_100 = (
            (top_100_count / same_sender_recipient * 100) if same_sender_recipient > 0 else 0
        )

        logger.info(
            "  ℹ INFO: %d swaps (%.2f%%) have same sender/recipient",
            same_sender_recipient,
            pct,
        )
        logger.info(
            "    - %d unique addresses (avg: %.1f swaps/address)",
            unique_addrs,
            same_sender_recipient / unique_addrs if unique_addrs > 0 else 0,
        )
        logger.info(
            "    - Top 10 addresses: %.1f%% | Top 100: %.1f%%",
            conc_10,
            conc_100,
        )

        if conc_100 > 90:
            logger.info(
                "    - HIGH CONCENTRATION: Likely MEV bots/arbitrageurs (normal for DEX)",
            )
        elif conc_10 > 50:
            logger.info(
                "    - MODERATE-HIGH: Some concentrated activity (typical for DEX)",
            )
        else:
            logger.info(
                "    - Low concentration: Diverse user base",
            )

    # 10. Summary
    logger.info("=" * 60)
    if issues_found:
        logger.warning("VALIDATION SUMMARY: %d issues found", len(issues_found))
        for issue in issues_found:
            logger.warning("  - %s", issue)
    else:
        logger.info("VALIDATION SUMMARY: All checks passed ✓")
    logger.info("=" * 60)


def filter_and_decode_usdc_swaps(
    input_dir: Path,
    output_file: Path,
    pools_file: Path,
) -> None:
    """Filter and decode USDC-paired swaps from raw swap data.

    Filter strategy:
    1. Direct USDC swaps: Include any swap where USDC is token0 or token1
    2. Indirect swaps: Include swaps between tokens A and B where BOTH A and B
       have direct pools with USDC

    Args:
        input_dir: Directory containing input parquet files.
        output_file: Output parquet file for filtered and decoded swaps.
        pools_file: Path to pools.json file.

    """
    usdc_paired_tokens, all_pool_tokens_map, uniswap_v3_pools = load_all_pools(
        pools_file,
    )

    # Load token decimals for later use
    decimals_map = load_token_decimals(uniswap_v3_pools)

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
    # 2. Sort by timestamp early to ensure temporal ordering
    # 3. Filter to V3 swaps
    # 4. Join with pool info
    # 5. Filter to USDC-paired swaps
    df = (
        pl.scan_parquet(input_dir / "*.parquet")
        .sort("block_timestamp")  # Sort immediately after loading
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

    # Filter to swaps based on USDC connectivity:
    # 1. Direct USDC swaps: Either token is USDC
    # 2. Indirect swaps: BOTH tokens have pools with USDC
    logger.info("Filtering to swaps with USDC connectivity...")
    usdc_lower = USDC_ADDRESS.lower()

    df_filtered = df_with_pools.filter(
        # Direct USDC swaps: either token is USDC
        (pl.col("token0") == usdc_lower)
        | (pl.col("token1") == usdc_lower)
        # Indirect swaps: both tokens have pools with USDC
        | (
            pl.col("token0").is_in(usdc_paired_tokens)
            & pl.col("token1").is_in(usdc_paired_tokens)
        ),
    )

    logger.info("Filtered to %s swaps with USDC connectivity", f"{len(df_filtered):,}")

    # Decode swap data
    logger.info("Decoding swap events...")

    # Extract sender, recipient, and keep pool address in normalized form
    df_decoded = df_filtered.with_columns(
        [
            # Extract sender and recipient from topics (skip 0x and padding)
            pl.col("topics")
            .list.get(1)
            .str.slice(26)
            .str.to_lowercase()
            .alias("sender"),
            pl.col("topics")
            .list.get(2)
            .str.slice(26)
            .str.to_lowercase()
            .alias("recipient"),
            # Keep pool address in normalized form
            pl.col("pool_or_manager_address").str.to_lowercase().alias("pool"),
        ],
    )

    # Add token decimals without decoding swap data
    logger.info("Adding token decimals...")
    df_final = df_decoded.with_columns(
        [
            pl.col("token0")
            .map_elements(
                lambda t: decimals_map.get(t, 18),
                return_dtype=pl.Int32,
            )
            .alias("token0_decimals"),
            pl.col("token1")
            .map_elements(
                lambda t: decimals_map.get(t, 18),
                return_dtype=pl.Int32,
            )
            .alias("token1_decimals"),
        ],
    )

    # Select final columns, keeping raw 'data' field for on-demand decoding
    df_final = df_final.select(
        [
            "block_timestamp",
            "block_number",
            "transaction_hash",
            "pool",
            "token0",
            "token1",
            "token0_decimals",
            "token1_decimals",
            "sender",
            "recipient",
            "data",  # Keep raw hex data for decoding on-demand
        ],
    )

    # Sort by timestamp
    df_final = df_final.sort("block_timestamp")

    # Deduplicate: Remove exact duplicate swaps (same tx + pool)
    # This can happen if raw data contains duplicates or multi-hop swaps
    logger.info("Deduplicating swaps...")
    rows_before = df_final.shape[0]
    df_final = df_final.unique(subset=["transaction_hash", "pool"], keep="first")
    rows_after = df_final.shape[0]
    if rows_before > rows_after:
        logger.warning(
            "Removed %d duplicate swaps (%.2f%%)",
            rows_before - rows_after,
            ((rows_before - rows_after) / rows_before) * 100,
        )
    else:
        logger.info("No duplicates found")

    # Ensure data is sorted after deduplication (for safety)
    df_final = df_final.sort("block_timestamp")

    # Analyze USDC coverage
    logger.info("USDC Coverage Analysis:")
    logger.info("  Total swaps: %s", f"{len(df_final):,}")

    usdc_as_token0 = len(df_final.filter(pl.col("token0") == USDC_ADDRESS.lower()))
    usdc_as_token1 = len(df_final.filter(pl.col("token1") == USDC_ADDRESS.lower()))
    logger.info("  USDC as token0: %s", f"{usdc_as_token0:,}")
    logger.info("  USDC as token1: %s", f"{usdc_as_token1:,}")

    swaps_with_usdc = df_final.filter(
        (pl.col("token0") == USDC_ADDRESS.lower())
        | (pl.col("token1") == USDC_ADDRESS.lower()),
    )
    logger.info(
        "  Swaps between USDC and a token: %s (%.2f%%)",
        f"{len(swaps_with_usdc):,}",
        len(swaps_with_usdc) / len(df_final) * 100 if len(df_final) > 0 else 0,
    )

    swaps_without_usdc = df_final.filter(
        (pl.col("token0") != USDC_ADDRESS.lower())
        & (pl.col("token1") != USDC_ADDRESS.lower()),
    )
    logger.info(
        "  Swaps where neither token is USDC: %s (%.2f%%)",
        f"{len(swaps_without_usdc):,}",
        len(swaps_without_usdc) / len(df_final) * 100 if len(df_final) > 0 else 0,
    )

    all_tokens = set(df_final["token0"]).union(set(df_final["token1"]))
    logger.info("  Total unique tokens in final dataset: %d", len(all_tokens))

    logger.info("Saving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(output_file)

    logger.info(
        "Done! Saved %s decoded swaps with USDC connectivity",
        f"{len(df_final):,}",
    )

    # Run validation on output
    validate_output(df_final)


@app.command()
def main(
    input_dir: Path = typer.Option(
        "data/swaps",
        "--input-dir",
        help="Directory containing input parquet files",
    ),
    output_file: Path = typer.Option(
        "data/usdc_paired_swaps.parquet",
        "--output-file",
        help="Output parquet file path",
    ),
    pools_file: Path = typer.Option(
        "data/pools.json",
        "--pools-file",
        help="JSON file with pool information",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
) -> None:
    """Filter and decode USDC-paired swaps from raw swap data."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Convert string paths to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_file)
    pools_path = Path(pools_file)

    filter_and_decode_usdc_swaps(input_path, output_path, pools_path)


if __name__ == "__main__":
    app()
