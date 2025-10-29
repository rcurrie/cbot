"""Wrangle Uniswap V3 Swap Events on the Ethereum Blockchain."""

# %%
import json
import math
from pathlib import Path

import polars as pl
import web3
from eth_hash.auto import keccak

WETH_ADDRESS = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2".lower()

# %%
# Filter to only Uniswap V3 pools on Ethereum with WETH as one of the tokens
# and save utility json lookup files
with Path("data/pools.json").open() as f:
    pools = json.load(f)["data"]

print(f"Pools: {len(pools)}")
print(f"Protocols: { {p['protocol'] for p in pools} }")
print(f"Blockchains: { {p['blockchain'] for p in pools} }")

# Generate a list of Uniswap V3 pools on Ethereum with WETH
ethereum_usp3_weth_pools = [
    p
    for p in pools
    if p["blockchain"] == "ethereum"
    and p["protocol"] == "usp3"
    and any(t["address"].lower() == WETH_ADDRESS for t in p["tokens"])
]
print(f"Ethereum Uniswap V3 WETH pools: {len(ethereum_usp3_weth_pools):,}")
with Path("data/ethereum_usp3_weth_pools.json").open("w") as f:
    json.dump(ethereum_usp3_weth_pools, f, indent=2)

weth_pools_by_address = {p["address"].lower(): p for p in ethereum_usp3_weth_pools}
with Path("data/weth_pools_by_address.json").open("w") as f:
    json.dump(weth_pools_by_address, f, indent=2)

weth_pools_addresses = pl.Series(
    [p["address"].lower() for p in ethereum_usp3_weth_pools],
)

weth_pool_addr_to_symbol = {
    p["address"].lower(): next(
        t["symbol"] for t in p["tokens"] if t["address"].lower() != WETH_ADDRESS
    )
    for p in ethereum_usp3_weth_pools
}
with Path("data/weth_pool_addr_to_symbol.json").open("w") as f:
    json.dump(weth_pool_addr_to_symbol, f, indent=2)


# %%
# ================================================================================
# Load and filter swap events to only Ethereum Uniswap V3 WETH pools
# ================================================================================
def compute_event_signature(event_signature: str) -> str:
    """Compute the keccak256 hash of an event signature."""
    event_signature_bytes = event_signature.encode("utf-8")
    event_signature_hash = keccak(event_signature_bytes)
    return web3.Web3.to_hex(event_signature_hash)


v3_topic_hash = compute_event_signature(
    "Swap(address,address,int256,int256,uint160,uint128,int24)",
)
print(f"Uniswap V3 Swap event topic hash: {v3_topic_hash}")

# Load previously ingested swap events from Parquet files, columns:
# block_timestamp,
# block_number,
# transaction_hash,
# address AS pool_or_manager_address,
# topics,
# data
# topics[0] keccak256 hash of the non-indexed event signature
# topics[1] Padded 32-byte representation of the sender address
# topics[2] Padded 32-byte representation of the recipient address
# topics[3] Always null
swaps_df = pl.scan_parquet("data/swaps/")
print(f"Total swap events loaded: {swaps_df.select(pl.len()).collect().item():,}")

print(
    f"Total unique swap signatures: {swaps_df.select(pl.col('topics').list.get(0).n_unique()).collect().item():,}",
)

swaps_df = swaps_df.filter(pl.col("topics").list.get(0) == v3_topic_hash).select(
    [
        pl.col("block_timestamp"),
        pl.col("transaction_hash"),
        pl.col("pool_or_manager_address").alias("pool_addr"),
        pl.col("data"),
    ],
)

print(f"Total Uniswap V3 Swaps: {swaps_df.select(pl.len()).collect().item():,}")

# Filter to only WETH pools
swaps_df = swaps_df.filter(pl.col("pool_addr").is_in(weth_pools_addresses.implode()))
print(f"Uniswap V3 Swaps of WETH pools: {swaps_df.collect().shape[0]:,}")


# %%
# ================================================================================
# Decode and normalize swap data to human-readable amounts and prices
# ================================================================================
# Decode swap data and normalize amounts/prices using token decimals
# The data field contains (non-indexed parameters encoded as ABI):
# - amount0 (int256): 32 bytes at offset 0
# - amount1 (int256): 32 bytes at offset 64
# - sqrtPriceX96 (uint160): 32 bytes at offset 128
# - liquidity (uint128): 32 bytes at offset 192
# - tick (int24): 32 bytes at offset 224
def decode_and_normalize_swap(
    pool_addr: str,
    data_hex: str,
) -> dict[str, str | float]:
    """Decode swap data and normalize amounts/prices to human-readable WETH values.

    For WETH pools, returns:
    - addr0, addr1: token addresses
    - amount_weth: WETH amount traded (normalized by decimals)
    - amount_token: other token amount traded (normalized by decimals)
    - price_in_weth: price of the non-WETH token in WETH
    """
    # Remove '0x' prefix
    data = data_hex.removeprefix("0x")

    # Extract 32-byte chunks (64 hex chars each)
    amount0_hex = data[0:64]
    amount1_hex = data[64:128]
    sqrt_price_x96_hex = data[128:192]

    # Convert to integers (signed for amounts, unsigned for price)
    amount0_raw = int.from_bytes(
        bytes.fromhex(amount0_hex),
        byteorder="big",
        signed=True,
    )
    amount1_raw = int.from_bytes(
        bytes.fromhex(amount1_hex),
        byteorder="big",
        signed=True,
    )
    sqrt_price_x96 = int.from_bytes(
        bytes.fromhex(sqrt_price_x96_hex),
        byteorder="big",
        signed=False,
    )
    assert sqrt_price_x96 > 0, "sqrtPriceX96 should be positive in a swap"

    # Get pool info
    pool = weth_pools_by_address.get(pool_addr.lower())
    if not pool:
        return {
            "addr0": "",
            "addr1": "",
            "amount_weth": 0.0,
            "amount_token": 0.0,
            "price_in_weth": 0.0,
        }

    token0 = pool["tokens"][0]
    token1 = pool["tokens"][1]
    token0_addr = token0["address"].lower()
    token1_addr = token1["address"].lower()
    decimals0 = int(token0["decimals"])
    decimals1 = int(token1["decimals"])

    # Normalize amounts by decimals
    amount0 = float(amount0_raw) / (10**decimals0)
    amount1 = float(amount1_raw) / (10**decimals1)

    # Calculate price from sqrtPriceX96
    # price1_per_token0 = (sqrt_price_x96 / 2^96)^2
    # This is the raw price ratio, needs decimal adjustment
    price1_per_token0_raw = (sqrt_price_x96 / (2**96)) ** 2

    # Adjust for decimals: price in real units = raw_price * (10^decimals0 / 10^decimals1)
    price1_per_token0 = price1_per_token0_raw * (10**decimals0) / (10**decimals1)

    # Determine which token is WETH and extract normalized values
    if token0_addr == WETH_ADDRESS:
        # token0 is WETH, token1 is the other token
        amount_weth = amount0
        amount_token = amount1
        price_in_weth = price1_per_token0  # price of token1 in terms of WETH
    elif token1_addr == WETH_ADDRESS:
        # token1 is WETH, token0 is the other token
        amount_weth = amount1
        amount_token = amount0
        price_in_weth = 1.0 / price1_per_token0 if price1_per_token0 != 0 else 0.0
    else:
        # Neither token is WETH (shouldn't happen with our filter)
        assert False, "Pool does not contain WETH"

    # Calculate log price (natural log) for price in WETH
    # Handle edge cases: log(0) -> -inf, log(negative) -> nan
    log_price_in_weth = math.log(price_in_weth) if price_in_weth > 0 else float("-inf")

    assert amount_weth is not None, "amount_weth should not be null"
    assert amount_token is not None, "amount_token should not be null"
    assert price_in_weth is not None, "price_in_weth should not be null"
    assert log_price_in_weth is not None, "log_price_in_weth should not be null"

    return {
        "addr0": token0_addr,
        "addr1": token1_addr,
        "amount_weth": amount_weth,
        "amount_token": amount_token,
        "price_in_weth": price_in_weth,
        "log_price_in_weth": log_price_in_weth,
    }


# Decode and normalize in a single pass
swaps_df = (
    swaps_df.with_columns(
        pl.struct(["pool_addr", "data"])
        .map_elements(
            lambda x: decode_and_normalize_swap(x["pool_addr"], x["data"]),
            return_dtype=pl.Struct(
                [
                    pl.Field("addr0", pl.String),
                    pl.Field("addr1", pl.String),
                    pl.Field("amount_weth", pl.Float64),
                    pl.Field("amount_token", pl.Float64),
                    pl.Field("price_in_weth", pl.Float64),
                    pl.Field("log_price_in_weth", pl.Float64),
                ],
            ),
        )
        .alias("decoded"),
    )
    .with_columns(
        pl.col("decoded").struct.field("addr0").alias("addr0"),
        pl.col("decoded").struct.field("addr1").alias("addr1"),
        pl.col("decoded").struct.field("amount_weth").alias("amount_weth"),
        pl.col("decoded").struct.field("amount_token").alias("amount_token"),
        pl.col("decoded").struct.field("price_in_weth").alias("price_in_weth"),
        pl.col("decoded").struct.field("log_price_in_weth").alias("log_price_in_weth"),
    )
    .drop("decoded", "data")
)

# Filter any rows where amount_weth or amount_token is null or non-positive
swaps_df = swaps_df.filter(
    (abs(pl.col("amount_weth")) > 0.0) & (abs(pl.col("amount_token")) > 0.0),
)
print(f"Non-zero uniswap v3 weth swaps: {swaps_df.collect().shape[0]:,}")


# %%
# Verify that all log_price_in_weth are finite (not -inf or nan)
num_invalid_prices = (
    swaps_df.filter(
        ~pl.col("log_price_in_weth").is_finite(),
    )
    .select(pl.len())
    .collect()
    .item()
)
print(f"Number of swaps with invalid log_price_in_weth: {num_invalid_prices}")

# Verify that all the absolute amounts are positive
num_invalid_amounts = (
    swaps_df.filter(
        (pl.col("amount_weth").abs() <= 0.0) | (pl.col("amount_token").abs() <= 0.0),
    )
    .select(pl.len())
    .collect()
    .item()
)
print(f"Number of swaps with non-positive amounts: {num_invalid_amounts}")

# %%
# Save to Parquet
swaps_df.collect().write_parquet(Path("data/swaps.parquet"))
print("Saved decoded WETH swaps to data/swaps.parquet")
