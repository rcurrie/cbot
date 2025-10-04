"""Wrangle Uniswap V3 Swap Events."""

# %%
import json
import math
from pathlib import Path

import polars as pl
import web3
from eth_hash.auto import keccak

WETH_ADDRESS = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2".lower()
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".lower()


# %%
# Get a list of all Uniswap V3 pools on Ethereum with WETH as one of the tokens
# Downloaded from https://reference-data-api.kaiko.io/v1/pools
with Path("data/pools.json").open() as f:
    pools = json.load(f)["data"]

print(f"Total pools loaded: {len(pools)}")
print(f"Unique protocols: { {p['protocol'] for p in pools} }")
print(f"Unique blockchains: { {p['blockchain'] for p in pools} }")

# %%
# Filter to Ethereium Blockchain and Uniswap V3 pools only
pools = [p for p in pools if p["blockchain"] == "ethereum" and p["protocol"] == "usp3"]
print(f"Uniswap V3 pools on Ethereum: {len(pools)}")

weth_pools = [
    p for p in pools if any(t["address"].lower() == WETH_ADDRESS for t in p["tokens"])
]
print(f"Uniswap V3 pools on Ethereum with WETH: {len(weth_pools)}")

weth_pools_by_address = {p["address"].lower(): p for p in weth_pools}
with Path("data/weth_pools_by_address.json").open("w") as f:
    json.dump(weth_pools_by_address, f, indent=2)

weth_pools_addresses = set(weth_pools_by_address.keys())

weth_pool_addr_to_symbol = {
    p["address"].lower(): next(
        (
            t["symbol"]
            for t in p["tokens"]
            if t["address"].lower() != WETH_ADDRESS
        )
    )
    for p in weth_pools
}
with Path("data/weth_pool_addr_to_symbol.json").open("w") as f:
    json.dump(weth_pool_addr_to_symbol, f, indent=2)

# %%
# Uniswap v3 swap event signature and topic hash and data decoder
def compute_event_signature(event_signature: str) -> str:
    """Compute the keccak256 hash of an event signature."""
    event_signature_bytes = event_signature.encode("utf-8")
    event_signature_hash = keccak(event_signature_bytes)
    return web3.Web3.to_hex(event_signature_hash)


v3_topic_hash = compute_event_signature(
    "Swap(address,address,int256,int256,uint160,uint128,int24)",
)
print(f"Uniswap V3 Swap event topic hash: {v3_topic_hash}")


# %%
# Load previously ingested swap events from Parquet files
# topics[0] keccak256 hash of the non-indexed event signature
# topics[1] Padded 32-byte representation of the sender address
# topics[2] Padded 32-byte representation of the recipient address
# topics[3] Always null
swaps_df = (
    pl.scan_parquet("data/swaps/")
    .filter(pl.col("topics").list.get(0) == v3_topic_hash)
    .with_columns(
        (pl.lit("0x") + pl.col("topics").list.get(2).str.slice(-40)).alias(
            "recipient_addr",
        ),
    )
    .select(
        [
            pl.col("block_timestamp"),
            # pl.col("block_number"),
            pl.col("transaction_hash"),
            pl.col("pool_or_manager_address").alias("pool_addr"),
            # pl.col("topics"),
            pl.col("data"),
        ],
    )
)

print("\nOptimized query result (first 5 rows):")
print(swaps_df.collect().head())

# %%
# Filter to only WETH pools
weth_swaps_df = swaps_df.filter(pl.col("pool_addr").is_in(weth_pools_addresses))

print("\nFiltered to WETH pools:")
print(f"Total swaps: {weth_swaps_df.select(pl.len()).collect().item()}")
print("\nFirst 5 WETH pool swaps:")
print(weth_swaps_df.collect().head())

# %%
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
        amount_weth = 0.0
        amount_token = 0.0
        price_in_weth = 0.0

    # Calculate log price (natural log) for price in WETH
    # Handle edge cases: log(0) -> -inf, log(negative) -> nan
    log_price_in_weth = math.log(price_in_weth) if price_in_weth > 0 else float("-inf")

    return {
        "addr0": token0_addr,
        "addr1": token1_addr,
        "amount_weth": amount_weth,
        "amount_token": amount_token,
        "price_in_weth": price_in_weth,
        "log_price_in_weth": log_price_in_weth,
    }


# %%
# Decode and normalize in a single pass
weth_swaps_decoded_df = (
    weth_swaps_df.with_columns(
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

print("\nDecoded WETH swaps with normalized amounts and prices:")
print(weth_swaps_decoded_df.collect().head())

# %%
# Resample to hourly data per pool with complete time grid
collected_swaps = weth_swaps_decoded_df.collect()

# Get time range
min_time = collected_swaps.select(pl.col("block_timestamp").min()).item()
max_time = collected_swaps.select(pl.col("block_timestamp").max()).item()

# Create hourly time grid
time_grid = pl.datetime_range(
    min_time.replace(minute=0, second=0, microsecond=0),
    max_time.replace(minute=0, second=0, microsecond=0),
    interval="1h",
    eager=True,
).alias("block_timestamp")

# Get list of all pools
all_pools = collected_swaps.select(pl.col("pool_addr").unique()).to_series()

# Create complete grid: all pools × all hours
complete_grid = (
    pl.DataFrame({"block_timestamp": time_grid})
    .join(
        pl.DataFrame({"pool_addr": all_pools}),
        how="cross",
    )
)

# Aggregate actual swaps per hour
hourly_swaps = (
    collected_swaps.sort("block_timestamp")
    .with_columns(
        pl.col("block_timestamp").dt.truncate("1h").alias("hour"),
    )
    .group_by(["pool_addr", "hour"])
    .agg(
        [
            pl.col("log_price_in_weth").last().alias("log_price"),
            pl.col("amount_weth").abs().sum().alias("amount_weth_sum"),
            pl.col("amount_token").abs().sum().alias("amount_token_sum"),
        ],
    )
    .rename({"hour": "block_timestamp"})
)

# Join with complete grid and forward fill
hourly_df = (
    complete_grid.join(hourly_swaps, on=["pool_addr", "block_timestamp"], how="left")
    .sort(["pool_addr", "block_timestamp"])
    .with_columns(
        [
            # Forward fill log_price per pool
            pl.col("log_price").forward_fill().over("pool_addr"),
            # Fill volume columns with 0 where no swaps occurred
            pl.col("amount_weth_sum").fill_null(0.0),
            pl.col("amount_token_sum").fill_null(0.0),
        ],
    )
    .with_columns(
        [
            # Combined volume
            (pl.col("amount_weth_sum") + pl.col("amount_token_sum")).alias("volume"),
        ],
    )
    .with_columns(
        [
            # Log volume
            pl.col("volume").log1p().alias("log_volume"),
        ],
    )
)

print("\nHourly resampled data:")
print(hourly_df.head(10))

# %%
hourly_df.write_parquet(
    "data/hourly_weth_swaps.parquet",
    compression="snappy",
)
# %%
