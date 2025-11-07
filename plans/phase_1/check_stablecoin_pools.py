"""Check what pools USDT and DAI are in."""

import json
import polars as pl

USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI_ADDRESS = "0x6b175474e89094c44da98b954eedeac495271d0f"

# Load pools
with open("data/pools.json") as f:
    pools_data = json.load(f)

all_pools = pools_data.get("data", [])
uniswap_v3_pools = [
    p
    for p in all_pools
    if p.get("protocol") == "usp3" and p.get("blockchain") == "ethereum"
]

print(f"Total Uniswap V3 Ethereum pools: {len(uniswap_v3_pools):,}")

# Find USDT pools
print("\n" + "=" * 70)
print("USDT POOLS")
print("=" * 70)

usdt_pools = [
    p for p in uniswap_v3_pools
    if any(t["address"].lower() == USDT_ADDRESS for t in p.get("tokens", []))
]

print(f"Total USDT pools: {len(usdt_pools)}")

# Count USDT-USDC pools
usdt_usdc_pools = [
    p for p in usdt_pools
    if any(t["address"].lower() == USDC_ADDRESS for t in p.get("tokens", []))
]
print(f"USDT-USDC pools: {len(usdt_usdc_pools)}")

# Show top USDT pools by token pairing
usdt_pairs = {}
for p in usdt_pools:
    tokens = p.get("tokens", [])
    if len(tokens) == 2:
        other_token = next(
            (t for t in tokens if t["address"].lower() != USDT_ADDRESS), None
        )
        if other_token:
            symbol = other_token.get("symbol", "UNKNOWN")
            usdt_pairs[symbol] = usdt_pairs.get(symbol, 0) + 1

print("\nTop USDT pairings:")
for symbol, count in sorted(usdt_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {symbol}: {count} pools")

# Find DAI pools
print("\n" + "=" * 70)
print("DAI POOLS")
print("=" * 70)

dai_pools = [
    p for p in uniswap_v3_pools
    if any(t["address"].lower() == DAI_ADDRESS for t in p.get("tokens", []))
]

print(f"Total DAI pools: {len(dai_pools)}")

# Count DAI-USDC pools
dai_usdc_pools = [
    p for p in dai_pools
    if any(t["address"].lower() == USDC_ADDRESS for t in p.get("tokens", []))
]
print(f"DAI-USDC pools: {len(dai_usdc_pools)}")

# Show top DAI pools by token pairing
dai_pairs = {}
for p in dai_pools:
    tokens = p.get("tokens", [])
    if len(tokens) == 2:
        other_token = next(
            (t for t in tokens if t["address"].lower() != DAI_ADDRESS), None
        )
        if other_token:
            symbol = other_token.get("symbol", "UNKNOWN")
            dai_pairs[symbol] = dai_pairs.get(symbol, 0) + 1

print("\nTop DAI pairings:")
for symbol, count in sorted(dai_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {symbol}: {count} pools")

# Load the actual swap data to see distribution
print("\n" + "=" * 70)
print("ACTUAL SWAP DISTRIBUTION")
print("=" * 70)

df_swaps = pl.read_parquet("data/usdc_paired_swaps.parquet")

# For USDT swaps, show token pairing distribution
usdt_swaps = df_swaps.filter(
    (pl.col("token0") == USDT_ADDRESS) | (pl.col("token1") == USDT_ADDRESS)
)

# Determine the other token in each swap
usdt_other_tokens = usdt_swaps.with_columns(
    pl.when(pl.col("token0") == USDT_ADDRESS)
    .then(pl.col("token1"))
    .otherwise(pl.col("token0"))
    .alias("other_token")
)

print(f"\nUSDT swap pairings (top 10):")
usdt_pairing_counts = (
    usdt_other_tokens.group_by("other_token")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(10)
)
for row in usdt_pairing_counts.iter_rows(named=True):
    addr = row["other_token"]
    count = row["count"]
    if addr == USDC_ADDRESS:
        print(f"  USDC: {count:,}")
    else:
        print(f"  {addr[:10]}...: {count:,}")

# For DAI swaps, show token pairing distribution
dai_swaps = df_swaps.filter(
    (pl.col("token0") == DAI_ADDRESS) | (pl.col("token1") == DAI_ADDRESS)
)

dai_other_tokens = dai_swaps.with_columns(
    pl.when(pl.col("token0") == DAI_ADDRESS)
    .then(pl.col("token1"))
    .otherwise(pl.col("token0"))
    .alias("other_token")
)

print(f"\nDAI swap pairings (top 10):")
dai_pairing_counts = (
    dai_other_tokens.group_by("other_token")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
    .head(10)
)
for row in dai_pairing_counts.iter_rows(named=True):
    addr = row["other_token"]
    count = row["count"]
    if addr == USDC_ADDRESS:
        print(f"  USDC: {count:,}")
    else:
        print(f"  {addr[:10]}...: {count:,}")
