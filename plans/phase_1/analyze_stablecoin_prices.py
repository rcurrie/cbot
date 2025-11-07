"""Analyze USDT and DAI prices to understand correlation issues."""

import polars as pl

# USDC contract address
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI_ADDRESS = "0x6b175474e89094c44da98b954eedeac495271d0f"

print("Loading swap data...")
df_swaps = pl.read_parquet("data/usdc_paired_swaps.parquet")

print("Loading price data...")
df_prices = pl.read_parquet("data/usdc_prices_timeseries.parquet")

print("\n" + "=" * 70)
print("USDT ANALYSIS")
print("=" * 70)

# Check USDT swaps
usdt_swaps = df_swaps.filter(
    (pl.col("token0") == USDT_ADDRESS) | (pl.col("token1") == USDT_ADDRESS)
)
print(f"\nTotal USDT swaps: {len(usdt_swaps):,}")

# Check how many are direct USDC-USDT swaps
usdt_usdc_direct = usdt_swaps.filter(
    ((pl.col("token0") == USDT_ADDRESS) & (pl.col("token1") == USDC_ADDRESS))
    | ((pl.col("token0") == USDC_ADDRESS) & (pl.col("token1") == USDT_ADDRESS))
)
print(f"Direct USDC-USDT swaps: {len(usdt_usdc_direct):,} ({len(usdt_usdc_direct)/len(usdt_swaps)*100:.1f}%)")

# Check USDT prices
usdt_prices = df_prices.filter(pl.col("token_address") == USDT_ADDRESS)
print(f"\nTotal USDT price observations: {len(usdt_prices):,}")
print(f"Price statistics:")
print(f"  Mean: {usdt_prices['price_in_usdc'].mean():.6f}")
print(f"  Median: {usdt_prices['price_in_usdc'].median():.6f}")
print(f"  Std: {usdt_prices['price_in_usdc'].std():.6f}")
print(f"  Min: {usdt_prices['price_in_usdc'].min():.6f}")
print(f"  Max: {usdt_prices['price_in_usdc'].max():.6f}")

print("\n" + "=" * 70)
print("DAI ANALYSIS")
print("=" * 70)

# Check DAI swaps
dai_swaps = df_swaps.filter(
    (pl.col("token0") == DAI_ADDRESS) | (pl.col("token1") == DAI_ADDRESS)
)
print(f"\nTotal DAI swaps: {len(dai_swaps):,}")

# Check how many are direct USDC-DAI swaps
dai_usdc_direct = dai_swaps.filter(
    ((pl.col("token0") == DAI_ADDRESS) & (pl.col("token1") == USDC_ADDRESS))
    | ((pl.col("token0") == USDC_ADDRESS) & (pl.col("token1") == DAI_ADDRESS))
)
print(f"Direct USDC-DAI swaps: {len(dai_usdc_direct):,} ({len(dai_usdc_direct)/len(dai_swaps)*100:.1f}%)")

# Check DAI prices
dai_prices = df_prices.filter(pl.col("token_address") == DAI_ADDRESS)
print(f"\nTotal DAI price observations: {len(dai_prices):,}")
print(f"Price statistics:")
print(f"  Mean: {dai_prices['price_in_usdc'].mean():.6f}")
print(f"  Median: {dai_prices['price_in_usdc'].median():.6f}")
print(f"  Std: {dai_prices['price_in_usdc'].std():.6f}")
print(f"  Min: {dai_prices['price_in_usdc'].min():.6f}")
print(f"  Max: {dai_prices['price_in_usdc'].max():.6f}")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("\nThe issue is likely that calculate_usdc_prices.py is calculating")
print("prices from ALL swaps in usdc_paired_swaps.parquet, including swaps")
print("that DON'T directly involve USDC (e.g., USDT-WETH, DAI-WETH).")
print("\nWe can only accurately price tokens from DIRECT USDC swaps.")
print("Indirect swaps (tokenA-tokenB where both have USDC pools) should")
print("be filtered out during price calculation.")
