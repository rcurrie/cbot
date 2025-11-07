"""Check for zero prices and investigate the cause."""

import polars as pl

# Load the prices
df = pl.read_parquet("data/usdc_prices_timeseries.parquet")

# Find zero prices
zero_prices = df.filter(pl.col("price_in_usdc") == 0.0)

print(f"Total observations: {len(df):,}")
print(f"Zero prices: {len(zero_prices):,}")
print()

if len(zero_prices) > 0:
    print("Sample zero price observations:")
    print(zero_prices.head(10))
    print()

    # Group by token
    zero_by_token = (
        zero_prices.group_by("token_address")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print("Zero prices by token:")
    print(zero_by_token)

# Also check for extremely low prices (< 1e-6)
very_low_prices = df.filter(
    (pl.col("price_in_usdc") > 0) & (pl.col("price_in_usdc") < 1e-6)
)
print(f"\nVery low prices (0 < price < 1e-6): {len(very_low_prices):,}")
if len(very_low_prices) > 0:
    print(very_low_prices.head(10))

# Check for NaN or infinity
nan_prices = df.filter(pl.col("price_in_usdc").is_nan())
inf_prices = df.filter(pl.col("price_in_usdc").is_infinite())
print(f"\nNaN prices: {len(nan_prices):,}")
print(f"Infinite prices: {len(inf_prices):,}")
