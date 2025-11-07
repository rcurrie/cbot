"""Check DAI data coverage to understand low correlation."""

import polars as pl

DAI_ADDRESS = "0x6b175474e89094c44da98b954eedeac495271d0f"

# Load prices
df_prices = pl.read_parquet("data/usdc_prices_timeseries.parquet")
dai_prices = df_prices.filter(pl.col("token_address") == DAI_ADDRESS)

print(f"Total DAI observations: {len(dai_prices):,}")
print(f"Date range: {dai_prices['block_timestamp'].min()} to {dai_prices['block_timestamp'].max()}")

# Resample to hourly to see coverage
dai_hourly = (
    dai_prices.sort("block_timestamp")
    .group_by_dynamic("block_timestamp", every="1h")
    .agg(
        [
            pl.col("price_in_usdc").mean().alias("price_mean"),
            pl.col("price_in_usdc").count().alias("obs_count"),
        ]
    )
)

print(f"\nHourly aggregation:")
print(f"  Total hours with data: {len(dai_hourly):,}")
print(f"  Hours with 0 obs: {len(dai_hourly.filter(pl.col('obs_count') == 0)):,}")
print(f"  Hours with 1-5 obs: {len(dai_hourly.filter((pl.col('obs_count') >= 1) & (pl.col('obs_count') <= 5))):,}")
print(f"  Hours with >5 obs: {len(dai_hourly.filter(pl.col('obs_count') > 5)):,}")

# Check gaps
print("\nChecking for data gaps...")

# Create full hour range
start_hour = dai_prices["block_timestamp"].min().replace(minute=0, second=0, microsecond=0)
end_hour = dai_prices["block_timestamp"].max().replace(minute=0, second=0, microsecond=0)

# Calculate expected hours
import pandas as pd
from datetime import timedelta

expected_hours = int((end_hour - start_hour).total_seconds() / 3600) + 1
print(f"Expected hours in range: {expected_hours:,}")
print(f"Actual hours with data: {len(dai_hourly):,}")
print(f"Coverage: {len(dai_hourly) / expected_hours * 100:.1f}%")

# Show distribution of observations per hour
print("\nObservations per hour distribution:")
obs_dist = (
    dai_hourly.group_by("obs_count")
    .agg(pl.len().alias("hour_count"))
    .sort("obs_count")
)
for row in obs_dist.iter_rows(named=True):
    print(f"  {row['obs_count']} obs: {row['hour_count']} hours")

# Check variance in hourly prices
hours_with_data = dai_hourly.filter(pl.col("obs_count") > 0)
print(f"\nPrice variance in hours with data:")
print(f"  Mean price: {hours_with_data['price_mean'].mean():.6f}")
print(f"  Std dev: {hours_with_data['price_mean'].std():.6f}")
print(f"  Min: {hours_with_data['price_mean'].min():.6f}")
print(f"  Max: {hours_with_data['price_mean'].max():.6f}")
print(f"  Range: {hours_with_data['price_mean'].max() - hours_with_data['price_mean'].min():.6f}")
