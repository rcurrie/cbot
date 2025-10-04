# %%
"""ARIMA modeling for high-volatility WETH swap pools."""

import json
from pathlib import Path

import polars as pl
from statsmodels.tsa.arima.model import ARIMA

with Path("data/weth_pool_addr_to_symbol.json").open() as f:
    weth_pool_addr_to_symbol = json.load(f)

# %%
# Load the hourly WETH swaps data
data_path = Path("data/hourly_weth_swaps.parquet")
swaps = pl.read_parquet(data_path)

# Display basic info
print(f"Loaded {swaps.shape[0]} swaps with columns: {swaps.columns}")

# %%
# Find most volatile pools with a minimum number of swaps in first 30 days
NUM_TOP_POOLS = 10
MIN_SWAPS_FOR_VOLATILITY = 100

min_timestamp = swaps.select(pl.col("block_timestamp").min()).item()
cutoff_timestamp = min_timestamp + pl.duration(days=30)

# Filter to first 30 days and exclude pools with any null values
first_30_days_df = swaps.filter(pl.col("block_timestamp") <= cutoff_timestamp)

pools_with_nulls = (
    first_30_days_df.group_by("pool_addr")
    .agg(pl.col("log_price").null_count().alias("null_count"))
    .filter(pl.col("null_count") > 0)
    .select("pool_addr")
)

first_30_days_df = first_30_days_df.join(
    pools_with_nulls,
    on="pool_addr",
    how="anti",
)

volatility_df = (
    first_30_days_df.group_by("pool_addr")
    .agg(
        pl.col("log_price").std().alias("volatility"),
        pl.col("log_price").count().alias("count"),
    )
    .filter(pl.col("count") >= MIN_SWAPS_FOR_VOLATILITY)
    .sort("volatility", descending=True)
    .drop_nulls()
    .limit(NUM_TOP_POOLS)
    .with_columns(
        pl.col("pool_addr")
        .map_elements(
            lambda addr: weth_pool_addr_to_symbol.get(addr, "UNKNOWN"),
            return_dtype=pl.String,
        )
        .alias("symbol"),
    )
)

print("\nTop Pools by Volatility (First 30 Days):")
print(volatility_df.select(["symbol", "pool_addr", "volatility"]))

# %%
# Fit ARIMA models on first 30 days and predict end of day 31 for top pools

# Calculate day 31 end timestamp
day_31_end = min_timestamp + pl.duration(days=31)

for pool_addr in volatility_df.select("pool_addr").to_series():
    symbol = weth_pool_addr_to_symbol.get(pool_addr, "UNKNOWN")
    print(f"\nFitting ARIMA model for pool: {symbol} ({pool_addr})")

    # Get log_price time series for this pool (first 30 days)
    pool_series = (
        first_30_days_df.filter(pl.col("pool_addr") == pool_addr)
        .sort("block_timestamp")
        .select("log_price")
        .to_series()
        .drop_nulls()
        .to_numpy()
    )

    assert pool_series.shape[0] == (30 * 24) + 1, "Expected 721 hourly observations for 30 days"

    # Fit ARIMA(1,1,1) model
    model = ARIMA(pool_series, order=(1, 1, 1))
    fitted = model.fit()

    # Forecast to end of day 31
    print(f"Next 5 steps forecast:\n{fitted.forecast(steps=5)}")
    print(swaps.filter(pl.col("pool_addr") == pool_addr)[721:726].select(
        "block_timestamp", "log_price", "volume"
    ))
