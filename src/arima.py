# %%
"""ARIMA modeling for WETH swap pools."""

import json
from pathlib import Path

import polars as pl
from statsmodels.tsa.arima.model import ARIMA

with Path("data/weth_pool_addr_to_symbol.json").open() as f:
    weth_pool_addr_to_symbol = json.load(f)

# %%
# Load the hourly WETH swaps data
swaps_df = pl.read_parquet(Path("data/hourly_weth_swaps.parquet"))

# Display basic info
print(f"Loaded {swaps_df.shape[0]:,} swaps with columns: {swaps_df.columns}")
print(
    f"Time range: {swaps_df.select(pl.col('block_timestamp').min()).item()} to {swaps_df.select(pl.col('block_timestamp').max()).item()} or {(swaps_df.select(pl.col('block_timestamp').max()).item() - swaps_df.select(pl.col('block_timestamp').min()).item()).days} days",
)

# %%

# %%
# Fit ARIMA models on first 30 days and predict following N hours

min_timestamp = swaps_df.select(pl.col("block_timestamp").min()).item()
day_30_end = min_timestamp + pl.duration(days=30)
train_df = swaps_df.filter(pl.col("block_timestamp") < day_30_end)

for pool_addr in swaps_df.select("pool_addr").to_series()[1:2]:
    symbol = weth_pool_addr_to_symbol.get(pool_addr, "UNKNOWN")
    print(f"\nFitting ARIMA model for pool: {symbol} ({pool_addr})")

    # Get log_price time series for this pool (first 30 days)
    pool_series = (
        train_df.filter(pl.col("pool_addr") == pool_addr)
        .sort("block_timestamp")
        .select("log_price")
        .to_series()
        .drop_nulls()  # ARIMA cannot handle nulls
        .to_numpy()
    )

    print(f"Number of hourly observations: {pool_series.shape[0]:,}")

    # assert pool_series.shape[0] == (30 * 24), (
    #     "Expected 721 hourly observations for 30 days"
    # )

    # Fit ARIMA(1,1,1) model
    model = ARIMA(pool_series, order=(1, 1, 1))
    fitted = model.fit()
    print(fitted.summary())

    # # Forecast to end of day 31
    # print(f"Next 5 steps forecast:\n{fitted.forecast(steps=5)}")
    # print(
    #     swaps.filter(pl.col("pool_addr") == pool_addr)[721:726].select(
    #         "block_timestamp",
    #         "log_price",
    #         "volume",
    #     ),
    # )

# %%
