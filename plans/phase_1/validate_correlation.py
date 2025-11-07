"""Validate correlation between our prices and CoinGecko."""

import os
from datetime import datetime, timezone

import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI_ADDRESS = "0x6b175474e89094c44da98b954eedeac495271d0f"


def fetch_coingecko_prices(token_id: str, start_date: str, end_date: str):
    """Fetch historical prices from CoinGecko API."""
    start_ts = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    end_ts = int(
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )

    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_ts,
        "to": end_ts,
    }
    headers = {
        "x-cg-demo-api-key": COINGECKO_API_KEY,
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    prices = data.get("prices", [])

    df = pl.DataFrame(
        {
            "timestamp": [
                datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc) for p in prices
            ],
            "price_usd": [p[1] for p in prices],
        }
    )

    return df


# Load our prices
print("Loading our calculated prices...")
df_prices = pl.read_parquet("data/usdc_prices_timeseries.parquet")

# Get date range
start_date = df_prices["block_timestamp"].min()
end_date = df_prices["block_timestamp"].max()
print(f"Date range: {start_date} to {end_date}")

# Test USDT
print("\n" + "=" * 70)
print("USDT VALIDATION")
print("=" * 70)

usdt_prices = df_prices.filter(pl.col("token_address") == USDT_ADDRESS)
print(f"Our USDT observations: {len(usdt_prices):,}")
print(f"Our USDT price: mean={usdt_prices['price_in_usdc'].mean():.6f}, std={usdt_prices['price_in_usdc'].std():.6f}")

print("\nFetching CoinGecko USDT prices...")
cg_usdt = fetch_coingecko_prices(
    "tether",
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
)
print(f"CoinGecko USDT observations: {len(cg_usdt):,}")
print(f"CoinGecko USDT price: mean={cg_usdt['price_usd'].mean():.6f}, std={cg_usdt['price_usd'].std():.6f}")

# Resample both to hourly and compare
usdt_hourly = (
    usdt_prices.sort("block_timestamp")
    .group_by_dynamic("block_timestamp", every="1h")
    .agg(pl.col("price_in_usdc").mean().alias("swap_price"))
    .with_columns(pl.col("block_timestamp").dt.truncate("1h").alias("hour"))
)

cg_usdt_hourly = (
    cg_usdt.with_columns(pl.col("timestamp").dt.truncate("1h").alias("hour"))
    .group_by("hour")
    .agg(pl.col("price_usd").mean().alias("cg_price"))
)

joined_usdt = usdt_hourly.join(cg_usdt_hourly, on="hour", how="inner")
print(f"\nMatched hours: {len(joined_usdt)}")

if len(joined_usdt) > 0:
    correlation = joined_usdt.select(pl.corr("swap_price", "cg_price").alias("corr"))[
        "corr"
    ][0]
    mape = (
        (joined_usdt["swap_price"] - joined_usdt["cg_price"]).abs()
        / joined_usdt["cg_price"]
    ).mean() * 100

    print(f"Correlation: {correlation:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Show some sample comparisons
    print("\nSample price comparisons (first 10 hours):")
    for row in joined_usdt.head(10).iter_rows(named=True):
        print(
            f"  {row['hour']}: Ours={row['swap_price']:.6f}, CG={row['cg_price']:.6f}, "
            f"Diff={abs(row['swap_price'] - row['cg_price']):.6f}"
        )

# Test DAI
print("\n" + "=" * 70)
print("DAI VALIDATION")
print("=" * 70)

dai_prices = df_prices.filter(pl.col("token_address") == DAI_ADDRESS)
print(f"Our DAI observations: {len(dai_prices):,}")
print(f"Our DAI price: mean={dai_prices['price_in_usdc'].mean():.6f}, std={dai_prices['price_in_usdc'].std():.6f}")

print("\nFetching CoinGecko DAI prices...")
cg_dai = fetch_coingecko_prices(
    "dai",
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
)
print(f"CoinGecko DAI observations: {len(cg_dai):,}")
print(f"CoinGecko DAI price: mean={cg_dai['price_usd'].mean():.6f}, std={cg_dai['price_usd'].std():.6f}")

# Resample both to hourly and compare
dai_hourly = (
    dai_prices.sort("block_timestamp")
    .group_by_dynamic("block_timestamp", every="1h")
    .agg(pl.col("price_in_usdc").mean().alias("swap_price"))
    .with_columns(pl.col("block_timestamp").dt.truncate("1h").alias("hour"))
)

cg_dai_hourly = (
    cg_dai.with_columns(pl.col("timestamp").dt.truncate("1h").alias("hour"))
    .group_by("hour")
    .agg(pl.col("price_usd").mean().alias("cg_price"))
)

joined_dai = dai_hourly.join(cg_dai_hourly, on="hour", how="inner")
print(f"\nMatched hours: {len(joined_dai)}")

if len(joined_dai) > 0:
    correlation = joined_dai.select(pl.corr("swap_price", "cg_price").alias("corr"))[
        "corr"
    ][0]
    mape = (
        (joined_dai["swap_price"] - joined_dai["cg_price"]).abs() / joined_dai["cg_price"]
    ).mean() * 100

    print(f"Correlation: {correlation:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Show some sample comparisons
    print("\nSample price comparisons (first 10 hours):")
    for row in joined_dai.head(10).iter_rows(named=True):
        print(
            f"  {row['hour']}: Ours={row['swap_price']:.6f}, CG={row['cg_price']:.6f}, "
            f"Diff={abs(row['swap_price'] - row['cg_price']):.6f}"
        )
