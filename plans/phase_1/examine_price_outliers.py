"""Examine price outliers in USDC prices timeseries."""

import polars as pl

# Load the prices
df = pl.read_parquet("data/usdc_prices_timeseries.parquet")

print(f"Total observations: {len(df):,}")
print(f"Unique tokens: {df['token_address'].n_unique()}")
print()

# Focus on USDT and DAI which showed poor correlation
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI_ADDRESS = "0x6b175474e89094c44da98b954eedeac495271d0f"

for name, address in [("USDT", USDT_ADDRESS), ("DAI", DAI_ADDRESS)]:
    print("=" * 70)
    print(f"{name} Analysis")
    print("=" * 70)

    df_token = df.filter(pl.col("token_address") == address)

    if len(df_token) == 0:
        print(f"No data for {name}")
        continue

    print(f"Observations: {len(df_token):,}")
    print(f"\nPrice Statistics:")
    print(f"  Mean: {df_token['price_in_usdc'].mean():.6f}")
    print(f"  Median: {df_token['price_in_usdc'].median():.6f}")
    print(f"  Std: {df_token['price_in_usdc'].std():.6f}")
    print(f"  Min: {df_token['price_in_usdc'].min():.6e}")
    print(f"  Max: {df_token['price_in_usdc'].max():.6e}")
    print(f"  1st percentile: {df_token['price_in_usdc'].quantile(0.01):.6e}")
    print(f"  99th percentile: {df_token['price_in_usdc'].quantile(0.99):.6f}")

    # Show extreme outliers
    print(f"\nTop 10 highest prices:")
    top_prices = df_token.sort("price_in_usdc", descending=True).head(10)
    for row in top_prices.iter_rows(named=True):
        print(f"  {row['price_in_usdc']:.6e} - {row['block_timestamp']}")

    print(f"\nTop 10 lowest prices:")
    bottom_prices = df_token.sort("price_in_usdc").head(10)
    for row in bottom_prices.iter_rows(named=True):
        print(f"  {row['price_in_usdc']:.6e} - {row['block_timestamp']}")

    # Check for extreme outliers using IQR
    q1 = df_token["price_in_usdc"].quantile(0.25)
    q3 = df_token["price_in_usdc"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    outliers_high = df_token.filter(pl.col("price_in_usdc") > upper_bound)
    outliers_low = df_token.filter(pl.col("price_in_usdc") < lower_bound)

    print(f"\nIQR-based outlier detection (3x IQR):")
    print(f"  Q1: {q1:.6f}, Q3: {q3:.6f}, IQR: {iqr:.6f}")
    print(f"  Lower bound: {lower_bound:.6e}")
    print(f"  Upper bound: {upper_bound:.6f}")
    print(f"  High outliers: {len(outliers_high):,} ({len(outliers_high)/len(df_token)*100:.2f}%)")
    print(f"  Low outliers: {len(outliers_low):,} ({len(outliers_low)/len(df_token)*100:.2f}%)")
    print()
