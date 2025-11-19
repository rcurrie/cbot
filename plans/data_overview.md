# Data Pipeline Overview

This document outlines the data processing pipeline transforming raw Uniswap V3 swaps into labeled, stationary time-series data ready for Temporal Graph Neural Network (TGNN) training.

## Pipeline Stages

The pipeline consists of 5 sequential stages, each producing a validated parquet artifact.

| Stage                | Script                           | Input                                                    | Output                               | Description                                                                                                   |
| -------------------- | -------------------------------- | -------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| 1. Filter & Decode   | `src/filter_and_decode_swaps.py` | Raw Swaps, `pools.json`                                  | `usdc_paired_swaps.parquet`          | Filters for V3 swaps with USDC connectivity (direct or indirect). Decodes events.                             |
| 2. Price Calculation | `src/calculate_usdc_prices.py`   | `usdc_paired_swaps.parquet`                              | `usdc_priced_swaps.parquet`          | Derives USDC prices for all tokens. Infers prices for indirect pairs using cached rates. Filters outliers.    |
| 3. Bar Generation    | `src/generate_usdc_bars.py`      | `usdc_priced_swaps.parquet`, `usdc_paired_swaps.parquet` | `usdc_bars.parquet`                  | Aggregates swaps into volume-based "Dollar Bars" (e.g., $100k). Computes signed flows (buy/sell pressure).    |
| 4. Stationarity      | `src/make_stationary.py`         | `usdc_bars.parquet`                                      | `log_fracdiff_price.parquet`         | Log-transforms prices and applies Fractional Differentiation to achieve stationarity while preserving memory. |
| 5. Labeling          | `src/label_triple_barrier.py`    | `log_fracdiff_price.parquet`                             | `labeled_log_fracdiff_price.parquet` | Generates labels (-1, 0, 1) using the Triple-Barrier Method. Computes sample weights for concurrency.         |

## Data Artifacts

### 1. `usdc_paired_swaps.parquet`

Raw decoded swap events relevant to the USDC ecosystem.

- **Key Columns**: `block_timestamp`, `pool`, `token0`, `token1`, `sender`, `recipient`, `data` (raw hex), `token0_decimals`, `token1_decimals`.

### 2. `usdc_priced_swaps.parquet`

Time-series of token prices derived from swaps.

- **Key Columns**: `block_timestamp`, `token_address`, `price_in_usdc`, `usdc_volume`.
- **Note**: Includes inferred prices for tokens traded against non-USDC partners (e.g., WETH).

### 3. `usdc_bars.parquet` (Feature Set Base)

The primary event-based dataset. Rows represent a completed volume bar for a specific pool.

- **Key Columns**:
  - `pool_id`: Unique pool identifier.
  - `src_token_id`, `dest_token_id`: Directional flow tokens.
  - `src_flow_usdc`, `dest_flow_usdc`: Net buying/selling pressure in USDC.
  - `src_price_usdc`, `dest_price_usdc`: Closing prices for the bar.
  - `bar_time_delta_sec`: Time taken to fill the bar.
  - `tick_count`: Number of swaps in the bar.

### 4. `log_fracdiff_price.parquet`

Stationary features ready for model input.

- **Key Columns**: `src_fracdiff`, `dest_fracdiff` (Fractionally differentiated log prices).
- **Method**: Finds minimum order $d$ (0.0-1.0) per token to pass ADF test ($p < 0.05$). Joins fracdiff series back to bars for both source and destination tokens.

### 5. `labeled_log_fracdiff_price.parquet` (Training Data)

Final dataset for Supervised Learning.

- **Key Columns**:
  - `label`: Target class. `1` (Profit), `-1` (Loss), `0` (Time-out).
  - `sample_weight`: Weight adjustment for overlapping labels (uniqueness).
  - `rolling_volatility`: Volatility used for dynamic barrier sizing.
  - `barrier_touch_bars`: Number of bars until the barrier was touched.
- **Method**: Triple-Barrier Method (Marcos López de Prado) applied to `src_fracdiff`.
  - **Upper Barrier**: Current Price + $C_1 \times \sigma$
  - **Lower Barrier**: Current Price - $C_2 \times \sigma$
  - **Vertical Barrier**: Dynamic Volume-Clock horizon (10% of daily volume).

## Methodological Notes

- **Dollar Bars**: We sample based on financial activity (volume) rather than time. This recovers normality in data distributions and synchronizes active/inactive periods.
- **Fractional Differentiation**: Standard differencing ($d=1$) destroys memory. We use the minimum $d$ necessary to achieve stationarity, preserving the maximum amount of history/trend information.
- **Double Lookup**: We calculate stationary series for all tokens globally and then join them back to the bars twice (for source and destination) to ensure the model has visibility into the state of both assets in the swap.
- **Triple Barrier Method**: A dynamic labeling technique that accounts for volatility. It avoids the pitfalls of fixed-horizon labeling.
  - **Volatility-Adjusted Time Horizons**: Instead of a fixed time barrier (e.g., 8 hours), we use a "Volume-Clock" approach. The vertical barrier is set to 10% of the token's Average Daily Volume (measured in bars). This ensures we hold liquid tokens for shorter periods (e.g., 30-60 mins) and illiquid tokens for longer periods (e.g., 4-8 hours), aligning with the market's information processing speed.

## Next Steps: TGNN Development

The `labeled_log_fracdiff_price.parquet` file is the direct input for the Temporal Graph Neural Network.

- **Nodes**: Tokens.
- **Edges**: Pools (weighted by flow/volume).
- **Node Features**: `src_fracdiff`, `dest_fracdiff`, `rolling_volatility`, `src_flow_usdc`.
- **Edge Features**: `bar_time_delta_sec`, `tick_count`.
