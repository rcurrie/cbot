# Milestone Plan

**General Information**

Transform the filtered swaps and usdc time series price files from phase 1 into dollar bars and train a baseline model.

Here are the updated milestones for your coding agent, reflecting the "pool-bar with signed flows" logic.

### Milestone 1: Generate Pool-Level Bars and Signed Messages ✅

- **Status:** COMPLETE
- **Inputs:**
  1. `usdc_paired_swaps.parquet` from phase 1 with a row per swap
  2. `usdc_prices_timeseries.parquet` with the prices from the swaps converted into usdc as the numéraire
  3. `pools.json` with information on each pool such as fee
- **Task:** 2. Define a global `target_usdc_bar_size` via command line with default $100k 3. Group the entire dataset by `pool_id`. 4. For each `pool_id` group: - Iterate through its swaps, sorted by `timestamp`. - Accumulate swaps into a bar until the cumulative `usdc_volume` exceeds `target_usdc_bar_size`. - When a bar is formed, get the `bar_close_timestamp`, `tick_count`, and `bar_time_delta_sec` (time since this pool's _last_ bar). - Inside this bar, determine the two tokens traded (e.g., `Token A` and `Token B`). - Determine the primary flow direction (src -> dest) based on the dominant trading direction in the bar: - If `|net_flow_A|` > `|net_flow_B|`, then Token A is the source and Token B is the destination - Otherwise, Token B is the source and Token A is the destination - Calculate the **signed net flow** for each token: - `src_flow_usdc` = net flow for the source token (positive when token flows in, negative when flows out) - `dest_flow_usdc` = net flow for the destination token (typically opposite sign of src*flow_usdc) - Get the `src_price_usdc` and `dest_price_usdc` from the \_last swap* in the bar (from the prices time series file) - **Generate one output row** for this bar event with both src and dest token information. 5. Collect all generated rows from all pools into a single DataFrame.
- **Output:** A new file, `usdc_bars.parquet`.
- **Output Schema (per row):**
  - `bar_close_timestamp`: Timestamp of the last swap in the bar.
  - `pool_id`: The pool that formed the bar.
  - `src_token_id`: The source token address (token being sold/flowing out).
  - `dest_token_id`: The destination token address (token being bought/flowing in).
  - `src_flow_usdc`: The signed net flow for the source token in USDC.
  - `dest_flow_usdc`: The signed net flow for the destination token in USDC.
  - `src_price_usdc`: The USDC price of the source token at the end of the bar.
  - `dest_price_usdc`: The USDC price of the destination token at the end of the bar.
  - `bar_time_delta_sec`: Time (in seconds) since this _pool's_ last bar.
  - `tick_count`: Total number of swaps in this bar.

---

### Milestone 2: Achieve Stationarity for Target Variable ✅

- **Status:** COMPLETE
- **Script:** [src/make_stationary.py](../src/make_stationary.py)
- **Input:** `usdc_bars.parquet` (from Milestone 1) with one row per bar event containing src and dest tokens.
- **Task:**
  1.  Load the `usdc_bars.parquet` file.
  2.  This file contains bar events with source and destination tokens. The **target variable** for prediction (`src_price_usdc`) must be made stationary (we're predicting the price movement of the source token—the token being sold/flowing out).
  3.  Create a new column `y_target_fracdiff` initialized to NaN.
  4.  **Group the DataFrame by `src_token_id`** (source token, as we're predicting its price movement after it's sold).
  5.  For _each source token group_ (e.g., all rows where `src_token_id == '0xc02aaa...'`), **sorted by `bar_close_timestamp`**:
      - Extract the `src_price_usdc` series.
      - Apply `np.log()` to get `log_price`.
      - Find the minimum fractional differentiation order `d` (0.0 to 1.0, step 0.05) for this token's `log_price` series that passes the `statsmodels.tsa.stattools.adfuller` test (p-value < 0.05 for stationarity).
      - If stationary d is found: apply fractional differentiation to generate `fracdiff_log_price`; assign back to `y_target_fracdiff` column for rows in this token group.
      - If no stationary d found: log the token as non-stationary and optionally skip it.
  6.  Drop all rows with `NaN` values in `y_target_fracdiff` (NaNs occur at the start of each token's fracdiff series due to the fixed-width window).
- **Output:** A new file, `log_fracdiff_price.parquet`, containing one row per bar event with the original columns plus `y_target_fracdiff` (the stationary target variable for the source token price).
