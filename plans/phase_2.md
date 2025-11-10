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
- **Task:** 2. Define a global `target_usdc_bar_size` via command line with default $100k 3. Group the entire dataset by `pool_id`. 4. For each `pool_id` group: - Iterate through its swaps, sorted by `timestamp`. - Accumulate swaps into a bar until the cumulative `usdc_volume` exceeds `target_usdc_bar_size`. - When a bar is formed, get the `bar_close_timestamp`, `tick_count`, and `bar_time_delta_sec` (time since this pool's _last_ bar). - Inside this bar, determine the two tokens traded (e.g., `Token A` and `Token B`). - Calculate the **signed net flow** for each token: - `net_flow_A` = `sum(usdc_volume where token_in == A)` - `sum(usdc_volume where token_out == A)` - `net_flow_B` = `sum(usdc_volume where token_in == B)` - `sum(usdc_volume where token_out == B)` - Get the `token_a_price` and `token_b_price` from the _last swap_ in the bar (from the prices time series file) - **Generate two output rows** for this single bar event (one for each token). 5. Collect all generated rows from all pools into a single DataFrame.
- **Output:** A new file, `master_message_log.parquet`.
- **Output Schema (per row):**
  - `bar_close_timestamp`: Timestamp of the last swap in the bar.
  - `pool_id`: The pool that formed the bar.
  - `token_id`: The token this message is _about_ (e.g., 'ETH').
  - `net_flow_usdc`: The signed net flow for `token_id` (e.g., `+12,050`).
  - `token_close_price_usdc`: The USDC price of `token_id` at the end of the bar.
  - `bar_time_delta_sec`: Time (in seconds) since this _pool's_ last bar.
  - `tick_count`: Total number of swaps in this bar.

---

### Milestone 2: Achieve Stationarity for Target Variable ✅

- **Status:** COMPLETE
- **Script:** [src/make_stationary.py](../src/make_stationary.py)
- **Results:**
  - Total messages: 270,615 (from 291,851 original, 7.3% NaN values dropped)
  - Unique tokens: 89 (6 tokens removed due to insufficient data)
  - Tokens achieving stationarity: 54/95 (56.8%)
  - Mean fractional differentiation order (d): 0.660
  - Median d: 1.000
- **Input:** `master_message_log.parquet` (from Milestone 1).
- **Task:**
  1.  Load the `master_message_log.parquet` file.
  2.  This file contains the "messages" (features) for the GNN and the raw data for the baseline. The **target variable** for prediction (`token_close_price_usdc`) must be made stationary.
  3.  Create a new, empty column `y_target_fracdiff`.
  4.  **Group the DataFrame by `token_id`**.
  5.  For _each token group_ (e.g., all rows where `token_id == 'ETH'`):
      - Extract the `token_close_price_usdc` series.
      - Apply `np.log()` to get `log_price`.
      - Find the minimum fractional differentiation order `d` (e.g., 0.1-1.0) for _this token's_ `log_price` series that passes the `statsmodels.tsa.stattools.adfuller` test (p-value < 0.05).
      - Apply this `d` to the `log_price` series to generate `fracdiff_log_price`.
      - Assign this `fracdiff_log_price` series back to the `y_target_fracdiff` column for the corresponding rows in the main DataFrame.
  6.  Drop all rows with `NaN` values (which will appear at the start of each token's `fracdiff_log_price` series).
- **Output:** A new file, `log_fracdiff_price.parquet`. This single file can now be used for both the baseline (by filtering) and the GNN (by using the full graph).
