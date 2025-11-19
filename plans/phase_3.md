# Milestone Plan

**General Information**

Transform the stationary price data from phase 2 into classification labels using the Triple-Barrier Method (TBM). This reframes the prediction problem from regression to classification by labeling each observation based on whether price hits a take-profit barrier, stop-loss barrier, or time limit first.

### Milestone 1: Implement Triple-Barrier Method with Concurrency Handling ✅

- **Status:** COMPLETE
- **Script:** [src/label_triple_barrier.py](../src/label_triple_barrier.py)
- **Results:**
  - Total labeled messages: 268,183 (from 270,615 original, 0.6% dropped)
  - Unique tokens: 38 (51 tokens dropped due to insufficient data)
  - Label distribution:
    - Positive labels (+1, take profit hit): 79,324 (29.6%)
    - Negative labels (-1, stop loss hit): 155,027 (57.8%)
    - Neutral labels (0, time limit hit): 33,832 (12.6%)
  - Mean sample weight: 0.179 (indicates healthy label overlap/concurrency)
  - Sample weight range: 0.038 to 1.000
- **Input:** `data/log_fracdiff_price.parquet` (from Phase 2 Milestone 2).
- **Command Line Arguments:**
  - `--upper_multiple` (C1): Multiplier for upper barrier (take profit). Default: 2.0
  - `--lower_multiple` (C2): Multiplier for lower barrier (stop loss). Default: 1.0
  - `--vertical_bars` (N): Number of bars into the future for time limit. Default: 25
  - `--volatility_window`: Rolling window for volatility calculation. Default: 20
- **Task:**
  1. Load the `log_fracdiff_price.parquet` file.
  2. **Group the DataFrame by `dest_token_id`** (each destination token must be processed independently - we're predicting the price movement of the token being bought).
  3. For _each token group_:
     - Sort by `bar_close_timestamp` to ensure chronological order.
     - Calculate rolling volatility: compute a rolling standard deviation of `y_target_fracdiff` over the last `--volatility_window` bars for each bar `t`.
     - For each bar `t`, set up three barriers:
       - **Upper Barrier (Take Profit):** `price_t * (1 + vol_t * C1)`
       - **Lower Barrier (Stop Loss):** `price_t * (1 - vol_t * C2)`
       - **Vertical Barrier (Time Limit):** `N` bars into the future
     - Look forward from bar `t` to determine which barrier is hit first:
       - Label = **`+1`** if upper barrier is hit first
       - Label = **`-1`** if lower barrier is hit first
       - Label = **`0`** if vertical barrier is hit first (neither profit/loss barrier reached)
     - Store the label and the number of bars until the barrier was hit (`barrier_touch_bars`).
  4. **Handle Label Overlap (Concurrency):**
     - Track which future bars are already "involved" in previous labels.
     - For each bar `t`, calculate a `sample_weight` based on the number of concurrent labels active at time `t`.
     - Formula: `sample_weight = 1.0 / num_concurrent_labels` where `num_concurrent_labels` is the count of how many other labels have `t` within their forward-looking window.
     - This weight will be used during model training to account for overlapping labels.
  5. Add new columns to the DataFrame:
     - `label`: The TBM classification label (`+1`, `-1`, or `0`)
     - `barrier_touch_bars`: Number of bars until barrier was touched
     - `sample_weight`: Concurrency-adjusted weight for this sample
  6. Drop rows where labels cannot be computed (e.g., last `N` bars of each token where we can't look forward).
- **Output:** A new file, `data/labeled_log_fracdiff_price.parquet`.
- **Output Schema (new columns added):**
  - All existing columns from `log_fracdiff_price.parquet`
  - `label`: Classification label (`-1`, `0`, or `+1`)
  - `barrier_touch_bars`: Integer, number of bars until barrier touch
  - `sample_weight`: Float, concurrency-adjusted weight (between 0.0 and 1.0)
  - `rolling_volatility`: Float, the rolling standard deviation used for this bar's barriers
- **Validation:**
  - Verify that the output file is created with all expected columns.
  - Check label distribution: print counts of each label class (`-1`, `0`, `+1`) per token and overall.
  - Verify that `sample_weight` values are reasonable (between 0 and 1, with mean typically > 0.5).
  - Plot example barrier events for a few tokens to visually confirm correct labeling.
  - Confirm that rows at the end of each token's time series are dropped appropriately (last `N` bars).

---

## Column Reference: `data/labeled_log_fracdiff_price.parquet`

| Column Name           | Type           | Source    | Description                                                                                                                              |
| --------------------- | -------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `bar_close_timestamp` | datetime64[ns] | Inherited | Unix timestamp of the bar close time                                                                                                     |
| `pool_id`             | str            | Inherited | Unique identifier for the pool                                                                                                           |
| `src_token_id`        | str            | Inherited | Source token address (token being sold/flowing out)                                                                                      |
| `dest_token_id`       | str            | Inherited | Destination token address (token being bought/flowing in)                                                                                |
| `src_flow_usdc`       | float64        | Inherited | Signed net flow in USDC for source token (typically negative when selling)                                                               |
| `dest_flow_usdc`      | float64        | Inherited | Signed net flow in USDC for destination token (typically positive when buying)                                                           |
| `src_price_usdc`      | float64        | Inherited | USDC price of source token at bar close                                                                                                  |
| `dest_price_usdc`     | float64        | Inherited | USDC price of destination token at bar close                                                                                             |
| `y_target_fracdiff`   | float64        | Inherited | Fractional differenced price target (log returns, stationary) - calculated on destination token price                                    |
| `label`               | int8           | **NEW**   | Triple-Barrier Method classification label: `+1` (take profit), `-1` (stop loss), `0` (time limit)                                       |
| `barrier_touch_bars`  | int16          | **NEW**   | Number of bars into the future until a barrier was touched                                                                               |
| `sample_weight`       | float32        | **NEW**   | Concurrency-adjusted weight (0.0–1.0) accounting for overlapping labels; used during model training to reduce bias from temporal overlap |
| `rolling_volatility`  | float32        | **NEW**   | Rolling standard deviation of `y_target_fracdiff` over `--volatility_window` bars; used to compute barrier distances                     |

### Notes:

- All rows where labels cannot be computed (last `N` bars per token) are **dropped**.
- `sample_weight` follows the formula: $\text{weight} = \frac{1}{n_{\text{concurrent}}}$ where $n_{\text{concurrent}}$ is the count of overlapping labels at time $t$.
- Barrier distances are computed as: Upper = $\text{price}_t \times (1 + \text{vol}_t \times C_1)$, Lower = $\text{price}_t \times (1 - \text{vol}_t \times C_2)$.
- Use `label` as the target for binary/multiclass classification models and `sample_weight` as sample weights during training.

---

**Next Steps:**

After Phase 3 is complete, we will have a properly labeled dataset ready for classification model training. The labels account for realistic trading scenarios (profit targets, stop losses, time limits) and the sample weights handle the temporal overlap inherent in financial data.
