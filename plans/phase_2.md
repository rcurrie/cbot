# Milestone Plan

**General Information**

Build a baseline forecasting workflow that transforms the phase_1 price history into WETH volume bars, derives stationary log-return series, and fits a simple ARIMA(0,1,0) model for 4-hour ahead evaluations. Deliver model artifacts and a benchmarking notebook that compares the ARIMA forecast against a random-walk baseline.

Prerequisites:
- `data/weth_prices_timeseries.parquet` is available from phase_1.
- Statsmodels and supporting dependencies are installed via `pyproject.toml`.
- All downstream data artifacts for this phase are written to `./data` and notebooks live under `./` or a dedicated notebooks directory.

**Milestone 1: Build Volume Bars**

- **Goal:** Convert the raw time-series into WETH volume bars aligned with the 4-hour prediction horizon.
- **Tasks:**
  1. Load `data/weth_prices_timeseries.parquet` with Polars.
  2. Compute rolling 4-hour WETH volume totals across the dataset.
  3. Derive the median 4-hour volume and use it as the fixed target volume per bar.
  4. Aggregate trades chronologically until the cumulative WETH volume reaches the target, emit a bar, and continue iterating.
  5. Persist the resulting bar series to `data/weth_volume_bars.parquet` with timestamps, open/high/low/close prices, and realized volume.
- **Validation:** Verify the bar file exists, inspect descriptive stats (e.g., bars per day) to ensure the derived volume keeps bar counts consistent with the 4-hour cadence, and spot-check bar boundaries around known regime shifts.

**Milestone 2: Stationarity Prep and Diagnostics**

- **Goal:** Produce log price and log-return series suitable for ARIMA modeling.
- **Tasks:**
  1. Load the volume bars and compute log prices.
  2. Generate first-differenced log returns to enforce stationarity.
  3. Run basic diagnostics (Augmented Dickey-Fuller or similar) to document stationarity assumptions.
  4. Save the transformed dataset to `data/weth_volume_bars_log_returns.parquet` with metadata columns (bar timestamp, log price, log return).
- **Validation:** Confirm diagnostics indicate stationarity, ensure no missing values in the transformed data, and document any anomalies for later review.

**Milestone 3: Train Baseline ARIMA Model**

- **Goal:** Fit an ARIMA(0,1,0) model on the log price series and produce forecasts aligned with the 4-hour horizon.
- **Tasks:**
  1. Use `statsmodels` to fit ARIMA(0,1,0) on the log price data.
  2. Generate out-of-sample forecasts for the next bar and for cumulative 4-hour windows.
  3. Serialize fitted model parameters and forecasts to `data/arima_baseline_model.json` and `data/arima_baseline_forecasts.parquet` (or equivalent structured formats).
  4. Capture model diagnostics (residual plots, Ljung-Box statistics) for inclusion in the notebook.
- **Validation:** Review residual diagnostics for autocorrelation, ensure forecasts are finite and within reasonable bounds, and validate serialization integrity by reloading artifacts in a dry run.

**Milestone 4: Benchmark Against Random Walk**

- **Goal:** Compare ARIMA forecasts to a random-walk baseline and document results in a notebook.
- **Tasks:**
  1. Define a random-walk baseline using Gaussian shocks scaled to recent log-return volatility.
  2. Compute performance metrics (MAE, RMSE, directional accuracy) for both ARIMA and the baseline over a held-out period.
  3. Generate visualizations (time-series overlays, error distributions) highlighting differences.
  4. Create a notebook `baseline_arima_vs_random_walk.ipynb` that walks through the preprocessing, model fitting, baseline generation, and evaluation.
  5. Summarize findings, noting where ARIMA outperforms or underperforms the baseline.
- **Validation:** Ensure the notebook executes top-to-bottom without errors, metrics show ARIMA provides measurable gains over the baseline, and plots render correctly. Document any limitations or edge cases uncovered during evaluation.

**Milestone 5: Phase Wrap-Up and Handoff**

- **Goal:** Consolidate outputs, document lessons, and define next steps for more advanced models.
- **Tasks:**
  1. Assemble a short report capturing data artifacts produced, key metrics, and known gaps.
  2. Log remaining questions or improvements (e.g., dynamic volume targets, richer feature sets) for future phases.
  3. Update the project README or dedicated phase tracker with links to artifacts and notebooks.
- **Validation:** Confirm stakeholders can reproduce results from documentation alone, verify all artifacts are version-controlled or referenced appropriately, and obtain sign-off before advancing to the next modeling phase.
