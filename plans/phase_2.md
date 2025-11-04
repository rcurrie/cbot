# Milestone Plan

**General Information**

Build a baseline forecasting workflow that transforms the phase_1 price history into WETH volume bars, derives stationary log-return series, and fits a simple ARIMA(0,1,0) model for 4-hour ahead evaluations. Deliver model artifacts and a benchmarking notebook that compares the ARIMA forecast against a random-walk baseline.

Prerequisites:

- `data/weth_prices_timeseries.parquet` is available from phase_1.
- Statsmodels and supporting dependencies are installed via `pyproject.toml`.
- All downstream data artifacts for this phase are written to `./data` and notebooks live under `./notebooks`.

**Milestone 1: Build Volume Bars**

- **Goal:** Convert the raw time-series into WETH volume bars aligned with the 4-hour prediction horizon. Note that each of these should be per token as we will be training ARIMA models per token below.
- **Tasks:**
  1. Load `data/weth_prices_timeseries.parquet` with Polars.
  2. Compute rolling 4-hour WETH volume totals across the dataset.
  3. Derive the median 4-hour volume and use it as the fixed target volume per bar.
  4. Aggregate trades chronologically until the cumulative WETH volume reaches the target, emit a bar, and continue iterating.
  5. Persist the resulting bar series to `data/weth_volume_bars.parquet` with timestamps, open/high/low/close prices, and realized volume.
- **Validation:** Verify the bar file exists, inspect descriptive stats (e.g., bars per day) for the same representative tokens used in validate_prices.ipynb (USDC, USDT, WBTC, DAI, LINK) to ensure the derived volume keeps bar counts consistent with the 4-hour cadence, and spot-check bar boundaries around known regime shifts.

**Milestone 2: Stationarity Prep and Diagnostics**

- **Goal:** Produce log price and log-return series suitable for ARIMA modeling.
- **Tasks:**
  1. Load the volume bars and compute log prices.
  2. Generate first-differenced log returns to enforce stationarity.
  3. Run basic diagnostics (Augmented Dickey-Fuller or similar) to document stationarity assumptions.
  4. Save the transformed dataset to `data/weth_volume_bars_log_returns.parquet` with metadata columns (bar timestamp, log price, log return).
- **Validation:** Confirm diagnostics indicate stationarity, ensure no missing values in the transformed data, and document any anomalies for later review.

**Milestone 3: Train ARIMA Models on Representative Tokens** ✅

- **Goal:** Create a script that trains ARIMA models on representative tokens from `data/weth_volume_bars_log_returns.parquet` to predict price movements over the next 4-6 volume bars (approximately 4-24 hours).
- **Status:** COMPLETE - Script successfully trains ARIMA models with command-line options, directional probabilities, and verbose diagnostics.
- **Tasks:**
  1. Create `src/train_arima_models.py` script that:
     - Loads `data/weth_volume_bars_log_returns.parquet` with filtering capability
     - Defaults to training on the 5 representative tokens (USDC, USDT, WBTC, DAI, LINK)
     - For each token, fits an appropriate ARIMA model (start with ARIMA(1,1,1) or use auto-selection) on the log price series
     - Generates out-of-sample forecasts for the next 4-6 volume bars with:
       - Predicted log price returns
       - Confidence intervals (e.g., 80% and 95%) for each prediction
       - Directional probability (likelihood of positive vs. negative movement)
     - Produces verbose output showing training progress and model diagnostics
     - Serializes model artifacts to `data/arima_models/` (per-token parameters) and forecasts to `data/arima_forecasts.parquet`
  2. Include command-line options to specify:
     - Which tokens to train on (default: 5 representative tokens)
     - Forecast horizon (default: 4-6 volume bars)
     - ARIMA model parameters or auto-selection method
- **Validation:** Run the script on the 5 representative tokens and verify that models train successfully, produce reasonable forecasts with confidence intervals, and save artifacts correctly. Check that verbose output provides clear diagnostic information.

**Milestone 4: Baseline Model Comparison** ✅

- **Goal:** Create a validation notebook that compares ARIMA forecasts against a random-walk baseline for the 5 representative tokens.
- **Status:** COMPLETE - Comprehensive comparison notebook with metrics, visualizations, and diagnostics successfully created and tested.
- **Tasks:**
  1. Create `notebooks/arima_baseline_comparison.ipynb` that:
     - Loads the ARIMA model forecasts from `data/arima_forecasts.parquet` for the 5 representative tokens
     - Defines a random-walk baseline using Gaussian shocks scaled to recent log-return volatility
     - Trains both models on a historical training set and evaluates on a held-out test period
     - Computes performance metrics (MAE, RMSE, directional accuracy) comparing ARIMA forecasts to the random-walk baseline
     - Generates visualizations showing:
       - Forecast accuracy over time for each token
       - Confidence intervals vs. actual outcomes
       - Directional accuracy comparison (ARIMA vs. baseline)
     - Documents model diagnostics (residual plots, Ljung-Box statistics) for each representative token
     - Produces a summary table comparing ARIMA and baseline performance
- **Validation:** Ensure the notebook executes without errors, visualizations clearly show model performance differences, ARIMA models demonstrate measurable improvement over random-walk baseline in directional accuracy, and confidence intervals are well-calibrated.

**Milestone 5: Trading Signal Generation for All Tokens**

- **Goal:** Enhance the training script to process all tokens, rank them by trading signal strength, and recommend top 5 tokens for long positions.
- **Tasks:**
  1. Extend `src/train_arima_models.py` to:
     - Support training on all tokens in the dataset (via command-line flag)
     - Calculate a trading signal score for each token that combines:
       - High probability of positive price movement over the next 4-6 bars
       - High confidence (narrow prediction intervals relative to expected return)
       - Sufficient historical data quality for reliable ARIMA fitting
     - Rank tokens by trading signal score
     - Output the top 5 tokens as buy-long candidates for 4-24 hour holding periods
     - Save ranking results to `data/arima_trading_signals.parquet`
  2. Ensure the script still defaults to training only the 5 representative tokens for quick validation
  3. Add logging and output that clearly shows:
     - Token ranking methodology
     - Trading signal scores and components for top tokens
     - Recommended positions with rationale
- **Validation:** Run the script with the all-tokens flag and verify it processes all tokens efficiently, produces a clear ranking with statistical justification for top 5 tokens, and the default behavior (5 representative tokens only) remains unchanged. Verify forecast horizons align with the 4-24 hour trading window.
