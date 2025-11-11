# Milestone Plan

**General Information**

Implement purged expanding-window cross-validation (CV) and comprehensive model evaluation. This phase builds on the labeled dataset from Phase 3 to train **per-token LSTM classifiers** using Prado's methodology for preventing data leakage through temporal purging, and evaluates model performance using trading-relevant metrics rather than standard accuracy.

**Important:** Each token gets its own LSTM model, trained independently on that token's time series data. This accounts for token-specific price dynamics and volatility patterns.

### Milestone 1: Implement and Validate Temporal Purging Logic

- **Script:** [src/utils/purging.py](../src/utils/purging.py) (utility module, not standalone script)
- **Purpose:** Create a reusable purging function that will be called by Milestone 2's expanding window CV framework.
- **Task:**
  1. **Implement `purge_temporal_leakage()` function** with signature:
     ```python
     def purge_temporal_leakage(
         train_df: pl.DataFrame,
         test_start_time: datetime,
         test_end_time: datetime
     ) -> pl.DataFrame:
         """
         Remove training samples whose label resolution time overlaps with test period.

         Args:
             train_df: Training set DataFrame (must have 'bar_close_timestamp' and 'barrier_touch_bars')
             test_start_time: Start timestamp of test set
             test_end_time: End timestamp of test set

         Returns:
             Purged training DataFrame with overlapping samples removed
         """
     ```
  2. **Purging algorithm:**
     - For each row `i` in `train_df`, calculate label resolution time:
       - Extract `bar_close_timestamp(i)` and `barrier_touch_bars(i)`
       - Calculate the time delta: look at the next `barrier_touch_bars(i)` rows to find the actual timestamp when label resolved
       - `t_end(i)` = timestamp at row `i + barrier_touch_bars(i)`
     - Remove any sample where `t_end(i) >= test_start_time`
     - Return purged DataFrame
  3. **Create validation script** `src/validate_purging.py`:
     - Load `labeled_log_fracdiff_price.parquet`
     - Select one example token with sufficient samples (e.g., the low-volatility token with 2769 samples)
     - Create a simple 70/30 train/test split by timestamp
     - Apply purging function
     - Generate detailed validation report
- **Output:**
  - `src/utils/purging.py` (reusable purging function)
  - `src/validate_purging.py` (validation script)
  - `data/validation/purging_validation_report.json` (validation metrics)
  - `data/validation/purging_timeline.png` (visualization)
- **Validation:**
  - Run `validate_purging.py` on example token
  - Verify no training sample has `t_end >= test_start_time`
  - Print purge statistics: original train size, purged train size, % removed
  - Sample 5-10 examples from the boundary region and manually verify purging logic
  - Visualize timeline showing:
    - Training period bars (blue)
    - Test period bars (green)
    - Purged training bars (red, with arrows showing their label resolution extending into test period)
    - Retained training bars near boundary (gray)
  - **Expected purge rate:** Typically 5-15% of training samples near the boundary should be purged

---

### Milestone 2: Build Expanding-Window CV Framework (Per-Token)

- **Status:** PENDING
- **Script:** [src/expanding_window_cv.py](../src/expanding_window_cv.py)
- **Input:** `data/labeled_log_fracdiff_price.parquet` (from Phase 3 Milestone 1).
- **Command Line Arguments:**
  - `--initial_train_pct`: Initial training set as percentage of token's data. Default: 0.50 (50%)
  - `--test_pct`: Test set size as percentage of token's data per fold. Default: 0.10 (10%)
  - `--num_folds`: Number of CV folds to generate. Default: 5
  - `--min_samples_per_token`: Minimum samples required per token to be included. Default: 500
- **Task:**
  1. Load the labeled dataset and sort by `bar_close_timestamp`.
  2. **Group by `token_id`** - each token will have its own set of CV folds.
  3. Filter out tokens with fewer than `min_samples_per_token` samples.
  4. **For each token independently:**
     - Sort token's data by timestamp
     - Calculate token-specific fold sizes:
       - `n_samples` = total samples for this token
       - `initial_train_size` = `int(n_samples * initial_train_pct)`
       - `test_size` = `int(n_samples * test_pct)`
     - For each fold `k` (from 0 to `num_folds - 1`):
       - Calculate fold boundaries by **row index within token**:
         - `train_start_idx = 0`
         - `train_end_idx = initial_train_size + (k * test_size)`
         - `test_start_idx = train_end_idx`
         - `test_end_idx = test_start_idx + test_size`
       - Extract train and test DataFrames by index slicing
       - Get actual timestamps:
         - `test_start_time` = `train_df['bar_close_timestamp'].max()` or `test_df['bar_close_timestamp'].min()`
         - `test_end_time` = `test_df['bar_close_timestamp'].max()`
       - **Apply purging:** Call `purge_temporal_leakage(train_df, test_start_time, test_end_time)` from Milestone 1
       - Save fold data with token_id in filename
  5. **Handle sample weights during purging:**
     - Ensure `sample_weight` column is preserved in purged datasets
     - Verify weight distribution doesn't change dramatically after purging
  6. **Skip tokens with insufficient data for all folds:**
     - If `train_end_idx + test_size > n_samples` for any fold, skip that token entirely
     - Log which tokens are skipped and why
- **Output:**
  - `data/cv_folds/` directory containing:
    - `{token_id}_fold_{k}_train_purged.parquet` (for each token and fold)
    - `{token_id}_fold_{k}_test.parquet` (for each token and fold)
    - `{token_id}_fold_{k}_metadata.json` (train size pre/post purge, test size, time ranges, purge stats, indices used)
  - `data/cv_folds/cv_summary.json` (overall CV configuration, list of included/excluded tokens with reasons, statistics per token)
- **Validation:**
  - Verify all folds are created for each qualifying token
  - Check that folds are properly expanding (train set grows by `test_size` each fold, test set stays constant) per token
  - Confirm no temporal leakage across any fold using automated check (call purging validation on each fold)
  - Print summary table per token:
    - Token ID (abbreviated), total samples, number of folds created
    - For each fold: fold number, train size (pre/post purge), purge %, test size, date ranges
  - Plot fold structure timeline for 3-5 sample tokens showing:
    - X-axis: timestamp
    - Y-axis: folds (stacked)
    - Color coding: training data (blue), purged data (red), test data (green)
  - Verify percentage-based sizing works across tokens with very different sample counts

---

### Milestone 3: Train Per-Token LSTM Classifiers with Cross-Validation

- **Status:** PENDING
- **Script:** [src/train_lstm_cv.py](../src/train_lstm_cv.py)
- **Input:** CV folds from `data/cv_folds/` (from Milestone 2).
- **Command Line Arguments:**
  - `--tokens`: Optional comma-separated list of token_ids to train. If not specified, trains all tokens.
  - `--test_subset`: Boolean flag to use a predefined test subset of 5 diverse tokens. Default: False
  - `--hidden_size`: LSTM hidden layer size. Default: 64
  - `--num_layers`: Number of LSTM layers. Default: 2
  - `--dropout`: Dropout rate. Default: 0.3
  - `--batch_size`: Training batch size. Default: 128
  - `--epochs`: Training epochs per fold. Default: 50
  - `--learning_rate`: Initial learning rate. Default: 0.001
  - `--early_stopping_patience`: Patience for early stopping. Default: 10
  - `--class_weights`: Whether to use class weights for imbalanced labels. Default: True
- **Test Subset Strategy:**
  - **Purpose:** Enable end-to-end testing of the full pipeline (Milestones 3-5) on a small, diverse set of tokens before committing to training all 38 tokens.
  - **Selected Tokens (5 total, spanning volatility spectrum):**
    1. **Low volatility:** `0x6b175474e89094c44da98b954eedeac495271d0f` (2769 samples, vol=0.000103)
    2. **Low-mid volatility:** `0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0` (1449 samples, vol=0.002702)
    3. **Mid volatility:** `0xa3931d71877c0e7a3148cb7eb4463d350b339d96` (96 samples, vol=0.003352) - edge case
    4. **High volatility:** `0xfe0c30065b384f05761f15d0cc86b07ac5df5ac3` (351 samples, vol=0.063403)
    5. **Highest volatility:** `0xd533a949740bb3306d119cc777fa900ba034cd52` (620 samples, vol=0.111757)
  - **Usage:** Run with `--test_subset` flag to train only these 5 tokens, verify all downstream steps work correctly, then remove flag to train all tokens.
- **Task:**
  1. **Load CV folds for selected tokens:**
     - If `--test_subset` is True, use the 5 predefined tokens above
     - If `--tokens` is specified, use that list
     - Otherwise, train all tokens from `data/cv_folds/cv_summary.json`
  2. **For each token, for each fold:**
     - Load `{token_id}_fold_{k}_train_purged.parquet` and corresponding test set
  3. **Prepare features per token:**
     - Features: `net_flow_usdc`, `bar_time_delta_sec`, `tick_count`, `rolling_volatility`
     - Target: `label` (-1, 0, +1)
     - Sample weights: `sample_weight` (from Phase 3)
  4. **Build token-specific LSTM architecture:**
     - Input layer: 4 features
     - LSTM layers with specified hidden size and dropout
     - Output layer: 3 classes (softmax activation)
     - **Each token gets its own model instance**
  5. **Training process per token-fold combination:**
     - Use `sample_weight` during training to handle label concurrency
     - Apply `class_weights` if enabled to handle class imbalance
     - Implement early stopping on validation loss (use 20% of train as validation)
     - Save model checkpoint at best validation performance
  6. **Track metrics per token-fold:**
     - Training loss curve
     - Validation loss curve
     - Number of epochs to convergence
     - Best validation loss achieved
- **Output:**
  - `models/{token_id}_lstm_fold_{k}.pt` (trained PyTorch model for each token-fold)
  - `models/{token_id}_lstm_fold_{k}_training_log.json` (loss curves, hyperparameters)
  - `models/lstm_cv_summary.json` (aggregated training statistics across all tokens/folds)
- **Validation:**
  - Verify models converge (loss decreases) for all token-fold combinations
  - Check for overfitting: compare train vs validation loss per token
  - Ensure sample weights are properly applied during training
  - Plot training curves for test subset tokens to visually inspect convergence
  - Sample predictions from each token-fold and verify output format (3-class probabilities)
  - Compare label distribution in predictions vs ground truth per token-fold
  - **Test subset validation:** Confirm all 5 test tokens complete training successfully before scaling to all tokens

---

### Milestone 4: Implement Trading-Focused Evaluation Metrics (Per-Token)

- **Status:** PENDING
- **Script:** [src/evaluate_trading_metrics.py](../src/evaluate_trading_metrics.py)
- **Input:**
  - Trained models from `models/{token_id}_lstm_fold_{k}.pt` (from Milestone 3)
  - Test sets from `data/cv_folds/{token_id}_fold_{k}_test.parquet` (from Milestone 2)
- **Command Line Arguments:**
  - `--tokens`: Optional comma-separated list of token_ids to evaluate. Default: all tokens with trained models
  - `--test_subset`: Boolean flag to evaluate only the 5 test tokens. Default: False
- **Task:**
  1. **For each token, for each fold:**
     - Load the trained model `{token_id}_lstm_fold_{k}.pt`
     - Load the corresponding test set `{token_id}_fold_{k}_test.parquet`
  2. Generate predictions on test set (3-class probabilities and hard predictions).
  3. **Calculate per-token-fold metrics:**
     - **Overall accuracy** (for reference only, not primary metric)
     - **Per-class metrics:**
       - Precision for class +1 (Long signals): TP_long / (TP_long + FP_long)
       - Recall for class +1 (Long signals): TP_long / (TP_long + FN_long)
       - F1-score for class +1: Harmonic mean of precision and recall
       - Precision for class -1 (Short signals): TP_short / (TP_short + FP_short)
       - Recall for class -1 (Short signals): TP_short / (TP_short + FN_short)
       - F1-score for class -1
     - **Confusion matrix** (3x3 matrix for -1, 0, +1)
  4. **Aggregate metrics:**
     - **Per-token aggregation:** Mean/std across folds for each token
     - **Global aggregation:** Mean/std across all token-fold combinations
     - Identify best/worst performing tokens
     - Identify best/worst performing folds
  5. **Save detailed results:**
     - Predictions with ground truth for error analysis per token-fold
     - Per-token summary statistics
- **Output:**
  - `results/{token_id}_fold_{k}_predictions.parquet` (test predictions + ground truth)
  - `results/{token_id}_fold_{k}_metrics.json` (all calculated metrics for this token-fold)
  - `results/{token_id}_aggregate_metrics.json` (mean/std across folds for this token)
  - `results/cv_aggregate_metrics.json` (global mean/std across all tokens and folds)
  - `results/confusion_matrices.pkl` (confusion matrices for all token-fold combinations)
- **Validation:**
  - Verify metrics sum correctly (e.g., precision/recall calculation)
  - Check that F1-scores are between precision and recall
  - Ensure predictions have valid probability distributions (sum to 1.0)
  - Sample manual calculation of metrics on a few token-fold examples
  - Compare label distribution in test set vs predictions to identify bias per token
  - Check for tokens with consistently poor performance (may indicate insufficient data or high noise)

---

### Milestone 5: Simulate Trading PnL and Generate Performance Reports (Per-Token)

- **Status:** PENDING
- **Script:** [src/simulate_trading_pnl.py](../src/simulate_trading_pnl.py)
- **Input:**
  - Predictions from `results/{token_id}_fold_{k}_predictions.parquet` (from Milestone 4)
  - Original price data from `data/usdc_prices_timeseries.parquet` (for PnL calculation)
  - Labeled data from `data/labeled_log_fracdiff_price.parquet` (for barrier validation)
- **Command Line Arguments:**
  - `--tokens`: Optional comma-separated list of token_ids to simulate. Default: all tokens
  - `--test_subset`: Boolean flag to simulate only the 5 test tokens. Default: False
  - `--position_size`: Position size per trade in USDC. Default: 1000
  - `--transaction_cost_bps`: Transaction cost in basis points. Default: 10 (0.1%)
  - `--hold_bars`: Number of bars to hold position. Default: 25 (same as vertical barrier)
- **Task:**
  1. **For each token, for each fold:**
     - Load predictions from `{token_id}_fold_{k}_predictions.parquet`
     - Match with actual price movements from `usdc_prices_timeseries.parquet`
  2. **Simulate trading strategy per token:**
     - For prediction = +1: Enter long position
     - For prediction = -1: Enter short position
     - For prediction = 0: No position (sit in cash)
  3. **Calculate PnL for each trade:**
     - Long PnL: `(price_exit - price_entry) / price_entry * position_size - transaction_cost`
     - Short PnL: `(price_entry - price_exit) / price_entry * position_size - transaction_cost`
     - Hold position for `hold_bars` or until barrier is hit
  4. **Generate equity curves:**
     - Per-token equity curves (aggregated across folds)
     - Per-fold equity curves (aggregated across tokens)
     - Separate curves for long-only, short-only, and combined strategies
  5. **Calculate trading statistics:**
     - **Per-token-fold level:**
       - Total return (%)
       - Sharpe ratio (assuming bar-level returns)
       - Maximum drawdown (%)
       - Win rate (% of profitable trades)
       - Average win vs average loss
       - Profit factor (gross profit / gross loss)
       - Number of trades (long, short, total)
     - **Aggregated per-token:** Mean/std of above metrics across folds
     - **Aggregated globally:** Mean/std across all tokens and folds
  6. **Per-token ranking and analysis:**
     - Rank tokens by total PnL, Sharpe ratio, win rate
     - Identify tokens that benefit from long vs short strategies
     - Correlate token volatility with trading performance
- **Output:**
  - `results/{token_id}_fold_{k}_trades.parquet` (all trades with entry/exit/PnL)
  - `results/{token_id}_fold_{k}_equity_curve.parquet` (time series of cumulative PnL)
  - `results/{token_id}_fold_{k}_trading_stats.json` (all trading statistics)
  - `results/{token_id}_aggregate_trading_stats.json` (aggregated across folds for this token)
  - `results/cv_aggregate_trading_stats.json` (aggregated across all tokens and folds)
  - `results/token_rankings.json` (tokens ranked by various metrics)
  - `results/visualizations/` directory with:
    - `equity_curves_by_fold.png` (all folds overlaid, tokens combined)
    - `equity_curves_by_token.png` (all tokens overlaid, folds combined)
    - `equity_curves_test_subset.png` (detailed view of 5 test tokens)
    - `drawdown_chart.png`
    - `returns_distribution.png`
    - `token_pnl_rankings.png` (bar chart of top/bottom 10 tokens)
    - `volatility_vs_performance.png` (scatter plot)
- **Validation:**
  - Verify PnL calculations with manual spot checks (5-10 trades across different tokens)
  - Ensure transaction costs are properly applied per trade
  - Check that equity curves start at 0 and accumulate logically
  - Verify Sharpe ratio calculation (annualized properly)
  - Confirm max drawdown represents true peak-to-trough per token
  - Cross-reference trade count with prediction count per token-fold
  - Sample trades and trace back to original price data for correctness
  - Compare actual label outcomes vs predicted outcomes for consistency
  - **Verify barrier concordance:** For 2-3 tokens, manually check that barrier touch events in predictions align with actual price movements in `usdc_prices_timeseries.parquet`
  - Check that high-volatility tokens don't dominate PnL due to position sizing (consider volatility-adjusted sizing in future)

---

**Next Steps:**

After Phase 4 is complete, we will have:

1. A robust cross-validation framework that prevents temporal data leakage through purging
2. Trained LSTM models evaluated on realistic out-of-sample test sets
3. Trading-focused metrics (precision, F1, PnL) that reflect real-world applicability
4. Clear understanding of model performance across different time periods (folds)
5. Foundation for comparing alternative models (GNN in future phases)

**Key Success Criteria:**

- Precision for long signals (+1) > 0.55 (better than coin flip after costs)
- F1-score for active signals (±1) > 0.45
- Positive cumulative PnL across majority of CV folds
- Sharpe ratio > 0.5 (indicating reasonable risk-adjusted returns)
- No evidence of temporal data leakage in validation checks
