# Pool State Features Implementation Plan

## Overview

Enrich the data pipeline with pool liquidity and tick movement features. This plan focuses ONLY on updating the pipeline up to `data/labeled_log_fracdiff_price.parquet` - the embedding/training stage will be handled separately later.

**Goal:** Add pool state features (liquidity, tick) to the labeled training dataset.

**Scope:**
- ✅ Modify: `generate_usdc_bars.py`, `make_stationary.py`, `label_triple_barrier.py`, `training_data_validation.py`
- ❌ Skip: `dex_contagion_trader.py`, model architecture, training, evaluation

**Approach:** Add features to data pipeline in two phases (liquidity first, then tick).

## Quick Summary

### What We're Adding

Two new pool state features to each bar:

1. **Liquidity** (`src_liquidity_close`, `dest_liquidity_close`)
   - Uniswap V3 pool liquidity at bar close
   - Normalized with log1p(x) for numerical stability
   - Represents market depth / execution capacity

2. **Tick Movement** (`src_tick_delta`, `dest_tick_delta`)
   - Change in pool tick from bar open to close
   - Normalized with z-score (mean=0, std=1)
   - Captures intra-pool price volatility

### Pipeline Changes

| Script | Changes Required | Complexity |
|--------|-----------------|------------|
| `generate_usdc_bars.py` | Import decode functions, track pool state, add columns to output | Medium |
| `make_stationary.py` | None (auto pass-through) | Trivial |
| `label_triple_barrier.py` | Add columns to output schema | Trivial |
| `training_data_validation.py` | Add columns to required list, add basic stats logging | Trivial |

### Expected Outcome

**Before:** 22 columns in `data/labeled_log_fracdiff_price.parquet`

**After:** 26 columns (22 original + 4 new pool state features)

**Runtime Impact:** Minimal (~5% slower due to hex decoding)

**Data Quality:** Liquidity <5% nulls, tick <10% nulls (expected)

---

## Phase 1: Add Liquidity Feature

**Goal:** Add pool liquidity to the labeled dataset for future embedding experiments

### 1.1 Modify generate_usdc_bars.py

**Location:** [src/generate_usdc_bars.py](../src/generate_usdc_bars.py)

**Import decoding function (add to imports):**
```python
from filter_and_decode_swaps import decode_liquidity
```

**Track liquidity per pool (add to state tracking):**
```python
# Inside DollarBarProcessor class, add to pool state:
self.pool_liquidity: dict[str, int] = {}  # pool_id -> last liquidity value
```

**Decode liquidity when processing swaps:**
```python
# In process_swap() method, after decoding amounts:
try:
    liquidity = decode_liquidity(swap["data"])
    self.pool_liquidity[pool_id] = liquidity
except Exception:
    # Silently skip - will use last known value or NaN
    pass
```

**Add liquidity to bar output:**
```python
# When closing a bar in _close_bar() method:
src_liquidity = self.pool_liquidity.get(pool_id, np.nan)
dest_liquidity = src_liquidity  # Same pool for both tokens

# Add to bar dictionary:
bar = {
    # ... existing fields ...
    "src_liquidity_close": np.log1p(float(src_liquidity)) if not np.isnan(src_liquidity) else np.nan,
    "dest_liquidity_close": np.log1p(float(dest_liquidity)) if not np.isnan(dest_liquidity) else np.nan,
}
```

**Lightweight validation:**
```python
# In main(), after creating output DataFrame, add:
logger.info("Liquidity feature validation:")
logger.info("  Null count src: %d", df["src_liquidity_close"].null_count())
logger.info("  Null count dest: %d", df["dest_liquidity_close"].null_count())
logger.info("  Median src: %.2e", df["src_liquidity_close"].median())
logger.info("  Median dest: %.2e", df["dest_liquidity_close"].median())
```

### 1.2 Modify make_stationary.py

**Location:** [src/make_stationary.py](../src/make_stationary.py)

**Pass through liquidity columns (no changes needed to core logic):**

Polars will automatically carry through the columns. Just verify in output:

```python
# In main(), after writing output, add log:
logger.info("Output includes liquidity columns: %s",
    "src_liquidity_close" in output_df.columns and "dest_liquidity_close" in output_df.columns)
```

### 1.3 Modify label_triple_barrier.py

**Location:** [src/label_triple_barrier.py](../src/label_triple_barrier.py)

**Pass through liquidity columns:**

Update the final column selection to include liquidity:

```python
# In main(), when selecting output columns:
output_cols = [
    # ... existing columns ...
    "src_liquidity_close",
    "dest_liquidity_close",
]
final_df = labeled_df.select(output_cols)
```

**Lightweight validation:**
```python
# Before writing output:
logger.info("Liquidity in labeled data - null counts: src=%d, dest=%d",
    final_df["src_liquidity_close"].null_count(),
    final_df["dest_liquidity_close"].null_count())
```

### 1.4 Update training_data_validation.py

**Location:** [src/training_data_validation.py](../src/training_data_validation.py)

**Add liquidity to required columns:**
```python
REQUIRED_COLUMNS = [
    # ... existing columns ...
    "src_liquidity_close",
    "dest_liquidity_close",
]
```

**Add liquidity validation (in validate_schema()):**
```python
# After checking required columns:
logger.info("\nLiquidity Distribution:")
for col in ["src_liquidity_close", "dest_liquidity_close"]:
    null_count = df[col].null_count()
    median = df[col].median()
    logger.info("  %s: %d nulls (%.1f%%), median=%.2e",
        col, null_count, 100 * null_count / len(df), median)

    # Check for negative values (should never happen with log1p)
    if (df[col].drop_nulls() < 0).any():
        logger.error("  ❌ ERROR: Negative liquidity values in %s", col)
```

### 1.5 Run Pipeline and Verify

**Execute pipeline up to labeled data:**
```bash
# Clean previous outputs
rm data/usdc_bars.parquet data/log_fracdiff_price.parquet data/labeled_log_fracdiff_price.parquet

# Run data pipeline (stop before training)
make generate-usdc-bars
make make-stationary
make label-triple-barrier
make training-data-validation
```

**Verify output has liquidity columns:**
```python
import polars as pl

# Check final labeled dataset
df = pl.read_parquet("data/labeled_log_fracdiff_price.parquet")

print("Schema:")
print(df.schema)

print("\nLiquidity columns present:")
print("src_liquidity_close" in df.columns)
print("dest_liquidity_close" in df.columns)

print("\nLiquidity statistics:")
print(df.select(["src_liquidity_close", "dest_liquidity_close"]).describe())

print("\nNull counts:")
print(df.select([
    pl.col("src_liquidity_close").null_count().alias("src_nulls"),
    pl.col("dest_liquidity_close").null_count().alias("dest_nulls"),
]))
```

**Expected output:**
- ✅ Both liquidity columns present in schema
- ✅ Null count < 5% of total rows
- ✅ Median liquidity ~14-18 (log1p of 1e6 to 1e8)
- ✅ No negative values

---

## Phase 2: Add Tick Movement

**Goal:** Add tick movement features to the labeled dataset

**Prerequisites:** Phase 1 complete (liquidity features added)

### 2.1 Modify generate_usdc_bars.py

**Track tick movement per pool:**
```python
# Add to DollarBarProcessor state:
self.pool_tick: dict[str, int] = {}  # pool_id -> last tick value
self.bar_tick_open: dict[str, int] = {}  # pool_id -> tick at bar open
```

**Decode tick when processing swaps:**
```python
# In process_swap(), after liquidity decoding:
try:
    from filter_and_decode_swaps import decode_tick
    tick = decode_tick(swap["data"])

    # Track tick at bar open
    if pool_id not in self.bar_tick_open:
        self.bar_tick_open[pool_id] = tick

    self.pool_tick[pool_id] = tick
except Exception:
    pass
```

**Calculate tick delta at bar close:**
```python
# In _close_bar():
tick_close = self.pool_tick.get(pool_id, np.nan)
tick_open = self.bar_tick_open.get(pool_id, np.nan)

if not np.isnan(tick_close) and not np.isnan(tick_open):
    tick_delta = tick_close - tick_open
else:
    tick_delta = np.nan

# Reset for next bar
self.bar_tick_open[pool_id] = tick_close

# Add to bar output:
bar = {
    # ... existing fields ...
    "src_tick_delta": float(tick_delta),
    "dest_tick_delta": float(tick_delta),  # Same pool
}
```

**Normalize tick deltas (after all bars created):**
```python
# In main(), before writing output:
# Standardize tick deltas per token (z-score normalization)
df = df.with_columns([
    ((pl.col("src_tick_delta") - pl.col("src_tick_delta").mean()) /
     pl.col("src_tick_delta").std()).alias("src_tick_delta"),
    ((pl.col("dest_tick_delta") - pl.col("dest_tick_delta").mean()) /
     pl.col("dest_tick_delta").std()).alias("dest_tick_delta"),
])
```

**Lightweight validation:**
```python
logger.info("Tick delta validation:")
logger.info("  Null count src: %d", df["src_tick_delta"].null_count())
logger.info("  Mean src: %.4f (should be ~0)", df["src_tick_delta"].mean())
logger.info("  Std src: %.4f (should be ~1)", df["src_tick_delta"].std())
```

### 2.2 Update Pipeline Stages

**make_stationary.py:** No changes (pass through automatically)

**label_triple_barrier.py:** Add to output columns:
```python
output_cols = [
    # ... existing ...
    "src_tick_delta",
    "dest_tick_delta",
]
```

**training_data_validation.py:** Add to required columns and basic range check:
```python
REQUIRED_COLUMNS += ["src_tick_delta", "dest_tick_delta"]

# In validation:
logger.info("Tick delta stats: mean=%.4f, std=%.4f",
    df["src_tick_delta"].mean(), df["src_tick_delta"].std())
```

### 2.3 Run Pipeline and Verify

**Execute pipeline:**
```bash
# Clean previous outputs
rm data/usdc_bars.parquet data/log_fracdiff_price.parquet data/labeled_log_fracdiff_price.parquet

# Run data pipeline
make generate-usdc-bars
make make-stationary
make label-triple-barrier
make training-data-validation
```

**Verify output has tick columns:**
```python
import polars as pl

df = pl.read_parquet("data/labeled_log_fracdiff_price.parquet")

print("Tick columns present:")
print("src_tick_delta" in df.columns)
print("dest_tick_delta" in df.columns)

print("\nTick delta statistics:")
print(df.select(["src_tick_delta", "dest_tick_delta"]).describe())

print("\nNormalization check (should be mean~0, std~1):")
print(f"Src tick delta: mean={df['src_tick_delta'].mean():.4f}, std={df['src_tick_delta'].std():.4f}")
print(f"Dest tick delta: mean={df['dest_tick_delta'].mean():.4f}, std={df['dest_tick_delta'].std():.4f}")
```

**Expected output:**
- ✅ Both tick columns present
- ✅ Mean ≈ 0 (normalized via z-score)
- ✅ Std ≈ 1 (normalized via z-score)
- ✅ Null count < 10% (ticks can be missing for some pools)

---

## Final Dataset Schema

After completing both phases, `data/labeled_log_fracdiff_price.parquet` should have:

**Original columns (unchanged):**
- `bar_close_timestamp`, `block_number`, `transaction_hash`, `pool_id`
- `src_token_id`, `dest_token_id`
- `src_flow_usdc`, `dest_flow_usdc`
- `src_price_usdc`, `dest_price_usdc`
- `bar_time_delta_sec`, `tick_count`
- `src_fracdiff`, `dest_fracdiff`
- `rolling_volatility`
- `label`, `dest_label`
- `sample_weight`, `dest_sample_weight`
- `barrier_touch_bars`, `dest_barrier_touch_bars`

**New columns (added):**
- `src_liquidity_close` - Log1p normalized pool liquidity (src token)
- `dest_liquidity_close` - Log1p normalized pool liquidity (dest token)
- `src_tick_delta` - Z-score normalized tick movement (src token)
- `dest_tick_delta` - Z-score normalized tick movement (dest token)

**Total:** 26 columns (22 original + 4 new pool state features)

---

## Rollback Plan

If you need to revert changes:

```bash
# Revert code changes
git checkout HEAD -- src/generate_usdc_bars.py
git checkout HEAD -- src/make_stationary.py
git checkout HEAD -- src/label_triple_barrier.py
git checkout HEAD -- src/training_data_validation.py

# Regenerate data with original pipeline
make generate-usdc-bars
make make-stationary
make label-triple-barrier
make training-data-validation
```

---

## Success Criteria

**Phase 1 Success:**
- ✅ `src_liquidity_close` and `dest_liquidity_close` columns present
- ✅ Null count < 5%
- ✅ Median liquidity ~14-18 (log1p scale)
- ✅ No negative values
- ✅ Pipeline runs without errors

**Phase 2 Success:**
- ✅ `src_tick_delta` and `dest_tick_delta` columns present
- ✅ Mean ≈ 0, Std ≈ 1 (proper z-score normalization)
- ✅ Null count < 10%
- ✅ Pipeline runs without errors

**Overall Success:**
- ✅ Final parquet has 26 columns
- ✅ All validation checks pass
- ✅ Data quality metrics within expected ranges
- ✅ Ready for embedding/training experiments

---

## Notes

**Why lightweight validation?**
- Existing scripts already have comprehensive schema checks, null handling, range validation
- We only need to verify new columns are present and reasonable
- Over-validation slows iteration - trust the existing infrastructure

**Why log1p for liquidity?**
- Liquidity values range from 1e6 to 1e15 (huge scale differences)
- log1p(x) = log(1+x) handles zeros gracefully and normalizes scale
- Standard practice for count-like features in ML
- Results in values roughly 14-20 for typical pools

**Why z-score for tick deltas?**
- Tick ranges vary by pool (-887,272 to +887,272 for full range)
- Most tick movements are small (±100 ticks typical)
- Z-score normalization centers at 0, scale to unit variance
- Makes feature comparable to fracdiff and volatility (which are also normalized)

**Why separate src/dest liquidity when they're the same?**
- Consistency with existing dual-token feature pattern
- Both tokens in a swap interact with the same pool → same liquidity
- Future-proofs for potential cross-pool scenarios
- Simplifies feature alignment in downstream embedding experiments

**Why stop at the labeled dataset?**
- Data pipeline changes are stable and reusable
- Embedding experiments are iterative and will test many configurations
- Separating concerns: data enrichment now, model architecture later
- Allows experimentation with different feature combinations without re-running expensive pipeline

**Next steps after this plan:**
- Experiment with different embedding dimensions (64D, 128D)
- Test ablation studies (liquidity-only, tick-only, both, neither)
- Modify `dex_contagion_trader.py` to consume new features
- Measure impact on Sharpe ratio, drawdown, win rate
