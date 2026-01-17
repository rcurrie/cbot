# Phase 5: Outlier Resolution Summary

## Problem

The training data validation identified extreme outliers in `src_flow_usdc`:

- **Range**: [-1.41e+26, 7.94e+23] USDC
- **Expected**: [-1e+08, 1e+08] USDC
- **Impact**: Values were **18 orders of magnitude** too large

## Root Cause Analysis

### Investigation Steps

1. **Traced outliers backward through pipeline**:

   - `labeled_log_fracdiff_price.parquet` → 11 extreme rows
   - `usdc_bars.parquet` → 5 extreme bars
   - `usdc_priced_swaps.parquet` → Identified source prices and volumes

2. **Identified two related issues**:

   **Issue 1: Extreme Low Prices**

   - Token 0xaaee1a (scam token): price = 0.000001 USDC
   - Token 0x514910 (LINK): price = 1.32e-18 USDC (outlier)
   - These prices passed through the IQR filter

   **Issue 2: Extreme Volumes**

   - Scam token volume: 1.41e+26 USDC at price 0.000001 USDC
   - Implied token amount: 9.99e+31 tokens (clearly wrong)

### Root Cause

The existing `filter_price_outliers()` function in [src/calculate_usdc_prices.py](../src/calculate_usdc_prices.py) had a **critical flaw**:

```python
# Old code (FLAWED)
lower_bound = Q1 - 3*IQR
upper_bound = Q3 + 3*IQR
```

**Problem**: When `Q1 - 3*IQR` is **negative**, extremely small positive prices (like 1.32e-18) pass the filter because they're greater than the negative bound.

**Example** (LINK token):

- Q1 = 18.50, Q3 = 24.74, IQR = 6.25
- Lower bound = 18.50 - 3(6.25) = **-0.24**
- Price of 1.32e-18 > -0.24 → **PASSES** (incorrectly)

Additionally, **no volume filtering** existed, so extreme volumes calculated from bad prices propagated through the entire pipeline.

## Solution

### Modified `filter_price_outliers()` in [src/calculate_usdc_prices.py](../src/calculate_usdc_prices.py)

Added **two-stage filtering** with absolute bounds:

#### Stage 1: Absolute Bounds (lines 151-197)

```python
# Absolute price bounds
absolute_min_price = 1e-9  # 0.000000001 USDC
absolute_max_price = 1e7   # 10 million USDC

# Absolute volume bounds
absolute_max_volume = 1e9  # 1 billion USDC per swap

# Filter observations outside absolute bounds
df_abs_filtered = df_prices.filter(
    (pl.col("price_in_usdc") >= absolute_min_price)
    & (pl.col("price_in_usdc") <= absolute_max_price)
    & (pl.col("usdc_volume") <= absolute_max_volume)
)
```

**Removed**: 19,652 observations (0.3%)

#### Stage 2: Enhanced IQR Filtering (lines 199-288)

```python
# Ensure lower_bound is never negative
pl.max_horizontal([
    pl.col("q1") - 3 * pl.col("iqr"),
    pl.lit(absolute_min_price),  # ← Floor at absolute minimum
]).alias("lower_bound")
```

**Removed**: 321,477 additional observations (4.9%)

**Total removed**: 341,129 observations (5.2% of original data)

### Added `validate_output()` Function (lines 291-448)

Comprehensive validation reporting:

1. Null value checks
2. Timestamp ordering
3. Time gap detection
4. Price range validation with warnings for extreme prices
5. Volume validation
6. Token-level price volatility analysis (Coefficient of Variation)
7. Low-value token identification (potential scam tokens)
8. Distribution summary statistics

## Results

### Before Fix

- **Total USDC volume**: $6.57e+27 (6.57 octillion - completely unrealistic)
- **Max src_flow_usdc**: 1.41e+26 USDC
- **Scam token volume**: $141 octillion
- **LINK token outlier**: 1.32e-18 USDC

### After Fix

- **Total USDC volume**: $72.4 billion (realistic for 2 months of Uniswap V3)
- **Max src_flow_usdc**: 7.54e+07 USDC
- **Scam token volume**: $4.3 million (still high but plausible for wash trading)
- **LINK token**: All extreme outliers removed

### Training Data Validation

```
Value range checks (excl. nulls):
  src_flow_usdc
    Range: [-1.8456e+07, 7.5428e+07] ✅ PASS
    (expected: [-1.0e+08, 1.0e+08])

  dest_flow_usdc
    Range: [-9.9999e+04, 9.9998e+04] ✅ PASS
    (expected: [-1.0e+05, 1.0e+05])
```

## Key Lessons

1. **IQR filtering alone is insufficient** for financial data with potential extreme outliers
2. **Absolute bounds** are necessary as a first-pass filter
3. **Volume validation** is as important as price validation
4. **Validation reporting** should happen at each pipeline stage to catch issues early
5. **Indirect swap pricing** (non-USDC pairs) is vulnerable to **price cache contamination** from bad prices

## Files Modified

1. [src/calculate_usdc_prices.py](../src/calculate_usdc_prices.py)

   - Enhanced `filter_price_outliers()` (lines 137-288)
   - Added `validate_output()` (lines 291-448)
   - Called validation before saving (line 839)

2. [src/filter_and_decode_swaps.py](../src/filter_and_decode_swaps.py)

   - Added `validate_output()` function (lines 254-495)
   - Added triple-redundant sorting
   - Added automatic deduplication

3. [src/training_data_validation.py](../src/training_data_validation.py)
   - Created comprehensive validation with 5 categories of checks

## Next Steps

Consider adding similar validation to:

- `src/generate_usdc_bars.py` - Validate bar generation
- `src/make_stationary.py` - Validate fractional differentiation
- `src/label_triple_barrier.py` - Validate labeling logic

## References

- **Issue**: Phase 5 outlier detection
- **Date**: 2025-12-17
- **Data pipeline**: [plans/data_overview.md](data_overview.md)
