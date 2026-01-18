# Phased Implementation Plan: Consensus Top 5 Improvements

**Timeline**: 6-8 weeks total
**Risk Level**: Low to Medium (incremental, reversible changes)
**Expected Impact**: 275% Sharpe improvement (0.8 → 3.0)

---

## PHASE 1: Quick Wins (Week 1)

**Goal**: Fix foundational data quality and label balance issues
**Expected Impact**: +38% Sharpe (0.8 → 1.1)
**Risk**: Low (parameter changes, fully reversible)

### Task 1.1: Adaptive Dollar Bar Thresholds (Day 1-2)

**File**: `src/generate_usdc_bars.py`

**Changes**:
1. Load token metadata with daily volume statistics
2. Calculate per-token threshold: `threshold = daily_volume_usdc * 0.001`
3. Replace fixed `DOLLAR_BAR_THRESHOLD = 100_000` with dynamic calculation

**Implementation**:
```python
# Add to generate_usdc_bars.py

def load_token_volumes(pools_path: Path) -> dict[str, float]:
    """Load average daily volume per token from pools metadata.

    Returns:
        Dict mapping token_address -> avg_daily_volume_usdc
    """
    with pools_path.open() as f:
        pools = json.load(f)

    token_volumes: dict[str, float] = {}
    for pool in pools:
        for token_key in ['token0', 'token1']:
            token_addr = pool[token_key]['address']
            # Estimate from pool TVL and turnover
            daily_vol = pool.get('volume_24h_usd', pool['tvl_usd'] * 0.1)

            if token_addr in token_volumes:
                token_volumes[token_addr] = max(token_volumes[token_addr], daily_vol)
            else:
                token_volumes[token_addr] = daily_vol

    return token_volumes

def get_dollar_bar_threshold(
    src_token: str,
    dest_token: str,
    token_volumes: dict[str, float],
    min_threshold: float = 1_000,
    max_threshold: float = 1_000_000,
) -> float:
    """Calculate adaptive dollar bar threshold for this token pair.

    Args:
        src_token: Source token address
        dest_token: Destination token address
        token_volumes: Dict of token -> daily volume
        min_threshold: Minimum bar size (for very illiquid tokens)
        max_threshold: Maximum bar size (for very liquid tokens)

    Returns:
        Dollar threshold for this pair (0.1% of daily volume)
    """
    src_vol = token_volumes.get(src_token, 100_000)
    dest_vol = token_volumes.get(dest_token, 100_000)

    # Use smaller of the two (less liquid token determines bar frequency)
    min_vol = min(src_vol, dest_vol)

    # 0.1% of daily volume
    threshold = min_vol * 0.001

    # Clamp to reasonable range
    return max(min_threshold, min(threshold, max_threshold))

# In main bar generation logic:
def generate_bars_for_pool(pool_swaps: pl.DataFrame, token_volumes: dict) -> pl.DataFrame:
    """Generate dollar bars with adaptive thresholds."""
    src_token = pool_swaps['src_token_id'][0]
    dest_token = pool_swaps['dest_token_id'][0]

    threshold = get_dollar_bar_threshold(src_token, dest_token, token_volumes)
    logger.info(f"Pool {pool_id[:10]}: Using ${threshold:,.0f} bar threshold")

    # Rest of bar generation logic uses this threshold...
```

**Validation**:
- Log distribution of thresholds across tokens
- Verify: Liquid tokens get larger bars, illiquid get smaller
- Check: Bar time deltas more uniform across tokens

**Acceptance Criteria**:
- [ ] Threshold varies by 10x+ across token spectrum
- [ ] Liquid tokens (USDC, WETH, WBTC): $100k-$1M bars
- [ ] Mid-cap tokens: $10k-$100k bars
- [ ] Small-cap tokens: $1k-$10k bars
- [ ] Bar time delta std dev reduced by 30%+

---

### Task 1.2: Symmetric Barriers + Longer Vertical (Day 3-4)

**File**: `src/label_triple_barrier.py`

**Changes**:
1. Make barriers symmetric: `UPPER_MULTIPLE = LOWER_MULTIPLE = 1.5`
2. Increase vertical barrier: `BARRIER_FRACTION = 0.5` (was 0.1)
3. Add adaptive scaling by token volatility

**Implementation**:
```python
# In label_triple_barrier.py

# Update constants
UPPER_MULTIPLE = 1.5  # Was 2.0
LOWER_MULTIPLE = 1.5  # Was 1.0 - NOW SYMMETRIC
BARRIER_FRACTION = 0.5  # Was 0.1 - NOW 50% of daily bars (4-6 hours)

def get_adaptive_barrier_multiple(
    token_volatility: float,
    vol_quintiles: list[float],
) -> float:
    """Scale barrier width by token volatility quintile.

    High-volatility tokens get wider barriers (longer holding periods).
    Low-volatility tokens get tighter barriers.

    Args:
        token_volatility: This token's rolling volatility
        vol_quintiles: [20th, 40th, 60th, 80th] percentile values from all tokens

    Returns:
        Barrier multiple in range [1.0, 2.0]
    """
    quintile = np.searchsorted(vol_quintiles, token_volatility)
    # Quintile 0 (lowest vol): 1.0σ
    # Quintile 4 (highest vol): 2.0σ
    return 1.0 + (quintile * 0.25)

# In _process_token_labels():
def _process_token_labels(
    df: pl.DataFrame,
    token_col: str,
    fracdiff_col: str,
    barrier_fraction: float,
    volatility_window: int,
    base_upper_multiple: float,  # Now just base value
    base_lower_multiple: float,  # Now just base value
    vol_quintiles: list[float],  # NEW: for adaptive scaling
) -> pl.DataFrame:
    """Process labels with adaptive barriers."""

    # ... existing setup code ...

    for token_tuple, token_group in token_groups:
        # ... existing code ...

        # Calculate token's average volatility
        token_vol = np.nanmean(volatilities)

        # Get adaptive multiple for this token
        adaptive_multiple = get_adaptive_barrier_multiple(token_vol, vol_quintiles)

        # Apply to barriers (symmetric)
        upper_multiple = base_upper_multiple * adaptive_multiple
        lower_multiple = base_lower_multiple * adaptive_multiple

        # Apply triple-barrier with adaptive multiples
        labels, barrier_touch_bars = apply_triple_barrier(
            prices,
            volatilities,
            upper_multiple,
            lower_multiple,
            vertical_bars,
        )
```

**Validation**:
- Check label distribution: Should be ~33% each for {-1, 0, +1}
- Verify: High-vol tokens have wider barriers
- Check: Timeout rate reduced (fewer label=0)

**Acceptance Criteria**:
- [ ] Label distribution: Down 30-35%, Stay 30-40%, Up 30-35% (balanced)
- [ ] Average barrier touch time: 2-4 hours (was 20 minutes)
- [ ] Timeout rate: <30% (was 70%)
- [ ] High-vol tokens: 1.5-2.0σ barriers
- [ ] Low-vol tokens: 1.0-1.25σ barriers

---

### Task 1.3: Re-run Pipeline with Phase 1 Changes (Day 5)

**Commands**:
```bash
# Regenerate bars with adaptive thresholds
make generate-usdc-bars

# Regenerate labels with symmetric barriers
make label-triple-barrier

# Validate output
uv run python src/training_data_validation.py --verbose
```

**Validation Metrics** (compare to baseline):
- Bar heterogeneity: Time delta std dev should decrease
- Label balance: Down/Stay/Up should be more balanced
- Data coverage: More tokens should pass stationarity tests

**Acceptance Criteria**:
- [ ] Pipeline completes without errors
- [ ] Bar time delta CoV (coefficient of variation) reduced by 30%+
- [ ] Label distribution within [30%, 40%] for each class
- [ ] No significant data loss (>95% of bars preserved)

---

### Phase 1 Summary

**Estimated Time**: 5 days
**Files Modified**: 2 (`generate_usdc_bars.py`, `label_triple_barrier.py`)
**Lines Changed**: ~200 lines
**Risk Level**: Low (parameter changes, re-run pipeline)
**Rollback**: Keep old parquet files for 30 days

**Success Metrics**:
- Sharpe ratio improvement: +20-30% (backtest validation)
- Label balance improved (from 80/20 to 33/33/33)
- Bar information content more uniform

---

## PHASE 2: Strategic Improvements (Weeks 2-4)

**Goal**: Optimize decision-making and adapt to regime dynamics
**Expected Impact**: +120% Sharpe (1.1 → 2.5)
**Risk**: Medium (model architecture changes)

### Task 2.1: Restore Ternary Classification (Week 2, Days 1-2)

**File**: `src/label_triple_barrier.py`, `src/dex_contagion_trader.py`

**Changes**:
1. Remove binary merge in label generation
2. Update model output to 3 classes
3. Update loss function for 3 classes

**Implementation**:

**In `label_triple_barrier.py`**:
```python
# REMOVE the binary merge (lines 477-500 in current code)
# Keep original ternary labels: -1, 0, +1

# Labels should stay as-is from triple-barrier method
# No conversion to binary
```

**In `dex_contagion_trader.py`**:
```python
# Update model configuration
NUM_CLASSES = 3  # Was 2 - now {-1: down, 0: stay, +1: up}

# In build_window() - map labels to 0, 1, 2 for PyTorch
# Original: -1, 0, +1
# PyTorch needs: 0, 1, 2
raw_src_labels = df_window["label"].to_numpy().astype(np.int64)
src_labels = np.where(
    raw_src_labels == -999,
    -999,  # Keep unlabeled marker
    raw_src_labels + 1,  # Map: -1→0, 0→1, +1→2
)

# Similar for dest labels
```

**Acceptance Criteria**:
- [ ] Model trains on 3 classes without errors
- [ ] Loss function handles 3-class labels
- [ ] Predictions return 3 probabilities per token

---

### Task 2.2: Implement Kelly Criterion Position Sizing (Week 2, Days 3-5)

**File**: `src/dex_contagion_trader.py`

**Changes**:
1. Modify prediction function to return probabilities
2. Calculate Kelly bet size from probabilities
3. Update position construction logic

**Implementation**:
```python
# In dex_contagion_trader.py

def calculate_kelly_position_size(
    prob_down: float,
    prob_stay: float,
    prob_up: float,
    min_bet: float = 0.05,
) -> tuple[float, str]:
    """Calculate Kelly bet size and direction.

    Kelly formula for equal payoffs: f* = p_win - p_lose
    Use half-Kelly for robustness: f* / 2

    Args:
        prob_down: P(price goes down)
        prob_stay: P(price stays neutral)
        prob_up: P(price goes up)
        min_bet: Minimum bet size threshold

    Returns:
        (bet_size, direction) where:
            bet_size: Fraction of capital to allocate [0.0, 1.0]
            direction: 'long' or 'skip'
    """
    # Expected edge (assuming equal +/- moves)
    edge = prob_up - prob_down

    # Kelly bet size (full Kelly)
    kelly_full = max(0, edge)

    # Half-Kelly for safety (more robust to estimation errors)
    kelly_half = kelly_full / 2

    # Only bet if edge is positive AND above minimum threshold
    if kelly_half >= min_bet:
        return kelly_half, 'long'
    else:
        return 0.0, 'skip'

def predict_with_kelly_sizing(
    model: TokenPredictorModel,
    data: DGData,
    le: LabelEncoder,
    device: str,
    max_positions: int = 10,  # Increased from 5
) -> list[tuple[str, float, dict]]:
    """Predict tokens and calculate Kelly position sizes.

    Returns:
        List of (token_addr, bet_size, metadata) sorted by bet_size descending
    """
    model.eval()
    dg = DGraph(data, device=device)
    loader = DGDataLoader(dg, batch_size=BATCH_SIZE, batch_unit=BATCH_UNIT)

    # Accumulate probabilities per token
    token_probabilities: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for batch_data in loader:
            if batch_data.src.shape[0] == 0 or batch_data.node_ids is None:
                continue

            move_batch_to_device(batch_data, device)
            logits = model(batch_data)

            if logits is None:
                continue

            # Convert logits to probabilities (3 classes)
            probs = torch.softmax(logits, dim=1)  # [N, 3]

            # Get node IDs for labeled nodes
            has_label = batch_data.dynamic_node_feats[:, 3] != -999
            labeled_node_ids = batch_data.node_ids[has_label].cpu().numpy()
            labeled_probs = probs.cpu().numpy()

            for node_id, prob in zip(labeled_node_ids, labeled_probs, strict=True):
                if node_id not in token_probabilities:
                    token_probabilities[node_id] = []
                token_probabilities[node_id].append(prob)

    # Calculate Kelly sizes
    positions = []
    for node_id, prob_list in token_probabilities.items():
        # Average probabilities across all predictions for this token
        avg_probs = np.mean(prob_list, axis=0)
        p_down, p_stay, p_up = avg_probs

        # Calculate Kelly bet size
        bet_size, direction = calculate_kelly_position_size(p_down, p_stay, p_up)

        if direction == 'long':
            token_addr = le.inverse_transform([node_id])[0]
            metadata = {
                'prob_down': float(p_down),
                'prob_stay': float(p_stay),
                'prob_up': float(p_up),
                'edge': float(p_up - p_down),
                'kelly_full': float(2 * bet_size),  # Recover full Kelly
            }
            positions.append((token_addr, bet_size, metadata))

    # Sort by bet size descending, take top N
    positions.sort(key=lambda x: x[1], reverse=True)
    return positions[:max_positions]
```

**In backtesting logic**:
```python
def calculate_portfolio_return_with_kelly(
    df_trade: pl.DataFrame,
    positions: list[tuple[str, float, dict]],
) -> tuple[float, dict]:
    """Calculate portfolio return with Kelly-weighted positions.

    Args:
        df_trade: Trading window data
        positions: List of (token_addr, bet_size, metadata)

    Returns:
        (portfolio_return, position_details)
    """
    total_capital = 1.0
    position_returns = {}

    for token_addr, kelly_bet_size, metadata in positions:
        # Allocate Kelly fraction of capital
        position_capital = total_capital * kelly_bet_size

        # Calculate token return
        token_return = calculate_token_return(df_trade, token_addr)

        # Position P&L
        position_pnl = position_capital * token_return
        position_returns[token_addr] = {
            'allocated_capital': position_capital,
            'return': token_return,
            'pnl': position_pnl,
            'metadata': metadata,
        }

    # Portfolio return = sum of position P&Ls / total capital
    portfolio_return = sum(p['pnl'] for p in position_returns.values()) / total_capital

    return portfolio_return, position_returns
```

**Acceptance Criteria**:
- [ ] Kelly sizing correctly maps probabilities to bet sizes
- [ ] High-confidence trades get larger allocations (50-80%)
- [ ] Marginal trades get small allocations (5-10%) or skipped
- [ ] Total allocation can be <100% (capital preservation when no edge)

---

### Task 2.3: Implement Regime Detection (Week 3)

**New File**: `src/detect_regime.py`

**Implementation**:
```python
"""Detect market regime for adaptive trading strategy.

Classifies current market state as bull, bear, or sideways based on
30-day volatility and price trend.
"""

import logging
from datetime import date, timedelta

import numpy as np
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_market_regime(
    df: pl.DataFrame,
    lookback_days: int = 30,
    vol_percentiles: tuple[float, float] = (0.2, 0.8),
) -> str:
    """Detect market regime from recent price history.

    Regime classification:
    - Bull: High volatility + upward trend
    - Bear: High volatility + downward trend
    - Sideways: Low volatility (regardless of trend)

    Args:
        df: DataFrame with bar_close_timestamp and src_fracdiff
        lookback_days: Number of days to analyze
        vol_percentiles: (low, high) percentile thresholds for volatility

    Returns:
        'bull' | 'bear' | 'sideways'
    """
    # Get recent data
    cutoff_date = df['bar_close_timestamp'].max() - timedelta(days=lookback_days)
    df_recent = df.filter(pl.col('bar_close_timestamp') >= cutoff_date)

    if len(df_recent) < 100:
        logger.warning(f"Insufficient data for regime detection ({len(df_recent)} bars)")
        return 'sideways'  # Conservative default

    # Calculate volatility (annualized)
    returns = df_recent['src_fracdiff'].diff().drop_nulls()
    volatility = float(returns.std() * np.sqrt(365))

    # Calculate trend (linear regression slope)
    prices = df_recent['src_price_usdc'].to_numpy()
    x = np.arange(len(prices))
    trend_slope = float(np.polyfit(x, prices, deg=1)[0])

    # Normalize slope by price level (percentage change per day)
    avg_price = float(np.mean(prices))
    trend_pct_per_day = (trend_slope / avg_price) * 100 if avg_price > 0 else 0

    # Historical percentiles (pre-computed or calculated from full dataset)
    # These should be computed once from full historical data
    vol_20th, vol_80th = vol_percentiles

    logger.info(
        f"Regime detection: vol={volatility:.2%}, trend={trend_pct_per_day:+.2%}/day"
    )

    # Classify regime
    if volatility > vol_80th:
        if trend_pct_per_day > 0.1:  # > 0.1% per day trend
            regime = 'bull'
        elif trend_pct_per_day < -0.1:
            regime = 'bear'
        else:
            regime = 'sideways'  # High vol but no clear trend
    else:
        regime = 'sideways'  # Low vol

    logger.info(f"Detected regime: {regime}")
    return regime

def compute_historical_vol_percentiles(
    df: pl.DataFrame,
    window_days: int = 30,
) -> tuple[float, float]:
    """Compute historical volatility percentiles for regime thresholds.

    Args:
        df: Full historical dataset
        window_days: Rolling window for volatility calculation

    Returns:
        (20th_percentile, 80th_percentile) volatility values
    """
    # Calculate rolling volatility across full history
    df_sorted = df.sort('bar_close_timestamp')

    # Group by date and calculate daily volatility
    daily_vols = []
    dates = df_sorted['bar_close_timestamp'].dt.date().unique().sort()

    for i in range(window_days, len(dates)):
        date_range = dates[i-window_days:i]
        window_data = df_sorted.filter(
            pl.col('bar_close_timestamp').dt.date().is_in(date_range)
        )

        if len(window_data) > 100:
            returns = window_data['src_fracdiff'].diff().drop_nulls()
            vol = float(returns.std() * np.sqrt(365))
            daily_vols.append(vol)

    # Calculate percentiles
    vol_20th = float(np.percentile(daily_vols, 20))
    vol_80th = float(np.percentile(daily_vols, 80))

    logger.info(f"Historical vol percentiles: 20th={vol_20th:.2%}, 80th={vol_80th:.2%}")

    return vol_20th, vol_80th
```

**Acceptance Criteria**:
- [ ] Regime detection runs without errors
- [ ] Historical data correctly classified (2020-2021=bull, 2022=bear, 2019=sideways)
- [ ] Volatility percentiles make sense (20th ~30%, 80th ~80% for crypto)

---

### Task 2.4: Train Regime-Specific Models (Week 4)

**File**: `src/dex_contagion_trader.py`

**Changes**:
1. Pre-compute regime labels for historical data
2. Train 3 separate models (one per regime)
3. Save models with regime suffix

**Implementation**:
```python
# In dex_contagion_trader.py

def train_regime_specific_models(
    df: pl.DataFrame,
    le: LabelEncoder,
    num_tokens: int,
    output_dir: Path,
    epochs: int = 10,
) -> dict[str, Path]:
    """Train separate models for each regime.

    Args:
        df: Full labeled dataset
        le: Label encoder
        num_tokens: Number of unique tokens
        output_dir: Directory to save models
        epochs: Training epochs per model

    Returns:
        Dict mapping regime -> model_path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute regime labels for all historical data
    logger.info("Computing historical regime labels...")
    vol_percentiles = compute_historical_vol_percentiles(df)

    # Add regime column to dataframe
    regime_labels = []
    dates = df['bar_close_timestamp'].dt.date().unique().sort()

    for date in dates:
        df_date = df.filter(pl.col('bar_close_timestamp').dt.date() == date)
        regime = detect_market_regime(df_date, vol_percentiles=vol_percentiles)
        regime_labels.extend([regime] * len(df_date))

    df = df.with_columns(pl.Series('regime', regime_labels))

    # Train model for each regime
    model_paths = {}

    for regime in ['bull', 'bear', 'sideways']:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {regime.upper()} model")
        logger.info(f"{'='*70}")

        # Filter to this regime only
        df_regime = df.filter(pl.col('regime') == regime)
        logger.info(f"Training samples: {len(df_regime):,}")

        if len(df_regime) < 1000:
            logger.warning(f"Insufficient data for {regime} regime, skipping")
            continue

        # Build training data for this regime
        data_regime = build_window(
            df_regime,
            start_date=df_regime['bar_close_timestamp'].min().date(),
            end_date=df_regime['bar_close_timestamp'].max().date(),
        )

        # Initialize model
        model = TokenPredictorModel(
            num_nodes=num_tokens,
            node_feat_dim=NODE_INPUT_DIM,
            node_embed_dim=NODE_EMBED_DIM,
            output_dim=NUM_CLASSES,
            device=DEVICE,
        ).to(DEVICE)

        # Train
        train_model(model, data_regime, DEVICE, epochs=epochs)

        # Save
        model_path = output_dir / f'tgcn_{regime}.pth'
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved {regime} model to {model_path}")

        model_paths[regime] = model_path

    return model_paths

# In backtest_slide() - use regime-specific models
def backtest_slide(...):
    # ... existing code ...

    for day_idx in range(TRAIN_WINDOW_DAYS, TRAIN_WINDOW_DAYS + trading_days):
        trade_date = dates_array[day_idx]

        # Detect current regime
        df_recent_30d = df.filter(
            (pl.col('trade_date') >= trade_date - timedelta(days=30))
            & (pl.col('trade_date') < trade_date)
        )
        current_regime = detect_market_regime(df_recent_30d)

        logger.info(f"Current regime: {current_regime}")

        # Load regime-specific model (or train new one)
        model_path = output_dir / f'tgcn_{current_regime}.pth'

        if model_path.exists():
            # Load pre-trained regime model
            model = TokenPredictorModel(...).to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded pre-trained {current_regime} model")
        else:
            # Train new model on regime-filtered data
            logger.info(f"Training new {current_regime} model")
            # ... filter training data by regime ...
            # ... train model ...

        # Rest of backtest logic uses regime-specific model
        # ...
```

**Acceptance Criteria**:
- [ ] 3 models train successfully (bull, bear, sideways)
- [ ] Models specialize: bull model learns momentum, bear learns mean-reversion
- [ ] Backtest correctly loads regime-appropriate model each day

---

### Phase 2 Summary

**Estimated Time**: 3 weeks
**Files Modified**: 2 (`label_triple_barrier.py`, `dex_contagion_trader.py`)
**New Files**: 1 (`detect_regime.py`)
**Lines Changed**: ~500 lines
**Risk Level**: Medium (model changes, regime logic)
**Rollback**: Keep Phase 1 models/data for comparison

**Success Metrics**:
- Sharpe ratio improvement: +120% vs Phase 1
- Position sizing: Variable (5-80% per position vs fixed 20%)
- Regime adaptation: Different strategies in bull vs bear

---

## PHASE 3: Execution Reality (Weeks 5-6)

**Goal**: Model execution costs to bridge backtest-to-live gap
**Expected Impact**: +20-30% Sharpe (2.5 → 3.0+)
**Risk**: Medium-High (external API dependencies)

### Task 3.1: Fetch Gas Price Data (Week 5, Days 1-2)

**New File**: `src/fetch_gas_prices.py`

**Implementation**:
```python
"""Fetch historical gas price data from Etherscan API."""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
import time

import polars as pl
import requests
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY')
ETHERSCAN_BASE_URL = 'https://api.etherscan.io/api'

def fetch_gas_prices(
    start_date: date,
    end_date: date,
    output_file: Path,
) -> None:
    """Fetch historical average gas prices from Etherscan.

    Note: Etherscan API returns daily averages, not hourly.
    For hourly data, would need to use different source (Alchemy, etc.)

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        output_file: Where to save parquet
    """
    logger.info(f"Fetching gas prices from {start_date} to {end_date}")

    # Etherscan daily gas oracle endpoint
    # Returns average gas price per day
    gas_data = []

    current_date = start_date
    while current_date <= end_date:
        # Etherscan format: Unix timestamp
        timestamp = int(datetime.combine(current_date, datetime.min.time()).timestamp())

        params = {
            'module': 'stats',
            'action': 'dailyavggasprice',
            'startdate': current_date.strftime('%Y-%m-%d'),
            'enddate': current_date.strftime('%Y-%m-%d'),
            'apikey': ETHERSCAN_API_KEY,
        }

        response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                # Parse result
                result = data['result']
                gas_data.append({
                    'date': current_date,
                    'avg_gas_gwei': float(result['avgGasPrice']) / 1e9,  # Wei to Gwei
                })
            else:
                logger.warning(f"No data for {current_date}: {data.get('message')}")
        else:
            logger.error(f"API error for {current_date}: {response.status_code}")

        current_date += timedelta(days=1)
        time.sleep(0.2)  # Rate limiting (5 requests/second max)

    # Convert to DataFrame
    df = pl.DataFrame(gas_data)

    # Expand daily to hourly (simple forward-fill for now)
    # In production, would use more granular data source
    hourly_data = []
    for row in df.iter_rows(named=True):
        base_date = row['date']
        gas_price = row['avg_gas_gwei']

        for hour in range(24):
            hourly_data.append({
                'timestamp': datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hour),
                'avg_gas_gwei': gas_price,
            })

    df_hourly = pl.DataFrame(hourly_data)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.write_parquet(output_file)
    logger.info(f"Saved {len(df_hourly)} hourly gas price records to {output_file}")

if __name__ == '__main__':
    # Example usage
    fetch_gas_prices(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        output_file=Path('data/gas_prices.parquet'),
    )
```

**Acceptance Criteria**:
- [ ] Gas data fetched successfully from Etherscan
- [ ] Data covers full date range
- [ ] Format: timestamp, avg_gas_gwei

---

### Task 3.2: Estimate Pool Liquidity Depth (Week 5, Days 3-5)

**New File**: `src/fetch_liquidity_depth.py`

**Implementation**:
```python
"""Estimate Uniswap V3 pool liquidity depth for slippage calculation."""

import logging
from pathlib import Path
from typing import Any

import polars as pl
from web3 import Web3
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# Simplified implementation - use pool TVL as proxy for depth
# Full implementation would query tick-level liquidity via Uniswap SDK

def estimate_slippage_from_tvl(
    pool_tvl_usd: float,
    swap_size_usd: float,
) -> float:
    """Estimate slippage using simple liquidity model.

    Price impact ~ sqrt(swap_size / liquidity)
    This is simplified; real implementation needs tick-level liquidity.

    Args:
        pool_tvl_usd: Total value locked in pool
        swap_size_usd: Size of swap in USD

    Returns:
        Estimated slippage percentage (e.g., 0.02 = 2%)
    """
    if pool_tvl_usd <= 0:
        return 1.0  # 100% slippage for zero liquidity (avoid division)

    # Simplified model: slippage proportional to sqrt(size/TVL)
    # Real Uniswap V3 math is more complex (tick-based)
    slippage = (swap_size_usd / pool_tvl_usd) ** 0.5

    # Cap at 100%
    return min(slippage, 1.0)

def fetch_pool_liquidity(
    pools_path: Path,
    output_file: Path,
) -> None:
    """Fetch TVL for all pools and estimate slippage.

    Args:
        pools_path: Path to pools.json (from kaiko API)
        output_file: Where to save depth estimates
    """
    import json

    logger.info(f"Loading pools from {pools_path}")
    with pools_path.open() as f:
        pools = json.load(f)

    depth_data = []

    for pool in pools:
        pool_id = pool['pool_address']
        tvl_usd = pool.get('tvl_usd', 0)

        # Estimate slippage for different swap sizes
        slippage_10k = estimate_slippage_from_tvl(tvl_usd, 10_000)
        slippage_100k = estimate_slippage_from_tvl(tvl_usd, 100_000)

        depth_data.append({
            'pool_id': pool_id,
            'tvl_usd': tvl_usd,
            'slippage_10k': slippage_10k,
            'slippage_100k': slippage_100k,
        })

    df = pl.DataFrame(depth_data)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_file)
    logger.info(f"Saved depth estimates for {len(df)} pools to {output_file}")

if __name__ == '__main__':
    fetch_pool_liquidity(
        pools_path=Path('data/pools.json'),
        output_file=Path('data/pool_depth.parquet'),
    )
```

**Note**: This is a simplified implementation using TVL as proxy. Full implementation would use Uniswap V3 SDK to query tick-level liquidity.

**Acceptance Criteria**:
- [ ] Depth estimates generated for all pools
- [ ] High-TVL pools have low slippage estimates
- [ ] Low-TVL pools have high slippage estimates
- [ ] Format: pool_id, tvl_usd, slippage_10k, slippage_100k

---

### Task 3.3: Join Execution Features to Bars (Week 6, Days 1-2)

**File**: `src/generate_usdc_bars.py`

**Changes**: Join gas and depth data to bars

**Implementation**:
```python
# In generate_usdc_bars.py

def add_execution_features(
    bars_df: pl.DataFrame,
    gas_prices_path: Path,
    pool_depth_path: Path,
) -> pl.DataFrame:
    """Join gas prices and liquidity depth to bars.

    Args:
        bars_df: Generated dollar bars
        gas_prices_path: Path to gas_prices.parquet
        pool_depth_path: Path to pool_depth.parquet

    Returns:
        Bars with execution features added
    """
    logger.info("Adding execution features to bars...")

    # Load gas prices
    gas_df = pl.read_parquet(gas_prices_path)

    # Join gas prices (by nearest hour)
    bars_with_gas = bars_df.join_asof(
        gas_df,
        left_on='bar_close_timestamp',
        right_on='timestamp',
        strategy='backward',  # Use most recent gas price
    )

    # Load pool depth
    depth_df = pl.read_parquet(pool_depth_path)

    # Join depth (by pool_id, static data)
    bars_with_execution = bars_with_gas.join(
        depth_df,
        on='pool_id',
        how='left',
    )

    # Fill missing values with conservative estimates
    bars_with_execution = bars_with_execution.with_columns([
        pl.col('avg_gas_gwei').fill_null(50),  # Assume 50 gwei if missing
        pl.col('slippage_100k').fill_null(0.10),  # Assume 10% slippage if missing
    ])

    logger.info("✅ Execution features added")
    return bars_with_execution
```

**Acceptance Criteria**:
- [ ] All bars have gas_gwei and slippage features
- [ ] No null values in execution columns
- [ ] Join preserves all bars (left join, fill missing)

---

### Task 3.4: Update Model to Use Execution Features (Week 6, Days 3-5)

**File**: `src/dex_contagion_trader.py`

**Changes**:
1. Increase `NODE_INPUT_DIM` from 3 to 5
2. Add gas and slippage to node features

**Implementation**:
```python
# In dex_contagion_trader.py

# Update feature dimension
NODE_INPUT_DIM = 5  # Was 3
# Now: [fracdiff, volatility, flow, gas_gwei, slippage]

# In build_window():
def build_window(...):
    # ... existing code ...

    # Extract execution features
    gas_gwei = df_window['avg_gas_gwei'].to_numpy()
    slippage = df_window['slippage_100k'].to_numpy()

    # Dynamic node features: [fracdiff, volatility, flow, gas, slippage, label, weight]
    dynamic_node_feats = np.zeros((num_events * 2, 7), dtype=np.float32)

    # Src node updates
    dynamic_node_feats[0::2, 0] = src_fracdiff  # fracdiff
    dynamic_node_feats[0::2, 1] = volatility     # volatility
    dynamic_node_feats[0::2, 2] = src_flow       # flow
    dynamic_node_feats[0::2, 3] = gas_gwei       # NEW: gas price
    dynamic_node_feats[0::2, 4] = slippage       # NEW: slippage
    dynamic_node_feats[0::2, 5] = src_labels     # label
    dynamic_node_feats[0::2, 6] = src_weights    # weight

    # Similar for dest nodes...

    # Model forward pass now receives 5 features (not 3)
    # node_features = batch.dynamic_node_feats[:, :5]  # [fracdiff, vol, flow, gas, slippage]
```

**Acceptance Criteria**:
- [ ] Model trains with 5 input features
- [ ] Execution features normalized appropriately
- [ ] Model learns to avoid high-gas or high-slippage trades

---

### Phase 3 Summary

**Estimated Time**: 2 weeks
**Files Modified**: 2 (`generate_usdc_bars.py`, `dex_contagion_trader.py`)
**New Files**: 2 (`fetch_gas_prices.py`, `fetch_liquidity_depth.py`)
**Lines Changed**: ~400 lines
**Risk Level**: Medium-High (API dependencies)
**Rollback**: Can disable features by setting to constant values

**Success Metrics**:
- Sharpe ratio improvement: +20-30% vs Phase 2
- Model avoids: High-gas periods, thin-liquidity pools
- Backtest-to-live gap: Reduced (execution costs modeled)

---

## VALIDATION & TESTING

### Per-Phase Backtests

After each phase, run full backtest and compare metrics:

**Metrics to Track**:
- Sharpe ratio (primary)
- Max drawdown
- Win rate
- Average win / average loss
- Total return
- Number of trades per day
- Average position size

**Comparison Format**:
```
Metric              Baseline  Phase1  Phase2  Phase3
-----------------   --------  ------  ------  ------
Sharpe Ratio        0.80      1.10    2.50    3.00
Max Drawdown        30%       25%     20%     18%
Win Rate            52%       55%     58%     60%
Avg Win             +15%      +15%    +25%    +22%
Avg Loss            -12%      -10%    -8%     -7%
Trades/Day          5         3       2       2
Avg Position Size   20%       18%     35%     30%
```

### Out-of-Sample Validation

**Reserve last 30 days** for final validation:
- Train on all data except last 30 days
- Backtest on last 30 days
- Compare to in-sample performance

**Acceptance Criteria**:
- Out-of-sample Sharpe within 20% of in-sample
- Out-of-sample max DD within 30% of in-sample
- No regime overfitting (test on all 3 regimes)

---

## ROLLBACK PLAN

### If Phase Fails

**Rollback Steps**:
1. Restore previous parquet files (bars, labels)
2. Restore previous model checkpoints
3. Re-run validation on previous phase
4. Debug issue in isolated environment

**Rollback Triggers**:
- Sharpe ratio decreases >10%
- Max drawdown increases >10%
- Pipeline errors or data corruption
- Model training instability

---

## TIMELINE SUMMARY

| Week | Phase | Tasks | Risk | Impact |
|------|-------|-------|------|--------|
| 1 | Phase 1 | Adaptive bars, symmetric barriers | Low | +38% Sharpe |
| 2 | Phase 2.1 | Ternary classification, Kelly sizing | Med | +50% Sharpe |
| 3 | Phase 2.2 | Regime detection logic | Med | +70% Sharpe |
| 4 | Phase 2.3 | Train regime models | Med | +70% Sharpe |
| 5 | Phase 3.1 | Fetch gas/depth data | High | +15% Sharpe |
| 6 | Phase 3.2 | Integrate execution features | High | +15% Sharpe |

**Total**: 6-8 weeks (including testing and validation)
**Final Expected Sharpe**: 3.0+ (from 0.8 baseline)

---

## NEXT STEPS

1. **Review this plan** with team/stakeholders
2. **Set up dev environment** (branch, backup data)
3. **Start Phase 1, Task 1.1** (adaptive bars)
4. **Daily progress tracking** (update this doc with actuals)
5. **Phase-end reviews** (backtest, validate, decide proceed/rollback)

**Ready to begin implementation when approved.**
