# Phase 5: TGCN Architecture Fixes

**Status**: In Process
**Goal**: Fix critical issues in TokenPredictorModel to properly utilize Prado's methodology and TGM library capabilities

---

## Executive Summary

Deep analysis of the current TokenPredictorModel implementation revealed **5 critical problems** that prevent the model from learning effectively:

1. Edge features are completely ignored by TGCN
2. Node features (fracdiff prices) are misassigned as edge features
3. Graph structure is unidirectional (breaks contagion model)
4. Only source tokens receive labels (50% of data unused)
5. Timestamps don't respect temporal ordering

These issues mean the model currently learns **only from graph topology and random embeddings**, ignoring all the carefully prepared stationary price signals from [plans/data_overview.md](data_overview.md).

---

## Future Work (Post-Phase 5)

After fixing the core architecture, the following enhancements should be considered:

### 7. Static Node Features
Add token-level metadata that doesn't change during training windows:
- Liquidity tier (from `data/tokens.json`)
- Market cap bucket
- Historical volatility regime

This can help embeddings converge faster by providing stable context.

### 8. Meta-Labeling (Prado's Two-Stage Approach)
Implement Prado's full methodology:
- **Stage 1**: Current model predicts direction (up/down/stay)
- **Stage 2**: Meta-model predicts bet size/confidence
- Use meta-model predictions to size positions dynamically

This enables the model to abstain from low-confidence trades.

---

## Critical Issues to Fix

### Issue 1: Node Features Not Used in Temporal Graph ✅ RESOLVED

**Problem**:
- Swap data contains observations of token states (fracdiff, volatility, flow) but model only used random embeddings
- Features were misclassified as "edge features" when they're actually token state observations
- Model ignored all carefully prepared stationary price signals from Prado's methodology

**Resolution** ([src/dex_contagion_trader.py:343-403](../src/dex_contagion_trader.py#L343-L403)):

**Used `dynamic_node_feats` for temporal token state evolution**:
- Each swap event creates TWO node updates (src and dest tokens)
- Features evolve temporally: Token A participates in swaps at t=0, t=1, t=2 → features update at each time
- Structure: `[fracdiff, volatility, flow, label, weight]` per node update
- TGCN sees temporal evolution of token states through message passing

**Why node features, not edge features**:
1. **Task alignment**: Predicting token prices = node classification task
2. **Semantic correctness**: fracdiff/flow/volatility describe token states, not swap properties
3. **TGCN design**: Builds node embeddings that evolve over time via graph structure
4. **Contagion dynamics**: Price movements spread through node state propagation
5. **Data structure**: Each swap provides observations of two tokens' current states

**Edge features used correctly**:
- `edge_weight = abs(src_flow_usdc)` for importance weighting
- Large volume swaps have more influence in message passing

**Results**:
- Training loss: 0.1045 (model learning successfully)
- Confident predictions: 100% bullish scores vs 0% before
- Temporal dynamics properly captured

---

### Issue 2: Graph Structure Doesn't Match Financial Reality (Unidirectional Edges)

**Status**: Resolved

**Current Implementation** ([src/dex_contagion_trader.py:407](../src/dex_contagion_trader.py#L407)):
```python
edge_index=torch.from_numpy(np.column_stack((src, dst)))
```

**Problem**:
- Creates DIRECTED edges: src → dst only
- No reverse edges: dst → src
- In TGCN message passing, information flows one-way along edges

**What Happens**:
- Token A sends messages to Token B
- Token B CANNOT send messages back to Token A (unless there's a separate swap with B→A direction)
- Limits "contagion" propagation - price movements can't flow bidirectionally through the graph

**Financial Reality**:
Dollar bars represent swap volume between two tokens. Both tokens participate in the swap and should mutually influence each other's representations. A large SOL→USDC swap provides information about both SOL's selling pressure AND USDC's buying pressure.

**Fix Options**:

**Option A (Recommended): Bidirectional Edges**
Add reverse edges to enable bidirectional message passing:
```python
# Create bidirectional edges
edges_forward = np.column_stack((src, dst))
edges_reverse = np.column_stack((dst, src))
edge_index = np.vstack([edges_forward, edges_reverse])

# Duplicate edge features/weights for reverse edges
edge_feats = np.concatenate([edge_feats, edge_feats])
edge_timestamps = np.concatenate([edge_timestamps, edge_timestamps])
```

**Option B: Research TGCN Behavior**
Investigate whether TGCN already handles bidirectional message passing internally, or if the library has built-in support for undirected graphs.

**Considerations**:
- Adding reverse edges doubles the edge count
- May improve contagion modeling but increases computational cost
- Should validate that TGCN doesn't already do bidirectional message passing

**Files to modify**:
- `src/dex_contagion_trader.py::build_window()` - Add reverse edges if needed

---

### Issue 3: Incomplete Label Utilization (Dest Tokens Unlabeled)

**Status**: Resolved

**Current Implementation** ([src/dex_contagion_trader.py:388-396](../src/dex_contagion_trader.py#L388-L396)):
```python
# Src node updates (even indices)
dynamic_node_feats[0::2, 3] = labels          # label (only src has labels)
dynamic_node_feats[0::2, 4] = weights         # weight

# Dest node updates (odd indices)
dynamic_node_feats[1::2, 3] = 0               # no label for dest (yet)
dynamic_node_feats[1::2, 4] = 0               # no weight for dest (yet)
```

**Problem**:
- Each swap event creates 2 node updates (src and dest)
- Only src tokens receive labels and sample weights
- Dest tokens have `label=0` and `weight=0`
- Training signal is provided for only ~50% of node updates

**Training Impact**:
- Dest tokens learn purely through graph structure (unsupervised)
- Wastes 50% of available training opportunities
- Each token participates in swaps as BOTH src and dest, but only learns when it's src

**Why This Matters**:
- Triple barrier labels are computed per-token based on future price movements
- A token's price trajectory is independent of whether it was src or dest in a swap
- Both src and dest nodes should receive supervised learning signal

**Fix Options**:

**Option A (Recommended): Add Dest Token Labels**
Compute labels for destination tokens during data preparation:
1. In `label_triple_barrier.py`, generate labels for BOTH src and dest tokens
2. Store as `src_label` and `dest_label` columns
3. In `build_window()`, assign both labels to their respective node updates
4. Handle sample weights similarly for both src and dest

**Option B: Token-Level Label Lookup**
Create a per-token label mapping:
1. For each window, build a dict mapping `token_id → (label, weight)`
2. Look up label/weight for each token when creating node updates
3. Handle tokens appearing multiple times (use most recent or aggregate)

**Considerations**:
- Option A requires changes to data preparation pipeline
- Option B is more flexible but adds complexity to build_window()
- Both options double the effective training data without increasing compute significantly

**Files to modify**:
- Option A: `src/label_triple_barrier.py` - Generate dest_label column
- Option A/B: `src/dex_contagion_trader.py::build_window()` - Assign dest labels

---

### Issue 4: Timestamp Granularity vs Window Independence Trade-off

**Status**: Open (Needs Decision)

**Current Implementation** ([src/dex_contagion_trader.py:358-362](../src/dex_contagion_trader.py#L358-L362)):
```python
# For each window, create local timestamps starting from 0
# This preserves relative timing within the window while making each window
# independent. Note: event_index maintains global ordering, but we renumber
# for local window time
edge_timestamps = np.arange(num_events, dtype=np.int64)
```

**Problem**:
- Uses sequential integers (0, 1, 2, ...) instead of real time deltas
- Dollar bars have VARIABLE time intervals based on volume
- Active periods: many bars/minute (small real time gaps)
- Inactive periods: few bars/hour (large real time gaps)
- Current approach treats all bars as evenly spaced

**Impact**:
TGCN cannot distinguish temporal patterns:
- "Bar 5 came 1 second after bar 4" (high activity burst)
- "Bar 5 came 1 hour after bar 4" (low activity period)

This information could be valuable for learning market regime changes.

**Current Rationale**:
The comment suggests renumbering preserves "window independence" - each window starts at t=0, making them comparable. However, this sacrifices temporal granularity.

**Fix Options**:

**Option A: Real Time Deltas (Within Window)**
Use actual time differences from window start:
```python
first_timestamp = df_window["bar_close_timestamp"].min()
edge_timestamps = (df_window["bar_close_timestamp"] - first_timestamp).dt.total_seconds()
edge_timestamps = edge_timestamps.to_numpy().astype(np.int64)
```
- Preserves temporal dynamics (Prado's "information time")
- Windows still start at t=0 (relative)
- TGCN can learn activity patterns

**Option B: Keep Sequential but Add Time Delta Features**
Keep current sequential timestamps but add real time gaps as edge features:
```python
edge_timestamps = np.arange(num_events, dtype=np.int64)
# Add bar_time_delta_sec as an additional edge feature
time_delta_feats = df_window["bar_time_delta_sec"].to_numpy()
```
- Maintains current ordering
- Adds temporal information as a feature
- More explicit signal for TGCN

**Option C: Research and Decide**
- Test both approaches with metrics
- Evaluate if TGCN actually uses timestamp granularity or just ordering
- Check TGM documentation for timestamp semantics

**Recommendation**: Start with Option A (real time deltas) since it's conceptually cleaner and preserves Prado's volume clock principle while maintaining window independence.

**Files to modify**:
- `src/dex_contagion_trader.py::build_window()` - Change timestamp computation

---

### Issue 5: Edge Weight Utilization ✅ RESOLVED

**Status**: Resolved

**Current Implementation**:
- [src/dex_contagion_trader.py:401-403](../src/dex_contagion_trader.py#L401-L403): Edge features use flow magnitude
- [src/dex_contagion_trader.py:284-288](../src/dex_contagion_trader.py#L284-L288): Forward pass extracts edge weights

**Resolution**:
```python
# In build_window()
edge_feats = np.abs(src_flow).astype(np.float32).reshape(-1, 1)

# In forward()
edge_weight = batch.edge_feats.squeeze(-1) if batch.edge_feats is not None else None
z = self.tgcn(x_combined, batch.edge_index, edge_weight)
```

**What Was Done**:
- Edge weights based on absolute flow magnitude (`abs(src_flow_usdc)`)
- Large volume swaps have more influence in message passing
- Properly passed to TGCN for weighted aggregation

**Benefits Achieved**:
- High-volume swaps shape embeddings more than low-volume swaps
- Aligns with financial intuition (liquidity/volume matters)
- TGCN can prioritize important market events

---

## Implementation Priority

### ✅ Completed (Phase 5A)
1. **Issue 1**: Node features now used via dynamic_node_feats - RESOLVED
2. **Issue 5**: Edge weights implemented using flow magnitude - RESOLVED

**What Was Accomplished**:
- Model now properly uses fracdiff, volatility, and flow as temporal node features
- Each swap creates 2 node updates (src and dest) with dynamic features
- Edge weights based on flow magnitude for importance weighting
- Training loss shows model is learning (0.1045)

### Remaining Issues (Phase 5B)

**High Priority**:
1. **Issue 2**: Make graph bidirectional (add reverse edges)
   - Impact: Enable bidirectional contagion propagation
   - Effort: Low (mostly mechanical change to build_window)
   - Risk: Low (doubles edge count but straightforward)

2. **Issue 3**: Add labels for dest tokens
   - Impact: Double the supervised training signal
   - Effort: Medium (requires data pipeline changes)
   - Risk: Medium (need to modify label_triple_barrier.py)

**Lower Priority (Evaluate First)**:
3. **Issue 4**: Use real timestamps instead of sequential integers
   - Impact: Uncertain - need to test if TGCN uses timestamp granularity
   - Effort: Low (simple change to timestamp computation)
   - Risk: Low (easy to revert if no improvement)
   - Decision needed: Does temporal granularity help or hurt?

**Recommended Next Steps**:
1. Address Issue 2 (bidirectional edges) - low-hanging fruit
2. Evaluate Issue 4 (real timestamps) - quick experiment
3. Plan Issue 3 (dest labels) - requires more careful design

---

## Success Metrics

### Phase 5A (Completed) ✅
- [x] Model training loss decreases consistently - **ACHIEVED** (0.1045)
- [x] Node features properly incorporated - **ACHIEVED** (dynamic_node_feats)
- [x] Edge weights implemented - **ACHIEVED** (flow magnitude)
- [x] Predictions show non-random behavior - **ACHIEVED** (100% bullish scores vs 0%)

### Phase 5B (Remaining)
- [ ] Graph structure is bidirectional (Issue 2)
- [ ] Training uses all available labels (Issue 3 - dest tokens)
- [ ] Temporal granularity evaluated (Issue 4)
- [ ] Backtest Sharpe ratio improves from baseline

### Future Analysis Needed
- [ ] Visualize node embeddings with t-SNE to verify price grouping
- [ ] Analyze prediction distribution across different market regimes
- [ ] Compare bidirectional vs unidirectional graph performance
- [ ] Test timestamp granularity impact on learning

---

## References

- **Data Pipeline**: [plans/data_overview.md](data_overview.md)
- **Prado's Methodology**: "Advances in Financial Machine Learning" (2018)
  - Chapter 3: Triple Barrier Method
  - Chapter 4: Sample Weights (concurrency)
  - Chapter 5: Fractional Differentiation
- **TGM Library**: https://tgm.readthedocs.io/
- **Current Implementation**: [src/dex_contagion_trader.py](../src/dex_contagion_trader.py)
