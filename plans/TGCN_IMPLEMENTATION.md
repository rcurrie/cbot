# TGCN Token Prediction Model - Implementation Guide

## Overview

Successfully implemented a **Temporal Graph Convolutional Network (TGCN)** model for predicting token price movements on the DEX swap network. The model leverages the TGM library to process temporal graph data and make node-level predictions.

## Architecture

### Data Structure (DGData)

The key design decision: **Separate node labels from node features**.

- **Nodes**: Tokens (all unique tokens in the swap network)
- **Edges**: Individual swap events with temporal information
- **Node Events**: One per swap, containing the label for the source token
- **Edge Features**: Financial metrics driving the transaction
  - `src_flow_usdc`: Buy/sell pressure for source token
  - `dest_flow_usdc`: Buy/sell pressure for destination token
  - `tick_count`: Number of ticks in the bar
  - `rolling_volatility`: Volatility used in triple-barrier calculation
  - `bar_time_delta_sec`: Time to fill the volume bar

### Node Labels (in `dynamic_node_feats`)

The **labels are stored as node events** in the `dynamic_node_feats` tensor:

```
dynamic_node_feats: [num_node_events, 1]
```

Labels are triple-barrier outputs:

- **-1 (class 0)**: Price moved down (loss)
- **0 (class 1)**: Timeout (stayed neutral)
- **1 (class 2)**: Price moved up (profit)

Shifted by +1 for CrossEntropyLoss compatibility: {-1, 0, 1} → {0, 1, 2}

**Why in node features, not edge features?**

- Targets must be associated with nodes for prediction
- TGM's `dynamic_node_feats` is the standard location for node-level targets
- During materialization, `batch.dynamic_node_feats` contains these labels for supervised learning

### Model Architecture

```
TokenPredictorModel
├── TGCN (Encoder)
│   ├── in_channels: 64 (static feature dimension)
│   ├── hidden_channels: 64
│   ├── out_channels: 64
│   └── num_nodes: # of unique tokens
└── NodePredictor (Decoder)
    ├── in_dim: 64
    ├── hidden_dim: 64
    └── out_dim: 3 (classes: down, stay, up)
```

## Key Implementation Details

### 1. DGData Construction

```python
data = DGData.from_raw(
    edge_timestamps=torch.from_numpy(timestamps_sec),
    edge_index=torch.from_numpy(np.column_stack((src, dst))),
    edge_feats=torch.from_numpy(edge_feats),
    node_timestamps=torch.from_numpy(node_timestamps),
    node_ids=torch.from_numpy(node_ids),
    dynamic_node_feats=torch.from_numpy(dynamic_node_feats),  # Labels stored here
    time_delta="s",
)
```

**Critical fields:**

- `edge_timestamps`: Swap execution times (in seconds since start)
- `edge_index`: Source → Destination token pairs
- `edge_feats`: Financial metrics of the swap
- `node_ids`: Source tokens (entities making predictions for)
- `dynamic_node_feats`: Labels (targets for prediction)

### 2. Static Node Features

Created as learned embeddings:

```python
static_node_feats = torch.randn((num_tokens, EMBED_DIM), device=DEVICE).float()
```

Could be enhanced with:

- Token metadata (decimals, liquidity)
- Historical volatility
- Network-based features (centrality, community)

### 3. Temporal Batching

The `DGDataLoader` slices the temporal graph into hourly windows:

```python
loader = DGDataLoader(dg, batch_unit="H")  # Hourly batches
```

Each batch contains edges and node events within that hour window.

### 4. Training Loop

- **Loss**: CrossEntropyLoss (multi-class classification)
- **Optimizer**: Adam (lr=1e-3)
- **Epochs**: 20
- **Batch Processing**:
  - Extract temporal window
  - Construct edge_index on GPU
  - Forward pass through TGCN
  - Predict only on nodes with labels (`batch.node_ids`)
  - Compute loss on `batch.dynamic_node_feats` (the labels)
  - Backprop and step

### 5. Evaluation & Recommendations

After training:

1. Run model in eval mode over all temporal windows
2. Aggregate predictions per token across all occurrences
3. Calculate **bullish score** = (# "up" predictions) / (total predictions)
4. Rank tokens by bullish score
5. Return top-20 tokens likely to increase

```python
bullish_score = up_count / len(pred_list)
```

## How Labels Are Used for Loss

During each batch:

```python
logits = model(batch, static_node_feats)  # [batch_size, 3]
y_true = batch.dynamic_node_feats.long().squeeze(-1)  # [batch_size]
loss = nn.CrossEntropyLoss()(logits, y_true)
```

The `batch.dynamic_node_feats` contains the materialized labels for nodes in that time window.

## Configuration

| Parameter         | Value | Purpose                      |
| ----------------- | ----- | ---------------------------- |
| `EMBED_DIM`       | 64    | Dimensionality of embeddings |
| `HIDDEN_DIM`      | 64    | TGCN hidden dimension        |
| `NUM_CLASSES`     | 3     | {down, stay, up}             |
| `BATCH_TIME_GRAN` | "H"   | Hourly temporal batches      |
| `LEARNING_RATE`   | 1e-3  | Adam optimizer rate          |
| `EPOCHS`          | 20    | Training iterations          |

## Recommendations for Future Improvements

### Model Enhancements

1. **Deeper Architecture**: Stack multiple TGCN layers for longer-range temporal dependencies
2. **Attention Mechanisms**: Add self-attention over temporal neighbors
3. **Multi-task Learning**: Predict both direction and magnitude
4. **Temporal Encoding**: Add sinusoidal position embeddings for time

### Feature Engineering

1. **Static Node Features**: Token metadata (decimals, supply, age)
2. **Liquidity Metrics**: 24h volume, bid-ask spread
3. **Network Features**: PageRank, centrality measures
4. **Cross-Asset Correlation**: Historical correlation with major tokens

### Data Pipeline

1. **Class Weighting**: Handle class imbalance in labels
2. **Temporal Splits**: Val/test on future time windows only (no lookahead)
3. **Sliding Window Validation**: Train on weeks N-8 to N, validate on N+1
4. **Ensemble Methods**: Train multiple seeds, average predictions

### Deployment

1. **Inference Pipeline**: Stream new swaps, update predictions in real-time
2. **Risk Management**: Combine with volatility forecasts for position sizing
3. **Backtesting Framework**: Evaluate recommendation profitability
4. **Active Learning**: Retrain on hardest examples

## Running the Model

```bash
# Requires labeled data at data/labeled_log_fracdiff_price.parquet
uv run python src/dex_contagion_trader.py
```

Outputs:

- Model checkpoint: `data/checkpoints/tgcn_model.pt`
- Recommendations: Logged to console with bullish scores

## Technical Notes

### Why Node Labels Belong in `dynamic_node_feats`

The TGM library design:

- `edge_feats`: Used for temporal memory updates in the encoder
- `dynamic_node_feats`: Node-level targets, materialized per batch for supervision
- `static_node_feats`: Invariant node properties

For node prediction tasks, labels are **node properties** that vary over time (they change after each bar), so they belong in `dynamic_node_feats`.

### CrossEntropyLoss Requirement

PyTorch's `CrossEntropyLoss` expects:

- Input: `[batch_size, num_classes]` logits
- Target: `[batch_size]` class indices (0-indexed, [0, num_classes))

Our label transformation: {-1, 0, 1} + 1 = {0, 1, 2} ✅

### Temporal Materialization

When `DGDataLoader` yields a batch:

- `batch.src, batch.dst, batch.time`: Edges in this time window
- `batch.node_ids`: Nodes with events (labels) in this window
- `batch.dynamic_node_feats`: The labels for those nodes

This ensures we **only compute loss on nodes that have labels in the current window**.
