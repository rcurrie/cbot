# cbot

### _Hunting for alpha with temporal GNNs in the DEX jungle_

Experimental ML-driven crypto trading system that models price contagion across DEX swap graphs using Temporal Graph Neural Networks. The pipeline is built on Marcos López de Prado's quantitative framework (_Advances in Financial Machine Learning_) — fractional differentiation, triple-barrier labeling, concurrency-weighted samples — applied to on-chain DEX data rather than traditional order book markets.

Thesis: Token prices propagate through the swap graph and we should be able to pickup alpha by capturing this contagion structure better than models that treat each token as an independent time series.

## Architecture

Each stage is a standalone CLI tool. Data flows through parquet files. The whole pipeline reruns in minutes.

```
BigQuery (Ethereum logs)
    │
    ▼
Ingest & Decode ──── Raw swap events, decoded from ABI
    │
    ▼
USDC Pricing ─────── Temporal price cache with 1hr TTL, indirect inference
    │
    ▼
Dollar Bars ──────── Adaptive thresholds (0.1% daily volume per pool)
    │
    ▼
Frac-Diff ────────── Min-d binary search, ADF p<0.05, preserves memory
    │
    ▼
Triple-Barrier ───── Symmetric ±2σ, 50% daily vertical, Prado weights
    │
    ▼
Walk-Forward ─────── Train N days → predict morning → trade 9am-5pm → step
```

## Stack

Python 3.14 · PyTorch Geometric · XGBoost · Polars · Typer · Rich · BigQuery · Modal (GPU training)

Very strict typing and style defined in AGENTS.md

## Data Pipeline

**Source:** Uniswap v3 swap events from Google BigQuery's public Ethereum dataset. ~500K labeled events across ~50 tokens and ~136 pools over a 3-month window.

**Dollar Bars** (Prado Ch. 2): Information-driven sampling where each bar represents a fixed dollar volume of trading activity. Adaptive thresholds scale per pool — WETH/USDC bars fire every few minutes at $500K; a small-cap token bars every few hours at $10K. This normalizes the information content per observation across the liquidity spectrum.

**Fractional Differentiation** (Prado Ch. 5): Log prices are non-stationary but first-differencing destroys memory. Fractional differentiation at minimum _d_ (binary search against ADF test) threads the needle — stationary series that retain predictive memory. Each token gets its own _d_; stablecoins land near 0.05, volatile tokens near 0.4.

**Triple-Barrier Labeling** (Prado Ch. 3): Each observation gets a ternary label {down, neutral, up} based on which barrier is touched first — symmetric profit-taking/stop-loss at ±2σ volatility, with a vertical barrier at 50% of daily bars (~4-6 hours). This captures real trading outcomes, not arbitrary return buckets.

**Concurrency Weights** (Prado Ch. 4): Labels from overlapping barrier windows share information. Sample weights = 1/concurrent_label_count, downweighting redundant observations. The effective training set is ~17K samples from 500K raw events — aggressive but honest.

## Model Explorations

### TGCN Trader (`dex_contagion_trader.py`)

Temporal Graph Convolutional Network with ternary classification and Kelly criterion position sizing. Trains daily on a 5-day sliding window, retraining from scratch each day to prevent look-ahead bias. Position size = (P(up) - P(down)) / 2, only entering when the model detects positive edge.

### Heterogeneous GATv2 (`ldr_tgn_trader.py`)

A more ambitious architecture: bipartite token↔pool graph with GATv2 dynamic attention, temporal memory via GRU, and initially Supervised Contrastive Loss for learning class-clustered embeddings.

**What we found:** The SupCon approach failed — with only ~50 tokens per batch aggregated from 256 events, there were literally zero positive pairs per anchor in every batch. The graph is too small (45 tokens, 136 pools, 138 unique edges) for contrastive learning to find structure. Switching to sample-weighted cross-entropy with LayerNorm throughout got the loss actually decreasing.

The deeper question is whether the graph structure adds signal at all. With so few nodes, all discriminative information may already live in the per-event tabular features.

### XGBoost Baseline (`baseline_xgboost.py`)

Same walk-forward protocol, same features, same Prado sample weights, trains in 5 seconds instead of 7 minutes per day. Exists to answer the question: does the graph add value, or is this a tabular problem wearing a graph costume?

## Running

See the Makefile for most of the ceremony...

```bash
# Full pipeline rebuild
make update-train-data

# Backtest the TGCN trader
uv run python src/dex_contagion_trader.py --epochs 50 --trading-days 10

# Backtest the GATv2 trader
uv run python src/ldr_tgn_trader.py --epochs 10 --trading-days 5

# XGBoost baseline comparison
uv run python src/baseline_xgboost.py --trading-days 30

# GPU training on Modal
uv run modal run src/modal_train.py --epochs 50 --trading-days 5
```

## Status

Active research. The data pipeline is solid and principled. The model architecture is the open question — whether graph structure captures exploitable contagion effects, or whether the alpha (if it exists) is accessible to simpler models operating on the same carefully prepared features.
