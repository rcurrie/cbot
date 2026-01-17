# Project Context

Machine Learning driven Crypto Trading System using Decentralized Exchange (DEX) token swaps, Temporal Graph Neural Networks (TGNN) and best practices from Marcos López de Prado publications and the book _Advances in Financial Machine Learning_.

## System Overview

This project implements a complete ML trading pipeline:

1. **Data Ingestion** - Extract Uniswap v3 swap events from Google BigQuery
2. **Feature Engineering** - Calculate USDC prices, generate dollar bars, apply fractional differentiation
3. **Labeling** - Triple-barrier method for supervised learning
4. **Training** - TGNN model for price prediction on temporal graphs
5. **Backtesting** - Validate strategy performance

## Key References

- **Advances in Financial Machine Learning** by Marcos López de Prado
  - Fractional Differentiation (Chapter 5): Make time series stationary while preserving memory
  - Triple-Barrier Method (Chapter 3): Label observations with directional bets using profit-taking, stop-loss, and time barriers
  - Meta-labeling (Chapter 3): Secondary model for position sizing
  - Sample Weights (Chapter 4): Account for label concurrency/overlap
  - Cross-validation (Chapter 7): Purging and embargo for financial time series

- **Temporal Graph Neural Networks**
  - Model token price contagion via swap events
  - Continuous-time dynamic graph structure
  - Each swap creates directed edges between token nodes

# CRITICAL: Python Execution

**NEVER use `python` directly. ALWAYS use `uv run python` for ALL Python execution.**

- Use `uv run python script.py` instead of `python script.py`
- Use `uv run pytest` instead of `pytest`
- Use `uv run ruff` instead of `ruff`
- For interactive Python: `uv run python`
- Only use uv, NEVER pip, to manage packages

# Coding Philosophy

- **Simplicity and reliability first** - this is a trading system where bugs are costly
- **Unix philosophy** - focused, composable command-line tools to form a pipeline
- **Be opinionated** - push back on questionable quant approaches
- **Minimal changes** - touch only what's required, no refactoring unless necessary
- **Ignore backwards compatibility** - this is new development, focus on the RIGHT way, we can rerun the pipeline anytime
- **Console output** - Informative, detailed, stylish, colorful, etc. by fully leveraging the rich library wherever possible

# Project Structure

- Plan documents with milestones in `plans/`
- One-off validation scripts go in `plans/<milestone_name>/` (not part of main system)
- Data artifacts used between scripts such as parquet files go in `data/`
- Scripts and notebooks run from the project root so use root-relative paths

# Type Safety & Linting

- Ruff for linting and strict except for exclusions configured in pyproject.toml
- To check for linter errors/warnings and fix: `uv run ruff check --fix --unsafe-fixes`
- Strict Mypy typing for all functions, variables, and definitions
- Use Pydantic for data validation

# Code Style

- Use `pathlib` over `os.path`
- Per-file logger instead of print (no f-strings in logging)
- Use logging format=`"%(asctime)s %(levelname)s %(message)s"` and datefmt=`"%Y-%m-%d %H:%M:%S"`
- Imperative style docstrings
- `logging.exception` in exception handlers, not `logging.error`
- Assertions are encouraged for cleaner data wrangling code
- Use uv to run scripts with python
- Typer for CLI tools
- Polars for data processing (not pandas)
- Load secrets from `.env` via dotenv
- Line lengths should be limited to 88

## Progress Bars for Long Operations

For operations that process data in batches or take significant time, use `rich.progress` to provide visual feedback:

```python
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

# Example: batch processing with progress bar
num_batches = (len(data) + batch_size - 1) // batch_size

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("Processing data...", total=num_batches)

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        progress.update(task, advance=1)
        # ... process batch ...
```

Benefits:
- Shows real-time progress with percentage and ETA
- Spinner indicates active processing
- Clean, professional appearance
- Minimal performance overhead
- See `calculate_usdc_prices.py` for a complete example

## Status Icons for Validation and Checks

Use consistent emoji icons to indicate status of validation checks and operations:

- **✅** - Success/Pass (green check mark)
- **⚠️** - Warning (yellow/orange warning triangle)
- **❌** - Error/Failure (red X)
- **ℹ️** or **i** - Informational (blue info icon)

### Pattern for Validation Functions

```python
def validate_data(df: pl.DataFrame) -> None:
    """Validate dataset with clear pass/warning/fail indicators."""
    logger.info("=" * 70)
    logger.info("DATA VALIDATION")
    logger.info("=" * 70)

    # Track issues for summary
    has_issues = False

    # Check 1: Basic validation
    logger.info("\n1. Checking for null values...")
    if df.null_count().sum_horizontal()[0] > 0:
        logger.warning("  ⚠️  WARN: Found null values in dataset")
        has_issues = True
    else:
        logger.info("  ✅ OK: No null values found")

    # Check 2: Critical validation
    logger.info("\n2. Checking data integrity...")
    if critical_check_fails():
        logger.error("  ❌ ERROR: Critical integrity check failed")
        has_issues = True
    else:
        logger.info("  ✅ OK: Data integrity verified")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    if has_issues:
        logger.warning("⚠️  Validation completed with warnings")
    else:
        logger.info("✅ All validation checks passed!")
    logger.info("=" * 70)
```

### Guidelines

1. **Be Consistent**: Always use the same icon for the same status type
2. **Add Context**: Include descriptive text after the icon (e.g., "OK:", "WARN:", "ERROR:")
3. **Summarize**: Provide an overall status summary at the end of validation sections
4. **Track State**: Use boolean flags to track warnings/errors for final summary
5. **Visual Hierarchy**: Use spacing and separators to make output scannable

Example: See `validate_output()` function in `calculate_usdc_prices.py` for a complete implementation

# Documentation Standards

## Script Docstrings

Each script in the main pipeline should have a module-level docstring that explains:

1. **What**: High-level purpose - what problem does this solve in the trading pipeline?
2. **Why**: Rationale - why this approach? Reference Prado or academic sources where relevant
3. **How**: Brief overview of methodology
4. **Input/Output**: What files it reads/writes

### Example Format

```python
"""Calculate USDC-denominated prices from swap events.

WHY: Direct USDC pairs provide ground truth prices. For non-USDC pairs (e.g., ETH/WBTC),
we must infer prices using a price cache to maintain temporal consistency. This approach
follows the "information-driven bars" concept from Prado (Ch. 2) - we sample based on
information flow (swaps) rather than time.

WHAT: Build a chronological price series for all tokens by:
1. Direct calculation for USDC-paired swaps
2. Inferred pricing for indirect swaps using cached prices
3. Outlier filtering using IQR method to remove data errors

HOW: Process swaps chronologically, maintaining a price cache that updates with each
direct USDC observation. For indirect swaps, cross-reference both tokens against cache
to infer prices.

INPUT: data/usdc_paired_swaps.parquet (filtered swap events)
OUTPUT: data/usdc_priced_swaps.parquet (price time series)

References:
- Prado AFML Ch. 2: Information-Driven Bars
"""
```

## Function Docstrings

Use imperative style with clear Args/Returns:

```python
def find_min_d_for_stationarity(
    log_price: np.ndarray,
    d_min: float = 0.0,
    d_max: float = 1.0,
) -> tuple[float, bool]:
    """Find minimum fractional differentiation order for stationarity.

    Uses ADF test to determine the minimum d that achieves stationarity
    while preserving maximum memory (Prado Ch. 5).

    Args:
        log_price: Log-transformed price series.
        d_min: Minimum d to search.
        d_max: Maximum d to search.

    Returns:
        Tuple of (minimum_d, is_stationary).
    """
```

# Tool Guidelines

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, so results ignore comments/strings, understand syntax, and can **safely rewrite** code.

**Use `ripgrep` when text is enough.** It's the fastest way to grep literals/regex across files.

### Snippets

Find structured code (ignores comments/strings):

```bash
uv run ast-grep -l TypeScript -p 'import $X from "$P"'
```

Codemod (only real `var` declarations become `let`):

```bash
uv run ast-grep -l JavaScript -p 'var $A = $B' -r 'let $A = $B' -U
```

Quick textual hunt:

```bash
uv run rg -n 'console\.log\(' -t js
```

Combine speed + precision:

```bash
uv run rg -l -t ts 'useQuery\(' | xargs ast-grep -l TypeScript -p 'useQuery($A)' -r 'useSuspenseQuery($A)' -U
```

### Mental model

- Unit of match: `ast-grep` = node; `rg` = line
- False positives: `ast-grep` low; `rg` depends on your regex
- Rewrites: `ast-grep` first-class; `rg` requires ad-hoc sed/awk and risks collateral edits

### Rule of thumb

- Need correctness over speed, or you'll **apply changes** → start with `ast-grep`
- Need raw speed or you're just **hunting text** → start with `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify with precision

# Data Pipeline Overview

The pipeline consists of these stages (see Makefile):

```
ingest-swaps → filter-and-decode-swaps → calculate-usdc-prices →
generate-usdc-bars → make-stationary → label-triple-barrier →
training-data-validation
```

## Stage Descriptions

1. **ingest-swaps** - Download raw Uniswap V3 swap events from BigQuery
2. **filter-and-decode-swaps** - Filter to USDC-connected pools, decode event data
3. **calculate-usdc-prices** - Calculate token prices in USDC, build price cache
4. **generate-usdc-bars** - Create dollar volume bars (Prado Ch. 2)
5. **make-stationary** - Apply fractional differentiation (Prado Ch. 5)
6. **label-triple-barrier** - Generate labels using triple-barrier method (Prado Ch. 3)
7. **training-data-validation** - Validate final training dataset quality

# Important Notes

## File Operations

NEVER DELETE A FILE WITHOUT EXPRESS PERMISSION FROM ME OR A DIRECT COMMAND FROM ME.

## Data Sources

- [Google BigQuery Crypto Datasets](https://cloud.google.com/blockchain-analytics/docs/supported-datasets)
- [Uniswap v4 Deployment Addresses](https://docs.uniswap.org/contracts/v4/deployments)
- [CoinGecko API](https://www.geckoterminal.com/dex-api)
- [Demeter Fetch for Block Index](https://github.com/zelos-alpha/demeter-fetch)

## Prior Art

- [DeFi Similarity Research](https://github.com/JunLLuo/DeFi-similarity)
- [Crypto Forecasting with GNNs](https://www.tandfonline.com/doi/full/10.1080/13504851.2022.2141436)
- [TGLite Framework](https://charithmendis.com/assets/pdf/asplos24-tglite.pdf)
