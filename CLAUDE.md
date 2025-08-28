# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for training on-chain crypto in order accumulate base tokens

## Development Commands

### Linting and Type Checking

```bash
# Run linting with automatic fixes
ruff check --fix .

# Run type checking
mypy .
```

All Python code must comply with the Ruff's linting rules.

### Running the Application

The project uses a Makefile with predefined targets:

```bash
# Update pools
make update-pools
```

Individual scripts can be run directly:

```bash
# Fetch pairs data
python pools.py

# Analyze cycles with custom parameters
python3 cycles.py -r 1000000

# Calculate spreads
python3 spreads.py -m 7 -i <input_file> -t
```

## Architecture Notes

### Key Components

- **pools.py**: Fetches pool/pair data from TheGraph subgraphs for various DEXs (Uniswap V3, PancakeSwap, QuickSwap, SushiSwap)

### Dependencies

- Python 3.11+
- Key packages: pandas, pyarrow, requests, python-dotenv
- Development: mypy, ruff

### Data Flow

1. Fetch pair/pool data from TheGraph API endpoints
2. Store data in JSON format (data/ directory)

### Environment Variables

The project uses `.env` file for configuration (loaded via python-dotenv).
