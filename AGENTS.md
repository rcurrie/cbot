# Project Context

Building a DeFi trading system from scratch, exploring embedding space concepts from computational biology applied to on-chain trading. The system includes data ingestion, backtesting, paper trading, and eventually live trading.

# Coding Philosophy

- **Simplicity and reliability first** - this is a trading system where bugs are costly
- **Unix philosophy** - focused, composable command-line tools
- **Be opinionated** - push back on questionable quant approaches
- **Minimal changes** - touch only what's required, no refactoring unless necessary

# Python Standards

**Type Safety & Linting:**

- Strict Mypy typing for all functions, variables, and definitions
- Ruff with ALL rules enabled (except S101 for assertions, PLR0913 for CLI entry points)
- Use Pydantic for data validation

**Code Style:**

- Use `pathlib` over `os.path`
- Per-file logger instead of print (no f-strings in logging)
- Imperative style docstrings
- `logging.exception` in exception handlers, not `logging.error`
- Assertions are encouraged for cleaner data wrangling code

**Dependencies:**

- Run Python via `uv run python src/script.py`
- Click for CLI tools
- Polars for data processing (not pandas)
- Load secrets from `.env` via dotenv

**Project Structure:**

- Save data artifacts to `./data` by default
- Jupyter notebooks run from project root (use `data/`, not `../data`)
- Virtual environment is in `.venv` with tools pre-installed
- One-off validation/debug scripts go in `plans/<milestone_name>/` (not part of main system)

# Development Workflow

1. **Before coding:** Read context, identify insertion points, assess what could break
2. **Run linting:** `ruff check --fix .`
3. **Run type checking:** `mypy .`
