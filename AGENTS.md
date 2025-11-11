# Project Context

Machine Learning driven Crypto Trading System using Decentralized Exchange (DEX) token swaps, Temporal Graph Neural Networks (TGNN) and best practices from Marcos López de Prado publications and the book Advances in Financial Machine Learning.

# Important

RULE NUMBER 1 (NEVER EVER EVER FORGET THIS RULE!!!): YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION FROM ME OR A DIRECT COMMAND FROM ME. EVEN A NEW FILE THAT YOU YOURSELF CREATED, SUCH AS A TEST CODE FILE. YOU HAVE A HORRIBLE TRACK RECORD OF DELETING CRITICALLY IMPORTANT FILES OR OTHERWISE THROWING AWAY TONS OF EXPENSIVE WORK THAT I THEN NEED TO PAY TO REPRODUCE. AS A RESULT, YOU HAVE PERMANENTLY LOST ANY AND ALL RIGHTS TO DETERMINE THAT A FILE OR FOLDER SHOULD BE DELETED. YOU MUST **ALWAYS** ASK AND _RECEIVE_ CLEAR, WRITTEN PERMISSION FROM ME BEFORE EVER EVEN THINKING OF DELETING A FILE OR FOLDER OF ANY KIND!!!

# Coding Philosophy

- **Simplicity and reliability first** - this is a trading system where bugs are costly
- **Unix philosophy** - focused, composable command-line tools to form a pipeline
- **Be opinionated** - push back on questionable quant approaches
- **Minimal changes** - touch only what's required, no refactoring unless necessary
- **Ignore Backwards Compatibility** - this is new development, focus on the RIGHT way, we can rerun the pipeline anytime.
- **Console Output** - Informative, detailed, stylish, colorful, etc. by fully leveraging the rich library wherever possible.

# Project Structure

- Plan documents with milestones in plans/
- One off validation scripts go in plans/<milestone_name>/ (not part of main system)
- Data artifacts used between scripts such as parquet files go in data/
- Scripts and notebooks run from the project root so use root relative paths

# Python Standards

- Use uv to run all command line tools
- Only use uv, NEVER pip.
- Virtual env managed by uv in .venv which you can assume is activated
- Python 3.14 and higher (NOT pip/poetry/conda!)

# Type Safety & Linting

- Ruff for linting and strict except for exclusions configured in pyproject.toml
- To check for linter errors/warnings and fix: `ruff check --fix --unsafe-fixes`
- Strict Mypy typing for all functions, variables, and definitions
- Use Pydantic for data validation

# Code Style

- Use `pathlib` over `os.path`
- Per-file logger instead of print (no f-strings in logging)
- Imperative style docstrings
- `logging.exception` in exception handlers, not `logging.error`
- Assertions are encouraged for cleaner data wrangling code
- Use uv to run scripts with python
- Typer for CLI tools
- Polars for data processing (not pandas)
- Load secrets from `.env` via dotenv

# Development Workflow

1. **Before coding:** Read context, identify insertion points, assess what could break
2. **Run linting:** `ruff check --fix .`
3. **Run type checking:**o `mypy .`

### ast-grep vs ripgrep (quick guidance)

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, so results ignore comments/strings, understand syntax, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms, rewrite call sites or variable kinds.
- Policy checks: enforce patterns across a repo (`scan` with rules + `test`).
- Editor/automation: LSP mode; `--json` output for tooling.

**Use `ripgrep` when text is enough.** It’s the fastest way to grep literals/regex across files.

- Recon: find strings, TODOs, log lines, config values, or non‑code assets.
- Pre-filter: narrow candidate files before a precise pass.

**Snippets**

Find structured code (ignores comments/strings):

```bash
ast-grep run -l TypeScript -p 'import $X from "$P"'
```

Codemod (only real `var` declarations become `let`):

```bash
ast-grep run -l JavaScript -p 'var $A = $B' -r 'let $A = $B' -U
```

Quick textual hunt:

```bash
rg -n 'console\.log\(' -t js
```

Combine speed + precision:

```bash
rg -l -t ts 'useQuery\(' | xargs ast-grep run -l TypeScript -p 'useQuery($A)' -r 'useSuspenseQuery($A)' -U
```

**Mental model**

- Unit of match: `ast-grep` = node; `rg` = line.
- False positives: `ast-grep` low; `rg` depends on your regex.
- Rewrites: `ast-grep` first-class; `rg` requires ad‑hoc sed/awk and risks collateral edits.

**Rule of thumb**

- Need correctness over speed, or you’ll **apply changes** → start with `ast-grep`.
- Need raw speed or you’re just **hunting text** → start with `rg`.
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify with precision.
