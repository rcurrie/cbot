# Coding Guidelines

The code must adhere to the following strict standards:

- Follow strict Ruff linting and style including comments
- Follow strict Mypy typing for all functions, variable and definitions
- Always prefer `pathlib` over `os.path` for file system operations.
- Always use a per file logger instead of print without f-strings.
- Use imperative style for all docstrings.
- Use logging.exception instead of logging.error in exception handlers
- Always get api keys, passwords etc... from .env via dotenv
- Any data artifacts should be saved to ./data by default
