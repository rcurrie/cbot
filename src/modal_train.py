"""Modal deployment for TGCN training.

Run training on Modal cloud with GPU:
    modal run src/modal_train.py --epochs 10 --trading-days 5

This script uploads data to Modal and runs training on cloud GPU.
Output streams directly to your terminal.
"""

import logging
from pathlib import Path

import modal
import typer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("dex-contagion-trader")

# Define Modal image with pinned dependencies
image = (
    modal.Image.debian_slim(python_version="3.14.2")
    .pip_install(
        "polars==1.35.2",
        "numpy==2.3.4",
        "scikit-learn==1.7.2",
        "torch==2.9.1",
        "torch-geometric==2.7.0",
        "tgm-lib==0.1.0b0",
        "typer==0.20.0",
    )
    .add_local_file("src/dex_contagion_trader.py", "/root/dex_contagion_trader.py")
)

# Create persistent volume for data
volume = modal.Volume.from_name("data", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600 * 4,  # 4 hour timeout
    gpu="T4",  # T4 GPU (cheapest option)
)
def train_remote(epochs: int, trading_days: int | None) -> None:
    """Run training on Modal GPU.

    Args:
        epochs: Number of training epochs per day.
        trading_days: Number of days to trade (None = all available).

    """
    import sys  # noqa: PLC0415
    from pathlib import Path as PathLib  # noqa: PLC0415

    # Add script to Python path
    sys.path.insert(0, "/root")

    # Import training functions
    from dex_contagion_trader import (  # noqa: PLC0415
        backtest_slide,
        load_and_filter_bars,
        load_token_metadata,
        prepare_data,
        validate_input_data,
    )

    # Load data from volume
    data_path = PathLib("/data/labeled_log_fracdiff_price.parquet")
    tokens_path = PathLib("/data/tokens.json")

    # Run full pipeline
    validate_input_data(data_path)

    tokens_metadata = load_token_metadata(tokens_path)
    df = load_and_filter_bars(data_path)
    df_prepared, le, num_tokens, unique_dates = prepare_data(df)

    backtest_slide(
        df_prepared,
        le,
        num_tokens,
        unique_dates,
        tokens_metadata,
        epochs,
        trading_days,
    )


@app.local_entrypoint()
def main(
    data_path: str = "data/labeled_log_fracdiff_price.parquet",
    tokens_path: str = "data/tokens.json",
    epochs: int = 10,
    trading_days: int | None = None,
) -> None:
    """Upload data and run training on Modal GPU.

    Args:
        data_path: Path to labeled training data parquet file.
        tokens_path: Path to tokens metadata JSON file.
        epochs: Number of training epochs per day.
        trading_days: Number of days to trade (None = all available).

    """
    logger.info("=" * 70)
    logger.info("MODAL CLOUD TRAINING")
    logger.info("=" * 70)

    # Validate local files
    data_file = Path(data_path)
    tokens_file = Path(tokens_path)

    if not data_file.exists():
        logger.error("❌ Data file not found: %s", data_path)
        raise typer.Exit(1)

    if not tokens_file.exists():
        logger.error("❌ Tokens file not found: %s", tokens_path)
        raise typer.Exit(1)

    # Upload to Modal volume
    logger.info("Uploading files to Modal...")
    logger.info("  📁 %s (%s)", data_path, _format_size(data_file))
    logger.info("  📁 %s (%s)", tokens_path, _format_size(tokens_file))

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(data_file), "labeled_log_fracdiff_price.parquet")
        batch.put_file(str(tokens_file), "tokens.json")

    logger.info("  ✅ Files uploaded")
    logger.info("")
    logger.info("Starting training on Modal GPU...")
    logger.info("=" * 70)

    # Run training (output streams to terminal)
    train_remote.remote(epochs, trading_days)

    logger.info("=" * 70)
    logger.info("✅ Training complete!")
    logger.info("=" * 70)


def _format_size(file: Path) -> str:
    """Format file size in human-readable format."""
    size = file.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
