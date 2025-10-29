"""Filter Uniswap V3 swap events from raw swap data.

Load all parquet files from data/swaps/, filter to only Uniswap V3 Swap events,
and save to a consolidated parquet file.
"""

import logging
from pathlib import Path

import click
import polars as pl
from web3 import Web3

logger = logging.getLogger(__name__)

# Uniswap V3 Swap event signature
# Swap(address indexed sender, address indexed recipient,
#      int256 amount0, int256 amount1, uint160 sqrtPriceX96,
#      uint128 liquidity, int24 tick)
V3_SWAP_SIGNATURE = Web3.keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)",
).hex()


def load_and_filter_swaps(
    input_dir: Path,
    output_file: Path,
) -> None:
    """Load all parquet files from input_dir and filter to V3 swaps.

    Args:
        input_dir: Directory containing input parquet files.
        output_file: Path to save filtered output parquet file.

    """
    logger.info("Loading swap data from %s...", input_dir)

    # Use lazy evaluation to read all parquet files efficiently
    # This leverages polars' multi-core processing capabilities
    v3_swaps = (
        pl.scan_parquet(input_dir / "*.parquet")
        .with_columns(
            # Extract first topic (event signature)
            pl.col("topics").list.get(0).alias("event_signature"),
        )
        .filter(
            # Filter to V3 swaps only
            pl.col("event_signature") == f"0x{V3_SWAP_SIGNATURE}",
        )
        .drop("event_signature")  # Drop temporary column
        .collect()  # Execute the lazy query
    )

    logger.info("Filtered to %s Uniswap V3 swap events", f"{len(v3_swaps):,}")

    # Save to output file
    logger.info("Saving to %s...", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    v3_swaps.write_parquet(output_file)

    logger.info("Done!")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/swaps"),
    help="Directory containing input parquet files",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/uniswap_v3_swaps.parquet"),
    help="Output parquet file path",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    input_dir: Path,
    output_file: Path,
    *,
    verbose: bool,
) -> None:
    """Filter Uniswap V3 swap events from raw swap data."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_and_filter_swaps(input_dir, output_file)


if __name__ == "__main__":
    main()
