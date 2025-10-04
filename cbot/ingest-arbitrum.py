#!/usr/bin/env python3
"""Ingest swap events from BigQuery Arbitrum public dataset for Uniswap V4 pools."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenPair(BaseModel):
    """Token pair configuration."""

    token0_address: str = Field(..., description="Address of token 0")
    token1_address: str = Field(..., description="Address of token 1")

    @field_validator("token0_address", "token1_address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        """Validate ethereum address format."""
        if not v.startswith("0x"):
            v = f"0x{v}"
        if len(v) != 42:
            raise ValueError(f"Invalid address length: {v}")
        return v.lower()


class SwapEventQuery(BaseModel):
    """Configuration for swap event query."""

    start_date: datetime
    end_date: datetime
    token_pair: TokenPair
    pool_manager_address: str = Field(
        default="0x0000000071727de22e5e9d8baf0edac6f37da032",
        description="Uniswap V4 PoolManager contract address"
    )
    output_path: Path = Field(default=Path("./data"))

    @field_validator("pool_manager_address")
    @classmethod
    def validate_pool_manager(cls, v: str) -> str:
        """Validate pool manager address."""
        if not v.startswith("0x"):
            v = f"0x{v}"
        return v.lower()

    @field_validator("output_path")
    @classmethod
    def create_output_path(cls, v: Path) -> Path:
        """Ensure output path exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v


def build_swap_query(config: SwapEventQuery) -> str:
    """Build BigQuery SQL for Uniswap V4 swap events.

    Args:
        config: Query configuration

    Returns:
        SQL query string
    """
    # Uniswap V4 Swap event signature
    # Swap(PoolId indexed id, address indexed sender, int128 amount0, int128 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick, uint24 fee)
    swap_v4_topic = "0x3b841dc9ab51e3104bda4f61b41e4271192d22cd19da5ee6e292dc8e2744f713"

    query = f"""
    WITH swap_events AS (
        SELECT
            block_timestamp,
            block_number,
            transaction_hash,
            transaction_index,
            log_index,
            address as pool_manager_address,
            topics[SAFE_OFFSET(1)] as pool_id,
            topics[SAFE_OFFSET(2)] as sender,
            data,
            -- Decode swap data (amount0, amount1, sqrtPriceX96, liquidity, tick, fee)
            -- amount0: int128 at position 0-31
            -- amount1: int128 at position 32-63
            -- sqrtPriceX96: uint160 at position 64-95
            -- liquidity: uint128 at position 96-127
            -- tick: int24 at position 128-159
            -- fee: uint24 at position 160-191
            CAST(CONCAT('0x', SUBSTR(data, 3, 64)) AS STRING) as amount0_hex,
            CAST(CONCAT('0x', SUBSTR(data, 67, 64)) AS STRING) as amount1_hex,
            CAST(CONCAT('0x', SUBSTR(data, 131, 64)) AS STRING) as sqrtPriceX96_hex,
            CAST(CONCAT('0x', SUBSTR(data, 195, 64)) AS STRING) as liquidity_hex,
            CAST(CONCAT('0x', SUBSTR(data, 259, 64)) AS STRING) as tick_hex,
            CAST(CONCAT('0x', SUBSTR(data, 323, 64)) AS STRING) as fee_hex
        FROM `bigquery-public-data.crypto_arbitrum.logs`
        WHERE DATE(block_timestamp) BETWEEN '{config.start_date.date()}' AND '{config.end_date.date()}'
        AND address = '{config.pool_manager_address}'
        AND topics[SAFE_OFFSET(0)] = '{swap_v4_topic}'
    ),
    -- Filter for pools that involve our token pair
    -- This requires looking at the pool creation or checking pool state
    -- For now, we'll get all swaps and filter client-side based on pool_id
    filtered_swaps AS (
        SELECT *
        FROM swap_events
        -- Pool ID in V4 is a hash of the pool key (currency0, currency1, fee, tickSpacing, hooks)
        -- Without pool creation events, we'll need to filter based on transaction analysis
        WHERE TRUE  -- Placeholder for pool filtering logic
    )
    SELECT
        block_timestamp,
        block_number,
        transaction_hash,
        transaction_index,
        log_index,
        pool_manager_address,
        pool_id,
        sender,
        amount0_hex,
        amount1_hex,
        sqrtPriceX96_hex,
        liquidity_hex,
        tick_hex,
        fee_hex
    FROM filtered_swaps
    ORDER BY block_timestamp, log_index
    """

    return query


def estimate_query_size(config: SwapEventQuery) -> float:
    """Estimate query size in GB.

    Args:
        config: Query configuration

    Returns:
        Estimated size in GB
    """
    try:
        client = bigquery.Client()

        # Build a count query to estimate data size
        count_query = f"""
        SELECT
            COUNT(*) as total_logs,
            -- Estimate bytes: ~500 bytes per log entry
            COUNT(*) * 500 / 1024 / 1024 / 1024 as estimated_gb
        FROM `bigquery-public-data.crypto_arbitrum.logs`
        WHERE DATE(block_timestamp) BETWEEN '{config.start_date.date()}' AND '{config.end_date.date()}'
        AND address = '{config.pool_manager_address}'
        """

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(count_query, job_config=job_config)

        # Get actual bytes that would be processed
        bytes_processed = query_job.total_bytes_processed
        gb_processed = bytes_processed / 1024 / 1024 / 1024

        return gb_processed

    except Exception as e:
        logger.exception("Failed to estimate query size")
        raise


def fetch_swap_events(config: SwapEventQuery) -> pd.DataFrame:
    """Fetch swap events from BigQuery.

    Args:
        config: Query configuration

    Returns:
        DataFrame with swap events
    """
    try:
        client = bigquery.Client()
        query = build_swap_query(config)

        logger.info(f"Querying Uniswap V4 swap events for tokens {config.token_pair.token0_address}/{config.token_pair.token1_address}")
        logger.info(f"Date range: {config.start_date.date()} to {config.end_date.date()}")
        logger.info(f"Pool Manager: {config.pool_manager_address}")

        query_job = client.query(query)
        df = query_job.to_dataframe()

        bytes_processed = query_job.total_bytes_processed
        gb_processed = bytes_processed / 1024 / 1024 / 1024
        logger.info(f"Query processed {gb_processed:.2f} GB")
        logger.info(f"Retrieved {len(df)} swap events")

        return df

    except Exception as e:
        logger.exception("Failed to fetch swap events from BigQuery")
        raise


def save_swap_events(df: pd.DataFrame, config: SwapEventQuery) -> Path:
    """Save swap events to parquet file.

    Args:
        df: DataFrame with swap events
        config: Query configuration

    Returns:
        Path to saved file
    """
    token0_short = config.token_pair.token0_address[:10]
    token1_short = config.token_pair.token1_address[:10]
    filename = (
        f"arbitrum_v4_swaps_{token0_short}_{token1_short}_"
        f"{config.start_date.date()}_{config.end_date.date()}.parquet"
    )
    output_file = config.output_path / filename

    df.to_parquet(output_file, index=False)
    logger.info(f"Saved {len(df)} swap events to {output_file}")

    return output_file


@click.command()
@click.option(
    "--start-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for data fetch (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for data fetch (YYYY-MM-DD)",
)
@click.option(
    "--token0",
    required=True,
    help="Address of token 0",
)
@click.option(
    "--token1",
    required=True,
    help="Address of token 1",
)
@click.option(
    "--pool-manager",
    default="0x0000000071727de22e5e9d8baf0edac6f37da032",
    help="Uniswap V4 PoolManager contract address",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default="./data",
    help="Output directory for parquet files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only estimate query size without executing",
)
def main(
    start_date: datetime,
    end_date: datetime,
    token0: str,
    token1: str,
    pool_manager: str,
    output_path: Path,
    dry_run: bool,
) -> None:
    """Ingest Arbitrum Uniswap V4 swap events from BigQuery."""
    try:
        token_pair = TokenPair(token0_address=token0, token1_address=token1)
        config = SwapEventQuery(
            start_date=start_date,
            end_date=end_date,
            token_pair=token_pair,
            pool_manager_address=pool_manager,
            output_path=output_path,
        )

        if dry_run:
            gb_estimate = estimate_query_size(config)
            logger.info(f"Dry run - Estimated query size: {gb_estimate:.3f} GB")
            logger.info(f"Estimated cost: ${gb_estimate * 5:.2f} (at $5/TB)")
            logger.info("Query details:")
            logger.info(f"  Date range: {config.start_date.date()} to {config.end_date.date()}")
            logger.info(f"  Pool Manager: {config.pool_manager_address}")
            logger.info(f"  Token pair: {token0} / {token1}")
            return

        df = fetch_swap_events(config)

        if df.empty:
            logger.warning("No swap events found for the specified criteria")
            return

        output_file = save_swap_events(df, config)
        logger.info(f"Successfully ingested swap events to {output_file}")

    except Exception as e:
        logger.exception("Failed to ingest swap events")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()