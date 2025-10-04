"""Ingest Uniswap V2/V3/V4 Swaps from Google BigQuery public blockchain tables."""

import logging
from datetime import datetime
from pathlib import Path

import click
import dotenv
import pandas as pd
from google.cloud import bigquery

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

BLOCKCHAINS = {
    "ethereum": {
        "dataset": "bigquery-public-data.goog_blockchain_ethereum_mainnet_us.logs",
        "address": "0x000000000004444c5dc75cB358380D2e3dE08A90",
    },
    "polygon": {
        "dataset": "bigquery-public-data.goog_blockchain_polygon_mainnet_us.logs",
        "address": "0x67366782805870060151383f4bbff9dab53e5cd6",
    },
    "arbitrum": {
        "dataset": "bigquery-public-data.goog_blockchain_arbitrum_one_us.logs",
        "address": "0x360e68faccca8ca495c1b759fd9eee466db9fb32",
    },
    "optimism": {
        "dataset": "bigquery-public-data.goog_blockchain_optimism_mainnet_us.logs",
        "address": "0x9a13f98cb987694c9f086b1f5eb990eea8264ec3",
    },
}


def query_all_swaps(
    client: bigquery.Client,
    chain: str,
    start_date: datetime,
    end_date: datetime,
    *,
    dry_run: bool = False,
) -> pd.DataFrame | None:
    """Query swap events from BigQuery.

    Args:
        client: BigQuery client instance.
        chain: Blockchain name (e.g., 'ethereum', 'polygon').
        start_date: Start date to query logs for.
        end_date: End date to query logs for.
        dry_run: If True, only analyze query without executing.

    Returns:
        DataFrame containing log events, or None if dry_run.

    """
    query = """
    SELECT
        block_timestamp,
        block_number,
        transaction_hash,
        address AS pool_or_manager_address,
        topics,
        data
    FROM
        `{}`
    WHERE
        topics[SAFE_OFFSET(0)] IN (
            -- V2 Swap topic
            '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',
            -- V3 Swap topic
            '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67',
            -- V4 Swap topic
            '0x32a8f1cc65ce5680dc7540e926b2482b7fd063d7cb362fcfc57008d2ed305743'
        )
        AND DATE(block_timestamp) BETWEEN @start_date AND @end_date
    ORDER BY
        block_timestamp ASC
    """.format(BLOCKCHAINS[chain]["dataset"])

    query_params = [
        bigquery.ScalarQueryParameter(
            "start_date",
            "DATE",
            start_date.strftime("%Y-%m-%d"),
        ),
        bigquery.ScalarQueryParameter(
            "end_date",
            "DATE",
            end_date.strftime("%Y-%m-%d"),
        ),
    ]

    job_config = bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=True,
        query_parameters=query_params,
    )

    logger.info(
        "Querying %s logs from %s to %s",
        chain,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )
    logger.debug("Query: %s", query)

    query_job = client.query(query, job_config=job_config)

    # Report statistics
    bytes_to_process = query_job.total_bytes_processed or 0
    gb_to_process = bytes_to_process / (1024**3)
    logger.info("=== DRY RUN ANALYSIS ===")
    logger.info(
        "Query will process: %.2f GB (%d bytes)",
        gb_to_process,
        bytes_to_process,
    )
    logger.info("Estimated cost: $%.4f (at $5/TB)", gb_to_process * 5 / 1024)
    logger.info("Query plan:")
    if hasattr(query_job, "query_plan") and query_job.query_plan:
        for stage in query_job.query_plan:
            logger.info("  Stage %d: %s", stage.id, stage.name)
    else:
        logger.info("  Query plan not available in dry run mode")
    if dry_run:
        return None

    df = query_job.to_dataframe()
    logger.info("Retrieved %d log events for %s", len(df), chain)
    return df


def save_to_parquet(
    df: pd.DataFrame,
    chain: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> Path:
    """Save DataFrame to parquet file.

    Args:
        df: DataFrame to save.
        chain: Blockchain name.
        start_date: Start date of the data.
        end_date: End date of the data.
        output_dir: Directory to save files to.

    Returns:
        Path to saved file.

    """
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    filename = f"{chain}_swaps_{start_str}_{end_str}.parquet"
    filepath = output_dir / filename

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    logger.info("Saved %d events to %s", len(df), filepath)

    return filepath


@click.command()
@click.option(
    "--chain",
    type=click.Choice(["ethereum", "polygon", "arbitrum", "optimism"]),
    default="ethereum",
    help="Blockchain to query",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    required=True,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/swaps"),
    help="Output directory for parquet files",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Analyze query without executing (shows data scan size and query plan)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose logging (debug level)",
)
def main(
    chain: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    dry_run: bool,  # noqa: FBT001
    verbose: bool,  # noqa: FBT001
) -> None:
    """Ingest Uniswap V4 log events from BigQuery and save to parquet files."""
    # Configure logging based on verbose flag
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(
        "Starting ingestion for %s from %s to %s",
        chain,
        start_date,
        end_date,
    )

    try:
        client = bigquery.Client()
        df = query_all_swaps(
            client,
            chain,
            start_date,
            end_date,
            dry_run=dry_run,
        )
        if df is not None and not df.empty:
            save_to_parquet(
                df,
                chain,
                start_date,
                end_date,
                output_dir,
            )
        elif df is not None:
            logger.warning(
                "No data found for %s",
                start_date.strftime("%Y-%m-%d"),
            )
    except Exception:
        logger.exception(
            "Error processing %s",
            start_date.strftime("%Y-%m-%d"),
        )
        raise

    logger.info("Ingestion completed successfully")


if __name__ == "__main__":
    main()
