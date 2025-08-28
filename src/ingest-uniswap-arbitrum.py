"""Fetch all Uniswap v3 pools from TheGraph on the Arbitrum network.

It queries the official Uniswap v3 subgraph, paginating through all available
pools and collecting detailed information for each one.

The collected data is saved to a JSON file named `uniswap_v3_arbitrum_pools.json`
in the project root.
"""

import json
import logging
import os
from pathlib import Path
from typing import TypedDict

import click
import requests
from dotenv import load_dotenv


# --- Type Definitions for Strict Typing ---
class TokenData(TypedDict):
    """Represent the data structure for a token in a pool."""

    id: str
    symbol: str
    name: str
    decimals: str


class PoolData(TypedDict):
    """Represent the data structure for a Uniswap v3 pool."""

    id: str
    token0: TokenData
    token1: TokenData
    feeTier: str
    liquidity: str
    sqrtPrice: str
    tick: str | None
    totalValueLockedToken0: str
    totalValueLockedToken1: str
    totalValueLockedUSD: str
    txCount: str
    createdAtTimestamp: str
    createdAtBlockNumber: str


# --- Environment and Constants ---

load_dotenv()  # Load environment variables from .env file

# TheGraph endpoint for Uniswap v3 on Arbitrum using the decentralized network
# https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV?view=About&chain=arbitrum-one
THEGRAPH_API_KEY = os.getenv("THEGRAPH_API_KEY")
assert THEGRAPH_API_KEY is not None, "THEGRAPH_API_KEY environment variable not set"

SUBGRAPH_URL = f"https://gateway.thegraph.com/api/{THEGRAPH_API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Path to the output file, located in the project root
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "uniswap_v3_arbitrum_pools.json"

# GraphQL query to fetch pools.
# We fetch 1000 at a time and paginate using the `id` field.
POOLS_QUERY_TEMPLATE = """
{{
  pools(
    first: 1000,
    orderBy: totalValueLockedUSD,
    orderDirection: desc,
    where: {{id_gt: "{last_id}"}}
  ) {{
    id
    token0 {{
      id
      symbol
      name
      decimals
    }}
    token1 {{
      id
      symbol
      name
      decimals
    }}
    feeTier
    liquidity
    sqrtPrice
    tick
    totalValueLockedToken0
    totalValueLockedToken1
    totalValueLockedUSD
    txCount
    createdAtTimestamp
    createdAtBlockNumber
  }}
}}
"""

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_all_pools(max_pools: int | None = None) -> list[PoolData]:
    """Fetch Uniswap v3 pools by paginating through TheGraph's API.

    Args:
        max_pools: The maximum number of pools to fetch. If None, fetches all.

    Returns:
        A list of dictionaries, where each dictionary represents a pool.

    """
    all_pools: list[PoolData] = []
    last_id = ""

    logger.info("Starting to fetch pools from TheGraph...")
    if max_pools is not None:
        logger.info("Fetching up to %d pools.", max_pools)

    while True:
        query = POOLS_QUERY_TEMPLATE.format(last_id=last_id)
        try:
            response = requests.post(SUBGRAPH_URL, json={"query": query}, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if "errors" in data:
                logger.error("GraphQL query errors: %s", data["errors"])
                break

            # The returned JSON for pools should match the PoolData structure
            pools: list[PoolData] = data.get("data", {}).get("pools", [])
            if not pools:
                logger.info("No more pools found. Pagination complete.")
                break

            all_pools.extend(pools)
            last_id = pools[-1]["id"]

            logger.info("%d pools fetched (total: %d).", len(pools), len(all_pools))

            if max_pools is not None and len(all_pools) >= max_pools:
                logger.info("Reached max pools limit of %d.", max_pools)
                all_pools = all_pools[:max_pools]
                break

        except requests.exceptions.RequestException:
            logger.exception("An HTTP error occurred")
            break
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON from response.")
            break

    return all_pools


@click.command()
@click.option(
    "--max-pools",
    type=int,
    default=1000,
    help="The maximum number of pools to fetch. Fetches all if not specified.",
)
def main(max_pools: int | None) -> None:
    """Fetch Uniswap v3 pools and save them to a file."""
    pools = fetch_all_pools(max_pools=max_pools)

    if not pools:
        logger.warning("No pools were fetched. Exiting.")
        return

    logger.info("Found a total of %d pools.", len(pools))

    # Save the data to a JSON file
    try:
        with OUTPUT_FILE.open("w") as f:
            # The list of TypedDicts can be directly serialized by json.dump
            json.dump(pools, f, indent=2)
        logger.info("Successfully saved pool data to %s", OUTPUT_FILE)
    except OSError:
        logger.exception("Error writing to file %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
