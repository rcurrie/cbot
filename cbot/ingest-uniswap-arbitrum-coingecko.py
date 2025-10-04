"""Fetch top Uniswap v3 pools by daily volume from CoinGecko.

Queries the CoinGecko Pro API for pools on Arbitrum, filtered by
Uniswap v3 DEX, ordered by 24h volume descending.

The collected data is saved to a JSON file named
`uniswap_v3_arbitrum_top_pools_by_volume.json` in the project root.
"""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import click
import requests


# --- Type Definitions for Strict Typing ---
class TokenData(TypedDict):
    """Represent the data structure for a token in a pool."""

    id: str
    symbol: str
    name: str
    decimals: int


class PoolData(TypedDict):
    """Represent the data structure for a pool."""

    id: str
    type: str
    attributes: dict[str, Any]
    relationships: dict[str, Any]


# --- Environment and Constants ---

BASE_URL = "https://api.geckoterminal.com/api/v2"


# Path to the output file, located in the project root
OUTPUT_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "uniswap_v3_arbitrum_top_pools_by_volume.json"
)

# API endpoint for pools
POOLS_URL = f"{BASE_URL}/networks/arbitrum/dexes/uniswap_v3_arbitrum/pools"

# Parameters for the query
PARAMS: dict[str, str | int] = {
    "order": "h24_volume_usd_desc",
    "include": "base_token,quote_token",
    "page": 1,
}

HEADERS = {
    "accept": "application/json",
}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_top_pools(max_pools: int = 100) -> list[dict[str, Any]]:
    """Fetch top Uniswap v3 pools by volume from CoinGecko API.

    Args:
        max_pools: The maximum number of pools to fetch.

    Returns:
        A list of pool data dictionaries.

    """
    all_pools: list[dict[str, Any]] = []
    params = PARAMS.copy()
    params["per_page"] = min(20, max_pools)  # API returns up to 20 per page

    logger.info("Starting to fetch top pools by volume from CoinGecko...")

    while len(all_pools) < max_pools:
        try:
            response = requests.get(
                POOLS_URL,
                headers=HEADERS,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            pools = data.get("data", [])
            if not pools:
                logger.info("No more pools found.")
                break

            all_pools.extend(pools)
            logger.info(
                "%d pools fetched (total: %d).",
                len(pools),
                len(all_pools),
            )

            per_page = cast("int", params["per_page"])
            if len(pools) < per_page:
                break  # No more pages

            params["page"] = cast("int", params["page"]) + 1

            if len(all_pools) >= max_pools:
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
    default=100,
    help="The maximum number of pools to fetch.",
)
def main(max_pools: int) -> None:
    """Fetch top Uniswap v3 pools by volume and save them to a file."""
    pools = fetch_top_pools(max_pools=max_pools)

    if not pools:
        logger.warning("No pools were fetched. Exiting.")
        return

    logger.info("Found a total of %d pools.", len(pools))

    # Save the data to a JSON file
    try:
        with OUTPUT_FILE.open("w") as f:
            json.dump(pools, f, indent=2)
        logger.info("Successfully saved pool data to %s", OUTPUT_FILE)
    except OSError:
        logger.exception("Error writing to file %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
