"""Invert pools.json into a tokens.json file.

Reads pools.json and creates a token-centric view with token address as key,
containing symbol, list of pools, decimals, blockchain, and other metadata.

Output: data/tokens.json
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def invert_pools_to_tokens() -> dict[str, dict[str, Any]]:
    """Invert pools.json structure into token-centric view.

    Returns:
        Dict mapping token addresses to token metadata including:
        - symbol: Token symbol
        - decimals: Token decimals (or "NULL" if not available)
        - blockchain: Blockchain (e.g., "arbitrum")
        - pools: List of pool objects containing this token
        - pool_count: Number of pools this token appears in

    """
    pools_path = Path("data/pools.json")

    if not pools_path.exists():
        logger.error("pools.json not found at %s", pools_path)
        msg = f"pools.json not found at {pools_path}"
        raise FileNotFoundError(msg)

    logger.info("Reading pools from %s", pools_path)

    with pools_path.open() as f:
        pools_data = json.load(f)

    pools = pools_data.get("data", [])
    logger.info("Loaded %d pools", len(pools))

    # Invert structure: token address -> token metadata + pools containing it
    tokens_dict: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "symbol": "",
            "decimals": None,
            "blockchain": "",
            "pools": [],
        },
    )

    for pool in pools:
        pool_tokens = pool.get("tokens", [])

        for token in pool_tokens:
            addr = token.get("address", "").lower()
            symbol = token.get("symbol", "").upper()
            decimals = token.get("decimals", "NULL")
            blockchain = token.get("blockchain", "")

            if not addr:
                logger.warning("Token without address in pool %s", pool.get("address"))
                continue

            # Update token metadata (use first occurrence for consistency)
            if not tokens_dict[addr]["symbol"]:
                tokens_dict[addr]["symbol"] = symbol
                tokens_dict[addr]["decimals"] = decimals
                tokens_dict[addr]["blockchain"] = blockchain

            # Add pool info to this token
            pool_info = {
                "pool_address": pool.get("address", ""),
                "pool_name": pool.get("name", ""),
                "blockchain": pool.get("blockchain", ""),
                "protocol": pool.get("protocol", ""),
                "type": pool.get("type", ""),
                "fee": pool.get("fee", ""),
                "tick_spacing": pool.get("tickSpacing", None),
            }

            # Only add if not already in pools list
            if not any(
                p["pool_address"] == pool_info["pool_address"]
                for p in tokens_dict[addr]["pools"]
            ):
                tokens_dict[addr]["pools"].append(pool_info)

    # Convert defaultdict to regular dict and add pool_count
    result = {}
    for addr, data in tokens_dict.items():
        result[addr] = {
            "symbol": data["symbol"],
            "decimals": data["decimals"],
            "blockchain": data["blockchain"],
            "pool_count": len(data["pools"]),
            "pools": data["pools"],
        }

    logger.info("Inverted %d unique tokens", len(result))

    return result


def save_tokens_json(tokens_dict: dict[str, dict[str, Any]]) -> None:
    """Save inverted tokens dictionary to data/tokens.json.

    Args:
        tokens_dict: Token dictionary to save

    """
    output_path = Path("data/tokens.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving tokens to %s", output_path)

    with output_path.open("w") as f:
        json.dump(tokens_dict, f, indent=2)

    logger.info("Saved %d tokens to %s", len(tokens_dict), output_path)


if __name__ == "__main__":
    tokens = invert_pools_to_tokens()
    save_tokens_json(tokens)
    logger.info("Done")
