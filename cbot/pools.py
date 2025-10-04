import hashlib
from typing import Any

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


def compute_keccak256(data: str) -> str:
    """Compute keccak256 hash of the given data string (hex without 0x)."""
    return "0x" + hashlib.sha3_256(bytes.fromhex(data)).hexdigest()


def parse_pool_key(data: str) -> dict[str, Any]:
    """Parse the PoolKey from the event data hex string (without 0x)."""
    if not data or len(data) != 320:
        return {}

    # Extract currency0 (last 40 chars of first 64-char field)
    currency0 = "0x" + data[24:64]

    # Extract currency1 (last 40 chars of second 64-char field, positions 64:128 -> 88:128)
    currency1 = "0x" + data[88:128]

    # Extract fee (uint24: last 6 chars of third 64-char field, positions 128:192 -> 186:192)
    fee_hex = data[186:192]
    fee = int(fee_hex, 16)

    # Extract tickSpacing (int24: last 6 chars of fourth 64-char field, positions 192:256 -> 250:256)
    tick_spacing_hex = data[250:256]
    bytes3 = bytes.fromhex(tick_spacing_hex)
    tick_spacing = int.from_bytes(bytes3, "big")
    if tick_spacing >= (1 << 23):
        tick_spacing -= 1 << 24

    # Extract hooks (last 40 chars of fifth 64-char field, positions 256:320 -> 280:320)
    hooks = "0x" + data[280:320]

    # Compute pool key hash (keccak256 of the entire data)
    pool_key_hash = compute_keccak256(data)

    return {
        "currency0": currency0,
        "currency1": currency1,
        "fee": fee,
        "tick_spacing": tick_spacing,
        "hooks": hooks,
        "pool_key_hash": pool_key_hash,
    }


def main() -> None:
    # Event signature for PoolCreated
    event_sig = "PoolCreated((address,address,uint24,int24,address))"
    pool_created_topic = "0x" + hashlib.sha3_256(event_sig.encode()).hexdigest()

    # Uniswap V4 PoolManager address on Ethereum Mainnet
    pool_manager_address = "0x000000000004444c5dc75cb358380d2e3de08a90"

    # BigQuery client (assumes GOOGLE_APPLICATION_CREDENTIALS is set)
    # client = bigquery.Client()

    credentials = service_account.Credentials.from_service_account_file(
        "keys/gcloud.json",
    )  # type: ignore[no-untyped-call]
    client = bigquery.Client(credentials=credentials)

    query = f"""
    SELECT
      block_timestamp,
      block_number,
      transaction_hash,
      data
    FROM
      `bigquery-public-data.goog_blockchain_ethereum_mainnet_us.logs`
    WHERE
      address = '{pool_manager_address.lower()}'
      AND topics[SAFE_OFFSET(0)] = '{pool_created_topic}'
      AND block_timestamp >= TIMESTAMP("2025-01-01")  -- Adjust start date as needed
    ORDER BY
      block_timestamp ASC
    """

    df = client.query(query).to_dataframe()

    parsed_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        key_info = parse_pool_key(row["data"])
        if key_info:
            key_info.update(
                {
                    "block_timestamp": row["block_timestamp"],
                    "block_number": row["block_number"],
                    "transaction_hash": row["transaction_hash"],
                },
            )
            parsed_rows.append(key_info)

    pools_df = pd.DataFrame(parsed_rows)
    pools_df.to_parquet("uniswap_v4_pools.parquet", index=False)
    print(f"Saved {len(pools_df)} Uniswap V4 pools to uniswap_v4_pools.parquet")


if __name__ == "__main__":
    main()
