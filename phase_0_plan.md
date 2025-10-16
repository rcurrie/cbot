# Milestone Plan

**General Information**

- data/swaps/ is already populated by ingest.py which should not be run as it cost $ unless you check with me.
- There are several reference json files in data/ that may be useful that are downloaded via curl in Makefile. data/pools.json has information on all DEX pools including Uniswap. data/tokens.json has information on tokens from uniswap and data/coins.json has information from coingecko. They are used to generate weth_pool_addr_to_symbol.json and weth_pools_by_address.json which may be used in any of these. For example to annotate a swap with its token symbols, or to find the token information given the address from a uniswap pair.

**Milestone 1: Data Ingestion and Uniswap V3 Filtering**

- **Goal:** Read the raw swap data, isolate the relevant Uniswap V3 events, and create a clean, consolidated dataset.
- **Tasks:**
  1.  Implement a script to recursively load all `.parquet` files from the `data/swaps/` directory using `polars`.
  2.  Apply filters to keep only transactions corresponding to Uniswap V3 "Swap" events.
  3.  Save the resulting filtered DataFrame to a single new parquet file, e.g., `data/uniswap_v3_swaps.parquet`.
- **Validation:** We will verify that the output file is created, contains significantly fewer rows than the source, and a manual inspection of a few sample rows will confirm they are indeed the correct Uniswap V3 swap events.

**Milestone 2: Swap Decoding and WETH-Pair Filtering**

- **Goal:** Decode the swap data into a structured format and limit our dataset to tokens that have a direct trading pair with WETH.
- **Tasks:**
  1.  Decode the swap data to extract structured fields: `token_in`, `token_out`, `amount_in`, `amount_out`, `pool`, and `trader_account`.
  2.  Identify all unique liquidity pools that pair a token directly with WETH.
  3.  Create a definitive list of all tokens that appear in these WETH-paired pools.
  4.  Filter the main swap dataset to only include swaps where _at least one_ of the tokens involved is in our WETH-paired list.
- **Validation:** We will generate and inspect the list of WETH-paired tokens. We'll then programmatically verify that all swaps in the filtered dataset involve at least one token from this list.

**Milestone 3: WETH Price Calculation and External Validation**

- **Goal:** Calculate the implied price of each token in terms of WETH from the swap data and validate these prices against a real-world source.
- **Tasks:**
  1.  Process the swaps in chronological order.
  2.  For each swap, calculate the price of the non-WETH token relative to WETH. For swaps between two non-WETH tokens, we will use the most recently calculated prices to infer the new price.
  3.  Store this time-series price data.
- **Validation:** We will create a Jupyter Notebook to fetch historical price data from the CoinGecko API for a selection of the tokens in our dataset. We will then plot our calculated, swap-derived prices against the CoinGecko prices to visually and statistically assess the accuracy of our pricing model.

**Milestone 4: Foundational Graph Construction**

- **Goal:** Build the initial static, multi-partite graph structure using PyTorch Geometric.
- **Tasks:**
  1.  Define a `HeteroData` schema for the graph with three node types: `Token`, `Pool`, and `Account`.
  2.  Assign unique integer IDs to every token, pool, and account entity.
  3.  Construct the graph's edge topology by creating directed edges for each swap, representing the flow of tokens (e.g., `(Account) -> (Pool)` and `(Pool) -> (Account)`).
- **Validation:** We will check the total number of nodes of each type and the number of edges. We will also manually inspect the connectivity for a few sample swaps to ensure the graph structure correctly represents the underlying transactions.

**Milestone 5: Temporal Feature Engineering**

- **Goal:** Convert the static graph into a temporal graph by adding time-aware edges and engineering node features that evolve over time.
- **Tasks:**
  1.  Attach a timestamp to each edge (or set of edges) corresponding to a swap, making the graph temporal.
  2.  Create and update node features chronologically with each swap:
      - **Token Nodes:** Price in WETH.
      - **Pool Nodes:** Total liquidity and token reserves, valued in WETH.
      - **Account Nodes:** Inferred holdings of each token based on swap activity, valued in WETH.
- **Validation:** We will query the graph at different timestamps. By selecting a specific node (e.g., a busy trading account or a popular liquidity pool), we will trace its feature history to confirm that the values are being updated correctly and logically based on the sequence of swaps.
