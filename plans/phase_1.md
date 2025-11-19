# Milestone Plan

**General Information**

Wrangle raw swap data downloaded from google big query into a series of USDC priced swaps with volume and information useful to develop a graph neural network in later phases.

- data/swaps/ is already populated by ingest_swaps.py which should not be run as it cost $ unless you check with me.
- data/pools.json has information on all DEX pools including Uniswap

**Milestone 1: Data Ingestion and Uniswap V3 Filtering**

- **Goal:** Read the raw swap data, isolate the relevant Uniswap V3 events, and create a clean, consolidated dataset.
- **Script:** [src/filter_and_decode_swaps.py](../src/filter_and_decode_swaps.py)
- **Tasks:**
  1.  Implement a script to recursively load all `.parquet` files from the `data/swaps/` directory using `polars`.
  2.  Apply filters to keep only transactions corresponding to Uniswap V3 "Swap" events.
  3.  Save the resulting filtered DataFrame to a single new parquet file, e.g., `data/uniswap_v3_swaps.parquet`.
- **Validation:** We will verify that the output file is created, contains significantly fewer rows than the source, and a manual inspection of a few sample rows will confirm they are indeed the correct Uniswap V3 swap events.

**Milestone 2: Swap Decoding and USDC-Pair Filtering**

- **Rational**
  [Why USDC? And only it now and not WETH as well?](https://gemini.google.com/app/44edb6acb67b153d)

- **Goal:** Decode the swap data into a structured format and limit our dataset to tokens that have a direct trading pair with USDC.
- **Tasks:**
  1.  Decode the swap data to extract structured fields: `token_in`, `token_out`, `amount_in`, `amount_out`, `pool`, and `trader_account`.
  2.  Identify all unique liquidity pools that pair a token directly with USDC.
  3.  Create a definitive list of all tokens that appear in these USDC-paired pools.
  4.  Filter the main swap dataset to only include swaps where _at least one_ of the tokens involved is in our USDC-paired list.
- **Validation:** We will generate and inspect the list of USDC-paired tokens. We'll then programmatically verify that all swaps in the filtered dataset involve at least one token from this list.

**Milestone 3: USDC Price Calculation and External Validation**

- **Goal:** Calculate the implied price of each token in terms of USDC from the swap data and validate these prices against a real-world source.
- **Tasks:**
  1.  Process the swaps in chronological order.
  2.  For each swap, calculate the price of the non-USDC token relative to USDC. For swaps between two non-USDC tokens, we will use the most recently calculated prices to infer the new price.
  3.  Store this time-series price data.
- **Validation:** We will create a Jupyter Notebook to fetch historical price data from the CoinGecko API for a selection of the tokens in our dataset. We will then plot our calculated, swap-derived prices against the CoinGecko prices to visually and statistically assess the accuracy of our pricing model.
