ingest-swaps:
	echo "Ingesting swaps data for July 2025 from Google BigQuery..."
	python src/ingest-swaps.py \
		--chain ethereum \
		--start-date 2025-07-01 \
		--end-date 2025-08-01 \
		--output-dir data/swaps/ \
		--verbose 

filter-swaps:
	echo "Filtering uniswap v3 swaps..."
	python src/filter_v3_swaps.py \
		--input-dir data/swaps/ \
		--output-file data/uniswap_v3_swaps.parquet \
		--verbose



ingest-pools:
	echo "Updating pools.json..."	
	curl --compressed -H "Accept: application/json" \
		"https://reference-data-api.kaiko.io/v1/pools" \
		> data/pools.json

decode-swaps:
	echo "Decoding uniswap v3 swaps..."
	python src/decode_swaps.py \
		--input-file data/uniswap_v3_swaps.parquet \
		--output-file data/weth_paired_swaps.parquet \
		--verbose



calculate-weth-prices:
	echo "Calculating WETH paired token prices..."
	python src/calculate_weth_prices.py \
		--input-file data/weth_paired_swaps.parquet \
		--output-file data/weth_prices_timeseries.parquet \
		--pools-file data/pools.json \
		--filter-outliers \
		--verbose
	echo "Now run validate_prices.ipynb notebook to validate the prices."








# # Reference json files
# ingest-coins:
# 	echo "Updating coins.json..."	
# 	curl --compressed -H "Accept: application/json" \
# 		"https://tokens.coingecko.com/uniswap/all.json" \
# 		--header "x-cg-pro-api-key: $$COINGECKO_API_KEY" \
# 		> data/coins.json

# ingest-tokens:
# 	echo "Updating tokens.json..."	
# 	curl --compressed -H 'Accept: application/json' \
# 		"https://ipfs.io/ipns/tokens.uniswap.org" \
# 		> data/tokens.json


# Reference prices for validation notebook
ingest-uni-prices:
	echo "Ingesting UNI prices for July 2025..."
	curl --compressed -H "Accept: application/json" \
  		"https://api.coingecko.com/api/v3/coins/uni/market_chart/range?vs_currency=eth&from=2025-07-01&to=2025-08-01" \
		--header "x-cg-pro-api-key: $$COINGECKO_API_KEY" \
		> data/uni.json
