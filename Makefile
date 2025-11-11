# Phase 0: Get swaps and reference data
ingest-swaps:
	echo "Ingesting swaps data for July 2025 from Google BigQuery..."
	python src/ingest-swaps.py \
		--chain ethereum \
		--start-date 2025-07-01 \
		--end-date 2025-08-01 \
		--output-dir data/swaps/ \
		--verbose 

ingest-pools:
	echo "Updating pools.json from kaiko.io..."	
	curl --compressed -H "Accept: application/json" \
		"https://reference-data-api.kaiko.io/v1/pools" \
		> data/pools.json


filter-and-decode-swaps:
	python src/filter_and_decode_swaps.py --verbose

calculate-usdc-prices:
	python src/calculate_usdc_prices.py --verbose


generate-volume-bars:
	python src/generate_pool_bars.py --verbose


make-stationary:
	python src/make_stationary.py --verbose

# Phase 3: From stationary prices to labeled data using Triple-Barrier Method
label-triple-barrier:
	python src/label_triple_barrier.py --verbose









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
# ingest-uni-prices:
# 	echo "Ingesting UNI prices for July 2025..."
# 	curl --compressed -H "Accept: application/json" \
#   		"https://api.coingecko.com/api/v3/coins/uni/market_chart/range?vs_currency=eth&from=2025-07-01&to=2025-08-01" \
# 		--header "x-cg-pro-api-key: $$COINGECKO_API_KEY" \
# 		> data/uni.json
