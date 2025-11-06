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
	echo "Updating pools.json..."	
	curl --compressed -H "Accept: application/json" \
		"https://reference-data-api.kaiko.io/v1/pools" \
		> data/pools.json


# Phase 1: From raw swaps to WETH paired token prices
filter-and-decode-swaps:
	python src/filter_and_decode_swaps.py \
	--input-dir data/swaps \
	--output-file data/usdc_paired_swaps.parquet \
	--pools-file data/pools.json \
	--verbose

calculate-usdc-prices:
	echo "Calculating USDC paired token prices..."
	python src/calculate_usdc_prices.py \
		--input-file data/usdc_paired_swaps.parquet \
		--output-file data/usdc_prices_timeseries.parquet \
		--pools-file data/pools.json \
		--filter-outliers \
		--verbose
	echo "Now run validate_prices.ipynb notebook to validate the prices."

# Phase 2: From WETH paired token prices to volume bars and ARIMA forecasts
generate-volume-bars:
	python src/generate_volume_bars.py

stationarity-prep:
	python src/stationarity_prep.py

train-arima-models:
	python src/train_arima_models.py
	echo "Run notebooks/validate_arima_models.ipynb to validate ARIMA models."




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
