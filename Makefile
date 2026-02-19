# Phase 0: Get swaps and reference data
ingest-swaps:
	echo "Ingesting swaps data from Google BigQuery..."
	python src/ingest_swaps.py \
		--chain ethereum \
		--start-date 2025-12-01 \
		--end-date 2026-01-01 \
		--output-dir data/swaps/ \
		--verbose 

ingest-pools:
	echo "Updating pools.json from kaiko.io..."	
	curl --compressed -H "Accept: application/json" \
		"https://reference-data-api.kaiko.io/v1/pools" \
		> data/pools.json


filter-and-decode-swaps:
	python src/filter_and_decode_swaps.py \
	--start-date 2025-10-01 \
	--end-date 2026-01-01 \
	--verbose

calculate-usdc-prices:
	python src/calculate_usdc_prices.py --verbose --validate

generate-usdc-bars:
	python src/generate_usdc_bars.py --verbose --validate

make-stationary:
	python src/make_stationary.py --verbose --validate

embed:
	python src/generate_embeddings.py \
		--epochs 50 \
		--patience 5

label-triple-barrier:
	python src/label_triple_barrier.py --verbose --validate

training-data-validation:
	python src/training_data_validation.py


update-embedding-data: filter-and-decode-swaps calculate-usdc-prices generate-usdc-bars make-stationary


update-train-data: filter-and-decode-swaps calculate-usdc-prices generate-usdc-bars make-stationary label-triple-barrier training-data-validation


backtest:
	python src/dex_contagion_trader.py \
		--save-embeddings \
		--epochs 50 \
		--trading-days 3 

modal-train:
	uv run modal run src/modal_train.py \
		--epochs 50 \
		--trading-days 5

ldr-tgn-trader:
	uv run python src/ldr_tgn_trader.py \
		--epochs 50 \
		--train-days 30 \
		--trading-days 15 \
		--save-embeddings









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
