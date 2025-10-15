ingest-coins:
	echo "Updating coins.json..."	
	curl --compressed -H "Accept: application/json" \
		"https://tokens.coingecko.com/uniswap/all.json" \
		--header "x-cg-pro-api-key: $$COINGECKO_API_KEY" \
		> data/coins.json

ingest-tokens:
	echo "Updating tokens.json..."	
	curl --compressed -H 'Accept: application/json' \
		"https://ipfs.io/ipns/tokens.uniswap.org" \
		> data/tokens.json

ingest-pools:
	echo "Updating pools.json..."	
	curl --compressed -H "Accept: application/json" \
		"https://reference-data-api.kaiko.io/v1/pools" \
		> data/pools.json

ingest: ingest-coins ingest-tokens ingest-pools


ingest-uni-prices:
	echo "Ingesting UNI prices for July 2025..."
	curl --compressed -H "Accept: application/json" \
  		"https://api.coingecko.com/api/v3/coins/uni/market_chart/range?vs_currency=eth&from=2025-07-01&to=2025-08-01" \
		--header "x-cg-pro-api-key: $$COINGECKO_API_KEY" \
		> data/uni.json


ingest-swaps:
	echo "Ingesting swaps data for July 2025..."
	python src/ingest.py \
		--chain ethereum \
		--start-date 2025-07-01 \
		--end-date 2025-08-01 \
		--output-dir data/swaps/ \
		--verbose 
