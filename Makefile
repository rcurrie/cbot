ingest-pools:
	echo "Updating pools.json..."	
	curl --compressed -H 'Accept: application/json' 'https://reference-data-api.kaiko.io/v1/pools' > data/pools.json

ingest-swaps:
	python src/ingest.py \
		--chain ethereum \
		--start-date 2025-07-01 \
		--end-date 2025-08-01 \
		--output-dir data/swaps/ \
		--verbose 