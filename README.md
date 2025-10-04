# CBOT

## Install

See [How to Install PyTorch Geometric with Apple Silicon Support](https://medium.com/@dessi.georgieva8/how-to-install-pytorch-geometric-with-apple-silicon-support-m1-m2-m3-39f1a5ad33b6) about installing Py

## Running

```
# Update list of top uniswap pools on arbitrum via coin gecko
python src/ingest-uniswap-arbitrum-coingecko.py --max-pools 200

# Get raw transactions using demeter-fetch from bigquery for USCD/WETH
demeter-fetch -c demeter-fetch-uscd-weth-config.toml
```

## Grok Chats

[Continuous-Time Temporal Graph Neural Networks been applied to DeFi for Trading](https://grok.com/share/bGVnYWN5_a5a49b8f-c662-46d0-aeb1-3b25d28468e8)

## Data Sources

[Google Big Query Crypt Supported Datasets](https://cloud.google.com/blockchain-analytics/docs/supported-datasets)

[Uniswap v4 Deployment Addresses](https://docs.uniswap.org/contracts/v4/deployments)

[Google Cloud Big Query Blockchain Analytics Dataset Schemas](https://cloud.google.com/blockchain-analytics/docs/schema)

[CoinGecko API](https://www.geckoterminal.com/dex-api)

[Demeter Fetch for Block Index](https://github.com/zelos-alpha/demeter-fetch?tab=readme-ov-file)

[How to Track liquidity for token pairs on Uniswap](https://bitquery.io/blog/how-to-track-liquidity-for-token-pair-on-uniswap)

[Query The Graph w/Python and Subgrounds](https://thegraph.com/docs/en/subgraphs/querying/python/)

[Cost Effective Uniswap Transaction Download Grok Chat](https://grok.com/share/bGVnYWN5_4cbd326d-2d56-49f8-b97b-5973fc17f3b3)

[BigQuery Blockchain Datasets](https://cloud.google.com/blockchain-analytics/docs/supported-datasets)

## References

[Exploring the Public Cryptocurrency Datasets Available in BigQuery](https://www.cloudskillsboost.google/focuses/8486?parent=catalog)

## Prior Art

[Investigating Similarities Across Decentralized Finance (DeFi) Services](https://github.com/JunLLuo/DeFi-similarity)

[Forecasting cryptocurrencies’ price with the financial stress index: a graph neural network prediction strategy](https://www.tandfonline.com/doi/full/10.1080/13504851.2022.2141436#abstract)

[TGLite: A Lightweight Programming Framework for Continuous-Time Temporal Graph Neural Networks](https://charithmendis.com/assets/pdf/asplos24-tglite.pdf)

## Other

[Uniswap Pools MCP Server](https://github.com/kukapay/uniswap-pools-mcp)

[Uniswap Pool Addresses REST Endpoint](https://docs.kaiko.com/rest-api/data-feeds/reference-data/free-tier/on-chain-pools)
