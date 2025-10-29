I'd like to develop a crypto trading bot that accumulates WETH by predicting token's that will appreciate against WETH over hours to a day, entering long positions in those tokens and then swapping back using a DEX to WETH at the end of the period. The long term goal is to accumulate WETH so this is a purely on-chain using only signals and data from the chain and dex pools. 

Initially we'll limit to Uniswap v3 pools on Etherium main net. Training data includes logs of all uniswap v3 swaps in parquet files downloaded from google big query as well as a list of all pools and their associated token addresses and symbols. I'd like to start by transforming the data into a format that could be used to develop a baseline (say ARIMA or logistic regression) and then level up from there. I have 2 months downloaded of raw uniswap v3 swap events and a list of all pools and their associated tokens. We should only focus on tokens that have a pools that can be directly or indirectly swaped to WETH, so we'll need to filter tokens that are in 'islands' on the chain isolated from WETH (Etherium Main Net has 4 islands of uniswap v3 and only one of them, the biggest, has WETH)

My initial thinking is to model this as a uni-partite graph with token's as nodes and swaps as directed edges ordered in time but maybe its better to do a bi-partite with token and pool nodes. 

As a related example end model take a look at this blog post that uses Temporal Graph Neural Networks for Multi-Product Time Series Forecasting:

https://pub.towardsai.net/temporal-graph-neural-networks-for-multi-product-time-series-forecasting-f4cc87f8354c

Initialy let's debate the graph structure, maybe starting on the simple side and then a basic data tranforms (ie normalizing prices against WETH, normalizing tokens that are high volume vs. low volume, whale transactions etc...) followed by a baseline model implementation.
