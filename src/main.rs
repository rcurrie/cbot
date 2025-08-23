use ethers::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::env;
use std::str::FromStr;
use std::sync::Arc;

// Generate Uniswap V3 Swap event ABI
abigen!(
    UniswapV3Pool,
    r#"[
        event Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
    ]"#
);

async fn fetch_historical_swaps(
    pool_address: &str,
    start_block: u64,
    end_block: u64,
) -> Result<Vec<SwapFilter>, Box<dyn std::error::Error>> {
    // Initialize Alchemy provider
    let alchemy_api_key = env::var("ALCHEMY_API_KEY").expect("ALCHEMY_API_KEY not set");
    let provider = Provider::<Http>::try_from(format!(
        "https://eth-mainnet.g.alchemy.com/v2/{}",
        alchemy_api_key
    ))?;
    let client = Arc::new(provider);

    // Define Swap event topic
    let swap_topic =
        H256::from_str("0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67")?;

    // Create filter for Uniswap V3 pool swaps
    let filter = Filter::new()
        .address(pool_address.parse::<Address>()?)
        .from_block(start_block)
        .to_block(end_block)
        .topic0(swap_topic);

    // Fetch logs
    let logs = client.get_logs(&filter).await?;

    // Convert logs to Swap events
    let swap_events: Vec<SwapFilter> = logs
        .into_iter()
        .filter_map(|log| {
            <SwapFilter as ethers::contract::EthLogDecode>::decode_log(&log.into()).ok()
        })
        .collect();

    Ok(swap_events)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();

    // Uniswap V3 ETH/USDC pool (0.3% fee, mainnet)
    let pool_address = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"; // ETH/USDC 0.3%

    // Block range (approx. Jan 2025 to Aug 2025; adjust as needed)
    let start_block = 0x11edd80; // Rough estimate, use Alchemy to map timestamps
    let end_block = 0x11edf73;

    // Fetch historical swaps
    let swaps = fetch_historical_swaps(pool_address, start_block, end_block).await?;

    // Process and print swaps
    for swap in &swaps {
        // Convert sqrtPriceX96 to ETH/USDC price
        // Formula: price = (sqrtPriceX96^2 / 2^192) * (10^6 / 10^18) for USDC/ETH (6 vs 18 decimals)
        let sqrt_price_x96: U256 = swap.sqrt_price_x96.into();
        let price = (sqrt_price_x96 * sqrt_price_x96 * U256::from(10).pow(U256::from(6)))
            / (U256::from(2).pow(U256::from(192)) * U256::from(10).pow(U256::from(18)));
        let price_decimal: Decimal =
            Decimal::from_str(&price.to_string()).unwrap_or(Decimal::ZERO) / dec!(1_000_000); // Adjust for decimals

        println!(
            "Swap: amount0={} ({}), amount1={} (USDC), price={} ETH/USDC, tick={}",
            swap.amount_0,
            if swap.amount_0 > I256::zero() {
                "ETH in"
            } else {
                "ETH out"
            },
            swap.amount_1,
            price_decimal,
            swap.tick
        );
    }

    println!("Total swaps fetched: {}", swaps.len());
    Ok(())
}
