"""Train ARIMA models on volume bar log returns for all tokens.

This script implements Phase 2 Milestone 3:
- Load weth_volume_bars_log_returns.parquet
- Train per-token ARIMA models
- Generate forecasts with confidence intervals and directional probabilities
- Serialize model artifacts and forecasts

Usage:
  python src/train_arima_models.py                           # Train on 5 representative tokens
  python src/train_arima_models.py --all-tokens              # Train on all tokens
  python src/train_arima_models.py --tokens 0xabc... 0xdef...  # Train on specific tokens
  python src/train_arima_models.py --forecast-horizon 10     # Custom forecast horizon
  python src/train_arima_models.py --arima-order 2 1 2      # Fixed ARIMA order
"""

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "arima_models"
MODELS_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "weth_volume_bars_log_returns.parquet"
FORECASTS_FILE = DATA_DIR / "arima_forecasts.parquet"
MODEL_SUMMARY_FILE = DATA_DIR / "arima_model_summary.json"

# Representative tokens for default training (USDC, USDT, WBTC, DAI, LINK)
REPRESENTATIVE_TOKENS = [
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
    "0x514910771af9ca656af840dff83e8264ecf986ca",  # LINK
]

# Token names for display
TOKEN_NAMES = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "WBTC",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
    "0x514910771af9ca656af840dff83e8264ecf986ca": "LINK",
}

# Forecasting parameters
DEFAULT_FORECAST_HORIZON = 6  # 4-24 hours worth of volume bars
MIN_TRAIN_BARS = 50  # Minimum bars required to train a model
TRAIN_TEST_SPLIT = 0.8  # Use 80% for training, 20% for validation


def load_data() -> pl.DataFrame:
    """Load the volume bars with log returns."""
    print(f"Loading data from {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} bars for {df['token_address'].n_unique()} tokens")
    return df


def prepare_token_data(
    df: pl.DataFrame, token_address: str
) -> tuple[pl.DataFrame, int]:
    """
    Prepare data for a single token.

    Returns:
        Tuple of (token_df, train_size) where train_size is the number of bars for training
    """
    token_df = (
        df.filter(pl.col("token_address") == token_address)
        .sort("bar_end")
        .with_row_index("index")
    )

    n_bars = len(token_df)
    train_size = int(n_bars * TRAIN_TEST_SPLIT)

    return token_df, train_size


def find_best_arima_order(
    train_series: pl.Series, max_p: int = 3, max_q: int = 3
) -> tuple[int, int, int]:
    """
    Find the best ARIMA(p,1,q) order using AIC.

    We fix d=1 for first-differencing (data is already differenced but ARIMA
    will handle the integration).
    """
    best_aic = float("inf")
    best_order = (1, 1, 1)  # Default fallback

    train_data = train_series.to_numpy()

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # Skip ARIMA(0,1,0) - too simple

            try:
                model = ARIMA(train_data, order=(p, 1, q))
                fitted = model.fit(method_kwargs={"warn_convergence": False})

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 1, q)
            except Exception:
                continue

    return best_order


def train_token_model(
    token_address: str,
    token_df: pl.DataFrame,
    train_size: int,
    forecast_horizon: int,
    arima_order: tuple[int, int, int] | None = None,
) -> dict:
    """
    Train ARIMA model for a single token.

    Returns:
        Dictionary with model results and forecasts
    """
    # Split into train/test
    train_df = token_df.head(train_size)
    test_df = token_df.tail(len(token_df) - train_size)

    # Get log price series for training
    train_series = train_df.select("log_price").to_series()

    # Find best model order or use provided order
    if arima_order is None:
        print("    Finding best ARIMA order...")
        order = find_best_arima_order(train_series)
        print(f"    Selected ARIMA{order}")
    else:
        order = arima_order
        print(f"    Using fixed ARIMA order: {order}")

    # Fit the model
    train_data = train_series.to_numpy()
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit(method_kwargs={"warn_convergence": False})

    # Generate forecasts
    forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI

    # Calculate forecast metrics
    forecast_std = forecast_result.se_mean

    # Calculate directional probabilities
    # Probability of positive movement = P(forecast > last_price)
    last_log_price = float(train_series[-1])
    directional_probs = []
    for i in range(forecast_horizon):
        # Z-score for no change (forecast equals last price)
        z_score = (last_log_price - forecast_mean[i]) / forecast_std[i]
        # Probability of price increase (forecast > last_price)
        prob_increase = float(stats.norm.cdf(-z_score))
        directional_probs.append(prob_increase)

    # Store model parameters
    model_info = {
        "token_address": token_address,
        "order": order,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "n_train_bars": train_size,
        "n_test_bars": len(test_df),
        "last_train_log_price": float(train_series[-1]),
        "train_end_timestamp": str(train_df["bar_end"][-1]),
    }

    # Prepare forecast data
    forecasts = []
    for i in range(forecast_horizon):
        forecasts.append({
            "token_address": token_address,
            "forecast_step": i + 1,
            "forecast_log_price": float(forecast_mean[i]),
            "forecast_lower_95": float(forecast_conf_int[i, 0]),
            "forecast_upper_95": float(forecast_conf_int[i, 1]),
            "forecast_std": float(forecast_std[i]),
            "prob_increase": directional_probs[i],
            "prob_decrease": 1 - directional_probs[i],
        })

    # Save model to disk
    model_path = MODELS_DIR / f"{token_address}.pkl"
    fitted_model.save(str(model_path))
    model_info["model_path"] = str(model_path)

    return {
        "model_info": model_info,
        "forecasts": forecasts,
    }


def compute_baseline_forecast(
    token_df: pl.DataFrame, train_size: int, forecast_horizon: int
) -> list[dict]:
    """
    Generate random walk baseline forecast.

    Uses recent volatility to generate Gaussian random walk predictions.
    """
    train_df = token_df.head(train_size)

    # Calculate recent log return volatility (last 20 bars)
    recent_returns = train_df.tail(20).select("log_return").to_series()
    volatility = float(recent_returns.std())

    last_log_price = float(train_df.select("log_price").to_series()[-1])

    # Generate random walk forecasts
    # For baseline, we assume no drift (expected return = 0)
    # Variance increases with sqrt(steps) for random walk
    forecasts = []
    for step in range(1, forecast_horizon + 1):
        step_std = volatility * (step**0.5)

        forecasts.append({
            "token_address": token_df["token_address"][0],
            "forecast_step": step,
            "baseline_log_price": last_log_price,
            "baseline_std": step_std,
            "baseline_lower_95": last_log_price - 1.96 * step_std,
            "baseline_upper_95": last_log_price + 1.96 * step_std,
        })

    return forecasts


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ARIMA models on token volume bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Train on 5 representative tokens
  %(prog)s --all-tokens                 # Train on all tokens
  %(prog)s --tokens 0xabc... 0xdef...   # Train on specific tokens
  %(prog)s --forecast-horizon 10        # 10-bar forecast horizon
  %(prog)s --arima-order 2 1 2          # Fixed ARIMA(2,1,2)
        """,
    )

    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument(
        "--all-tokens",
        action="store_true",
        help="Train on all tokens (default: 5 representative tokens)",
    )
    token_group.add_argument(
        "--tokens",
        nargs="+",
        metavar="ADDRESS",
        help="Train on specific token addresses",
    )

    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON,
        metavar="N",
        help=f"Forecast N volume bars ahead (default: {DEFAULT_FORECAST_HORIZON})",
    )

    parser.add_argument(
        "--arima-order",
        type=int,
        nargs=3,
        metavar=("P", "D", "Q"),
        help="Use fixed ARIMA(p,d,q) order instead of auto-selection",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed model diagnostics",
    )

    return parser.parse_args()


def get_token_name(token_address: str) -> str:
    """Get token name or abbreviated address."""
    return TOKEN_NAMES.get(token_address, token_address[:10])


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 80)
    print("ARIMA Model Training Pipeline - Phase 2 Milestone 3")
    print("=" * 80)

    # Load data
    df = load_data()

    # Determine which tokens to process
    if args.tokens:
        tokens_to_process = args.tokens
        print(f"\nTraining on {len(tokens_to_process)} specified tokens")
    elif args.all_tokens:
        tokens_to_process = df.select("token_address").unique().to_series().to_list()
        print(f"\nTraining on ALL {len(tokens_to_process)} tokens")
    else:
        tokens_to_process = REPRESENTATIVE_TOKENS
        print("\nTraining on 5 representative tokens (USDC, USDT, WBTC, DAI, LINK)")
        print("Use --all-tokens to train on all tokens")

    # Filter tokens with sufficient data
    bars_per_token = (
        df.group_by("token_address")
        .agg(pl.len().alias("n_bars"))
        .filter(pl.col("n_bars") >= MIN_TRAIN_BARS)
    )
    valid_tokens = set(bars_per_token.select("token_address").to_series().to_list())

    # Filter requested tokens by validity
    tokens_to_train = [t for t in tokens_to_process if t in valid_tokens]

    if len(tokens_to_train) < len(tokens_to_process):
        skipped = len(tokens_to_process) - len(tokens_to_train)
        print(
            f"Skipping {skipped} tokens with insufficient data "
            f"(< {MIN_TRAIN_BARS} bars)",
        )

    print(f"Training {len(tokens_to_train)} tokens")
    print(f"Forecast horizon: {args.forecast_horizon} bars")

    if args.arima_order:
        print(f"Using fixed ARIMA order: {tuple(args.arima_order)}")
    else:
        print("Using automatic ARIMA order selection")

    # Train models for each token
    all_model_info = []
    all_forecasts = []
    all_baselines = []

    for i, token_address in enumerate(tokens_to_train, 1):
        token_name = get_token_name(token_address)
        print(f"\n[{i}/{len(tokens_to_train)}] Processing {token_name}...")

        try:
            # Prepare data
            token_df, train_size = prepare_token_data(df, token_address)

            # Train ARIMA model
            arima_order = tuple(args.arima_order) if args.arima_order else None
            result = train_token_model(
                token_address,
                token_df,
                train_size,
                args.forecast_horizon,
                arima_order,
            )
            all_model_info.append(result["model_info"])
            all_forecasts.extend(result["forecasts"])

            # Generate baseline forecast
            baseline_forecasts = compute_baseline_forecast(
                token_df,
                train_size,
                args.forecast_horizon,
            )
            all_baselines.extend(baseline_forecasts)

            # Print summary
            model_info = result["model_info"]
            print(f"    Order: ARIMA{model_info['order']}")
            print(f"    AIC: {model_info['aic']:.2f}, BIC: {model_info['bic']:.2f}")
            print(f"    Training bars: {model_info['n_train_bars']}")

            if args.verbose:
                # Show forecast summary
                first_forecast = result["forecasts"][0]
                print(
                    f"    First forecast: {first_forecast['forecast_log_price']:.4f} "
                    f"[{first_forecast['forecast_lower_95']:.4f}, "
                    f"{first_forecast['forecast_upper_95']:.4f}]",
                )
                print(
                    f"    Prob(increase): {first_forecast['prob_increase']:.1%}, "
                    f"Prob(decrease): {first_forecast['prob_decrease']:.1%}",
                )

        except Exception as e:
            print(f"    ERROR: {str(e)}")
            if args.verbose:
                traceback.print_exc()
            continue

    # Save forecasts
    print(f"\n{'=' * 80}")
    print("Saving results...")

    if all_forecasts:
        # Combine ARIMA and baseline forecasts
        arima_df = pl.DataFrame(all_forecasts)
        baseline_df = pl.DataFrame(all_baselines)

        # Join on token_address and forecast_step
        forecasts_df = arima_df.join(
            baseline_df,
            on=["token_address", "forecast_step"],
            how="left",
        )

        forecasts_df.write_parquet(FORECASTS_FILE)
        print(f"Saved forecasts to {FORECASTS_FILE}")

        # Save model summary
        MODEL_SUMMARY_FILE.write_text(
            json.dumps(all_model_info, indent=2),
            encoding="utf-8",
        )
        print(f"Saved model summary to {MODEL_SUMMARY_FILE}")

        print(f"\n{'=' * 80}")
        print("Training complete!")
        print(f"Models trained: {len(all_model_info)}")
        print(f"Total forecasts: {len(all_forecasts)}")
        print(f"{'=' * 80}")
    else:
        print("ERROR: No models were successfully trained!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
