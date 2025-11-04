"""Train baseline ARIMA(0,1,0) model on log price series.

Fits ARIMA(0,1,0) model per token, generates out-of-sample forecasts for
4-hour horizon, and serializes model parameters and diagnostics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import polars as pl
from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore[import-untyped]
from statsmodels.tsa.arima.model import (  # type: ignore[import-untyped]
    ARIMA,
    ARIMAResults,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

VALIDATION_TOKEN_ADDRESSES: dict[str, str] = {
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "WBTC": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "LINK": "0x514910771af9ca656af840dff83e8264ecf986ca",
}


@dataclass(slots=True, frozen=True)
class ARIMADiagnostics:
    """Model diagnostics for fitted ARIMA model."""

    token_address: str
    n_observations: int
    aic: float
    bic: float
    log_likelihood: float
    residual_mean: float
    residual_std: float
    ljung_box_statistic: float
    ljung_box_p_value: float


@dataclass(slots=True, frozen=True)
class ARIMAForecast:
    """Forecast result for a single token."""

    token_address: str
    forecast_start_bar: int
    forecast_values: list[float]
    forecast_steps: int


class ARIMATrainingError(Exception):
    """Base exception for ARIMA training."""


class InsufficientDataError(ARIMATrainingError):
    """Raised when insufficient data for model fitting."""

    def __init__(self, token_address: str, n_bars: int, min_required: int) -> None:
        """Initialize with token and data count."""
        super().__init__(
            f"Token {token_address} has {n_bars} bars, need at least {min_required}",
        )


def load_log_returns(input_file: Path) -> pl.DataFrame:
    """Load and validate the log returns parquet file."""
    logger.info("Loading log returns from %s", input_file)
    frame = pl.read_parquet(input_file)

    expected_columns = {"token_address", "bar_start", "log_price", "log_return"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        msg = f"Missing expected columns: {', '.join(sorted(missing))}"
        raise ARIMATrainingError(msg)

    return frame.with_columns(
        [
            pl.col("token_address").str.to_lowercase(),
            pl.col("bar_start").cast(pl.Datetime(time_unit="us")),
            pl.col("log_price").cast(pl.Float64),
        ],
    ).sort(["token_address", "bar_start"])


def fit_arima_010(
    log_prices: NDArray[np.float64],
    token_address: str,
    *,
    min_observations: int = 24,
) -> ARIMAResults:
    """Fit ARIMA(0,1,0) model on log price series.

    Args:
        log_prices: Array of log prices for a single token.
        token_address: Token address for logging.
        min_observations: Minimum number of observations required.

    Returns:
        Fitted ARIMA model results.

    Raises:
        InsufficientDataError: If insufficient observations for fitting.

    """
    n_obs = len(log_prices)
    if n_obs < min_observations:
        raise InsufficientDataError(token_address, n_obs, min_observations)

    logger.info("Fitting ARIMA(0,1,0) for token %s with %d bars", token_address, n_obs)

    # ARIMA(0,1,0) is equivalent to a random walk model
    model = ARIMA(log_prices, order=(0, 1, 0))
    fitted = model.fit()

    logger.info(
        "Fitted ARIMA(0,1,0) for %s: AIC=%.2f, BIC=%.2f",
        token_address,
        fitted.aic,
        fitted.bic,
    )

    return fitted


def compute_diagnostics(
    fitted: ARIMAResults,
    token_address: str,
) -> ARIMADiagnostics:
    """Compute model diagnostics from fitted ARIMA model.

    Args:
        fitted: Fitted ARIMA model.
        token_address: Token address for the model.

    Returns:
        ARIMADiagnostics object with test statistics.

    """
    residuals = fitted.resid

    # Ljung-Box test for autocorrelation in residuals
    # Test up to lag 10 or 10% of data, whichever is smaller
    n_lags = min(10, max(1, len(residuals) // 10))
    ljung_box_result = acorr_ljungbox(residuals, lags=[n_lags], return_df=True)
    lb_stat = float(ljung_box_result["lb_stat"].iloc[0])
    lb_pvalue = float(ljung_box_result["lb_pvalue"].iloc[0])

    logger.info(
        "Diagnostics for %s: Residual mean=%.6f, std=%.6f, LB(lag=%d)=%.2f, p=%.4f",
        token_address,
        residuals.mean(),
        residuals.std(),
        n_lags,
        lb_stat,
        lb_pvalue,
    )

    return ARIMADiagnostics(
        token_address=token_address,
        n_observations=fitted.nobs,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        log_likelihood=float(fitted.llf),
        residual_mean=float(residuals.mean()),
        residual_std=float(residuals.std()),
        ljung_box_statistic=lb_stat,
        ljung_box_p_value=lb_pvalue,
    )


def generate_forecast(
    fitted: ARIMAResults,
    token_address: str,
    n_bars_in_sample: int,
    forecast_steps: int = 1,
) -> ARIMAForecast:
    """Generate out-of-sample forecast.

    Args:
        fitted: Fitted ARIMA model.
        token_address: Token address for the forecast.
        n_bars_in_sample: Number of bars used for training.
        forecast_steps: Number of steps ahead to forecast.

    Returns:
        ARIMAForecast object with predictions.

    """
    forecast = fitted.forecast(steps=forecast_steps)
    forecast_list = forecast.tolist()

    logger.info(
        "Generated %d-step forecast for %s: %s",
        forecast_steps,
        token_address,
        forecast_list,
    )

    return ARIMAForecast(
        token_address=token_address,
        forecast_start_bar=n_bars_in_sample,
        forecast_values=forecast_list,
        forecast_steps=forecast_steps,
    )


def save_model_parameters(
    diagnostics_list: list[ARIMADiagnostics],
    output_file: Path,
) -> None:
    """Save model parameters and diagnostics to JSON.

    Args:
        diagnostics_list: List of diagnostics for all fitted models.
        output_file: Path to output JSON file.

    """
    output_data = {
        "model_spec": {
            "order": [0, 1, 0],
            "description": "Random walk baseline (ARIMA(0,1,0))",
        },
        "diagnostics": [
            {
                "token_address": d.token_address,
                "n_observations": d.n_observations,
                "aic": d.aic,
                "bic": d.bic,
                "log_likelihood": d.log_likelihood,
                "residual_mean": d.residual_mean,
                "residual_std": d.residual_std,
                "ljung_box_statistic": d.ljung_box_statistic,
                "ljung_box_p_value": d.ljung_box_p_value,
            }
            for d in diagnostics_list
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved model parameters to %s", output_file)


def save_forecasts(
    forecasts_list: list[ARIMAForecast],
    output_file: Path,
) -> None:
    """Save forecasts to parquet file.

    Args:
        forecasts_list: List of forecast objects.
        output_file: Path to output parquet file.

    """
    if not forecasts_list:
        logger.warning("No forecasts to save.")
        return

    # Flatten forecast values into rows
    rows: list[dict[str, Any]] = []
    for forecast in forecasts_list:
        for step_idx, value in enumerate(forecast.forecast_values, start=1):
            rows.append({
                "token_address": forecast.token_address,
                "forecast_start_bar": forecast.forecast_start_bar,
                "forecast_step": step_idx,
                "forecast_log_price": value,
            })

    forecasts_df = pl.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    forecasts_df.write_parquet(output_file)

    logger.info("Saved %d forecast rows to %s", len(rows), output_file)


def save_residuals(
    residuals_data: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Save residual diagnostics to parquet file.

    Args:
        residuals_data: List of residual diagnostic dictionaries.
        output_file: Path to output parquet file.

    """
    if not residuals_data:
        logger.warning("No residuals to save.")
        return

    residuals_df = pl.DataFrame(residuals_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    residuals_df.write_parquet(output_file)

    logger.info("Saved residual diagnostics to %s", output_file)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_volume_bars_log_returns.parquet"),
    show_default=True,
    help="Input parquet file containing log returns.",
)
@click.option(
    "--model-file",
    type=click.Path(path_type=Path),
    default=Path("data/arima_baseline_model.json"),
    show_default=True,
    help="Output JSON file for model parameters and diagnostics.",
)
@click.option(
    "--forecasts-file",
    type=click.Path(path_type=Path),
    default=Path("data/arima_baseline_forecasts.parquet"),
    show_default=True,
    help="Output parquet file for forecasts.",
)
@click.option(
    "--residuals-file",
    type=click.Path(path_type=Path),
    default=Path("data/arima_baseline_residuals.parquet"),
    show_default=True,
    help="Output parquet file for residual diagnostics.",
)
@click.option(
    "--forecast-steps",
    type=int,
    default=1,
    show_default=True,
    help="Number of steps ahead to forecast.",
)
@click.option(
    "--min-observations",
    type=int,
    default=24,
    show_default=True,
    help="Minimum number of bars required to fit model.",
)
@click.option(
    "--tokens",
    multiple=True,
    help="Optional subset of token addresses to process.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging output.",
)
def main(  # noqa: PLR0913

    input_file: Path,
    model_file: Path,
    forecasts_file: Path,
    residuals_file: Path,
    forecast_steps: int,
    min_observations: int,
    tokens: tuple[str, ...],
    *,
    verbose: bool,
) -> None:
    """Train ARIMA(0,1,0) baseline models and generate forecasts."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load log returns data
    log_returns_df = load_log_returns(input_file)

    # Filter to specific tokens if requested
    if tokens:
        token_set = {t.lower() for t in tokens}
        log_returns_df = log_returns_df.filter(
            pl.col("token_address").is_in(token_set),
        )
        logger.info("Filtered to %d tokens", len(token_set))

    # Get unique tokens
    unique_tokens = log_returns_df["token_address"].unique().to_list()
    logger.info("Processing %d unique tokens", len(unique_tokens))

    diagnostics_list: list[ARIMADiagnostics] = []
    forecasts_list: list[ARIMAForecast] = []
    residuals_data: list[dict[str, Any]] = []

    # Process each token
    for token_address in unique_tokens:
        token_df = log_returns_df.filter(
            pl.col("token_address") == token_address,
        ).sort("bar_start")

        # Extract log prices (include log_price column which has log returns computed)
        log_prices = token_df["log_price"].to_numpy()
        n_bars = len(log_prices)

        try:
            # Fit ARIMA(0,1,0) model
            fitted = fit_arima_010(
                log_prices,
                token_address,
                min_observations=min_observations,
            )

            # Compute diagnostics
            diagnostics = compute_diagnostics(fitted, token_address)
            diagnostics_list.append(diagnostics)

            # Generate forecast
            forecast = generate_forecast(
                fitted,
                token_address,
                n_bars,
                forecast_steps=forecast_steps,
            )
            forecasts_list.append(forecast)

            # Store residuals with bar timestamps for later analysis
            residuals = fitted.resid
            bar_starts = token_df["bar_start"].to_list()

            # Residuals start from second observation due to differencing
            for i, (bar_start, residual) in enumerate(
                zip(bar_starts[1:], residuals, strict=False),
                start=1,
            ):
                residuals_data.append({
                    "token_address": token_address,
                    "bar_index": i,
                    "bar_start": bar_start,
                    "residual": float(residual),
                })

        except InsufficientDataError as e:
            logger.warning("Skipping token due to insufficient data: %s", e)
            continue

    # Save all artifacts
    logger.info("Saving artifacts for %d tokens...", len(diagnostics_list))

    save_model_parameters(diagnostics_list, model_file)
    save_forecasts(forecasts_list, forecasts_file)
    save_residuals(residuals_data, residuals_file)

    # Log summary statistics
    logger.info(
        "Successfully fitted %d models with %d total forecast rows",
        len(diagnostics_list),
        len(forecasts_list) * forecast_steps,
    )

    # Log validation tokens if processed
    validation_addresses = {
        addr.lower() for addr in VALIDATION_TOKEN_ADDRESSES.values()
    }
    for diag in diagnostics_list:
        if diag.token_address in validation_addresses:
            symbol = next(
                (
                    sym
                    for sym, addr in VALIDATION_TOKEN_ADDRESSES.items()
                    if addr.lower() == diag.token_address
                ),
                "UNKNOWN",
            )
            logger.info(
                "%s: n=%d, AIC=%.2f, BIC=%.2f, LB_p=%.4f",
                symbol,
                diag.n_observations,
                diag.aic,
                diag.bic,
                diag.ljung_box_p_value,
            )


if __name__ == "__main__":
    main()
