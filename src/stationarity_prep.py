"""Stationarity preparation for ARIMA modeling.

Load volume bars, compute log prices and log returns, run stationarity
diagnostics, and persist transformed data for downstream modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import polars as pl
from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

VALIDATION_TOKEN_ADDRESSES: dict[str, str] = {
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "WBTC": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "DAI": "0x6b175474e89094c44da98b954eedeac495271d0f",
    "LINK": "0x514910771af9ca656af840dff83e8264ecf986ca",
}


@dataclass(slots=True, frozen=True)
class ADFResult:
    """Results from Augmented Dickey-Fuller test."""

    token_address: str
    adf_statistic: float
    p_value: float
    n_lags_used: int
    n_observations: int
    critical_value_1pct: float
    critical_value_5pct: float
    critical_value_10pct: float
    is_stationary: bool


class StationarityPrepError(Exception):
    """Base exception for stationarity preparation."""


class MissingColumnsError(StationarityPrepError):
    """Raised when the input parquet is missing required columns."""

    def __init__(self, missing: list[str]) -> None:
        """Build the exception message from the missing column names."""
        missing_sorted = ", ".join(sorted(missing))
        super().__init__(f"Missing expected columns: {missing_sorted}")


class NoDataError(StationarityPrepError):
    """Raised when no valid data remains after transformation."""

    def __init__(self) -> None:
        """Initialize with a descriptive message for no data."""
        super().__init__("No valid data after transformation; inspect source data.")


def load_volume_bars(input_file: Path) -> pl.DataFrame:
    """Load and validate the volume bars parquet file."""
    frame = pl.read_parquet(input_file)

    expected_columns = {
        "token_address",
        "bar_start",
        "bar_end",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "realized_volume",
    }
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise MissingColumnsError(list(missing))

    return frame.with_columns(
        [
            pl.col("token_address").str.to_lowercase(),
            pl.col("bar_start").cast(pl.Datetime(time_unit="us")),
            pl.col("bar_end").cast(pl.Datetime(time_unit="us")),
            pl.col("open_price").cast(pl.Float64),
            pl.col("high_price").cast(pl.Float64),
            pl.col("low_price").cast(pl.Float64),
            pl.col("close_price").cast(pl.Float64),
            pl.col("realized_volume").cast(pl.Float64),
        ],
    ).sort(["token_address", "bar_start"])


def compute_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Compute log prices and first-differenced log returns per token.

    Args:
        df: DataFrame with volume bars including close_price column.

    Returns:
        DataFrame with added log_price and log_return columns.

    """
    logger.info("Computing log prices and log returns...")

    # Compute log price (natural log of close price)
    df_with_log = df.with_columns(
        pl.col("close_price").log().alias("log_price"),
    )

    # Compute log return as first difference of log price within each token
    df_with_returns = df_with_log.with_columns(
        pl.col("log_price")
        .diff()
        .over("token_address")
        .alias("log_return"),
    )

    # Filter out rows with null log_return (first bar per token)
    # and any infinite or NaN values
    clean_df = df_with_returns.filter(
        pl.col("log_return").is_not_null()
        & pl.col("log_return").is_finite()
        & pl.col("log_price").is_not_null()
        & pl.col("log_price").is_finite(),
    )

    logger.info(
        "Computed log returns for %d bars (%d original)",
        clean_df.height,
        df.height,
    )

    return clean_df


def run_adf_test(
    df: pl.DataFrame,
    tokens: Iterable[str] | None = None,
    *,
    alpha: float = 0.05,
) -> list[ADFResult]:
    """Run Augmented Dickey-Fuller test for each token's log return series.

    Args:
        df: DataFrame with log_return column.
        tokens: Optional subset of tokens to test. If None, test all tokens.
        alpha: Significance level for stationarity determination (default 0.05).

    Returns:
        List of ADFResult objects with test statistics and p-values.

    """
    logger.info("Running Augmented Dickey-Fuller tests...")

    results: list[ADFResult] = []
    token_list = (
        list(tokens) if tokens else df["token_address"].unique().to_list()
    )

    for token_address in token_list:
        token_df = df.filter(pl.col("token_address") == token_address.lower())

        if token_df.is_empty():
            logger.warning("No data for token %s, skipping ADF test.", token_address)
            continue

        log_returns = token_df["log_return"].to_numpy()

        # Need at least 12 observations for meaningful ADF test
        if len(log_returns) < 12:  # noqa: PLR2004
            logger.warning(
                "Insufficient data for token %s (%d bars), skipping ADF test.",
                token_address,
                len(log_returns),
            )
            continue

        # Run ADF test with automatic lag selection
        adf_stat, p_value, n_lags, n_obs, critical_values, _ = adfuller(
            log_returns,
            autolag="AIC",
        )

        is_stationary = p_value < alpha

        results.append(
            ADFResult(
                token_address=token_address.lower(),
                adf_statistic=float(adf_stat),
                p_value=float(p_value),
                n_lags_used=int(n_lags),
                n_observations=int(n_obs),
                critical_value_1pct=float(critical_values["1%"]),
                critical_value_5pct=float(critical_values["5%"]),
                critical_value_10pct=float(critical_values["10%"]),
                is_stationary=is_stationary,
            ),
        )

        status = "STATIONARY" if is_stationary else "NON-STATIONARY"
        logger.info(
            "Token %s: ADF=%.4f, p-value=%.4f, lags=%d, status=%s",
            token_address,
            adf_stat,
            p_value,
            n_lags,
            status,
        )

    return results


def log_adf_summary(results: list[ADFResult]) -> None:
    """Log summary statistics for ADF test results."""
    if not results:
        logger.warning("No ADF test results to summarize.")
        return

    total = len(results)
    stationary_count = sum(1 for r in results if r.is_stationary)
    non_stationary_count = total - stationary_count

    logger.info(
        "ADF Summary: %d/%d tokens stationary (%.1f%%)",
        stationary_count,
        total,
        100.0 * stationary_count / total if total > 0 else 0.0,
    )
    logger.info("Non-stationary tokens: %d", non_stationary_count)

    # Log validation tokens
    validation_addresses = {
        addr.lower() for addr in VALIDATION_TOKEN_ADDRESSES.values()
    }
    for result in results:
        if result.token_address in validation_addresses:
            symbol = next(
                (
                    sym
                    for sym, addr in VALIDATION_TOKEN_ADDRESSES.items()
                    if addr.lower() == result.token_address
                ),
                "UNKNOWN",
            )
            status = "STATIONARY" if result.is_stationary else "NON-STATIONARY"
            logger.info(
                "%s: ADF=%.4f, p=%.4f, %s",
                symbol,
                result.adf_statistic,
                result.p_value,
                status,
            )


def save_adf_results(results: list[ADFResult], output_file: Path) -> None:
    """Save ADF test results to a parquet file."""
    if not results:
        logger.warning("No ADF results to save.")
        return

    results_df = pl.DataFrame(
        {
            "token_address": [r.token_address for r in results],
            "adf_statistic": [r.adf_statistic for r in results],
            "p_value": [r.p_value for r in results],
            "n_lags_used": [r.n_lags_used for r in results],
            "n_observations": [r.n_observations for r in results],
            "critical_value_1pct": [r.critical_value_1pct for r in results],
            "critical_value_5pct": [r.critical_value_5pct for r in results],
            "critical_value_10pct": [r.critical_value_10pct for r in results],
        },
    ).with_columns(
        pl.Series(
            "is_stationary",
            [r.is_stationary for r in results],
            dtype=pl.Boolean,
        ),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(output_file)
    logger.info("Saved ADF results to %s", output_file)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_volume_bars.parquet"),
    show_default=True,
    help="Input parquet file containing volume bars.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/weth_volume_bars_log_returns.parquet"),
    show_default=True,
    help="Output parquet file for log returns.",
)
@click.option(
    "--adf-results-file",
    type=click.Path(path_type=Path),
    default=Path("data/adf_test_results.parquet"),
    show_default=True,
    help="Output parquet file for ADF test results.",
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
def main(
    input_file: Path,
    output_file: Path,
    adf_results_file: Path,
    tokens: tuple[str, ...],
    *,
    verbose: bool,
) -> None:
    """Entry point for stationarity preparation."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading volume bars from %s", input_file)
    bars_df = load_volume_bars(input_file)

    # Filter to specific tokens if requested
    if tokens:
        token_set = {t.lower() for t in tokens}
        bars_df = bars_df.filter(pl.col("token_address").is_in(token_set))
        logger.info("Filtered to %d tokens", len(token_set))

    # Compute log prices and returns
    log_returns_df = compute_log_returns(bars_df)

    if log_returns_df.is_empty():
        raise NoDataError

    # Run ADF tests
    adf_results = run_adf_test(log_returns_df, tokens=tokens)
    log_adf_summary(adf_results)

    # Save transformed data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_returns_df.write_parquet(output_file)
    logger.info("Wrote %d log return bars to %s", log_returns_df.height, output_file)

    # Save ADF results
    save_adf_results(adf_results, adf_results_file)

    # Log missing value check
    null_count = log_returns_df.null_count().sum_horizontal()[0]
    logger.info("Missing values in output: %d", null_count)


if __name__ == "__main__":
    main()
