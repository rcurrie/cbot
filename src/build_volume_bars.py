"""Build WETH volume bars with a fixed 4-hour target volume.

Load the phase_1 price history, derive per-token target volumes, and
construct chronological volume bars with OHLC pricing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import polars as pl

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


@dataclass(slots=True)
class VolumeBar:
    """Represents a single volume bar for a token."""

    token_address: str
    bar_start: datetime
    bar_end: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    realized_volume: float


FOUR_HOURS = "4h"
FLOAT_EPSILON = 1e-12


class VolumeBarError(Exception):
    """Base exception for volume bar construction."""


class MissingColumnsError(VolumeBarError):
    """Raised when the input parquet is missing required columns."""

    def __init__(self, missing: list[str]) -> None:
        """Build the exception message from the missing column names."""
        missing_sorted = ", ".join(sorted(missing))
        super().__init__(f"Missing expected columns: {missing_sorted}")


class NoTargetVolumesError(VolumeBarError):
    """Raised when no positive target volumes can be derived."""

    def __init__(self) -> None:
        """Initialize with a descriptive message for missing target volumes."""
        super().__init__("No positive target volumes available; inspect source data.")


class InvalidTimestampError(VolumeBarError):
    """Raised when a timestamp is not a datetime instance."""

    def __init__(self, value: object) -> None:
        """Include the offending value type in the error message."""
        super().__init__(f"block_timestamp must be datetime, received {type(value)!r}")


class NoVolumeBarsError(VolumeBarError):
    """Raised when bar construction yields no output."""

    def __init__(self) -> None:
        """Provide guidance when no bars are produced."""
        super().__init__(
            "No volume bars were generated; verify input data and targets.",
        )


def load_price_timeseries(input_file: Path) -> pl.DataFrame:
    """Load and normalize the price time series parquet file."""
    frame = pl.read_parquet(input_file)

    expected_columns = {
        "block_timestamp",
        "token_address",
        "price_in_weth",
        "weth_volume",
    }
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise MissingColumnsError(list(missing))

    return frame.with_columns(
        [
            pl.col("block_timestamp").cast(pl.Datetime(time_unit="us")),
            pl.col("token_address").str.to_lowercase(),
            pl.col("price_in_weth").cast(pl.Float64),
            pl.col("weth_volume").cast(pl.Float64),
        ],
    ).sort(["token_address", "block_timestamp"])


def compute_target_volumes(df: pl.DataFrame) -> dict[str, float]:
    """Compute median 4-hour WETH volume per token."""
    logger.info("Computing 4-hour rolling volume medians...")

    aggregated = (
        df.group_by_dynamic(
            index_column="block_timestamp",
            every=FOUR_HOURS,
            period=FOUR_HOURS,
            group_by="token_address",
        )
        .agg(pl.col("weth_volume").sum().alias("volume_4h"))
        .filter(pl.col("volume_4h") > 0)
    )

    medians = (
        aggregated.group_by("token_address")
        .agg(pl.col("volume_4h").median().alias("target_volume"))
        .filter(pl.col("target_volume") > 0)
    )

    target_map = dict(
        zip(
            medians["token_address"].to_list(),
            medians["target_volume"].to_list(),
            strict=True,
        ),
    )

    if not target_map:
        raise NoTargetVolumesError

    logger.info("Computed targets for %d tokens", len(target_map))
    return target_map


def _build_volume_bars_for_token(  # noqa: PLR0915
    token_address: str,
    df_token: pl.DataFrame,
    target_volume: float,
) -> list[VolumeBar]:
    """Construct volume bars for a single token."""
    if target_volume <= FLOAT_EPSILON:
        logger.warning(
            "Skipping token %s due to non-positive target volume %.6e",
            token_address,
            target_volume,
        )
        return []

    bars: list[VolumeBar] = []
    current_volume = 0.0
    bar_start: datetime | None = None
    bar_end: datetime | None = None
    bar_open: float | None = None
    bar_high: float | None = None
    bar_low: float | None = None
    bar_close: float | None = None

    def finalize_bar(realized_volume: float) -> None:
        nonlocal bar_start
        nonlocal bar_end
        nonlocal bar_open
        nonlocal bar_high
        nonlocal bar_low
        nonlocal bar_close
        nonlocal current_volume
        assert bar_start is not None
        assert bar_end is not None
        assert bar_open is not None
        assert bar_high is not None
        assert bar_low is not None
        assert bar_close is not None
        bars.append(
            VolumeBar(
                token_address=token_address,
                bar_start=bar_start,
                bar_end=bar_end,
                open_price=bar_open,
                high_price=bar_high,
                low_price=bar_low,
                close_price=bar_close,
                realized_volume=realized_volume,
            ),
        )
        current_volume = 0.0
        bar_start = None
        bar_end = None
        bar_open = None
        bar_high = None
        bar_low = None
        bar_close = None

    sorted_rows = df_token.select(
        [
            pl.col("block_timestamp"),
            pl.col("price_in_weth"),
            pl.col("weth_volume"),
        ],
    )

    for row in sorted_rows.iter_rows(named=True):
        timestamp = row["block_timestamp"]
        price = float(row["price_in_weth"])
        volume_remaining = float(row["weth_volume"])

        if volume_remaining <= FLOAT_EPSILON:
            continue

        if not isinstance(timestamp, datetime):
            raise InvalidTimestampError(timestamp)

        while volume_remaining > FLOAT_EPSILON:
            if bar_start is None:
                bar_start = timestamp
                bar_open = price
                bar_high = price
                bar_low = price
                current_volume = 0.0

            assert bar_open is not None
            assert bar_high is not None
            assert bar_low is not None

            needed = max(target_volume - current_volume, FLOAT_EPSILON)
            take = min(volume_remaining, needed)
            current_volume += take
            volume_remaining -= take

            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            bar_close = price
            bar_end = timestamp

            if current_volume + FLOAT_EPSILON >= target_volume:
                finalize_bar(target_volume)

    if bar_start is not None and current_volume > FLOAT_EPSILON:
        finalize_bar(current_volume)

    return bars


def build_volume_bars(
    df_prices: pl.DataFrame,
    target_volumes: dict[str, float],
    *,
    tokens: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Build volume bars for the provided token selection."""
    records: list[VolumeBar] = []

    token_iterable = tokens if tokens is not None else target_volumes.keys()
    for token_address in token_iterable:
        address_lower = token_address.lower()
        token_df = df_prices.filter(pl.col("token_address") == address_lower)
        if token_df.is_empty():
            logger.info("No trades found for token %s, skipping.", address_lower)
            continue

        target_volume = target_volumes.get(address_lower)
        if target_volume is None:
            logger.warning("No target volume for token %s, skipping.", address_lower)
            continue

        logger.info(
            "Building volume bars for %s with target %.6f WETH...",
            address_lower,
            target_volume,
        )
        records.extend(
            _build_volume_bars_for_token(address_lower, token_df, target_volume),
        )

    if not records:
        raise NoVolumeBarsError

    return (
        pl.DataFrame(
            {
                "token_address": [bar.token_address for bar in records],
                "bar_start": [bar.bar_start for bar in records],
                "bar_end": [bar.bar_end for bar in records],
                "open_price": [bar.open_price for bar in records],
                "high_price": [bar.high_price for bar in records],
                "low_price": [bar.low_price for bar in records],
                "close_price": [bar.close_price for bar in records],
                "realized_volume": [bar.realized_volume for bar in records],
            },
        )
        .with_columns(
            [
                pl.col("bar_start").cast(pl.Datetime(time_unit="us")),
                pl.col("bar_end").cast(pl.Datetime(time_unit="us")),
                pl.col("open_price").cast(pl.Float64),
                pl.col("high_price").cast(pl.Float64),
                pl.col("low_price").cast(pl.Float64),
                pl.col("close_price").cast(pl.Float64),
                pl.col("realized_volume").cast(pl.Float64),
            ],
        )
        .sort(["token_address", "bar_start"])
    )


def log_validation_stats(bars_df: pl.DataFrame) -> None:
    """Log descriptive stats for representative validation tokens."""
    for symbol, address in VALIDATION_TOKEN_ADDRESSES.items():
        token_bars = bars_df.filter(pl.col("token_address") == address.lower())
        if token_bars.is_empty():
            logger.info("No volume bars generated for %s (%s)", symbol, address)
            continue

        daily_counts = (
            token_bars.with_columns(
                pl.col("bar_start").dt.truncate("1d").alias("bar_day"),
            )
            .group_by("bar_day")
            .agg(pl.len().alias("bars"))
        )

        mean_bars = float(daily_counts["bars"].mean())
        median_bars = float(daily_counts["bars"].median())

        logger.info(
            "%s: %d bars across %d days (mean %.2f, median %.2f per day)",
            symbol,
            token_bars.height,
            daily_counts.height,
            mean_bars,
            median_bars,
        )


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/weth_prices_timeseries.parquet"),
    show_default=True,
    help="Input parquet file containing WETH price time series.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/weth_volume_bars.parquet"),
    show_default=True,
    help="Output parquet file for generated volume bars.",
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
    tokens: tuple[str, ...],
    *,
    verbose: bool,
) -> None:
    """Entry point for building volume bars."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading price time series from %s", input_file)
    prices_df = load_price_timeseries(input_file)

    target_volumes = compute_target_volumes(prices_df)

    subset = list(tokens) if tokens else None
    bars_df = build_volume_bars(prices_df, target_volumes, tokens=subset)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    bars_df.write_parquet(output_file)
    logger.info("Wrote %d volume bars to %s", bars_df.height, output_file)

    log_validation_stats(bars_df)


if __name__ == "__main__":
    main()
