"""Training Data Validation Script.

This script performs comprehensive sanity checks on the final labeled_log_fracdiff_price.parquet
dataset before it's used for TGNN training. Based on plans/data_overview.md.

Validation Categories:
1. Data Integrity: Non-null, finite values, no duplicates
2. Statistical Sanity: Range checks, distribution health
3. Feature Relationships: Correlations and logical consistency
4. Label Quality: Balance, coverage, sample weights
5. Temporal Consistency: Time ordering, no gaps
"""

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# %%
# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/labeled_log_fracdiff_price.parquet")
PLOTS_DIR = Path("plots/validation")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Expected value ranges based on data_overview.md and statistical analysis
EXPECTED_RANGES = {
    "src_flow_usdc": (-1e8, 1e8),  # Flow should be bounded (extreme outliers filtered)
    "dest_flow_usdc": (-1e5, 1e5),  # Capped at ±99999 per earlier processing
    "src_fracdiff": (-5.0, 10.0),  # Fractionally differentiated prices
    "dest_fracdiff": (-5.0, 10.0),
    "rolling_volatility": (0.0, 1.0),  # Volatility should be positive and < 100%
    "bar_time_delta_sec": (0.0, 86400.0),  # Up to 1 day seems reasonable
    "tick_count": (1, 10000),  # At least 1 swap per bar
    "label": (-1.0, 1.0),  # Triple-barrier labels: -1, 0, 1
    "sample_weight": (0.0, 1.0),  # Normalized weights
}


# %%
# Validation Functions


def check_data_integrity(df: pl.DataFrame) -> dict[str, any]:
    """Check for nulls, NaNs, infinities, and duplicates.

    Args:
        df: The labeled training dataset

    Returns:
        Dictionary with validation results

    """
    logger.info("=" * 60)
    logger.info("1. DATA INTEGRITY CHECKS")
    logger.info("=" * 60)

    results = {}

    # Shape
    logger.info(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    results["total_rows"] = df.shape[0]
    results["total_columns"] = df.shape[1]

    # Null counts for critical columns
    critical_cols = [
        "src_flow_usdc",
        "dest_flow_usdc",
        "src_fracdiff",
        "dest_fracdiff",
        "rolling_volatility",
    ]

    logger.info("\nNull/NaN counts for critical columns:")
    for col in critical_cols:
        null_count = df[col].null_count()
        nan_count = (
            df.filter(~pl.col(col).is_finite()).shape[0]
            if df.schema[col]
            in (
                pl.Float32,
                pl.Float64,
            )
            else 0
        )
        total_invalid = null_count + nan_count
        pct = (total_invalid / df.shape[0]) * 100

        status = "✓ PASS" if total_invalid == 0 else "✗ WARN"
        logger.info(f"  {col:25} {total_invalid:>10,} ({pct:>6.2f}%) {status}")

        results[f"{col}_nulls"] = null_count
        results[f"{col}_nans"] = nan_count

    # Check for duplicate rows (by timestamp + pool + tokens)
    dup_count = (
        df.group_by(
            ["bar_close_timestamp", "pool_id", "src_token_id", "dest_token_id"],
        )
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .shape[0]
    )

    status = "✓ PASS" if dup_count == 0 else "✗ FAIL"
    logger.info(f"\nDuplicate rows (by time+pool+tokens): {dup_count:,} {status}")
    results["duplicate_rows"] = dup_count

    # Check timestamp ordering
    is_sorted = df["bar_close_timestamp"].is_sorted()
    status = "✓ PASS" if is_sorted else "✗ WARN"
    logger.info(f"Timestamps sorted: {is_sorted} {status}")
    results["timestamps_sorted"] = is_sorted

    return results


def check_statistical_sanity(df: pl.DataFrame) -> dict[str, any]:
    """Check value ranges, outliers, and distributions.

    Args:
        df: The labeled training dataset

    Returns:
        Dictionary with validation results

    """
    logger.info("\n" + "=" * 60)
    logger.info("2. STATISTICAL SANITY CHECKS")
    logger.info("=" * 60)

    results = {}

    # Value range checks
    logger.info("\nValue range checks (excl. nulls):")
    for col, (expected_min, expected_max) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue

        # Get stats (excluding nulls/NaNs)
        stats = (
            df.select(pl.col(col))
            .filter(
                pl.col(col).is_not_null() & pl.col(col).is_finite(),
            )
            .select(
                [
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                ]
            )
            .to_dicts()[0]
        )

        actual_min = stats["min"]
        actual_max = stats["max"]

        # Check if within expected range (with some tolerance for outliers)
        min_ok = actual_min >= expected_min * 10  # 10x tolerance for minimums
        max_ok = actual_max <= expected_max * 10  # 10x tolerance for maximums

        status = "✓ PASS" if (min_ok and max_ok) else "✗ WARN"

        logger.info(f"  {col:25}")
        logger.info(
            f"    Range: [{actual_min:>12.4e}, {actual_max:>12.4e}] "
            f"(expected: [{expected_min:.1e}, {expected_max:.1e}]) {status}",
        )
        logger.info(
            f"    Mean: {stats['mean']:>12.4e}, Std: {stats['std']:>12.4e}",
        )

        results[f"{col}_min"] = actual_min
        results[f"{col}_max"] = actual_max
        results[f"{col}_mean"] = stats["mean"]
        results[f"{col}_std"] = stats["std"]

    # Check for concerning patterns
    logger.info("\nPattern checks:")

    # 1. Excessive zeros in flow columns
    for col in ["src_flow_usdc", "dest_flow_usdc"]:
        zero_count = df.filter(pl.col(col) == 0.0).shape[0]
        zero_pct = (zero_count / df.shape[0]) * 100
        status = "✗ WARN" if zero_pct > 50 else "✓ PASS"
        logger.info(f"  {col} zeros: {zero_count:,} ({zero_pct:.2f}%) {status}")
        results[f"{col}_zero_pct"] = zero_pct

    # 2. Extremely low volatility (might indicate stagnant tokens)
    low_vol_count = df.filter(pl.col("rolling_volatility") < 1e-10).shape[0]
    low_vol_pct = (low_vol_count / df.shape[0]) * 100
    status = "✗ WARN" if low_vol_pct > 10 else "✓ PASS"
    logger.info(
        f"  Near-zero volatility: {low_vol_count:,} ({low_vol_pct:.2f}%) {status}",
    )
    results["low_volatility_pct"] = low_vol_pct

    # 3. Check fracdiff stationarity (should be centered near 0 with low variance)
    for col in ["src_fracdiff", "dest_fracdiff"]:
        valid_df = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
        if valid_df.shape[0] > 0:
            mean = valid_df[col].mean()
            std = valid_df[col].std()
            # Stationary series should have mean near 0
            status = "✓ PASS" if abs(mean) < 2.0 else "✗ WARN"
            logger.info(
                f"  {col} stationarity: mean={mean:.4f}, std={std:.4f} {status}",
            )
            results[f"{col}_stationarity_ok"] = abs(mean) < 2.0

    return results


def check_feature_relationships(df: pl.DataFrame) -> dict[str, any]:
    """Check correlations and logical consistency between features.

    Args:
        df: The labeled training dataset

    Returns:
        Dictionary with validation results

    """
    logger.info("\n" + "=" * 60)
    logger.info("3. FEATURE RELATIONSHIP CHECKS")
    logger.info("=" * 60)

    results = {}

    # Select numeric features for correlation (excluding nulls)
    feature_cols = [
        "src_flow_usdc",
        "dest_flow_usdc",
        "src_fracdiff",
        "rolling_volatility",
    ]

    # Filter to rows with all features valid
    valid_df = df.select(feature_cols).filter(
        pl.all_horizontal(pl.all().is_not_null() & pl.all().is_finite()),
    )

    logger.info(f"\nRows with all features valid: {valid_df.shape[0]:,}")

    if valid_df.shape[0] > 1000:
        # Compute correlation matrix (sample if too large)
        sample_size = min(100000, valid_df.shape[0])
        corr_df = valid_df.sample(n=sample_size, seed=42)

        # Convert to numpy and compute correlation manually
        data_array = corr_df.to_numpy()
        corr_matrix_values = np.corrcoef(data_array.T)

        # Create a pandas-like structure for easier indexing
        import pandas as pd

        corr_matrix = pd.DataFrame(
            corr_matrix_values,
            index=feature_cols,
            columns=feature_cols,
        )

        logger.info("\nCorrelation matrix:")
        logger.info(corr_matrix.to_string())

        # Save correlation heatmap
        plt.figure(figsize=(10, 8))
        if HAS_SEABORN:
            sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
        else:
            # Fallback to matplotlib imshow if seaborn not available
            plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(
                range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45
            )
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plot_path = PLOTS_DIR / "correlation_matrix.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Correlation heatmap saved to {plot_path}")
        plt.close()

        # Check for concerning correlations
        # High correlation between src_flow and dest_flow would be suspicious
        src_dest_corr = corr_matrix.loc["src_flow_usdc", "dest_flow_usdc"]
        status = "✓ PASS" if abs(src_dest_corr) < 0.5 else "✗ WARN"
        logger.info(
            f"\nsrc_flow vs dest_flow correlation: {src_dest_corr:.3f} {status}",
        )
        results["src_dest_flow_corr"] = src_dest_corr

    # Logical consistency checks
    logger.info("\nLogical consistency:")

    # 1. Check if labels align with price movements
    # (Positive label should correlate with positive fracdiff changes)
    labeled_df = df.filter(
        pl.col("label").is_not_null()
        & pl.col("label").is_finite()
        & pl.col("src_fracdiff").is_not_null()
        & pl.col("src_fracdiff").is_finite(),
    )

    if labeled_df.shape[0] > 0:
        label_counts = labeled_df.group_by("label").agg(pl.len().alias("count"))
        logger.info("\nLabel distribution:")
        for row in label_counts.sort("label").iter_rows(named=True):
            pct = (row["count"] / labeled_df.shape[0]) * 100
            logger.info(
                f"  Label {row['label']:>4.0f}: {row['count']:>10,} ({pct:>5.2f}%)"
            )
        results["label_distribution"] = label_counts.to_dicts()

    return results


def check_label_quality(df: pl.DataFrame) -> dict[str, any]:
    """Check label balance, coverage, and sample weights.

    Args:
        df: The labeled training dataset

    Returns:
        Dictionary with validation results

    """
    logger.info("\n" + "=" * 60)
    logger.info("4. LABEL QUALITY CHECKS")
    logger.info("=" * 60)

    results = {}

    # Label coverage (how many rows have labels)
    src_labeled = df.filter(
        pl.col("label").is_not_null() & pl.col("label").is_finite(),
    ).shape[0]
    dest_labeled = df.filter(
        pl.col("dest_label").is_not_null() & pl.col("dest_label").is_finite(),
    ).shape[0]

    src_pct = (src_labeled / df.shape[0]) * 100
    dest_pct = (dest_labeled / df.shape[0]) * 100

    logger.info(f"Source label coverage: {src_labeled:,} ({src_pct:.2f}%)")
    logger.info(f"Dest label coverage: {dest_labeled:,} ({dest_pct:.2f}%)")

    # At least one label coverage (important for training)
    either_labeled = df.filter(
        (pl.col("label").is_not_null() & pl.col("label").is_finite())
        | (pl.col("dest_label").is_not_null() & pl.col("dest_label").is_finite()),
    ).shape[0]
    either_pct = (either_labeled / df.shape[0]) * 100

    status = "✓ PASS" if either_pct > 10 else "✗ FAIL"
    logger.info(
        f"At least one label: {either_labeled:,} ({either_pct:.2f}%) {status}",
    )
    results["label_coverage_pct"] = either_pct

    # Class balance for source labels
    if src_labeled > 0:
        src_label_df = df.filter(
            pl.col("label").is_not_null() & pl.col("label").is_finite(),
        )

        label_dist = src_label_df.group_by("label").agg(pl.len().alias("count"))

        logger.info("\nSource label class balance:")
        min_count = float("inf")
        max_count = 0

        for row in label_dist.sort("label").iter_rows(named=True):
            count = row["count"]
            pct = (count / src_labeled) * 100
            min_count = min(min_count, count)
            max_count = max(max_count, count)
            logger.info(f"  Label {row['label']:>4.0f}: {count:>10,} ({pct:>5.2f}%)")

        # Check for severe imbalance (smallest class < 10% of largest)
        imbalance_ratio = min_count / max_count if max_count > 0 else 0
        status = "✗ WARN" if imbalance_ratio < 0.1 else "✓ PASS"
        logger.info(
            f"\nClass balance ratio (min/max): {imbalance_ratio:.3f} {status}",
        )
        results["class_balance_ratio"] = imbalance_ratio

    # Sample weight checks
    weight_cols = ["sample_weight", "dest_sample_weight"]
    logger.info("\nSample weight statistics:")

    for col in weight_cols:
        valid_weights = df.filter(
            pl.col(col).is_not_null() & pl.col(col).is_finite(),
        )

        if valid_weights.shape[0] > 0:
            stats = valid_weights.select(
                [
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).mean().alias("mean"),
                ]
            ).to_dicts()[0]

            # Weights should be in [0, 1] and have reasonable mean
            status = (
                "✓ PASS"
                if (0 <= stats["min"] <= 1 and 0 <= stats["max"] <= 1)
                else "✗ WARN"
            )

            logger.info(
                f"  {col:25} min={stats['min']:.4f}, "
                f"max={stats['max']:.4f}, mean={stats['mean']:.4f} {status}",
            )
            results[f"{col}_stats"] = stats

    # Barrier touch statistics
    logger.info("\nBarrier touch statistics:")
    barrier_stats = (
        df.filter(
            pl.col("barrier_touch_bars").is_not_null()
            & pl.col("barrier_touch_bars").is_finite(),
        )
        .select(
            [
                pl.col("barrier_touch_bars").min().alias("min"),
                pl.col("barrier_touch_bars").max().alias("max"),
                pl.col("barrier_touch_bars").mean().alias("mean"),
                pl.col("barrier_touch_bars").median().alias("median"),
            ]
        )
        .to_dicts()[0]
    )

    logger.info(
        f"  min={barrier_stats['min']:.1f}, max={barrier_stats['max']:.1f}, "
        f"mean={barrier_stats['mean']:.1f}, median={barrier_stats['median']:.1f}",
    )
    results["barrier_touch_stats"] = barrier_stats

    return results


def check_temporal_consistency(df: pl.DataFrame) -> dict[str, any]:
    """Check time ordering, gaps, and temporal patterns.

    Args:
        df: The labeled training dataset

    Returns:
        Dictionary with validation results

    """
    logger.info("\n" + "=" * 60)
    logger.info("5. TEMPORAL CONSISTENCY CHECKS")
    logger.info("=" * 60)

    results = {}

    # Time range
    min_time = df["bar_close_timestamp"].min()
    max_time = df["bar_close_timestamp"].max()
    duration = max_time - min_time

    logger.info(f"Time range: {min_time} to {max_time}")
    logger.info(f"Duration: {duration}")
    results["time_range"] = (str(min_time), str(max_time))

    # Check for large time gaps (might indicate missing data)
    sorted_df = df.sort("bar_close_timestamp")
    time_diffs = sorted_df.select(
        pl.col("bar_close_timestamp").diff().alias("time_diff"),
    ).filter(pl.col("time_diff").is_not_null())

    if time_diffs.shape[0] > 0:
        gap_stats = time_diffs.select(
            [
                pl.col("time_diff").min().alias("min_gap"),
                pl.col("time_diff").max().alias("max_gap"),
                pl.col("time_diff").mean().alias("mean_gap"),
            ]
        ).to_dicts()[0]

        logger.info("\nTime gaps between consecutive bars:")
        logger.info(f"  Min gap: {gap_stats['min_gap']}")
        logger.info(f"  Max gap: {gap_stats['max_gap']}")
        logger.info(f"  Mean gap: {gap_stats['mean_gap']}")

        # Flag if max gap > 1 hour (might indicate data issues)
        max_gap_seconds = gap_stats["max_gap"].total_seconds()
        status = "✗ WARN" if max_gap_seconds > 3600 else "✓ PASS"
        logger.info(f"  Max gap in seconds: {max_gap_seconds:.0f} {status}")
        results["max_gap_seconds"] = max_gap_seconds

    # Distribution of events over time (check for temporal bias)
    logger.info("\nTemporal distribution:")

    # Count events per day
    daily_counts = sorted_df.group_by_dynamic(
        "bar_close_timestamp",
        every="1d",
    ).agg(pl.len().alias("count"))

    logger.info(f"  Number of days with data: {daily_counts.shape[0]}")
    logger.info(
        f"  Mean events per day: {daily_counts['count'].mean():.0f}",
    )
    logger.info(
        f"  Median events per day: {daily_counts['count'].median():.0f}",
    )

    # Check for days with very low activity (< 10% of median)
    median_daily = daily_counts["count"].median()
    low_activity_days = daily_counts.filter(
        pl.col("count") < median_daily * 0.1,
    ).shape[0]

    status = "✗ WARN" if low_activity_days > daily_counts.shape[0] * 0.1 else "✓ PASS"
    logger.info(
        f"  Days with low activity (< 10% median): {low_activity_days} {status}",
    )
    results["low_activity_days"] = low_activity_days

    return results


def generate_summary_plots(df: pl.DataFrame) -> None:
    """Generate summary visualizations for key columns.

    Args:
        df: The labeled training dataset

    """
    logger.info("\n" + "=" * 60)
    logger.info("6. GENERATING SUMMARY PLOTS")
    logger.info("=" * 60)

    # Create a 2x3 subplot for key distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Data Distribution Summary", fontsize=16)

    plot_cols = [
        ("src_flow_usdc", "Source Flow (USDC)", True),
        ("dest_flow_usdc", "Dest Flow (USDC)", True),
        ("src_fracdiff", "Source Fracdiff", False),
        ("dest_fracdiff", "Dest Fracdiff", False),
        ("rolling_volatility", "Rolling Volatility", True),
        ("label", "Label Distribution", False),
    ]

    for idx, (col, title, use_log) in enumerate(plot_cols):
        ax = axes[idx // 3, idx % 3]

        # Get valid data
        valid_data = df.filter(
            pl.col(col).is_not_null() & pl.col(col).is_finite(),
        )[col].to_numpy()

        if len(valid_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            continue

        # Sample if too large
        if len(valid_data) > 100000:
            valid_data = np.random.choice(valid_data, 100000, replace=False)

        # Plot histogram
        if col == "label":
            # Bar plot for discrete labels
            unique, counts = np.unique(valid_data, return_counts=True)
            ax.bar(unique, counts, color="steelblue")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
        else:
            # Histogram for continuous features
            ax.hist(
                valid_data, bins=50, alpha=0.7, color="steelblue", edgecolor="black"
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

            if use_log:
                ax.set_yscale("log")

        ax.set_title(title)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "distribution_summary.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Distribution summary saved to {plot_path}")
    plt.close()

    # Create scatter plot: flow vs fracdiff
    logger.info("Generating flow vs fracdiff scatter plot...")

    sample_df = df.filter(
        pl.col("src_flow_usdc").is_not_null()
        & pl.col("src_flow_usdc").is_finite()
        & pl.col("src_fracdiff").is_not_null()
        & pl.col("src_fracdiff").is_finite(),
    ).sample(n=min(10000, df.shape[0]), seed=42)

    if sample_df.shape[0] > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(
            sample_df["src_flow_usdc"].to_numpy(),
            sample_df["src_fracdiff"].to_numpy(),
            alpha=0.3,
            s=10,
        )
        plt.xlabel("Source Flow (USDC)")
        plt.ylabel("Source Fracdiff")
        plt.title("Flow vs Fracdiff Relationship")
        plt.grid(alpha=0.3)

        plot_path = PLOTS_DIR / "flow_vs_fracdiff.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Flow vs fracdiff plot saved to {plot_path}")
        plt.close()


# %%
# Main orchestration


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    default=DATA_PATH,
    help="Path to labeled_log_fracdiff_price.parquet",
)
@click.option(
    "--skip-plots",
    is_flag=True,
    help="Skip plot generation for faster execution",
)
def main(data_path: Path, skip_plots: bool) -> None:
    """Run comprehensive validation on training data."""
    logger.info("=" * 60)
    logger.info("TRAINING DATA VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")

    # Load data
    logger.info("\nLoading data...")
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Run validation checks
    all_results = {}

    all_results["integrity"] = check_data_integrity(df)
    all_results["statistical"] = check_statistical_sanity(df)
    all_results["relationships"] = check_feature_relationships(df)
    all_results["labels"] = check_label_quality(df)
    all_results["temporal"] = check_temporal_consistency(df)

    # Generate plots
    if not skip_plots:
        generate_summary_plots(df)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Review the output above for any ✗ FAIL or ✗ WARN markers.")
    logger.info(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
