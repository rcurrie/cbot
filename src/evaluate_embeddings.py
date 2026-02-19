"""Evaluate token embeddings from TGCN trader for downstream tasks.

WHY: After training the TGCN model, we want to understand if the learned token
embeddings capture meaningful structure about token behavior. Good embeddings
should cluster similar tokens together and enable downstream predictions.

WHAT: Evaluate embeddings via:
1. Dimensionality reduction (PCA, t-SNE) for visualization
2. Clustering analysis (K-means silhouette score)
3. Token similarity analysis (cosine similarity between tokens)
4. Correlation with token metrics (flow volume, volatility, price movement)

HOW:
1. Load saved embeddings from .npz files
2. Compute embedding statistics and quality metrics
3. Analyze token clusters and similarities
4. Correlate embeddings with actual token behavior from data

INPUT: data/embeddings/tgcn_embeddings_*.npz (saved embeddings)
       data/labeled_log_fracdiff_price.parquet (for token metrics)
       data/tokens.json (token metadata)
OUTPUT: Console report with embedding quality metrics

"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import typer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DIVIDER_LENGTH = 70

app = typer.Typer(
    help="Evaluate token embeddings from TGCN trader.",
)


def load_embeddings(embeddings_path: Path) -> dict[str, Any]:
    """Load embeddings from .npz file.

    Args:
        embeddings_path: Path to the .npz file.

    Returns:
        Dictionary with embeddings and metadata.

    """
    data = np.load(embeddings_path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "token_addresses": data["token_addresses"],
        "trade_date": str(data["trade_date"]),
        "embed_dim": int(data["embed_dim"]),
        "num_tokens": int(data["num_tokens"]),
    }


def load_token_metadata(tokens_path: Path) -> dict[str, dict[str, Any]]:
    """Load token metadata from tokens.json.

    Args:
        tokens_path: Path to tokens.json file.

    Returns:
        Dictionary mapping token address to metadata.

    """
    with tokens_path.open() as f:
        metadata: dict[str, dict[str, Any]] = json.load(f)
    return metadata


def compute_token_metrics(
    df: pl.DataFrame,
    token_addresses: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics for each token from swap data.

    Args:
        df: DataFrame with labeled bars.
        token_addresses: Array of token addresses.

    Returns:
        Dictionary mapping token address to metrics dict.

    """
    metrics: dict[str, dict[str, float]] = {}

    for token_addr in token_addresses:
        # Get swaps where token is source
        token_df = df.filter(pl.col("src_token_id") == token_addr)

        if len(token_df) == 0:
            metrics[token_addr] = {
                "total_flow": 0.0,
                "mean_flow": 0.0,
                "flow_volatility": 0.0,
                "mean_fracdiff": 0.0,
                "fracdiff_volatility": 0.0,
                "num_swaps": 0,
                "up_ratio": 0.0,
                "down_ratio": 0.0,
            }
            continue

        flows = token_df["src_flow_usdc"].to_numpy()
        fracdiffs = token_df["src_fracdiff"].to_numpy()
        labels = token_df["label"].fill_null(0).to_numpy()

        # Filter out NaN values
        flows = flows[~np.isnan(flows)]
        fracdiffs = fracdiffs[~np.isnan(fracdiffs)]

        total_labels = len(labels)
        up_count = np.sum(labels == 1)
        down_count = np.sum(labels == -1)

        metrics[token_addr] = {
            "total_flow": float(np.sum(np.abs(flows))) if len(flows) > 0 else 0.0,
            "mean_flow": float(np.mean(np.abs(flows))) if len(flows) > 0 else 0.0,
            "flow_volatility": float(np.std(flows)) if len(flows) > 1 else 0.0,
            "mean_fracdiff": float(np.mean(fracdiffs)) if len(fracdiffs) > 0 else 0.0,
            "fracdiff_volatility": (
                float(np.std(fracdiffs)) if len(fracdiffs) > 1 else 0.0
            ),
            "num_swaps": len(token_df),
            "up_ratio": float(up_count / total_labels) if total_labels > 0 else 0.0,
            "down_ratio": (
                float(down_count / total_labels) if total_labels > 0 else 0.0
            ),
        }

    return metrics


def analyze_embedding_statistics(embeddings: np.ndarray) -> dict[str, float]:
    """Compute basic statistics about embeddings.

    Args:
        embeddings: Token embeddings array [num_tokens, embed_dim].

    Returns:
        Dictionary of statistics.

    """
    # Normalize embeddings for analysis
    norms = np.linalg.norm(embeddings, axis=1)

    # Compute pairwise cosine similarities
    cos_sim = cosine_similarity(embeddings)
    # Get upper triangle (excluding diagonal)
    upper_tri = cos_sim[np.triu_indices(len(embeddings), k=1)]

    return {
        "num_tokens": embeddings.shape[0],
        "embed_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_cosine_sim": float(np.mean(upper_tri)),
        "std_cosine_sim": float(np.std(upper_tri)),
        "min_cosine_sim": float(np.min(upper_tri)),
        "max_cosine_sim": float(np.max(upper_tri)),
    }


def analyze_clusters(
    embeddings: np.ndarray,
    n_clusters_range: tuple[int, int] = (2, 10),
) -> dict[str, Any]:
    """Analyze embedding clusters using K-means.

    Args:
        embeddings: Token embeddings array.
        n_clusters_range: Range of cluster counts to try.

    Returns:
        Dictionary with clustering analysis results.

    """
    # Standardize embeddings for clustering
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    results: dict[str, Any] = {
        "silhouette_scores": {},
        "best_k": 2,
        "best_silhouette": -1.0,
    }

    min_k, max_k = n_clusters_range
    max_k = min(max_k, len(embeddings) - 1)  # Can't have more clusters than samples

    if max_k < min_k:
        logger.warning("Not enough tokens for clustering analysis")
        return results

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)
        score = silhouette_score(embeddings_scaled, labels)
        results["silhouette_scores"][k] = float(score)

        if score > results["best_silhouette"]:
            results["best_silhouette"] = float(score)
            results["best_k"] = k
            results["best_labels"] = labels.tolist()

    return results


def analyze_pca(embeddings: np.ndarray) -> dict[str, Any]:
    """Analyze embeddings using PCA.

    Args:
        embeddings: Token embeddings array.

    Returns:
        Dictionary with PCA analysis results.

    """
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Full PCA for variance analysis
    pca_full = PCA()
    pca_full.fit(embeddings_scaled)

    # Compute cumulative variance explained
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Find number of components for 90% and 95% variance
    n_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 0.95) + 1)

    # 2D projection for visualization
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings_scaled)

    # Compute cumulative variance at 5 and 10 components
    cum_var_5 = (
        float(cumulative_var[4])
        if len(cumulative_var) > 4
        else float(cumulative_var[-1])
    )
    cum_var_10 = (
        float(cumulative_var[9])
        if len(cumulative_var) > 9
        else float(cumulative_var[-1])
    )

    return {
        "variance_explained_ratio": pca_full.explained_variance_ratio_[:10].tolist(),
        "cumulative_variance_5": cum_var_5,
        "cumulative_variance_10": cum_var_10,
        "n_components_90pct": n_90,
        "n_components_95pct": n_95,
        "embeddings_2d": embeddings_2d,
        "pca_2d_variance": float(sum(pca_2d.explained_variance_ratio_)),
    }


def find_similar_tokens(
    embeddings: np.ndarray,
    token_addresses: np.ndarray,
    token_metadata: dict[str, dict[str, Any]],
    top_k: int = 5,
) -> list[tuple[str, str, str, str, float]]:
    """Find most similar token pairs by embedding cosine similarity.

    Args:
        embeddings: Token embeddings array.
        token_addresses: Array of token addresses.
        token_metadata: Token metadata for display names.
        top_k: Number of top pairs to return.

    Returns:
        List of (sym1, sym2, addr1, addr2, similarity) tuples.

    """
    cos_sim = cosine_similarity(embeddings)

    # Get all pairs with their similarities
    pairs: list[tuple[str, str, str, str, float]] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            addr1 = str(token_addresses[i])
            addr2 = str(token_addresses[j])
            meta1 = token_metadata.get(addr1.lower(), {})
            meta2 = token_metadata.get(addr2.lower(), {})
            sym1 = meta1.get("symbol", addr1[:10])
            sym2 = meta2.get("symbol", addr2[:10])
            pairs.append((sym1, sym2, addr1, addr2, float(cos_sim[i, j])))

    # Sort by similarity descending
    pairs.sort(key=lambda x: x[4], reverse=True)

    return pairs[:top_k]


def find_dissimilar_tokens(
    embeddings: np.ndarray,
    token_addresses: np.ndarray,
    token_metadata: dict[str, dict[str, Any]],
    top_k: int = 5,
) -> list[tuple[str, str, str, str, float]]:
    """Find most dissimilar token pairs by embedding cosine similarity.

    Args:
        embeddings: Token embeddings array.
        token_addresses: Array of token addresses.
        token_metadata: Token metadata for display names.
        top_k: Number of top pairs to return.

    Returns:
        List of (sym1, sym2, addr1, addr2, similarity) tuples.

    """
    cos_sim = cosine_similarity(embeddings)

    # Get all pairs with their similarities
    pairs: list[tuple[str, str, str, str, float]] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            addr1 = str(token_addresses[i])
            addr2 = str(token_addresses[j])
            meta1 = token_metadata.get(addr1.lower(), {})
            meta2 = token_metadata.get(addr2.lower(), {})
            sym1 = meta1.get("symbol", addr1[:10])
            sym2 = meta2.get("symbol", addr2[:10])
            pairs.append((sym1, sym2, addr1, addr2, float(cos_sim[i, j])))

    # Sort by similarity ascending (most different first)
    pairs.sort(key=lambda x: x[4])

    return pairs[:top_k]


def correlate_with_metrics(
    embeddings: np.ndarray,
    token_addresses: np.ndarray,
    token_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Correlate embedding dimensions with token metrics.

    Uses PCA to find main embedding directions and correlates with metrics.

    Args:
        embeddings: Token embeddings array.
        token_addresses: Array of token addresses.
        token_metrics: Token metrics dictionary.

    Returns:
        Dictionary of correlation results.

    """
    # Get metrics as arrays aligned with embeddings
    metric_names = [
        "total_flow", "mean_flow", "flow_volatility", "num_swaps", "up_ratio",
    ]
    metric_arrays: dict[str, np.ndarray] = {}

    for metric_name in metric_names:
        values = []
        for addr in token_addresses:
            addr_str = str(addr)
            if addr_str in token_metrics:
                values.append(token_metrics[addr_str].get(metric_name, 0.0))
            else:
                values.append(0.0)
        metric_arrays[metric_name] = np.array(values)

    # PCA to get main embedding directions
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=min(5, embeddings.shape[1]))
    pca_embeddings = pca.fit_transform(embeddings_scaled)

    # Correlate each PC with each metric
    correlations: dict[str, float] = {}
    for i in range(pca_embeddings.shape[1]):
        pc = pca_embeddings[:, i]
        for metric_name, metric_values in metric_arrays.items():
            # Skip if metric has no variance
            if np.std(metric_values) < 1e-10:
                continue
            corr = np.corrcoef(pc, metric_values)[0, 1]
            if not np.isnan(corr):
                correlations[f"PC{i+1}_vs_{metric_name}"] = float(corr)

    # Find strongest correlations
    if correlations:
        abs_corrs = {k: abs(v) for k, v in correlations.items()}
        max_corr_key = max(abs_corrs, key=abs_corrs.get)  # type: ignore[arg-type]
        correlations["strongest_correlation"] = correlations[max_corr_key]
        correlations["strongest_correlation_name"] = max_corr_key  # type: ignore[assignment]

    return correlations


@app.command()
def main(
    embeddings_path: Path = typer.Option(
        ...,
        "--embeddings",
        "-e",
        help="Path to embeddings .npz file",
    ),
    data_path: Path = typer.Option(
        Path("data/labeled_log_fracdiff_price.parquet"),
        "--data",
        "-d",
        help="Path to labeled training data parquet file",
    ),
    tokens_path: Path = typer.Option(
        Path("data/tokens.json"),
        "--tokens",
        "-t",
        help="Path to tokens metadata JSON file",
    ),
) -> None:
    """Evaluate token embeddings quality and structure."""
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EMBEDDING EVALUATION")
    logger.info("=" * DIVIDER_LENGTH)

    # Load embeddings
    logger.info("Loading embeddings from %s", embeddings_path)
    emb_data = load_embeddings(embeddings_path)
    embeddings = emb_data["embeddings"]
    token_addresses = emb_data["token_addresses"]

    logger.info("  Trade date: %s", emb_data["trade_date"])
    logger.info("  Num tokens: %d", emb_data["num_tokens"])
    logger.info("  Embed dim: %d", emb_data["embed_dim"])

    # Load token metadata
    logger.info("Loading token metadata from %s", tokens_path)
    token_metadata = load_token_metadata(tokens_path)

    # Load swap data for metrics
    logger.info("Loading swap data from %s", data_path)
    df = pl.read_parquet(data_path)
    logger.info("  Loaded %s bars", f"{len(df):,}")

    # Compute token metrics
    logger.info("Computing token metrics...")
    token_metrics = compute_token_metrics(df, token_addresses)

    # Basic statistics
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EMBEDDING STATISTICS")
    logger.info("=" * DIVIDER_LENGTH)

    stats = analyze_embedding_statistics(embeddings)
    logger.info("  Mean embedding norm: %.4f", stats["mean_norm"])
    logger.info("  Std embedding norm: %.4f", stats["std_norm"])
    logger.info("  Norm range: [%.4f, %.4f]", stats["min_norm"], stats["max_norm"])
    logger.info("")
    logger.info("  Mean cosine similarity: %.4f", stats["mean_cosine_sim"])
    logger.info("  Std cosine similarity: %.4f", stats["std_cosine_sim"])
    logger.info(
        "  Similarity range: [%.4f, %.4f]",
        stats["min_cosine_sim"],
        stats["max_cosine_sim"],
    )

    # Interpret cosine similarity
    if stats["mean_cosine_sim"] > 0.9:
        logger.warning("  ⚠️  High similarity - embeddings may be collapsing")
    elif stats["mean_cosine_sim"] < 0.1:
        logger.info("  ✅ Good diversity - embeddings are well-separated")
    else:
        logger.info("  ✅ Moderate similarity - reasonable spread")

    # PCA analysis
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("PCA ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)

    pca_results = analyze_pca(embeddings)
    var_5 = pca_results["cumulative_variance_5"] * 100
    var_10 = pca_results["cumulative_variance_10"] * 100
    logger.info("  Variance explained by first 5 PCs: %.1f%%", var_5)
    logger.info("  Variance explained by first 10 PCs: %.1f%%", var_10)
    n_90 = pca_results["n_components_90pct"]
    n_95 = pca_results["n_components_95pct"]
    var_2d = pca_results["pca_2d_variance"] * 100
    logger.info("  Components for 90%% variance: %d", n_90)
    logger.info("  Components for 95%% variance: %d", n_95)
    logger.info("  2D projection variance: %.1f%%", var_2d)

    # Interpret PCA
    if pca_results["n_components_90pct"] <= 5:
        logger.info("  ✅ Embeddings are low-dimensional - clear structure")
    elif pca_results["n_components_90pct"] <= 20:
        logger.info("  ✅ Moderate dimensionality - some structure present")
    else:
        logger.warning("  ⚠️  High dimensionality - embeddings may be noisy")

    # Clustering analysis
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("CLUSTERING ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)

    cluster_results = analyze_clusters(embeddings)
    logger.info("  Best number of clusters: %d", cluster_results["best_k"])
    logger.info("  Best silhouette score: %.4f", cluster_results["best_silhouette"])

    if "silhouette_scores" in cluster_results:
        logger.info("  Silhouette scores by k:")
        for k, score in sorted(cluster_results["silhouette_scores"].items()):
            logger.info("    k=%d: %.4f", k, score)

    # Interpret silhouette
    if cluster_results["best_silhouette"] > 0.5:
        logger.info("  ✅ Strong cluster structure")
    elif cluster_results["best_silhouette"] > 0.25:
        logger.info("  ✅ Moderate cluster structure")
    elif cluster_results["best_silhouette"] > 0:
        logger.warning("  ⚠️  Weak cluster structure")
    else:
        logger.warning("  ❌ No meaningful clusters found")

    # Similar tokens
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("MOST SIMILAR TOKEN PAIRS")
    logger.info("=" * DIVIDER_LENGTH)

    similar_pairs = find_similar_tokens(
        embeddings,
        token_addresses,
        token_metadata,
        top_k=5,
    )
    for sym1, sym2, _, _, sim in similar_pairs:
        logger.info("  %s <-> %s: %.4f", sym1, sym2, sim)

    # Dissimilar tokens
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("MOST DISSIMILAR TOKEN PAIRS")
    logger.info("=" * DIVIDER_LENGTH)

    dissimilar_pairs = find_dissimilar_tokens(
        embeddings,
        token_addresses,
        token_metadata,
        top_k=5,
    )
    for sym1, sym2, _, _, sim in dissimilar_pairs:
        logger.info("  %s <-> %s: %.4f", sym1, sym2, sim)

    # Correlation with metrics
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EMBEDDING-METRIC CORRELATIONS")
    logger.info("=" * DIVIDER_LENGTH)

    correlations = correlate_with_metrics(embeddings, token_addresses, token_metrics)

    if "strongest_correlation_name" in correlations:
        logger.info(
            "  Strongest correlation: %s = %.4f",
            correlations["strongest_correlation_name"],
            correlations["strongest_correlation"],
        )

    # Show top correlations
    corr_items = [
        (k, v)
        for k, v in correlations.items()
        if k not in ("strongest_correlation", "strongest_correlation_name")
    ]
    corr_items.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info("  Top correlations (by absolute value):")
    for name, corr in corr_items[:10]:
        icon = "+" if corr > 0 else "-"
        logger.info("    %s: %s%.4f", name, icon, abs(corr))

    # Interpret correlations
    if corr_items and abs(corr_items[0][1]) > 0.5:
        logger.info("  ✅ Strong correlation with token metrics")
    elif corr_items and abs(corr_items[0][1]) > 0.3:
        logger.info("  ✅ Moderate correlation with token metrics")
    else:
        logger.warning("  ⚠️  Weak correlation with token metrics")

    # Summary
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("SUMMARY")
    logger.info("=" * DIVIDER_LENGTH)

    quality_score = 0
    max_score = 4

    # Score based on metrics
    if stats["mean_cosine_sim"] < 0.8:
        quality_score += 1
        logger.info("  ✅ Embedding diversity: GOOD")
    else:
        logger.info("  ⚠️  Embedding diversity: POOR (high similarity)")

    if pca_results["n_components_90pct"] <= 20:
        quality_score += 1
        logger.info("  ✅ Dimensionality: GOOD")
    else:
        logger.info("  ⚠️  Dimensionality: POOR (too spread)")

    if cluster_results["best_silhouette"] > 0.1:
        quality_score += 1
        logger.info("  ✅ Cluster structure: GOOD")
    else:
        logger.info("  ⚠️  Cluster structure: POOR")

    if corr_items and abs(corr_items[0][1]) > 0.2:
        quality_score += 1
        logger.info("  ✅ Metric correlation: GOOD")
    else:
        logger.info("  ⚠️  Metric correlation: POOR")

    logger.info("")
    logger.info("  Overall quality score: %d/%d", quality_score, max_score)

    if quality_score >= 3:
        logger.info("  ✅ Embeddings show meaningful structure!")
    elif quality_score >= 2:
        logger.info("  ⚠️  Embeddings have some useful signal")
    else:
        logger.info("  ❌ Embeddings may not be learning useful representations")

    logger.info("=" * DIVIDER_LENGTH)


if __name__ == "__main__":
    app()
