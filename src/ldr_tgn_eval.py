"""Evaluate LDR-TGN embeddings for linear discriminability and cluster quality.

WHY: The MCR² loss aims to learn Linear Discriminative Representations (LDR) where
classes form orthogonal subspaces. To verify this, we need to test:
1. Linear Separability: Can a simple linear classifier (logistic regression) predict
   labels from embeddings? High accuracy = good LDR quality.
2. Cluster Structure: Do embeddings visually separate by class in UMAP projection?
   Clear clusters = the model learned meaningful structure.
3. Attention Patterns: Are GATv2 attention weights sparse and meaningful?
   Sparse attention = model is selectively aggregating information.

These diagnostics help identify if the MCR² loss is working as intended and guide
hyperparameter tuning.

WHAT: Three evaluation methods:
1. Linear Probe: Train logistic regression on frozen embeddings, report accuracy
2. UMAP Visualization: 2D projection colored by label to visualize cluster separation
3. Attention Analysis: Extract and analyze GATv2 attention weights (entropy, sparsity)

HOW:
1. Load saved embeddings from data/ldr_embeddings/*.npz
2. Split into train/test for linear probe evaluation
3. Fit UMAP for dimensionality reduction and visualization
4. Compute cluster metrics (silhouette score, Davies-Bouldin index)
5. Generate visualization plots saved to data/ldr_embeddings/eval/

INPUT: data/ldr_embeddings/*.npz (saved embeddings from ldr_tgn_trader.py)
OUTPUT: Evaluation metrics (console) and plots (data/ldr_embeddings/eval/)

References:
- Yu et al. "Learning Diverse and Discriminative Representations via the
  Principle of Maximal Coding Rate Reduction" (NeurIPS 2020)
- McInnes et al. "UMAP: Uniform Manifold Approximation and Projection"
- Rousseeuw "Silhouettes: A graphical aid to interpretation of cluster analysis"

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
import umap
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
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
    help="Evaluate LDR-TGN embeddings for linear discriminability and cluster quality.",
)

# Label names for visualization
LABEL_NAMES = {0: "Down", 1: "Neutral", 2: "Up"}
LABEL_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#27ae60"}  # Red, Gray, Green


def load_embeddings(
    embeddings_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load embeddings from .npz file.

    Args:
        embeddings_path: Path to embeddings .npz file.

    Returns:
        Tuple of (embeddings, token_addresses, labels, trade_date).

    """
    data = np.load(embeddings_path, allow_pickle=True)

    embeddings = data["embeddings"]
    token_addresses = data["token_addresses"]
    labels = data["labels"]
    trade_date = str(data["trade_date"])

    logger.info("Loaded embeddings from %s", embeddings_path)
    logger.info("  Shape: %s", embeddings.shape)
    logger.info("  Trade date: %s", trade_date)

    return embeddings, token_addresses, labels, trade_date


def filter_labeled_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    token_addresses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter to only include labeled embeddings.

    Args:
        embeddings: Full embedding matrix.
        labels: Label array (may contain -1 for unlabeled).
        token_addresses: Token address array.

    Returns:
        Filtered (embeddings, labels, token_addresses).

    """
    mask = labels >= 0
    n_filtered = mask.sum()
    n_total = len(labels)

    logger.info("Filtering labeled embeddings: %d / %d", n_filtered, n_total)

    return embeddings[mask], labels[mask], token_addresses[mask]


def evaluate_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate linear separability with logistic regression.

    A well-trained MCR² model should produce embeddings that are linearly
    separable - i.e., a simple linear classifier should achieve high accuracy.

    Args:
        embeddings: Embedding matrix [n_samples, embed_dim].
        labels: Label array [n_samples].
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with accuracy and per-class metrics.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("LINEAR PROBE EVALUATION")
    logger.info("=" * DIVIDER_LENGTH)

    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for label, count in zip(unique_labels, counts, strict=False):
        pct = 100 * count / len(labels)
        label_name = LABEL_NAMES.get(label, str(label))
        logger.info("  %s: %d (%.1f%%)", label_name, count, pct)

    # Check for minimum samples per class
    min_samples_per_class = 2
    if min(counts) < min_samples_per_class:
        logger.warning(
            "Some classes have < %d samples, skipping probe", min_samples_per_class,
        )
        return {"accuracy": 0.0, "error": "insufficient_samples"}

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Split data
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings_scaled,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as e:
        logger.warning("Could not stratify split: %s", e)
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings_scaled,
            labels,
            test_size=test_size,
            random_state=random_state,
        )

    logger.info("Train/Test split: %d / %d", len(x_train), len(x_test))

    # Fit logistic regression
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)

    # Predict
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("")
    logger.info("Results:")
    if accuracy > 0.6:
        logger.info("  OK Accuracy: %.2f%%", accuracy * 100)
    elif accuracy > 0.4:
        logger.info("  WARN Accuracy: %.2f%%", accuracy * 100)
    else:
        logger.info("  FAIL Accuracy: %.2f%%", accuracy * 100)

    # Detailed classification report
    logger.info("")
    logger.info("Classification Report:")
    target_names = [LABEL_NAMES.get(lbl, str(lbl)) for lbl in sorted(unique_labels)]
    report = classification_report(y_test, y_pred, target_names=target_names)
    for line in report.split("\n"):
        if line.strip():
            logger.info("  %s", line)

    # Confusion matrix
    logger.info("")
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info("  Predicted ->  %s", "  ".join(target_names))
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:3d}" for v in row)
        logger.info("  Actual %s: %s", target_names[i], row_str)

    logger.info("=" * DIVIDER_LENGTH)

    return {
        "accuracy": accuracy,
        "n_train": len(x_train),
        "n_test": len(x_test),
    }


def compute_cluster_metrics(  # noqa: C901
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        embeddings: Embedding matrix [n_samples, embed_dim].
        labels: Label array [n_samples].

    Returns:
        Dictionary with silhouette score and other metrics.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("CLUSTER QUALITY METRICS")
    logger.info("=" * DIVIDER_LENGTH)

    # Check we have enough samples and at least 2 classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Need at least 2 classes for cluster metrics")
        return {"silhouette": 0.0, "error": "insufficient_classes"}

    if len(embeddings) < 3:
        logger.warning("Need at least 3 samples for cluster metrics")
        return {"silhouette": 0.0, "error": "insufficient_samples"}

    # Silhouette Score: measures how similar an object is to its own cluster
    # vs other clusters. Range [-1, 1], higher is better.
    try:
        sil_score = silhouette_score(embeddings, labels)
        if sil_score > 0.5:
            logger.info("  OK Silhouette Score: %.3f (strong structure)", sil_score)
        elif sil_score > 0.25:
            logger.info("  WARN Silhouette Score: %.3f (weak structure)", sil_score)
        else:
            logger.info("  FAIL Silhouette Score: %.3f (no structure)", sil_score)
    except ValueError as e:
        logger.warning("Could not compute silhouette score: %s", e)
        sil_score = 0.0

    # Compute per-class cohesion (mean intra-class distance)
    logger.info("")
    logger.info("Per-class cohesion (lower = tighter clusters):")
    cohesion_scores = {}
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        if len(class_embeddings) > 1:
            # Compute mean pairwise distance within class
            distances = pdist(class_embeddings, metric="euclidean")
            mean_dist = np.mean(distances)
            cohesion_scores[label] = mean_dist
            logger.info("  %s: %.3f", LABEL_NAMES.get(label, str(label)), mean_dist)
        else:
            cohesion_scores[label] = 0.0
            logger.info("  %s: N/A (single sample)", LABEL_NAMES.get(label, str(label)))

    # Compute inter-class separation (distance between class centroids)
    logger.info("")
    logger.info("Inter-class separation (higher = better separated):")
    centroids = {}
    for label in unique_labels:
        centroids[label] = embeddings[labels == label].mean(axis=0)

    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1 :]:
            dist = np.linalg.norm(centroids[label1] - centroids[label2])
            logger.info(
                "  %s <-> %s: %.3f",
                LABEL_NAMES.get(label1, str(label1)),
                LABEL_NAMES.get(label2, str(label2)),
                dist,
            )

    logger.info("=" * DIVIDER_LENGTH)

    return {
        "silhouette": sil_score,
        "cohesion": cohesion_scores,
    }


def visualize_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    token_addresses: np.ndarray,  # noqa: ARG001
    trade_date: str,
    output_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> Path:
    """Create UMAP visualization of embeddings.

    Args:
        embeddings: Embedding matrix [n_samples, embed_dim].
        labels: Label array [n_samples].
        token_addresses: Token address array (for future token labeling).
        trade_date: Trade date string for filename.
        output_dir: Directory to save plot.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed.

    Returns:
        Path to saved plot.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("UMAP VISUALIZATION")
    logger.info("=" * DIVIDER_LENGTH)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fit UMAP
    logger.info(
        "Fitting UMAP (n_neighbors=%d, min_dist=%.2f)...", n_neighbors, min_dist,
    )
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",  # Cosine for normalized embeddings
        random_state=random_state,
    )

    try:
        embedding_2d = reducer.fit_transform(embeddings)
    except (ValueError, RuntimeError) as e:
        logger.warning("UMAP failed: %s", e)
        return output_dir / "umap_failed.png"

    # Create figure
    _fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Colored by label
    ax1 = axes[0]
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax1.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=LABEL_COLORS.get(label, "#333333"),
            label=LABEL_NAMES.get(label, str(label)),
            alpha=0.6,
            s=50,
        )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title(f"LDR-TGN Embeddings by Label\n{trade_date}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Density plot
    ax2 = axes[1]
    ax2.hexbin(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        gridsize=30,
        cmap="YlOrRd",
        mincnt=1,
    )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_title(f"Embedding Density\n{trade_date}")
    plt.colorbar(ax2.collections[0], ax=ax2, label="Count")

    plt.tight_layout()

    # Save
    output_path = output_dir / f"umap_{trade_date.replace('-', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved UMAP plot to: %s", output_path)
    logger.info("=" * DIVIDER_LENGTH)

    return output_path


def analyze_embedding_geometry(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Analyze the geometric properties of embeddings.

    For MCR² to work well, embeddings should:
    1. Be approximately unit normalized
    2. Have orthogonal class subspaces
    3. Have high within-class correlation

    Args:
        embeddings: Embedding matrix [n_samples, embed_dim].
        labels: Label array [n_samples].

    Returns:
        Dictionary with geometric metrics.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("EMBEDDING GEOMETRY ANALYSIS")
    logger.info("=" * DIVIDER_LENGTH)

    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    logger.info("Embedding norms:")
    logger.info("  Mean: %.3f (ideal: 1.0 for MCR2)", mean_norm)
    logger.info("  Std: %.3f (ideal: 0.0)", std_norm)

    # Normalize for subspace analysis
    embeddings_norm = embeddings / (norms[:, np.newaxis] + 1e-8)

    # Compute class means (subspace directions)
    unique_labels = np.unique(labels)
    class_means = {}
    for label in unique_labels:
        class_embeddings = embeddings_norm[labels == label]
        class_means[label] = class_embeddings.mean(axis=0)
        class_means[label] /= np.linalg.norm(class_means[label]) + 1e-8

    # Check orthogonality of class subspaces
    logger.info("")
    logger.info("Class subspace orthogonality (ideal: cos=0 for orthogonal):")
    orthogonality_scores = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i + 1 :]:
            cos_sim = np.dot(class_means[label1], class_means[label2])
            orthogonality_scores.append(abs(cos_sim))
            logger.info(
                "  %s <-> %s: cos=%.3f",
                LABEL_NAMES.get(label1, str(label1)),
                LABEL_NAMES.get(label2, str(label2)),
                cos_sim,
            )

    mean_orthogonality = np.mean(orthogonality_scores) if orthogonality_scores else 0.0

    # Check within-class alignment
    logger.info("")
    logger.info("Within-class alignment (ideal: cos=1.0 for tight clusters):")
    alignment_scores = {}
    for label in unique_labels:
        class_embeddings = embeddings_norm[labels == label]
        if len(class_embeddings) > 1:
            # Compute mean cosine similarity to class mean
            cos_sims = class_embeddings @ class_means[label]
            mean_cos = np.mean(cos_sims)
            alignment_scores[label] = mean_cos
            logger.info("  %s: %.3f", LABEL_NAMES.get(label, str(label)), mean_cos)
        else:
            alignment_scores[label] = 1.0
            logger.info("  %s: N/A (single sample)", LABEL_NAMES.get(label, str(label)))

    # Compute effective dimension (using SVD)
    logger.info("")
    logger.info("Embedding space dimensionality:")
    _u, s, _ = np.linalg.svd(embeddings_norm, full_matrices=False)
    # Effective dimension via participation ratio
    s_normalized = s / s.sum()
    effective_dim = 1.0 / np.sum(s_normalized**2)
    logger.info("  Effective dimension: %.1f / %d", effective_dim, embeddings.shape[1])
    logger.info("  Top 5 singular values: %s", ", ".join(f"{v:.3f}" for v in s[:5]))

    logger.info("=" * DIVIDER_LENGTH)

    return {
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "mean_orthogonality": mean_orthogonality,
        "alignment_scores": alignment_scores,
        "effective_dimension": effective_dim,
    }


def evaluate_embeddings_file(
    embeddings_path: Path,
    output_dir: Path,
    skip_umap: bool = False,
) -> dict:
    """Run full evaluation on a single embeddings file.

    Args:
        embeddings_path: Path to embeddings .npz file.
        output_dir: Directory for output plots.
        skip_umap: Skip UMAP visualization.

    Returns:
        Dictionary with all evaluation metrics.

    """
    # Load embeddings
    embeddings, token_addresses, labels, trade_date = load_embeddings(embeddings_path)

    # Filter to labeled only
    embeddings, labels, token_addresses = filter_labeled_embeddings(
        embeddings, labels, token_addresses,
    )

    if len(embeddings) < 5:
        logger.warning(
            "Insufficient labeled samples (%d), skipping eval", len(embeddings),
        )
        return {"error": "insufficient_samples", "n_samples": len(embeddings)}

    results = {"trade_date": trade_date, "n_samples": len(embeddings)}

    # 1. Linear probe
    probe_results = evaluate_linear_probe(embeddings, labels)
    results["linear_probe"] = probe_results

    # 2. Cluster metrics
    cluster_results = compute_cluster_metrics(embeddings, labels)
    results["cluster_metrics"] = cluster_results

    # 3. Geometry analysis
    geometry_results = analyze_embedding_geometry(embeddings, labels)
    results["geometry"] = geometry_results

    # 4. UMAP visualization
    if not skip_umap:
        umap_path = visualize_umap(
            embeddings, labels, token_addresses, trade_date, output_dir,
        )
        results["umap_path"] = str(umap_path)

    return results


def evaluate_all_embeddings(
    embeddings_dir: Path,
    output_dir: Path,
    skip_umap: bool = False,
) -> None:
    """Evaluate all embedding files in a directory.

    Args:
        embeddings_dir: Directory containing .npz embedding files.
        output_dir: Directory for output plots.
        skip_umap: Skip UMAP visualization.

    """
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("LDR-TGN EMBEDDING EVALUATION")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("Embeddings directory: %s", embeddings_dir)
    logger.info("Output directory: %s", output_dir)

    # Find all embedding files
    embedding_files = sorted(embeddings_dir.glob("ldr_embeddings_*.npz"))

    if not embedding_files:
        logger.error("No embedding files found in %s", embeddings_dir)
        return

    logger.info("Found %d embedding files", len(embedding_files))

    # Evaluate each file
    all_results = []
    for emb_file in embedding_files:
        logger.info("")
        logger.info("#" * DIVIDER_LENGTH)
        logger.info("Processing: %s", emb_file.name)
        logger.info("#" * DIVIDER_LENGTH)

        results = evaluate_embeddings_file(emb_file, output_dir, skip_umap)
        all_results.append(results)

    # Summary across all dates
    logger.info("")
    logger.info("=" * DIVIDER_LENGTH)
    logger.info("SUMMARY ACROSS ALL DATES")
    logger.info("=" * DIVIDER_LENGTH)

    valid_results = [r for r in all_results if "error" not in r]
    if not valid_results:
        logger.warning("No valid results to summarize")
        return

    accuracies = [
        r["linear_probe"]["accuracy"]
        for r in valid_results
        if "accuracy" in r["linear_probe"]
    ]
    silhouettes = [
        r["cluster_metrics"]["silhouette"]
        for r in valid_results
        if "silhouette" in r["cluster_metrics"]
    ]

    if accuracies:
        logger.info("Linear Probe Accuracy:")
        logger.info("  Mean: %.2f%%", np.mean(accuracies) * 100)
        logger.info("  Std: %.2f%%", np.std(accuracies) * 100)
        logger.info("  Min: %.2f%%", np.min(accuracies) * 100)
        logger.info("  Max: %.2f%%", np.max(accuracies) * 100)

    if silhouettes:
        logger.info("")
        logger.info("Silhouette Score:")
        logger.info("  Mean: %.3f", np.mean(silhouettes))
        logger.info("  Std: %.3f", np.std(silhouettes))

    logger.info("=" * DIVIDER_LENGTH)


@app.command()
def main(
    embeddings_dir: Path = typer.Option(
        Path("data/ldr_embeddings"),
        "--embeddings-dir",
        "-e",
        help="Directory containing embedding .npz files",
    ),
    output_dir: Path = typer.Option(
        Path("data/ldr_embeddings/eval"),
        "--output-dir",
        "-o",
        help="Directory to save evaluation outputs",
    ),
    single_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Evaluate a single embedding file instead of all",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help="Skip UMAP visualization (faster)",
    ),
) -> None:
    """Evaluate LDR-TGN embeddings for linear discriminability.

    Runs linear probe, cluster metrics, geometry analysis, and UMAP visualization.
    """
    if single_file is not None:
        if not single_file.exists():
            logger.error("File not found: %s", single_file)
            raise typer.Exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_embeddings_file(single_file, output_dir, skip_umap)
    else:
        if not embeddings_dir.exists():
            logger.error("Directory not found: %s", embeddings_dir)
            raise typer.Exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_all_embeddings(embeddings_dir, output_dir, skip_umap)


if __name__ == "__main__":
    app()
