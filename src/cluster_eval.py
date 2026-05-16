"""
cluster_eval.py — Internal Clustering Evaluation (Project 2)
=============================================================
Evaluates clustering quality using the Dunn Index, an internal measure
that requires no external reference data (Lecture 10, §6).

The Dunn Index rewards:
  - High inter-cluster distance (clusters far apart)
  - Low intra-cluster diameter (clusters tightly packed)

A higher Dunn Index indicates better clustering quality.

Formula (Lecture 10):
    Dunn = min_inter_dist(Ci, Cj) / max_intra_diam(Ck)

Where distances are derived from the similarity matrix as:
    dist(a, b) = 1 - sim(a, b)

NOTE: Precision, Recall, and F-value have been intentionally removed.
Those metrics required a geographic reference clustering (ground truth),
which is external, subjective, and inconsistent with pure unsupervised
evaluation. The Dunn Index is fully self-contained and metric-agnostic.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Holds the result of an internal cluster evaluation pass."""
    dunn_index: float          # Higher is better; > 1 means well-separated
    min_inter_dist: float      # Smallest distance between any two clusters
    max_intra_diam: float      # Largest diameter (spread) among all clusters
    n_clusters: int            # Number of clusters evaluated
    n_objects: int             # Total number of clustered objects


# ---------------------------------------------------------------------------
# Distance helpers (derived from similarity)
# ---------------------------------------------------------------------------

def _dist(a: str, b: str, matrix: dict) -> float:
    """Convert similarity to distance: dist = 1 - sim."""
    return 1.0 - matrix[a][b]


def _inter_cluster_distance(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    """
    Minimum inter-cluster distance between two clusters.
    Uses the Single-Link (min) method — most conservative lower bound,
    making the Dunn Index harder to inflate artificially.
    """
    return min(
        _dist(a, b, matrix)
        for a in cluster_a
        for b in cluster_b
    )


def _intra_cluster_diameter(cluster: list[str], matrix: dict) -> float:
    """
    Cluster diameter: maximum distance between any two objects in the cluster.
    A singleton cluster has diameter 0 (no pairs to compare).
    """
    if len(cluster) < 2:
        return 0.0
    return max(
        _dist(a, b, matrix)
        for i, a in enumerate(cluster)
        for b in cluster[i + 1:]
    )


# ---------------------------------------------------------------------------
# Public evaluation function
# ---------------------------------------------------------------------------

def evaluate(clusters: list[list[str]], matrix: dict) -> EvaluationResult:
    """
    Compute the Dunn Index for a given set of clusters.

    Args:
        clusters: List of clusters, each a list of country names.
        matrix:   Pairwise similarity matrix as a nested dict.

    Returns:
        EvaluationResult with the Dunn Index and its components.

    Raises:
        ValueError: If fewer than 2 clusters are provided (Dunn undefined).
    """
    # Remove empty clusters defensively
    non_empty = [c for c in clusters if c]

    if len(non_empty) < 2:
        raise ValueError(
            "Dunn Index requires at least 2 non-empty clusters."
        )

    # Compute all pairwise inter-cluster distances
    inter_distances = [
        _inter_cluster_distance(non_empty[i], non_empty[j], matrix)
        for i in range(len(non_empty))
        for j in range(i + 1, len(non_empty))
    ]

    # Compute all intra-cluster diameters
    intra_diameters = [
        _intra_cluster_diameter(cluster, matrix)
        for cluster in non_empty
    ]

    min_inter = min(inter_distances)
    max_intra = max(intra_diameters)

    # Guard against degenerate case (all singletons → max_intra = 0)
    dunn = (min_inter / max_intra) if max_intra > 0.0 else float("inf")

    total_objects = sum(len(c) for c in non_empty)

    return EvaluationResult(
        dunn_index=round(dunn, 4),
        min_inter_dist=round(min_inter, 4),
        max_intra_diam=round(max_intra, 4),
        n_clusters=len(non_empty),
        n_objects=total_objects,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_evaluation(result: EvaluationResult, algorithm_name: str) -> None:
    """Print a formatted evaluation summary to stdout."""
    print(f"\n{algorithm_name} Evaluation (Dunn Index):")
    print(f"  Clusters evaluated    : {result.n_clusters}")
    print(f"  Objects clustered     : {result.n_objects}")
    print(f"  Min inter-cluster dist: {result.min_inter_dist:.4f}")
    print(f"  Max intra-cluster diam: {result.max_intra_diam:.4f}")
    print(f"  Dunn Index            : {result.dunn_index:.4f}  (higher is better)")
