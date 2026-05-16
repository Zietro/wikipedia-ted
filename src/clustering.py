import random
from dataclasses import dataclass, field


@dataclass
class MergeStep:
    cluster_a: list[str]
    cluster_b: list[str]
    similarity: float
    merged: list[str]


@dataclass
class Dendrogram:
    merges: list[MergeStep] = field(default_factory=list)

    def cut_at_threshold(self, threshold: float) -> list[list[str]]:
        clusters = []
        remaining = set()
        for merge in self.merges:
            if merge.similarity >= threshold:
                remaining.add(tuple(sorted(merge.merged)))
            else:
                for member in merge.cluster_a:
                    remaining.discard(tuple(sorted(merge.cluster_a)))
                for member in merge.cluster_b:
                    remaining.discard(tuple(sorted(merge.cluster_b)))
                remaining.add(tuple(sorted(merge.cluster_a)))
                remaining.add(tuple(sorted(merge.cluster_b)))

        seen = set()
        result = []
        for cluster in remaining:
            key = tuple(sorted(cluster))
            if key not in seen:
                seen.add(key)
                result.append(list(cluster))
        return result

    def cut_at_k(self, k: int) -> list[list[str]]:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        all_countries = self.merges[-1].merged if self.merges else []
        if k >= len(all_countries):
            return [[c] for c in all_countries]

        active: list[list[str]] = [[c] for c in all_countries]

        for merge in self.merges:
            if len(active) <= k:
                break
            a_key = tuple(sorted(merge.cluster_a))
            b_key = tuple(sorted(merge.cluster_b))
            active = [c for c in active if tuple(sorted(c)) not in (a_key, b_key)]
            active.append(merge.merged)

        return active


@dataclass
class AgglomerativeResult:
    dendrogram: Dendrogram
    flat_clusters: list[list[str]]
    linkage: str = "average"


@dataclass
class KMeansResult:
    clusters: list[list[str]]
    medoids: list[str]
    intra_cluster_similarity: float
    k: int
    iterations_used: int


def _average_link_similarity(
    cluster_a: list[str],
    cluster_b: list[str],
    matrix: dict,
) -> float:
    total = sum(matrix[a][b] for a in cluster_a for b in cluster_b)
    return total / (len(cluster_a) * len(cluster_b))


def agglomerative(
    matrix: dict,
    countries: list[str],
    k: int | None = None,
    threshold: float | None = None,
) -> AgglomerativeResult:
    if k is None and threshold is None:
        raise ValueError("Provide either k or threshold to determine where to cut.")

    clusters: list[list[str]] = [[c] for c in countries]
    dendrogram = Dendrogram()

    while len(clusters) > 1:
        best_sim = -1.0
        best_i = 0
        best_j = 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = _average_link_similarity(clusters[i], clusters[j], matrix)
                if sim > best_sim:
                    best_sim = sim
                    best_i = i
                    best_j = j

        merged = clusters[best_i] + clusters[best_j]
        dendrogram.merges.append(MergeStep(
            cluster_a=list(clusters[best_i]),
            cluster_b=list(clusters[best_j]),
            similarity=round(best_sim, 4),
            merged=merged,
        ))

        clusters = [c for idx, c in enumerate(clusters) if idx not in (best_i, best_j)]
        clusters.append(merged)

    if k is not None:
        flat = dendrogram.cut_at_k(k)
    else:
        flat = dendrogram.cut_at_threshold(threshold)

    return AgglomerativeResult(dendrogram=dendrogram, flat_clusters=flat)


def _compute_medoid(cluster: list[str], matrix: dict) -> str:
    best_country = cluster[0]
    best_avg = -1.0
    for candidate in cluster:
        avg = sum(matrix[candidate][other] for other in cluster) / len(cluster)
        if avg > best_avg:
            best_avg = avg
            best_country = candidate
    return best_country


def _assign_to_medoids(
    countries: list[str],
    medoids: list[str],
    matrix: dict,
) -> list[list[str]]:
    clusters: list[list[str]] = [[] for _ in medoids]
    for country in countries:
        best_idx = max(range(len(medoids)), key=lambda i: matrix[country][medoids[i]])
        clusters[best_idx].append(country)
    return clusters


def _total_intra_similarity(clusters: list[list[str]], medoids: list[str], matrix: dict) -> float:
    total = 0.0
    for cluster, medoid in zip(clusters, medoids):
        total += sum(matrix[c][medoid] for c in cluster)
    return round(total, 4)


def _kmeans_single_run(
    matrix: dict,
    countries: list[str],
    k: int,
    max_iterations: int,
    seed: int,
) -> KMeansResult:
    rng = random.Random(seed)
    medoids = rng.sample(countries, k)

    for iteration in range(1, max_iterations + 1):
        clusters = _assign_to_medoids(countries, medoids, matrix)
        clusters = [c if c else [medoids[i]] for i, c in enumerate(clusters)]

        new_medoids = [_compute_medoid(cluster, matrix) for cluster in clusters]

        if new_medoids == medoids:
            return KMeansResult(
                clusters=clusters,
                medoids=new_medoids,
                intra_cluster_similarity=_total_intra_similarity(clusters, new_medoids, matrix),
                k=k,
                iterations_used=iteration,
            )
        medoids = new_medoids

    clusters = _assign_to_medoids(countries, medoids, matrix)
    clusters = [c if c else [medoids[i]] for i, c in enumerate(clusters)]
    return KMeansResult(
        clusters=clusters,
        medoids=medoids,
        intra_cluster_similarity=_total_intra_similarity(clusters, medoids, matrix),
        k=k,
        iterations_used=max_iterations,
    )


def kmeans(
    matrix: dict,
    countries: list[str],
    k: int,
    max_iterations: int = 100,
    n_runs: int = 5,
) -> KMeansResult:
    if k <= 0 or k > len(countries):
        raise ValueError(f"k must be between 1 and {len(countries)}.")

    best_result = None
    for run in range(n_runs):
        result = _kmeans_single_run(matrix, countries, k, max_iterations, seed=run)
        if best_result is None or result.intra_cluster_similarity > best_result.intra_cluster_similarity:
            best_result = result

    return best_result