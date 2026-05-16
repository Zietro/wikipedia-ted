from dataclasses import dataclass


REFERENCE_CLUSTERS = [
    ["France", "Germany", "United Kingdom", "Italy", "Spain", "Switzerland", "Sweden"],
    ["Brazil", "Colombia", "Mexico", "Argentina"],
    ["Lebanon", "Syria", "Jordan", "Egypt", "Saudi Arabia", "Algeria", "Turkey"],
    ["China", "Japan", "South Korea", "Indonesia", "India"],
    ["Nigeria", "South Africa", "Kenya", "Ethiopia"],
    ["United States", "Canada", "Australia", "New Zealand"],
    ["Russia"],
]


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f_value: float
    correctly_grouped: int
    total_grouped: int
    total_reference_pairs: int


def _get_pairs(cluster: list[str]) -> set[frozenset]:
    pairs = set()
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            pairs.add(frozenset([cluster[i], cluster[j]]))
    return pairs


def _build_reference_pairs(countries: list[str]) -> set[frozenset]:
    pairs = set()
    for cluster in REFERENCE_CLUSTERS:
        members = [c for c in cluster if c in countries]
        pairs |= _get_pairs(members)
    return pairs


def _build_algorithm_pairs(clusters: list[list[str]]) -> set[frozenset]:
    pairs = set()
    for cluster in clusters:
        pairs |= _get_pairs(cluster)
    return pairs


def evaluate(clusters: list[list[str]], countries: list[str]) -> EvaluationResult:
    reference_pairs = _build_reference_pairs(countries)
    algorithm_pairs = _build_algorithm_pairs(clusters)

    correctly_grouped = len(reference_pairs & algorithm_pairs)
    total_grouped = len(algorithm_pairs)
    total_reference = len(reference_pairs)

    precision = correctly_grouped / total_grouped if total_grouped > 0 else 0.0
    recall = correctly_grouped / total_reference if total_reference > 0 else 0.0
    f_value = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return EvaluationResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f_value=round(f_value, 4),
        correctly_grouped=correctly_grouped,
        total_grouped=total_grouped,
        total_reference_pairs=total_reference,
    )


def print_evaluation(result: EvaluationResult, algorithm_name: str) -> None:
    print(f"\n{algorithm_name} Evaluation:")
    print(f"  Correctly grouped pairs : {result.correctly_grouped}")
    print(f"  Total grouped pairs     : {result.total_grouped}")
    print(f"  Total reference pairs   : {result.total_reference_pairs}")
    print(f"  Precision               : {result.precision:.4f}")
    print(f"  Recall                  : {result.recall:.4f}")
    print(f"  F-value                 : {result.f_value:.4f}")