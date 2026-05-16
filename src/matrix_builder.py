import json
import os

from src.preprocessor import load_tree
from src.ted import compute_ted
from models.tree import TreeUtils

MATRIX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "similarity_matrix.json")

WORKING_SET = [
    "Lebanon", "Syria", "Jordan", "Egypt", "Saudi Arabia", "Turkey",
    "France", "Germany", "United Kingdom", "Italy", "Spain", "Switzerland",
    "United States", "Canada", "Mexico", "Brazil", "Argentina",
    "China", "Japan", "India", "South Korea", "Indonesia",
    "Nigeria", "South Africa", "Kenya", "Ethiopia",
    "Australia", "New Zealand",
    "Russia", "Sweden",
]


def build_matrix(countries: list[str], overwrite: bool = False) -> dict:
    if not overwrite and os.path.exists(MATRIX_PATH):
        print(f"Loading cached matrix from {MATRIX_PATH}")
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Loading trees for {len(countries)} countries...")
    trees = {}
    skipped = []
    for country in countries:
        try:
            trees[country] = load_tree(country)
            print(f"  [OK] {country}")
        except Exception as e:
            skipped.append(country)
            print(f"  [SKIP] {country}: {e}")

    loaded = list(trees.keys())
    n = len(loaded)
    total_pairs = n * (n - 1) // 2

    print(f"\nComputing {total_pairs} pairs across {n} countries...")

    matrix = {c: {c: 1.0} for c in loaded}

    computed = 0
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = loaded[i], loaded[j]
            result = compute_ted(trees[c1], trees[c2], c1, c2)
            score = round(result.similarity, 4)
            matrix[c1][c2] = score
            matrix[c2][c1] = score
            computed += 1
            if computed % 10 == 0 or computed == total_pairs:
                print(f"  [{computed}/{total_pairs}] {c1} vs {c2}: {score:.4f}")

    validate_matrix(matrix, loaded)

    os.makedirs(os.path.dirname(MATRIX_PATH), exist_ok=True)
    with open(MATRIX_PATH, "w", encoding="utf-8") as f:
        json.dump({"countries": loaded, "skipped": skipped, "matrix": matrix}, f, indent=2)

    print(f"\nMatrix saved to {MATRIX_PATH}")
    return matrix


def validate_matrix(matrix: dict, countries: list[str]) -> None:
    n = len(countries)
    expected_pairs = n * (n - 1) // 2

    missing = [c for c in countries if c not in matrix]
    assert not missing, f"Missing countries in matrix: {missing}"

    bad_diagonal = [c for c in countries if matrix[c][c] != 1.0]
    assert not bad_diagonal, f"Diagonal not 1.0 for: {bad_diagonal}"

    asymmetric = [
        (c1, c2) for c1 in countries for c2 in countries
        if abs(matrix[c1][c2] - matrix[c2][c1]) > 1e-9
    ]
    assert not asymmetric, f"Asymmetric pairs: {asymmetric[:5]}"

    out_of_bounds = [
        (c1, c2) for c1 in countries for c2 in countries
        if not (0.0 <= matrix[c1][c2] <= 1.0)
    ]
    assert not out_of_bounds, f"Out of bounds scores: {out_of_bounds[:5]}"

    computed_pairs = sum(
        1 for i, c1 in enumerate(countries)
        for c2 in countries[i + 1:]
        if c1 in matrix and c2 in matrix[c1]
    )
    assert computed_pairs == expected_pairs, (
        f"Pair count mismatch: expected {expected_pairs}, found {computed_pairs}"
    )

    print(f"\nValidation passed: {n} countries, {computed_pairs} pairs, all checks OK.")


def load_matrix() -> tuple[dict, list[str]]:
    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(
            f"No matrix found at {MATRIX_PATH}. Run build_matrix() first."
        )
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["matrix"], data["countries"]


if __name__ == "__main__":
    build_matrix(WORKING_SET, overwrite=False)