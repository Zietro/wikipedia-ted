from src.matrix_builder import build_matrix

WORKING_SET = [
    "Lebanon", "Syria", "Jordan", "Egypt", "Saudi Arabia", "Turkey",
    "France", "Germany", "United Kingdom", "Italy", "Spain", "Switzerland",
    "United States", "Canada", "Mexico", "Brazil", "Argentina", "Colombia",
    "China", "Japan", "India", "South Korea", "Indonesia",
    "Nigeria", "South Africa", "Kenya", "Ethiopia",
    "Australia", "New Zealand",
    "Russia", "Sweden",
]

matrix = build_matrix(WORKING_SET, overwrite=True)