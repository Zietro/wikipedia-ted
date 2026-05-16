from src.matrix_builder import build_matrix
from src.collector import UN_MEMBER_STATES

# We use UN_MEMBER_STATES instead of the limited WORKING_SET
matrix = build_matrix(UN_MEMBER_STATES, overwrite=True)