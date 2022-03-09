import os.path

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)

WORD_LIST_FILE = os.path.join(DATA_DIR, "words.txt")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")

PATTERN_MATRIX = os.path.join(DATA_DIR, "words.txt")