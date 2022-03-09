import os.path

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)

POSSIBLE_WORDS_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")

PATTERN_MATRIX = os.path.join(DATA_DIR, "possible_words.txt")