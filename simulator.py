import itertools as it
import numpy as np

from constants import *

POSSIBLE_WORDS = []
PATTERN_GRID_DATA = dict()

def get_possible_words():
    global POSSIBLE_WORDS
    if POSSIBLE_WORDS:
        return POSSIBLE_WORDS

    with open(POSSIBLE_WORDS_LIST_FILE, "r") as f:
        possible_words = [line.rstrip() for line in f]
    f.close()

    POSSIBLE_WORDS += possible_words
    return POSSIBLE_WORDS


def word_to_int_array(word):
    return np.array([ord(l)for l in word], dtype=np.uint8)

def generate_pattern_matrix(words):
    n = len(words)
    word_length = len(words[0])
    int_words = np.array([word_to_int_array(w) for w in words], dtype=np.uint8)

    """
        So here we create an 8*8 boolean matrix, by comparing every word against 
        every word in the list. If the letter are equal and are in the same 
        position in both words, then some cell in the main diagonal will be 
        set to true. If both words share a letter in common but not in the same
        position, some cell below or above the main diagonal will be set to true.
    """

    word_equality = np.zeros((n, n, word_length, word_length), dtype=bool)
    for i, j in it.product(range(word_length), range(word_length)):
        word_equality[:, :, i, j] = np.equal.outer(int_words[:, i], int_words[:, j])

    """
        Now that we have the matrix with all the comparisons, we need to create the
        pattern arrays between two words and set a number if it is an exact match,
        a miss or if it does not exist.
    """
    patterns = np.zeros((n, n, word_length), dtype=np.uint8)

    # First we look for exact matches and remove all "bad misses", and mark
    # those positions as green
    for i in range(word_length):
        matches = word_equality[:, :, i, i].flatten()
        patterns[:, :, i].flat[matches] = 2

        for k in range(word_length):
            word_equality[:, :, k, i].flat[matches] = False
            word_equality[:, :, i, k].flat[matches] = False

    # Now misplaced letters, and mark those positions as yellow
    for i, j in it.product(range(word_length), range(word_length)):
        matches = word_equality[:, :, i, j].flatten()
        patterns[:, :, i].flat[matches] = 1

        for k in range(word_length):
            word_equality[:, :, k, j].flat[matches] = False
            word_equality[:, :, i, k].flat[matches] = False

    return np.dot(
        patterns,
        (3 ** np.arange(word_length)).astype(np.uint8)
    )


def generate_full_pattern_matrix():
    words = get_possible_words()
    pattern_matrix = generate_pattern_matrix(words)

    # Save to file
    np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID_DATA:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            """
            log.info("\n".join([
                "Generating pattern matrix. This takes a minute, but",
                "the result will be saved to file so that it only",
                "needs to be computed once.",
            ]))
            """
            generate_full_pattern_matrix()
        PATTERN_GRID_DATA['grid'] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA['words_to_index'] = dict(zip(
            get_possible_words(), it.count()
        ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]
    return full_grid[np.ix_(indices1, indices2)]


def get_pattern(guess, answer):
    return get_pattern_matrix([guess], [answer])[0, 0]


if __name__ == "__main__":
    print(get_pattern("91+9=100", "9+91=100"))