from tqdm import tqdm as ProgressDisplay
from scipy.stats import entropy
import itertools as it
import random

import numpy as np

from constants import *

WORD_LIST = []
PATTERN_GRID_DATA = dict()


def get_all_words():
    global WORD_LIST
    if WORD_LIST:
        return WORD_LIST

    with open(WORD_LIST_FILE, "r") as f:
        possible_words = [line.rstrip() for line in f]
    f.close()

    WORD_LIST += possible_words
    return WORD_LIST


def word_to_int_array(word):
    return np.array([ord(l) for l in word], dtype=np.uint8)


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
    words = get_all_words()
    pattern_matrix = generate_pattern_matrix(words)

    # Save to file
    np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    return pattern_matrix


def get_pattern_matrix(words1, words2):
    if not PATTERN_GRID_DATA:
        if not os.path.exists(PATTERN_MATRIX_FILE):
            generate_full_pattern_matrix()
        PATTERN_GRID_DATA['grid'] = np.load(PATTERN_MATRIX_FILE)
        PATTERN_GRID_DATA['words_to_index'] = dict(zip(
            get_all_words(), it.count()
        ))

    full_grid = PATTERN_GRID_DATA['grid']
    words_to_index = PATTERN_GRID_DATA['words_to_index']

    indices1 = [words_to_index[w] for w in words1]
    indices2 = [words_to_index[w] for w in words2]

    return full_grid[np.ix_(indices1, indices2)]


def get_pattern(guess, answer):
    return get_pattern_matrix([guess], [answer])[0, 0]


def get_possible_words(guess, pattern, possibilities):
    all_patterns = get_pattern_matrix([guess], possibilities).flatten()
    return list(np.array(possibilities)[all_patterns == pattern])


def get_pattern_distributions(allowed_words, possible_words, weights):
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^8 nerdle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.
    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    For example if A = words[a] and B = words[b] form the same pattern with C=words[c]
    then when C is the answer and A, B the possibilities, then the probabilities will be
    lower in the distribution since both puts the same amount of information for the
    given answer.
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)

    n = len(allowed_words)
    distributions = np.zeros((n, 3**8))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def get_bucket_sizes(allowed_words, possible_words):
    """
    Returns a (len(allowed_words), 243) shape array reprenting the size of
    word buckets associated with each guess in allowed_words
    """
    weights = np.ones(len(possible_words))
    return get_pattern_distributions(allowed_words, possible_words, weights)


def get_bucket_counts(allowed_words, possible_words):
    """
    Returns the number of separate buckets that each guess in allowed_words
    would separate possible_words into
    """
    bucket_sizes = get_bucket_sizes(allowed_words, possible_words)
    return (bucket_sizes > 0).sum(1)


def get_score_lower_bounds(allowed_words, possible_words):
    """
    Assuming a uniform distribution on how likely each element
    of possible_words is, this gives a lower bound on the
    possible score for each word in allowed_words
    """
    bucket_counts = get_bucket_counts(allowed_words, possible_words)
    N = len(possible_words)
    # Probabilities of getting it in 1
    p1s = np.array([w in possible_words for w in allowed_words]) / N
    # Probabilities of getting it in 2
    p2s = bucket_counts / N - p1s
    # Otherwise, assume it's gotten in 3 (which is optimistics)
    p3s = 1 - bucket_counts / N
    return p1s + 2 * p2s + 3 * p3s


def get_entropies(allowed_words, possible_words):
    weights = np.ones(len(possible_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy(distributions, base=2, axis=0)


def optimal_guess(allowed_words, possible_words):
    """
    if len(possible_words) == 1:
        return possible_words[0]
    ents = get_entropies(allowed_words, possible_words)
    return allowed_words[np.argmax(ents)]
    """

    expected_scores = get_score_lower_bounds(allowed_words, possible_words)
    return allowed_words[np.argmin(expected_scores)]


def pattern_to_int_list(pattern):
    result = []
    curr = pattern
    for x in range(8):
        result.append(curr % 3)
        curr = curr // 3
    return result


def pattern_to_string(pattern):
    d = {0: "⬛", 1: "🟨", 2: "🟩"}
    return "".join(d[x] for x in pattern_to_int_list(pattern))


def simulate():
    words = get_all_words()

    first_guess = words[random.randint(0, len(words)-1)]

    patterns = []
    scores = np.zeros(0, dtype=int)
    scores_dist = dict()

    for answer in ProgressDisplay(words, leave=False, desc=" Trying all nerdle answers"):
        score = 0
        guesses = []
        possibilities = words
        guess = first_guess

        while answer != guess:
            guesses.append(guess)
            pattern = get_pattern(guess, answer)
            patterns.append(pattern)

            possibilities = get_possible_words(guess, pattern, possibilities)
            guess = optimal_guess(words, possibilities)
            score += 1

        scores = np.append(scores, [score])
        score_dist = [
            int((scores == i).sum())
            for i in range(1, scores.max() + 1)
        ]

    final_result = dict(
        score_distribution=score_dist,
        # total_guesses=int(total_guesses),
        average_score=float(scores.mean()),
        # game_results=game_results,
    )

    return final_result

