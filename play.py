from simulator import *


class Play:
    _answer = ""
    _guesses = dict()
    _score = 1
    _possibilities = []
    _entropies = []
    _expected_scores = []
    _pattern_dist = None
    _finished = False

    @property
    def expected_score(self):
        return dict(zip(self._possibilities, self._expected_scores))

    @property
    def pattern_distribution(self):
        return (self._pattern_dist > 0).sum(1)

    def reset(self):
        self._answer = ""
        self._guesses = dict()
        self._score = 1
        self._entropies = []
        self._expected_scores = []
        self._pattern_dist = None
        self._finished = False

    def play(self):
        self.reset()

        words = get_all_words()
        self._possibilities = words
        # self.entropies = get_entropies(words, words)
        self._expected_scores = get_score_lower_bounds(words, words)
        self._pattern_dist = get_pattern_distributions(words, self._possibilities, np.ones(len(words), dtype=np.uint8))

        self._answer = words[random.randint(0, len(words) - 1)]

    def guess(self, guess):
        if len(guess) != 8:
            print("Answers must have exactly 8 characters.")
            return

        if guess == self._answer:
            self._finished = True
            print("Congratulations!! You've found the correct answer!")

        pattern = get_pattern(guess, self._answer)
        self._guesses[guess] = pattern_to_string(pattern)
        print(self._guesses)

        words = get_all_words()
        self._possibilities = get_possible_words(guess, pattern, self._possibilities)
        # self.entropies = get_entropies(words, self.possibilities)
        self._pattern_dist = get_pattern_distributions(words, self._possibilities, np.ones(len(self._possibilities), dtype=np.uint8))
        self._expected_scores = get_score_lower_bounds(words, self._possibilities)
        self._score += 1

        if self._score > 6:
            print("Don't worry, you can always play.py again!\n The correct answer was: {}".format(self._answer))
            self._finished = True



